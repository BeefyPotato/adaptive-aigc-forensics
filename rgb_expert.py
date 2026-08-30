"""Frozen Community Forensics RGB expert and shared inference preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "config" / "community-forensics-models.json"
SUPPORTED_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
TRACK5_PARAMETER_LIMIT = 2_000_000_000


class RgbBackend(Protocol):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray: ...


@lru_cache(maxsize=1)
def load_model_metadata(path: Path = METADATA_PATH) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "community-forensics-models-v1":
        raise ValueError("Unsupported Community Forensics metadata schema.")
    for field in (
        "upstream_repository",
        "upstream_revision",
        "license",
        "preprocessing_version",
        "score_direction",
        "known_provenance_limitations",
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"Model metadata field {field} must be non-empty.")
    if not isinstance(metadata.get("numeric_tolerance"), (int, float)) or metadata["numeric_tolerance"] <= 0:
        raise ValueError("Model metadata numeric_tolerance must be positive.")
    for resolution in ("384", "224"):
        model = metadata.get("models", {}).get(resolution)
        if not isinstance(model, dict):
            raise ValueError(f"Missing Community Forensics {resolution} metadata.")
        if model.get("input_resolution") != int(resolution):
            raise ValueError(f"Community Forensics {resolution} input resolution is incompatible.")
        if not isinstance(model.get("parameter_count"), int) or model["parameter_count"] >= TRACK5_PARAMETER_LIMIT:
            raise ValueError(f"Community Forensics {resolution} exceeds the Track 5 parameter limit.")
        for field, length in (("revision", 40), ("sha256", 64)):
            value = model.get(field, "")
            if len(value) != length or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Community Forensics {resolution} {field} must be lowercase hexadecimal.")
    return metadata


def verify_checkpoint(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Checkpoint checksum mismatch: received {actual}; expected {expected_sha256}.")


def preprocess_image(path: Path | str, *, resolution: int = 384) -> np.ndarray:
    metadata = load_model_metadata()
    model = metadata["models"].get(str(resolution))
    if model is None:
        raise ValueError("Community Forensics resolution must be 384 or 224.")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        scale = model["resize_short_edge"] / min(width, height)
        resized = image.resize(
            (int(width * scale), int(height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
        left = (resized.width - resolution) // 2
        top = (resized.height - resolution) // 2
        cropped = resized.crop((left, top, left + resolution, top + resolution))
        rgb = np.asarray(cropped, dtype=np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray((chw - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32)


def _sigmoid(logit: float) -> float:
    if not math.isfinite(logit):
        raise ValueError("RGB expert produced a non-finite logit.")
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _predict_paths(
    paths: list[Path], backend: RgbBackend, *, resolution: int, batch_size: int
) -> list[tuple[Path, float, float]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    results = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        batch = np.stack([preprocess_image(path, resolution=resolution) for path in batch_paths])
        logits = np.asarray(backend.predict_logits(batch), dtype=np.float64).reshape(-1)
        if len(logits) != len(batch_paths):
            raise ValueError("RGB backend must return one logit per image.")
        for path, logit in zip(batch_paths, logits, strict=True):
            value = float(logit)
            results.append((path, value, _sigmoid(value)))
    return results


def discover_images(directory: Path | str) -> list[Path]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def infer_directory(
    directory: Path | str,
    backend: RgbBackend,
    *,
    resolution: int = 384,
    batch_size: int = 8,
) -> list[dict]:
    root = Path(directory).resolve()
    predictions = [
        {"image_path": path.relative_to(root).as_posix(), "pred": probability}
        for path, _, probability in _predict_paths(
            discover_images(root), backend, resolution=resolution, batch_size=batch_size
        )
    ]
    return predictions


def predict_experiment_observations(
    observations: Iterable[dict],
    backend: RgbBackend,
    *,
    resolution: int = 384,
    batch_size: int = 8,
) -> list[dict]:
    records = list(observations)
    for index, record in enumerate(records):
        missing = {"source_id", "variant_id", "image_path"} - record.keys()
        if missing:
            raise ValueError(f"Experiment observation {index} is missing {', '.join(sorted(missing))}.")
    paths = [Path(record["image_path"]).resolve() for record in records]
    predictions = _predict_paths(paths, backend, resolution=resolution, batch_size=batch_size)
    metadata = load_model_metadata()
    model = metadata["models"][str(resolution)]
    artifacts = [
        {
            "artifact_schema_version": "rgb-logit-v1",
            "preprocessing_version": metadata["preprocessing_version"],
            "checkpoint_revision": model["revision"],
            "source_id": record["source_id"],
            "variant_id": record["variant_id"],
            "rgb_logit": logit,
            "pred": probability,
            "numeric_tolerance": metadata["numeric_tolerance"],
        }
        for record, (_, logit, probability) in zip(records, predictions, strict=True)
    ]
    for artifact in artifacts:
        identity = json.dumps(
            {
                "artifact_schema_version": artifact["artifact_schema_version"],
                "checkpoint_revision": artifact["checkpoint_revision"],
                "preprocessing_version": artifact["preprocessing_version"],
                "variant_id": artifact["variant_id"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        artifact["cache_key"] = "rgb-cache-v1-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return artifacts


class CommunityForensicsBackend:
    """Lazy PyTorch backend so metadata/preprocessing work without the GPU stack."""

    def __init__(self, checkpoint: Path, *, resolution: int = 384, device: str = "auto"):
        try:
            import torch
            import torch.nn as nn
            import timm
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-rgb.txt before loading Community Forensics."
            ) from error

        metadata = load_model_metadata()
        model_metadata = metadata["models"].get(str(resolution))
        if model_metadata is None:
            raise ValueError("Community Forensics resolution must be 384 or 224.")
        verify_checkpoint(checkpoint, model_metadata["sha256"])
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        architecture = f"vit_small_patch16_{resolution}.augreg_in21k_ft_in1k"
        class ViTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.vit = timm.create_model(architecture, pretrained=False)
                self.vit.head = nn.Linear(384, 1)

            def forward(self, inputs):
                return self.vit(inputs)

        model = ViTClassifier()
        state = load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        model.eval().to(selected_device)
        self._torch = torch
        self._model = model
        self._device = selected_device

    @property
    def device(self) -> str:
        return self._device

    @property
    def peak_memory_bytes(self) -> int | None:
        if not self._device.startswith("cuda"):
            return None
        return int(self._torch.cuda.max_memory_allocated(self._device))

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(batch).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(tensor).reshape(-1)
        return logits.detach().cpu().numpy()


def download_checkpoint(*, resolution: int = 384, cache_directory: Path | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install requirements-rgb.txt before downloading checkpoints.") from error
    model = load_model_metadata()["models"][str(resolution)]
    path = Path(
        hf_hub_download(
            repo_id=model["repository"],
            filename=model["checkpoint_name"],
            revision=model["revision"],
            cache_dir=str(cache_directory) if cache_directory else None,
        )
    )
    verify_checkpoint(path, model["sha256"])
    return path
