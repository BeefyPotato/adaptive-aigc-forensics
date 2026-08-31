"""Fail-closed directory inference for the frozen Issue #7 submission policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from fusion_pipeline import calibrated_logit, read_static_fallback_generation
from rgb_expert import discover_images, preprocess_image
from safe_output import atomic_write_bytes
from shared_observation import prepare_shared_expert_rgb
from signal_expert import extract_signal_representation


GENERATION_REVISION = "static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181"
BUNDLE_REVISION = "static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179"
BUNDLE_SHA256 = "9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2"
RGB_WEIGHT = 0.677
SIGNAL_WEIGHT = 0.323
SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}


class LogitBackend(Protocol):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray: ...


def _sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Frozen fusion produced a non-finite logit.")
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def _light_calibrated_logit(value: float, calibrator: dict) -> float:
    slope, intercept = calibrator.get("slope"), calibrator.get("intercept")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in (value, slope, intercept)):
        raise ValueError("Calibrator and expert logits must be finite numbers.")
    return float(slope) * float(value) + float(intercept)


def frozen_probability(rgb_logit: float, signal_logit: float, bundle: dict, *, validate_calibrators: bool = True) -> float:
    if bundle.get("selected_fallback_type") != "learned-static-fusion":
        raise ValueError("Frozen submission policy must be learned-static-fusion.")
    weight = bundle.get("static_weight", {})
    if weight.get("rgb_weight") != RGB_WEIGHT or weight.get("signal_weight") != SIGNAL_WEIGHT:
        raise ValueError("Frozen submission weights are stale or incompatible.")
    apply = calibrated_logit if validate_calibrators else _light_calibrated_logit
    rgb = apply(rgb_logit, bundle.get("rgb_calibrator", {}))
    signal = apply(signal_logit, bundle.get("signal_calibrator", {}))
    return _sigmoid(RGB_WEIGHT * rgb + SIGNAL_WEIGHT * signal)


def load_frozen_bundle(generation_directory: Path | str) -> dict:
    loaded = read_static_fallback_generation(
        generation_directory,
        expected_generation_revision=GENERATION_REVISION,
    )
    bundle = loaded["bundle"]
    bundle_path = Path(generation_directory).absolute() / "static-fallback-bundle.json"
    if bundle.get("bundle_revision") != BUNDLE_REVISION:
        raise ValueError("Frozen submission bundle revision is stale or incompatible.")
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != BUNDLE_SHA256:
        raise ValueError("Frozen submission bundle checksum is stale or incompatible.")
    frozen_probability(0.0, 0.0, bundle)
    provenance = bundle.get("provenance", {})
    if provenance.get("signal_experiment_profile") != "hackathon-v1":
        raise ValueError("Frozen signal experiment profile is stale or incompatible.")
    return bundle


def _preprocess_rgb(path: Path) -> np.ndarray:
    return preprocess_image(path, resolution=384)


def _preprocess_signal(path: Path) -> np.ndarray:
    rgb = prepare_shared_expert_rgb(path, resolution=384)
    return extract_signal_representation(rgb)["features"]


def _validate_input_paths(root: Path, paths: list[Path]) -> list[str]:
    relative = []
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
            name = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise ValueError("Supported input image resolves outside the input directory.") from error
        if name in seen:
            raise ValueError("Input images contain a duplicate normalized relative path.")
        seen.add(name)
        relative.append(name)
    return relative


def _backend_logits(backend: LogitBackend, batch: np.ndarray, expected: int, expert: str) -> np.ndarray:
    values = np.asarray(backend.predict_logits(batch), dtype=np.float64).reshape(-1)
    if len(values) != expected:
        raise ValueError(f"{expert} backend must return one logit per image.")
    if not np.isfinite(values).all():
        raise ValueError(f"{expert} backend produced a non-finite logit.")
    return values


def run_submission(
    image_directory: Path | str,
    generation_directory: Path | str,
    output_path: Path | str,
    rgb_backend: LogitBackend,
    signal_backend: LogitBackend,
    *,
    batch_size: int = 8,
) -> list[dict]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    requested_root = Path(image_directory).absolute()
    try:
        root = requested_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("Input directory does not exist.") from error
    if root != requested_root or not root.is_dir():
        if root != requested_root:
            raise ValueError("Input directory must not be redirected.")
        raise ValueError("Input path must be a directory.")
    bundle = load_frozen_bundle(generation_directory)  # validate every artifact before decoding
    paths = discover_images(root)
    names = _validate_input_paths(root, paths)
    records = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        try:
            rgb_batch = np.stack([_preprocess_rgb(path) for path in batch_paths])
            signal_batch = np.stack([_preprocess_signal(path) for path in batch_paths])
        except (OSError, ValueError) as error:
            raise ValueError("Supported input image is unreadable or corrupt.") from error
        rgb = _backend_logits(rgb_backend, rgb_batch, len(batch_paths), "RGB")
        signal = _backend_logits(signal_backend, signal_batch, len(batch_paths), "Signal")
        for index, (rgb_logit, signal_logit) in enumerate(zip(rgb, signal, strict=True), start=offset):
            records.append({"image_path": names[index], "pred": frozen_probability(float(rgb_logit), float(signal_logit), bundle, validate_calibrators=False)})
    payload = (json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    destination = Path(output_path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, payload)
    return records


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    if requested not in SUPPORTED_DEVICES:
        raise ValueError("Device must be one of auto, cpu, or cuda.")
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA was explicitly requested but is unavailable.")
    return "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
