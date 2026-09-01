"""Fail-closed directory inference for the frozen Issue #7 submission policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from fusion_pipeline import (
    CORRECTED_COMPLETION_SCHEMA,
    calibrated_logit,
    validate_bundle,
)
from rgb_expert import discover_images, load_model_metadata, preprocess_image
from safe_output import atomic_write_bytes
from signal_expert import decode_expert_rgb, extract_signal_representation


GENERATION_REVISION = "static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181"
BUNDLE_REVISION = "static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179"
BUNDLE_SHA256 = "9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2"
RGB_CHECKPOINT_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
SIGNAL_MODEL_SHA256 = "cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5"
RGB_WEIGHT = 0.677
SIGNAL_WEIGHT = 0.323
WEIGHT_ABSOLUTE_TOLERANCE = 1e-15
SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}
EXPECTED_PROVENANCE = {
    "corruption_version": "track5-corruption-v1+sharp-0.35.4",
    "manifest_schema_version": "track5-manifest-v1",
    "manifest_sha256": "c9ea2d3b616b37844d21602a95e5f90c824a692ad609d31bbe0b982c5f45228a",
    "rgb_checkpoint_revision": "6076002bf0d9dd37537f965ee2f06f826c333b61",
    "rgb_checkpoint_sha256": RGB_CHECKPOINT_SHA256,
    "rgb_preprocessing_version": "community-forensics-eval-v1",
    "rgb_score_direction": "positive-logit-means-ai-generated",
    "shared_observation_preprocessing_version": "shared-preprocessing-v1",
    "signal_acceptance_scope": "issue-6-timeboxed-acceptance",
    "signal_checkpoint_revision": "signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e",
    "signal_experiment_profile": "hackathon-v1",
    "signal_feature_extraction_version": "signal-feature-extraction-v1",
    "signal_normalization_revision": "signal-normalization-v1-25b16b78f7ecb5e02572e03650537e8b5e266f2f3e49a911a2ae2e2e11d45e80",
    "signal_representation_version": "signal-representation-v1",
    "signal_resolution": 384,
}
EXPECTED_GENERATION_ARTIFACTS = {
    "matched-fusion-training-logits.json",
    "matched-internal-validation-logits.json",
    "calibrated-fusion-training-logits.json",
    "calibrated-internal-validation-logits.json",
    "static-fallback-bundle.json",
}


class LogitBackend(Protocol):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray: ...


def _canonical_sha256(value: object) -> str:
    contents = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path, context: str) -> str:
    requested = path.absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{context} is missing.") from error
    if resolved != requested or not resolved.is_file():
        raise ValueError(f"{context} is redirected or invalid.")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"{context} cannot be read.") from error
    return digest.hexdigest()


def _read_direct_json(root: Path, name: str, context: str) -> tuple[dict, bytes]:
    path = root / name
    try:
        resolved = path.resolve(strict=True)
        if resolved != path or not resolved.is_file():
            raise ValueError(f"{context} is redirected or invalid.")
        contents = resolved.read_bytes()
        payload = json.loads(contents)
    except ValueError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is missing or invalid.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object.")
    return payload, contents


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
    observed_weights = (weight.get("rgb_weight"), weight.get("signal_weight"))
    expected_weights = (RGB_WEIGHT, SIGNAL_WEIGHT)
    if any(
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=WEIGHT_ABSOLUTE_TOLERANCE,
        )
        for observed, expected in zip(observed_weights, expected_weights, strict=True)
    ):
        raise ValueError("Frozen submission weights are stale or incompatible.")
    apply = calibrated_logit if validate_calibrators else _light_calibrated_logit
    rgb = apply(rgb_logit, bundle.get("rgb_calibrator", {}))
    signal = apply(signal_logit, bundle.get("signal_calibrator", {}))
    return _sigmoid(RGB_WEIGHT * rgb + SIGNAL_WEIGHT * signal)


def load_frozen_bundle(generation_directory: Path | str) -> dict:
    requested = Path(generation_directory).absolute()
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError("Frozen submission bundle directory is missing.") from error
    if root != requested or not root.is_dir():
        raise ValueError("Frozen submission bundle directory is redirected or invalid.")

    completion, _ = _read_direct_json(
        root, "static-fallback.complete.json", "Frozen submission receipt"
    )
    if completion.get("completion_schema_version") != CORRECTED_COMPLETION_SCHEMA:
        raise ValueError("Frozen submission receipt schema is incompatible.")
    completion_identity = dict(completion)
    revision = completion_identity.pop("generation_revision", None)
    expected_revision = "static-fallback-generation-v2-" + _canonical_sha256(
        completion_identity
    )
    if revision != expected_revision or revision != GENERATION_REVISION:
        raise ValueError("Frozen submission generation revision is stale or incompatible.")
    if _canonical_sha256(completion.get("provenance")) != _canonical_sha256(
        EXPECTED_PROVENANCE
    ):
        raise ValueError("Frozen submission receipt provenance is stale or incompatible.")
    bindings = completion.get("artifacts")
    if not isinstance(bindings, dict) or set(bindings) != EXPECTED_GENERATION_ARTIFACTS:
        raise ValueError("Frozen submission receipt artifact bindings are incomplete.")
    for name, binding in bindings.items():
        if (
            not isinstance(binding, dict)
            or binding.get("path") != name
            or not isinstance(binding.get("file_sha256"), str)
            or len(binding["file_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["file_sha256"])
        ):
            raise ValueError("Frozen submission receipt artifact binding is invalid.")

    bundle, bundle_bytes = _read_direct_json(
        root, "static-fallback-bundle.json", "Frozen submission bundle"
    )
    bundle_digest = hashlib.sha256(bundle_bytes).hexdigest()
    bundle_binding = bindings["static-fallback-bundle.json"]
    if (
        bundle_digest != BUNDLE_SHA256
        or bundle_digest != completion.get("bundle_sha256")
        or bundle_digest != bundle_binding.get("file_sha256")
    ):
        raise ValueError("Frozen submission bundle checksum is stale or incompatible.")
    if (
        bundle.get("bundle_revision") != BUNDLE_REVISION
        or completion.get("bundle_revision") != BUNDLE_REVISION
    ):
        raise ValueError("Frozen submission bundle revision is stale or incompatible.")
    bundle = validate_bundle(bundle)
    if (
        _canonical_sha256(bundle.get("provenance"))
        != _canonical_sha256(EXPECTED_PROVENANCE)
        or _canonical_sha256(bundle.get("provenance"))
        != _canonical_sha256(completion.get("provenance"))
    ):
        raise ValueError("Frozen submission bundle provenance is stale or incompatible.")
    frozen_probability(0.0, 0.0, bundle, validate_calibrators=False)
    return bundle


def validate_submission_artifacts(
    generation_directory: Path | str,
    *,
    rgb_checkpoint: Path | str,
    signal_model: Path | str,
) -> dict:
    """Fail closed on every frozen deployment binding before model construction."""
    bundle = load_frozen_bundle(generation_directory)
    signal_binding = bundle.get("input_cache_bindings", {}).get("signal_model")
    expected_signal_binding = {
        "path": "upstream/signal-model.json",
        "file_sha256": SIGNAL_MODEL_SHA256,
        "checkpoint_revision": EXPECTED_PROVENANCE["signal_checkpoint_revision"],
        "normalization_revision": EXPECTED_PROVENANCE["signal_normalization_revision"],
    }
    if signal_binding != expected_signal_binding:
        raise ValueError("Frozen signal model binding is stale or incompatible.")
    if _sha256_file(Path(signal_model), "Frozen signal model") != SIGNAL_MODEL_SHA256:
        raise ValueError("Frozen signal model checksum is stale or incompatible.")

    metadata = load_model_metadata()
    model = metadata.get("models", {}).get("384", {})
    expected_runtime = {
        "checkpoint_revision": model.get("revision"),
        "checkpoint_sha256": model.get("sha256"),
        "preprocessing_version": metadata.get("preprocessing_version"),
        "score_direction": metadata.get("score_direction"),
        "shared_observation_preprocessing_version": "shared-preprocessing-v1",
        "input_resolution": model.get("input_resolution"),
        "resize_short_edge": model.get("resize_short_edge"),
    }
    expected_frozen = {
        "checkpoint_revision": EXPECTED_PROVENANCE["rgb_checkpoint_revision"],
        "checkpoint_sha256": RGB_CHECKPOINT_SHA256,
        "preprocessing_version": EXPECTED_PROVENANCE["rgb_preprocessing_version"],
        "score_direction": EXPECTED_PROVENANCE["rgb_score_direction"],
        "shared_observation_preprocessing_version": EXPECTED_PROVENANCE[
            "shared_observation_preprocessing_version"
        ],
        "input_resolution": 384,
        "resize_short_edge": 440,
    }
    normalizer = bundle.get("rgb_normalizer", {})
    if expected_runtime != expected_frozen or any(
        normalizer.get(key) != value for key, value in expected_frozen.items()
    ):
        raise ValueError("Frozen RGB checkpoint/preprocessing binding is incompatible.")
    if _sha256_file(Path(rgb_checkpoint), "Frozen RGB checkpoint") != RGB_CHECKPOINT_SHA256:
        raise ValueError("Frozen RGB checkpoint checksum is stale or incompatible.")
    return bundle


def _preprocess_rgb(path: Path) -> np.ndarray:
    return preprocess_image(path, resolution=384)


def _preprocess_signal(path: Path) -> np.ndarray:
    rgb = decode_expert_rgb(path, resolution=384)
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
    bundle = load_frozen_bundle(generation_directory)  # revalidate deployment contract before decoding
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
