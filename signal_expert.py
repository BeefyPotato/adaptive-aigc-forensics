"""Deterministic low-level signal expert for AI-generated image detection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shared_observation import (
    decode_materialized_png_bytes,
    prepare_shared_expert_rgb,
    prepare_shared_expert_rgb_array,
)


SIGNAL_REPRESENTATION_VERSION = "signal-representation-v1"
SIGNAL_ARTIFACT_VERSION = "signal-feature-logit-v1"
NORMALIZATION_SCHEMA_VERSION = "signal-normalization-v1"
CHECKPOINT_SCHEMA_VERSION = "signal-mlp-v1"
NORMALIZATION_REVISION_PREFIX = "signal-normalization-v1"
CHECKPOINT_REVISION_PREFIX = "signal-checkpoint-v1"
SELECTION_METRIC_VERSION = "signal-condition-balanced-bce-v1"
SELECTION_METRIC_NAME = "condition-balanced-validation-bce"
IMPLEMENTATION_HASH_CONTRACT_VERSION = "utf8-lf-normalized-sha256-v1"
EXPERIMENT_SCOPE_BY_PROFILE = {
    "custom-v1": "non-acceptance",
    "hackathon-v1": "issue-6-timeboxed-acceptance",
    "issue-6-full-v1": "issue-6-full-acceptance",
}
FEATURE_NAMES = tuple(
    [f"fourier_radial_log_energy_{index:02d}" for index in range(16)]
    + [
        "neighbour_horizontal_abs_mean", "neighbour_horizontal_abs_std",
        "neighbour_vertical_abs_mean", "neighbour_vertical_abs_std",
        "neighbour_diagonal_down_abs_mean", "neighbour_diagonal_up_abs_mean",
    ]
    + ["residual_abs_mean", "residual_std", "residual_excess_kurtosis", "residual_sign_change_rate"]
)
RESIDUAL_KERNEL = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
ALLOWED_TRAIN_SPLIT = "expert-training"
ALLOWED_SELECTION_SPLIT = "internal-validation"
CANONICAL_VALIDATION_CONDITIONS = (
    ("clean", "clean"),
    *(("jpeg", f"quality-{quality}") for quality in (90, 70, 50, 30)),
    *(("blur", f"sigma-{sigma}") for sigma in (0.5, 1, 2)),
    *(("resize", f"factor-{factor}") for factor in (0.5, 0.25)),
    *(("noise", f"sigma-{sigma}") for sigma in (0.02, 0.05, 0.1)),
    *(("color", f"{property_}-{factor}") for property_ in ("brightness", "contrast", "saturation") for factor in (0.8, 1.2)),
    ("crop", "center-0.8"),
)
CANONICAL_VALIDATION_FAMILIES = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
REQUIRED_MANIFEST_METADATA_FIELDS = (
    "manifest_schema_version",
    "materialization_schema_version",
    "manifest_sha256",
    "corruption_version",
    "shared_observation_preprocessing_version",
    "materialized_encoding",
    "signal_representation_version",
    "feature_extraction_version",
)
REQUIRED_EXPERIMENT_PROVENANCE_FIELDS = (
    "experiment_profile",
    "acceptance_scope",
    "training_plan_sha256",
    "training_feature_records_sha256",
    "validation_feature_records_sha256",
    "signal_feature_extraction_version",
    "resolution",
    "feature_extraction",
)
REQUIRED_TRAINING_PROVENANCE_FIELDS = (
    "experiment_profile",
    "acceptance_scope",
    "training_plan_sha256",
    "training_feature_records_sha256",
    "signal_feature_extraction_version",
    "resolution",
    "feature_extraction",
)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Signal artifact provenance must be finite canonical JSON.") from error


def _strict_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int or int/float aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _content_revision(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _validated_manifest_metadata(manifest_metadata: dict) -> dict:
    if not isinstance(manifest_metadata, dict):
        raise ValueError("Signal artifact manifest_metadata must be an object.")
    for field in REQUIRED_MANIFEST_METADATA_FIELDS:
        value = manifest_metadata.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Signal artifact manifest_metadata requires {field}.")
    if manifest_metadata["manifest_schema_version"] != "track5-manifest-v1":
        raise ValueError("Signal artifact manifest_schema_version is incompatible.")
    if manifest_metadata["materialization_schema_version"] != "track5-materialized-observations-v1":
        raise ValueError("Signal artifact materialization_schema_version is incompatible.")
    if manifest_metadata["signal_representation_version"] != SIGNAL_REPRESENTATION_VERSION:
        raise ValueError("Signal artifact signal_representation_version is incompatible.")
    digest = manifest_metadata["manifest_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Signal artifact manifest_sha256 must be a lowercase SHA-256 digest.")
    _canonical_json(manifest_metadata)
    return json.loads(_canonical_json(manifest_metadata))


def _normalization_identity(normalization: dict) -> dict:
    return {key: value for key, value in normalization.items() if key != "normalization_revision"}


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return value


def _validated_experiment_provenance(experiment_provenance: dict) -> dict:
    if not isinstance(experiment_provenance, dict):
        raise ValueError("Signal checkpoint experiment provenance must be an object.")
    for field in REQUIRED_EXPERIMENT_PROVENANCE_FIELDS:
        if field not in experiment_provenance:
            raise ValueError(f"Signal checkpoint experiment provenance requires {field}.")
    for field in (
        "training_plan_sha256",
        "training_feature_records_sha256",
        "validation_feature_records_sha256",
    ):
        _require_sha256(experiment_provenance[field], f"Signal checkpoint experiment provenance {field}")
    experiment_profile = experiment_provenance["experiment_profile"]
    acceptance_scope = experiment_provenance["acceptance_scope"]
    if EXPERIMENT_SCOPE_BY_PROFILE.get(experiment_profile) != acceptance_scope:
        raise ValueError(
            "Signal checkpoint experiment profile and acceptance scope are stale or incompatible."
        )
    feature_version = experiment_provenance["signal_feature_extraction_version"]
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("Signal checkpoint experiment provenance requires signal_feature_extraction_version.")
    resolution = experiment_provenance["resolution"]
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal checkpoint experiment provenance resolution must be 224 or 384.")
    _validated_feature_extraction_snapshot(
        experiment_provenance["feature_extraction"],
        feature_version=feature_version,
        resolution=resolution,
    )
    return json.loads(_canonical_json(experiment_provenance))


def _validated_feature_extraction_snapshot(
    snapshot: dict,
    *,
    feature_version: str,
    resolution: int,
) -> dict:
    if not isinstance(snapshot, dict):
        raise ValueError("Signal feature extraction provenance must be an object.")
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal feature extraction expected resolution must be 224 or 384.")
    snapshot_resolution = snapshot.get("resolution")
    if (
        snapshot.get("feature_extraction_version") != feature_version
        or snapshot.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION
        or snapshot.get("feature_names") != list(FEATURE_NAMES)
        or type(snapshot_resolution) is not int
        or not _strict_json_equal(snapshot_resolution, resolution)
        or snapshot.get("implementation_hash_contract_version")
        != IMPLEMENTATION_HASH_CONTRACT_VERSION
    ):
        raise ValueError("Signal feature extraction provenance or hash contract is stale or incompatible.")
    runtime_versions = snapshot.get("runtime_versions")
    if not isinstance(runtime_versions, dict) or not runtime_versions or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or not value
        for name, value in runtime_versions.items()
    ):
        raise ValueError("Signal feature extraction runtime provenance is incomplete.")
    implementation = snapshot.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Signal feature extraction implementation provenance is incomplete.")
    for name, digest in implementation.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Signal feature extraction implementation path is invalid.")
        _require_sha256(digest, f"Signal feature extraction implementation {name}")
    return json.loads(_canonical_json(snapshot))


def _validated_training_provenance(training_provenance: dict) -> dict:
    if not isinstance(training_provenance, dict):
        raise ValueError("Signal normalization training provenance must be an object.")
    if set(training_provenance) != set(REQUIRED_TRAINING_PROVENANCE_FIELDS):
        raise ValueError(
            "Signal normalization training provenance must contain exactly the experiment "
            "profile/scope, training plan, training feature digest, feature version, and resolution."
        )
    experiment_profile = training_provenance["experiment_profile"]
    acceptance_scope = training_provenance["acceptance_scope"]
    if EXPERIMENT_SCOPE_BY_PROFILE.get(experiment_profile) != acceptance_scope:
        raise ValueError(
            "Signal normalization experiment profile and acceptance scope are stale or incompatible."
        )
    _require_sha256(
        training_provenance["training_plan_sha256"],
        "Signal normalization training provenance training_plan_sha256",
    )
    _require_sha256(
        training_provenance["training_feature_records_sha256"],
        "Signal normalization training provenance training_feature_records_sha256",
    )
    feature_version = training_provenance["signal_feature_extraction_version"]
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("Signal normalization training provenance requires a feature version.")
    resolution = training_provenance["resolution"]
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal normalization training provenance resolution must be 224 or 384.")
    _validated_feature_extraction_snapshot(
        training_provenance["feature_extraction"],
        feature_version=feature_version,
        resolution=resolution,
    )
    return json.loads(_canonical_json(training_provenance))


def _training_provenance_from_experiment(experiment_provenance: dict) -> dict:
    experiment = _validated_experiment_provenance(experiment_provenance)
    return _validated_training_provenance({
        field: experiment[field] for field in REQUIRED_TRAINING_PROVENANCE_FIELDS
    })


def _records_sha256(records: list[dict]) -> str:
    return hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()


def _contains_non_numeric(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_non_numeric(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in "iuf" or any(
            _contains_non_numeric(item) for item in value.reshape(-1).tolist()
        )
    return (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
    )


def _validate_experiment_record_digests(
    experiment_provenance: dict,
    *,
    training_records: list[dict] | None = None,
    validation_records: list[dict] | None = None,
) -> dict:
    validated = _validated_experiment_provenance(experiment_provenance)
    for records, field, context in (
        (
            training_records,
            "training_feature_records_sha256",
            "Signal training feature records",
        ),
        (
            validation_records,
            "validation_feature_records_sha256",
            "Signal validation feature records",
        ),
    ):
        if records is not None and _records_sha256(records) != validated[field]:
            raise ValueError(f"{context} do not match experiment provenance {field}.")
    return validated


def _validate_training_record_digest(training_provenance: dict, records: list[dict]) -> dict:
    validated = _validated_training_provenance(training_provenance)
    if _records_sha256(records) != validated["training_feature_records_sha256"]:
        raise ValueError(
            "Signal normalization records do not match training provenance "
            "training_feature_records_sha256."
        )
    return validated


def _positive_integer_weights(records: list[dict], context: str) -> np.ndarray:
    weights = [record.get("sample_weight", 1) for record in records]
    if any(
        isinstance(weight, (bool, np.bool_))
        or not isinstance(weight, (int, np.integer))
        or weight <= 0
        for weight in weights
    ):
        raise ValueError(f"{context} sample weights must be positive integers.")
    return np.asarray(weights, dtype=np.float64)


def _feature_matrix(records: list[dict], context: str) -> np.ndarray:
    if any(record.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION for record in records):
        raise ValueError(f"{context} has a stale or incompatible representation version.")
    if any(_contains_non_numeric(record.get("features")) for record in records):
        raise ValueError(f"{context} requires strictly numeric feature values.")
    try:
        features = np.asarray([record.get("features") for record in records], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} requires finite 26-value feature records.") from error
    if features.shape != (len(records), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError(f"{context} requires finite 26-value feature records.")
    return features


def _binary_labels(records: list[dict]) -> np.ndarray:
    labels = [record.get("authenticity_label") for record in records]
    if any(
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (int, np.integer))
        or label not in (0, 1)
        for label in labels
    ):
        raise ValueError("Signal training and validation require binary labels 0 or 1.")
    return np.asarray(labels, dtype=np.float64)


def _reject_evaluation_only_records(records: list[dict], context: str) -> None:
    for index, record in enumerate(records):
        if record.get("dataset") == "organizer-demonstration":
            raise ValueError(f"{context} cannot use organizer-demonstration record {index}.")
        if record.get("usage") == "evaluation-only":
            raise ValueError(f"{context} cannot use evaluation-only record {index}.")


def _validate_feature_record_relationships(
    records: list[dict],
    *,
    manifest_metadata: dict,
    expected_split: str,
    context: str,
) -> None:
    seen_variants = set()
    label_by_source = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{context} record {index} must be an object.")
        for field in ("source_id", "variant_id", "condition_family"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"{context} record {index} requires {field}.")
        if record["variant_id"] in seen_variants:
            raise ValueError(f"{context} records have duplicate variant_id {record['variant_id']!r}.")
        seen_variants.add(record["variant_id"])
        if record.get("split") != expected_split:
            raise ValueError(f"{context} records must use only {expected_split} observations.")
        if record.get("severity") is None:
            raise ValueError(f"{context} record {index} requires severity.")
        _require_sha256(record.get("materialized_sha256"), f"{context} record {index}.materialized_sha256")
        if record.get("materialized_encoding") != manifest_metadata["materialized_encoding"]:
            raise ValueError(f"{context} record {index} has incompatible materialized_encoding.")
        _feature_matrix([record], f"{context} record {index}")
        label = int(_binary_labels([record])[0])
        source_id = record["source_id"]
        if source_id in label_by_source and label_by_source[source_id] != label:
            raise ValueError(f"{context} source {source_id!r} changes authenticity label.")
        label_by_source[source_id] = label


def decode_expert_rgb(path: Path | str, *, resolution: int = 384) -> np.ndarray:
    """Decode the shared post-corruption checkpoint crop into RGB [0, 1]."""
    return prepare_shared_expert_rgb(path, resolution=resolution).astype(np.float64) / 255.0


def decode_expert_rgb_bytes(
    value: bytes,
    *,
    resolution: int = 384,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    """Decode the exact checksum-verified canonical PNG bytes into expert RGB."""
    native_rgb = decode_materialized_png_bytes(
        value,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    return prepare_shared_expert_rgb_array(
        native_rgb,
        resolution=resolution,
    ).astype(np.float64) / 255.0


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 2:
        raise ValueError("Signal input must be an HxWx3 native-resolution RGB observation.")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Signal RGB values must be finite and within [0, 1].")
    return np.ascontiguousarray(array @ np.asarray([0.2126, 0.7152, 0.0722]))


def _convolve_reflect(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="reflect")
    output = np.zeros_like(image)
    for offset, weight in enumerate(kernel):
        slices = [slice(None), slice(None)]
        slices[axis] = slice(offset, offset + image.shape[axis])
        output += weight * padded[tuple(slices)]
    return output


def _residual(luminance: np.ndarray) -> np.ndarray:
    smooth = _convolve_reflect(_convolve_reflect(luminance, RESIDUAL_KERNEL, 1), RESIDUAL_KERNEL, 0)
    return luminance - smooth


def _neighbour_high_pass(luminance: np.ndarray) -> np.ndarray:
    """Return a same-sized forward-difference magnitude diagnostic map."""
    horizontal = np.zeros_like(luminance)
    vertical = np.zeros_like(luminance)
    horizontal[:, :-1] = np.diff(luminance, axis=1)
    vertical[:-1, :] = np.diff(luminance, axis=0)
    return np.hypot(horizontal, vertical)


def _fourier_features(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = luminance - luminance.mean()
    window = np.outer(np.hanning(luminance.shape[0]), np.hanning(luminance.shape[1]))
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2)
    yy, xx = np.indices(spectrum.shape, dtype=np.float64)
    radius = np.sqrt((yy - (spectrum.shape[0] - 1) / 2) ** 2 + (xx - (spectrum.shape[1] - 1) / 2) ** 2)
    normalized = radius / max(float(radius.max()), np.finfo(np.float64).eps)
    bins = np.minimum((normalized * 16).astype(np.int64), 15)
    values = np.asarray([
        spectrum[bins == index].mean() if np.any(bins == index) else 0.0
        for index in range(16)
    ])
    return values, spectrum


def extract_signal_representation(rgb: np.ndarray, *, include_maps: bool = False) -> dict:
    luminance = rgb_to_luminance(rgb)
    fourier, spectrum = _fourier_features(luminance)
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    diagonal_down = np.abs(luminance[1:, 1:] - luminance[:-1, :-1])
    diagonal_up = np.abs(luminance[1:, :-1] - luminance[:-1, 1:])
    neighbour = np.asarray([
        horizontal.mean(), horizontal.std(), vertical.mean(), vertical.std(),
        diagonal_down.mean(), diagonal_up.mean(),
    ])
    residual = _residual(luminance)
    residual_std = float(residual.std())
    residual_kurtosis = (
        0.0
        if residual_std <= np.finfo(np.float64).eps
        else float(np.mean(((residual - residual.mean()) / residual_std) ** 4) - 3)
    )
    sign_changes = np.concatenate([
        (residual[:, 1:] * residual[:, :-1] < 0).reshape(-1),
        (residual[1:, :] * residual[:-1, :] < 0).reshape(-1),
    ])
    residual_features = np.asarray([
        np.abs(residual).mean(), residual_std, residual_kurtosis,
        sign_changes.mean(),
    ])
    features = np.concatenate([fourier, neighbour, residual_features]).astype(np.float64)
    if features.shape != (26,) or not np.isfinite(features).all():
        raise ValueError("Signal representation must contain 26 finite values.")
    result = {"version": SIGNAL_REPRESENTATION_VERSION, "feature_names": FEATURE_NAMES, "features": features}
    if include_maps:
        result["maps"] = {
            "luminance": luminance,
            "spectrum": spectrum,
            "high_pass": _neighbour_high_pass(luminance),
            "residual": residual,
        }
    return result


def fit_normalization(
    records: Iterable[dict],
    *,
    manifest_metadata: dict,
    training_provenance: dict,
) -> dict:
    records = list(records)
    _reject_evaluation_only_records(records, "Signal normalization")
    if not records or any(record.get("split") != ALLOWED_TRAIN_SPLIT for record in records):
        raise ValueError("Normalization may be fit only on expert-training observations.")
    validated_metadata = _validated_manifest_metadata(manifest_metadata)
    validated_training_provenance = _validate_training_record_digest(
        training_provenance,
        records,
    )
    if (
        validated_training_provenance["signal_feature_extraction_version"]
        != validated_metadata["feature_extraction_version"]
        or validated_training_provenance["feature_extraction"][
            "signal_representation_version"
        ]
        != validated_metadata["signal_representation_version"]
    ):
        raise ValueError("Signal normalization feature provenance contradicts manifest metadata.")
    _validate_feature_record_relationships(
        records,
        manifest_metadata=validated_metadata,
        expected_split=ALLOWED_TRAIN_SPLIT,
        context="Normalization",
    )
    source_ids = {record["source_id"] for record in records}
    features = _feature_matrix(records, "Normalization")
    weights = _positive_integer_weights(records, "Normalization")
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.0
    normalization = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "fit_split": ALLOWED_TRAIN_SPLIT,
        "source_count": len(source_ids),
        "observation_count": int(weights.sum()),
        "unique_observation_count": len(records),
        "manifest_metadata": validated_metadata,
        "training_provenance": validated_training_provenance,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    normalization["normalization_revision"] = _content_revision(
        NORMALIZATION_REVISION_PREFIX,
        normalization,
    )
    return normalization


def _validate_normalization(
    normalization: dict,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
) -> tuple[np.ndarray, np.ndarray]:
    expected_metadata = _validated_manifest_metadata(manifest_metadata)
    if not isinstance(normalization, dict):
        raise ValueError("Signal normalization artifact must be an object.")
    if normalization.get("schema_version") != NORMALIZATION_SCHEMA_VERSION or normalization.get("fit_split") != ALLOWED_TRAIN_SPLIT:
        raise ValueError("Signal normalization schema or fit split is incompatible.")
    if normalization.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION:
        raise ValueError("Signal normalization representation version is stale.")
    if normalization.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Signal normalization feature order is stale or incompatible.")
    if not _strict_json_equal(normalization.get("manifest_metadata"), expected_metadata):
        raise ValueError("Signal normalization manifest metadata is stale.")
    if not _strict_json_equal(
        normalization.get("training_provenance"),
        _training_provenance_from_experiment(expected_experiment_provenance),
    ):
        raise ValueError("Signal normalization training provenance is stale or incompatible.")
    if _contains_non_numeric(normalization.get("mean")) or _contains_non_numeric(
        normalization.get("scale")
    ):
        raise ValueError("Signal normalization values must be strictly numeric.")
    try:
        mean = np.asarray(normalization.get("mean"), dtype=np.float64)
        scale = np.asarray(normalization.get("scale"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Signal normalization must contain 26 finite mean/positive scale values.") from error
    if mean.shape != (26,) or scale.shape != (26,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Signal normalization must contain 26 finite mean/positive scale values.")
    for field in ("source_count", "observation_count", "unique_observation_count"):
        value = normalization.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Signal normalization {field} must be a positive integer.")
    if normalization["unique_observation_count"] > normalization["observation_count"]:
        raise ValueError("Signal normalization observation counts are incompatible.")
    expected_revision = _content_revision(
        NORMALIZATION_REVISION_PREFIX,
        _normalization_identity(normalization),
    )
    if normalization.get("normalization_revision") != expected_revision:
        raise ValueError("Signal normalization revision is stale or incompatible.")
    return mean, scale


@dataclass
class SignalMLP:
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float

    def logits(self, features: np.ndarray) -> np.ndarray:
        hidden = np.tanh(np.asarray(features) @ self.input_weights + self.input_bias)
        return hidden @ self.output_weights + self.output_bias


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def _validate_canonical_selection_matrix(records: list[dict]) -> None:
    expected_conditions = set(CANONICAL_VALIDATION_CONDITIONS)
    conditions_by_source = {}
    label_by_source = {}
    for index, record in enumerate(records):
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Internal-validation record {index} requires source_id.")
        condition = (record.get("condition_family"), record.get("severity"))
        if condition not in expected_conditions:
            raise ValueError(
                f"Internal-validation record {index} is outside the canonical condition matrix."
            )
        source_conditions = conditions_by_source.setdefault(source_id, set())
        if condition in source_conditions:
            raise ValueError(
                f"Internal-validation source {source_id!r} repeats a canonical condition."
            )
        source_conditions.add(condition)
        label = record["authenticity_label"]
        if source_id in label_by_source and label_by_source[source_id] != label:
            raise ValueError(f"Internal-validation source {source_id!r} changes authenticity label.")
        label_by_source[source_id] = label
    for source_id, conditions in conditions_by_source.items():
        if conditions != expected_conditions:
            missing = sorted(expected_conditions - conditions)
            raise ValueError(
                f"Internal-validation source {source_id!r} is missing the canonical condition matrix"
                f" including {missing[0] if missing else 'an expected condition'}."
            )
    if set(label_by_source.values()) != {0, 1}:
        raise ValueError("The canonical internal-validation condition matrix requires both authenticity classes.")


def _condition_balanced_bce(logits: np.ndarray, labels: np.ndarray, records: list[dict]) -> float:
    losses = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    family_losses = []
    for family in CANONICAL_VALIDATION_FAMILIES:
        severity_losses = []
        for condition_family, severity in CANONICAL_VALIDATION_CONDITIONS:
            if condition_family != family:
                continue
            indexes = [
                index
                for index, record in enumerate(records)
                if record["condition_family"] == family and record["severity"] == severity
            ]
            severity_losses.append(float(losses[indexes].mean()))
        family_losses.append(sum(severity_losses) / len(severity_losses))
    return sum(family_losses) / len(family_losses)


def train_signal_mlp(
    training_records: Iterable[dict],
    validation_records: Iterable[dict],
    normalization: dict,
    *,
    manifest_metadata: dict,
    experiment_provenance: dict,
    seed: int = 61,
    epochs: int = 200,
    learning_rate: float = 0.02,
) -> tuple[SignalMLP, dict]:
    training, validation = list(training_records), list(validation_records)
    _reject_evaluation_only_records(training, "Signal training")
    _reject_evaluation_only_records(validation, "Signal checkpoint selection")
    if not training or any(row.get("split") != ALLOWED_TRAIN_SPLIT for row in training):
        raise ValueError("Signal MLP weights may use only expert-training observations.")
    if not validation or any(row.get("split") != ALLOWED_SELECTION_SPLIT for row in validation):
        raise ValueError("Signal checkpoint selection may use only internal-validation observations.")
    validated_metadata = _validated_manifest_metadata(manifest_metadata)
    _validate_feature_record_relationships(
        training,
        manifest_metadata=validated_metadata,
        expected_split=ALLOWED_TRAIN_SPLIT,
        context="Signal training",
    )
    _validate_feature_record_relationships(
        validation,
        manifest_metadata=validated_metadata,
        expected_split=ALLOWED_SELECTION_SPLIT,
        context="Signal validation",
    )
    if {row["source_id"] for row in training} & {row["source_id"] for row in validation}:
        raise ValueError("Signal training and validation sources must be disjoint.")
    if isinstance(epochs, (bool, np.bool_)) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("Signal training epochs must be a positive integer.")
    if (
        isinstance(learning_rate, (bool, np.bool_))
        or not isinstance(learning_rate, (int, float, np.integer, np.floating))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("Signal training learning rate must be positive and finite.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Signal training seed must be a non-negative integer.")
    validated_experiment = _validate_experiment_record_digests(
        experiment_provenance,
        training_records=training,
        validation_records=validation,
    )
    mean, scale = _validate_normalization(
        normalization,
        manifest_metadata,
        validated_experiment,
    )
    x = (_feature_matrix(training, "Signal training") - mean) / scale
    y = _binary_labels(training)
    vx = (_feature_matrix(validation, "Signal validation") - mean) / scale
    vy = _binary_labels(validation)
    _validate_canonical_selection_matrix(validation)
    training_weights = _positive_integer_weights(training, "Signal training")
    rng = np.random.default_rng(seed)
    model = SignalMLP(rng.normal(0, 0.1, (26, 16)), np.zeros(16), rng.normal(0, 0.1, 16), 0.0)
    best, best_loss, best_epoch = None, math.inf, -1
    for epoch in range(epochs):
        with np.errstate(over="ignore", invalid="ignore"):
            hidden = np.tanh(x @ model.input_weights + model.input_bias)
            logits = hidden @ model.output_weights + model.output_bias
            if not np.isfinite(logits).all():
                raise ValueError("Signal training diverged to non-finite logits.")
            probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
            delta = (probability - y) * training_weights / training_weights.sum()
            output_gradient = hidden.T @ delta
            bias_gradient = float(delta.sum())
            hidden_gradient = np.outer(delta, model.output_weights) * (1 - hidden ** 2)
            input_gradient = x.T @ hidden_gradient
            input_bias_gradient = hidden_gradient.sum(axis=0)
            model = SignalMLP(
                model.input_weights - learning_rate * input_gradient,
                model.input_bias - learning_rate * input_bias_gradient,
                model.output_weights - learning_rate * output_gradient,
                model.output_bias - learning_rate * bias_gradient,
            )
            try:
                model = _validated_model(model)
            except ValueError as error:
                raise ValueError("Signal training diverged to non-finite weights.") from error
            validation_logits = model.logits(vx)
            if not np.isfinite(validation_logits).all():
                raise ValueError("Signal training diverged to non-finite validation logits.")
            validation_loss = _condition_balanced_bce(validation_logits, vy, validation)
        if not math.isfinite(validation_loss):
            raise ValueError("Signal training diverged to a non-finite validation loss.")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best = SignalMLP(*(np.copy(value) if isinstance(value, np.ndarray) else value for value in (model.input_weights, model.input_bias, model.output_weights, model.output_bias)))
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "selected_epoch": best_epoch,
        "validation_bce": best_loss,
        "selection_metric": SELECTION_METRIC_NAME,
        "selection_metric_version": SELECTION_METRIC_VERSION,
        "selection_score": best_loss,
        "training_split": ALLOWED_TRAIN_SPLIT,
        "selection_split": ALLOWED_SELECTION_SPLIT,
        "hidden_units": 16,
        "training_source_count": len({row["source_id"] for row in training}),
        "validation_source_count": len({row["source_id"] for row in validation}),
        "training_observation_count": int(training_weights.sum()),
        "training_unique_observation_count": len(training),
        "validation_observation_count": len(validation),
        "validation_condition_count": len(CANONICAL_VALIDATION_CONDITIONS),
        "manifest_metadata": validated_metadata,
    }
    return best, metadata


def _validated_model(model: SignalMLP) -> SignalMLP:
    try:
        raw_values = (
            model.input_weights,
            model.input_bias,
            model.output_weights,
            model.output_bias,
        )
    except AttributeError as error:
        raise ValueError("Signal checkpoint weights are incompatible.") from error
    if any(_contains_non_numeric(value) for value in raw_values):
        raise ValueError("Signal checkpoint weights must be strictly numeric.")
    try:
        validated = SignalMLP(
            np.asarray(model.input_weights, dtype=np.float64),
            np.asarray(model.input_bias, dtype=np.float64),
            np.asarray(model.output_weights, dtype=np.float64),
            float(model.output_bias),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Signal checkpoint weights are incompatible.") from error
    if (
        validated.input_weights.shape != (26, 16)
        or validated.input_bias.shape != (16,)
        or validated.output_weights.shape != (16,)
    ):
        raise ValueError("Signal checkpoint weight dimensions are incompatible.")
    if (
        not np.isfinite(validated.input_weights).all()
        or not np.isfinite(validated.input_bias).all()
        or not np.isfinite(validated.output_weights).all()
        or not math.isfinite(validated.output_bias)
    ):
        raise ValueError("Signal checkpoint weights must be finite.")
    return validated


def _validate_checkpoint_metadata(metadata: dict, manifest_metadata: dict) -> None:
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Signal checkpoint schema is stale or incompatible.")
    if metadata.get("manifest_metadata") != _validated_manifest_metadata(manifest_metadata):
        raise ValueError("Signal checkpoint manifest metadata is stale or incompatible.")
    if metadata.get("training_split") != ALLOWED_TRAIN_SPLIT:
        raise ValueError("Signal checkpoint training split is incompatible.")
    if metadata.get("selection_split") != ALLOWED_SELECTION_SPLIT:
        raise ValueError("Signal checkpoint selection split is incompatible.")
    if type(metadata.get("hidden_units")) is not int or metadata.get("hidden_units") != 16:
        raise ValueError("Signal checkpoint hidden-unit count is incompatible.")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Signal checkpoint seed must be a non-negative integer.")
    learning_rate = metadata.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("Signal checkpoint learning rate must be positive and finite.")
    count_fields = (
        "training_source_count",
        "validation_source_count",
        "training_observation_count",
        "training_unique_observation_count",
        "validation_observation_count",
    )
    for field in count_fields:
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Signal checkpoint {field} must be a positive integer.")
    if not (
        metadata["training_source_count"]
        <= metadata["training_unique_observation_count"]
        <= metadata["training_observation_count"]
    ):
        raise ValueError("Signal checkpoint training counts are incompatible.")
    if (
        metadata["validation_source_count"] > metadata["validation_observation_count"]
        or metadata["validation_observation_count"]
        != metadata["validation_source_count"] * len(CANONICAL_VALIDATION_CONDITIONS)
    ):
        raise ValueError("Signal checkpoint validation counts are incompatible.")
    epochs = metadata.get("epochs")
    selected_epoch = metadata.get("selected_epoch")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs <= 0
        or isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 0 <= selected_epoch < epochs
    ):
        raise ValueError("Signal checkpoint epoch metadata is incompatible.")
    validation_bce = metadata.get("validation_bce")
    if (
        isinstance(validation_bce, bool)
        or not isinstance(validation_bce, (int, float))
        or not math.isfinite(validation_bce)
        or validation_bce < 0
    ):
        raise ValueError("Signal checkpoint validation loss must be finite and non-negative.")
    if metadata.get("selection_metric") != SELECTION_METRIC_NAME:
        raise ValueError("Signal checkpoint selection metric is stale or incompatible.")
    if metadata.get("selection_metric_version") != SELECTION_METRIC_VERSION:
        raise ValueError("Signal checkpoint selection metric version is stale or incompatible.")
    selection_score = metadata.get("selection_score")
    if (
        isinstance(selection_score, bool)
        or not isinstance(selection_score, (int, float))
        or not math.isfinite(selection_score)
        or selection_score < 0
        or selection_score != validation_bce
    ):
        raise ValueError("Signal checkpoint selection score is stale or incompatible.")
    validation_condition_count = metadata.get("validation_condition_count")
    if (
        type(validation_condition_count) is not int
        or validation_condition_count != len(CANONICAL_VALIDATION_CONDITIONS)
    ):
        raise ValueError("Signal checkpoint validation condition count is stale or incompatible.")


def _validate_checkpoint_experiment_links(metadata: dict, experiment_provenance: dict) -> None:
    experiment = _validated_experiment_provenance(experiment_provenance)
    manifest_metadata = _validated_manifest_metadata(metadata.get("manifest_metadata"))
    if (
        experiment["signal_feature_extraction_version"]
        != manifest_metadata["feature_extraction_version"]
        or experiment["feature_extraction"]["signal_representation_version"]
        != manifest_metadata["signal_representation_version"]
    ):
        raise ValueError(
            "Signal checkpoint experiment feature versions contradict manifest metadata."
        )
    for field in (
        "training_feature_records_sha256",
        "validation_feature_records_sha256",
    ):
        if metadata.get(field) != experiment[field]:
            raise ValueError(
                f"Signal checkpoint has contradictory or incompatible {field}."
            )
    feature_extraction = metadata.get("feature_extraction")
    if not _strict_json_equal(feature_extraction, experiment["feature_extraction"]):
        raise ValueError(
            "Signal checkpoint feature extraction metadata contradicts experiment provenance."
        )
    for name, split, kind, observation_count_field, unique_count_field in (
        (
            "training_selection",
            ALLOWED_TRAIN_SPLIT,
            "balanced-sampler",
            "training_observation_count",
            "training_unique_observation_count",
        ),
        (
            "validation_selection",
            ALLOWED_SELECTION_SPLIT,
            "complete-condition-matrix",
            "validation_observation_count",
            "validation_observation_count",
        ),
    ):
        selection = metadata.get(name)
        if not isinstance(selection, dict):
            raise ValueError(f"Signal checkpoint requires {name} metadata.")
        sample_count = selection.get("sample_count")
        unique_count = selection.get("unique_observation_count")
        shard_count = selection.get("shard_count")
        if (
            selection.get("split") != split
            or selection.get("kind") != kind
            or selection.get("plan_sha256") != experiment["training_plan_sha256"]
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or not _strict_json_equal(sample_count, metadata.get(observation_count_field))
            or isinstance(unique_count, bool)
            or not isinstance(unique_count, int)
            or unique_count <= 0
            or not _strict_json_equal(unique_count, metadata.get(unique_count_field))
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count <= 0
        ):
            raise ValueError(
                f"Signal checkpoint {name} contradicts its experiment or checkpoint counts."
            )


def _validate_normalization_checkpoint_counts(normalization: dict, metadata: dict) -> None:
    for normalization_field, checkpoint_field in (
        ("source_count", "training_source_count"),
        ("observation_count", "training_observation_count"),
        ("unique_observation_count", "training_unique_observation_count"),
    ):
        if not _strict_json_equal(
            normalization.get(normalization_field), metadata.get(checkpoint_field)
        ):
            raise ValueError(
                "Signal checkpoint normalization counts contradict training checkpoint counts."
            )


def write_model_bundle(
    path: Path | str,
    model: SignalMLP,
    metadata: dict,
    normalization: dict,
    *,
    experiment_provenance: dict,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Signal checkpoint metadata must be an object.")
    manifest_metadata = metadata.get("manifest_metadata")
    _validate_checkpoint_metadata(metadata, manifest_metadata)
    validated_experiment = _validated_experiment_provenance(experiment_provenance)
    _validate_checkpoint_experiment_links(metadata, validated_experiment)
    _validate_normalization(normalization, manifest_metadata, validated_experiment)
    _validate_normalization_checkpoint_counts(normalization, metadata)
    validated_model = _validated_model(model)
    reserved = {
        "checkpoint_revision",
        "experiment_provenance",
        "feature_names",
        "normalization",
        "normalization_revision",
        "signal_representation_version",
        "weights",
    }
    if reserved & metadata.keys():
        raise ValueError("Signal checkpoint metadata contains reserved artifact fields.")
    payload = {
        **json.loads(_canonical_json(metadata)),
        "experiment_provenance": validated_experiment,
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "normalization_revision": normalization["normalization_revision"],
        "normalization": json.loads(_canonical_json(normalization)),
        "weights": {
            "input": validated_model.input_weights.tolist(),
            "input_bias": validated_model.input_bias.tolist(),
            "output": validated_model.output_weights.tolist(),
            "output_bias": validated_model.output_bias,
        },
    }
    payload["checkpoint_revision"] = _content_revision(CHECKPOINT_REVISION_PREFIX, payload)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validated_model_bundle_payload(
    payload: dict,
    *,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
) -> tuple[SignalMLP, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Signal checkpoint artifact must be an object.")
    _validate_checkpoint_metadata(payload, manifest_metadata)
    expected_experiment = _validated_experiment_provenance(expected_experiment_provenance)
    if not _strict_json_equal(payload.get("experiment_provenance"), expected_experiment):
        raise ValueError("Signal checkpoint experiment provenance is stale or incompatible.")
    _validate_checkpoint_experiment_links(payload, expected_experiment)
    if payload.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION:
        raise ValueError("Signal checkpoint representation version is stale or incompatible.")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Signal checkpoint feature order is stale or incompatible.")
    normalization = payload.get("normalization")
    _validate_normalization(normalization, manifest_metadata, expected_experiment)
    _validate_normalization_checkpoint_counts(normalization, payload)
    if payload.get("normalization_revision") != normalization["normalization_revision"]:
        raise ValueError("Signal checkpoint normalization revision is stale or incompatible.")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Signal checkpoint weights are incompatible.")
    model = _validated_model(
        SignalMLP(
            weights.get("input"),
            weights.get("input_bias"),
            weights.get("output"),
            weights.get("output_bias"),
        )
    )
    expected_revision = _content_revision(
        CHECKPOINT_REVISION_PREFIX,
        {key: value for key, value in payload.items() if key != "checkpoint_revision"},
    )
    if payload.get("checkpoint_revision") != expected_revision:
        raise ValueError("Signal checkpoint revision is stale or incompatible.")
    return model, payload


def read_model_bundle(
    path: Path | str,
    *,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
) -> tuple[SignalMLP, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validated_model_bundle_payload(
        payload,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=expected_experiment_provenance,
    )


def _require_revision(revision: str, prefix: str, context: str) -> str:
    expected_prefix = f"{prefix}-"
    if (
        not isinstance(revision, str)
        or not revision.startswith(expected_prefix)
        or len(revision) != len(expected_prefix) + 64
        or any(character not in "0123456789abcdef" for character in revision[len(expected_prefix):])
    ):
        raise ValueError(f"{context} must be a content-derived {prefix} revision.")
    return revision


def _validated_feature_record(record: dict, index: int, manifest_metadata: dict) -> tuple[dict, np.ndarray]:
    if not isinstance(record, dict):
        raise ValueError(f"Signal feature record {index} must be an object.")
    _reject_evaluation_only_records([record], "Signal prediction cache")
    _validate_feature_record_relationships(
        [record],
        manifest_metadata=manifest_metadata,
        expected_split=ALLOWED_SELECTION_SPLIT,
        context="Signal feature",
    )
    features = _feature_matrix([record], f"Signal feature record {index}")[0]
    return record, features


def _cache_identity(
    record: dict,
    *,
    manifest_metadata: dict,
    checkpoint_revision: str,
    normalization_revision: str,
    signal_logit: float,
    pred: float,
) -> dict:
    return {
        "artifact_schema_version": SIGNAL_ARTIFACT_VERSION,
        "checkpoint_revision": checkpoint_revision,
        "normalization_revision": normalization_revision,
        "manifest_metadata": manifest_metadata,
        "manifest_sha256": manifest_metadata["manifest_sha256"],
        "corruption_version": manifest_metadata["corruption_version"],
        "shared_observation_preprocessing_version": manifest_metadata[
            "shared_observation_preprocessing_version"
        ],
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "source_id": record["source_id"],
        "variant_id": record["variant_id"],
        "materialized_sha256": record["materialized_sha256"],
        "materialized_encoding": record["materialized_encoding"],
        "signal_logit": signal_logit,
        "pred": pred,
    }


def cache_signal_predictions(
    records: Iterable[dict],
    checkpoint_bundle: dict,
    *,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
) -> list[dict]:
    metadata = _validated_manifest_metadata(manifest_metadata)
    records = list(records)
    validated_records = [
        _validated_feature_record(candidate, index, metadata)
        for index, candidate in enumerate(records)
    ]
    validated_experiment = _validate_experiment_record_digests(
        expected_experiment_provenance,
        validation_records=records,
    )
    validated_model, validated_bundle = _validated_model_bundle_payload(
        checkpoint_bundle,
        manifest_metadata=metadata,
        expected_experiment_provenance=validated_experiment,
    )
    normalization = validated_bundle["normalization"]
    mean, scale = _validate_normalization(
        normalization,
        metadata,
        validated_experiment,
    )
    normalization_revision = _require_revision(
        normalization.get("normalization_revision"),
        NORMALIZATION_REVISION_PREFIX,
        "Signal normalization revision",
    )
    checkpoint_revision = validated_bundle["checkpoint_revision"]
    output = []
    seen_variants = set()
    for index, (record, features) in enumerate(validated_records):
        if record["variant_id"] in seen_variants:
            raise ValueError(f"Signal feature records have duplicate variant_id {record['variant_id']!r}.")
        seen_variants.add(record["variant_id"])
        logit = float(validated_model.logits(((features - mean) / scale)[None, :])[0])
        if not math.isfinite(logit):
            raise ValueError(f"Signal prediction for {record['variant_id']} is not finite.")
        probability = 1 / (1 + math.exp(-max(-40, min(40, logit))))
        identity = _cache_identity(
            record,
            manifest_metadata=metadata,
            checkpoint_revision=checkpoint_revision,
            normalization_revision=normalization_revision,
            signal_logit=logit,
            pred=probability,
        )
        output.append({
            **identity,
            "split": record["split"],
            "authenticity_label": record["authenticity_label"],
            "condition_family": record["condition_family"],
            "severity": record["severity"],
            "features": features.tolist(),
            "signal_logit": logit,
            "pred": probability,
            "cache_key": "signal-cache-v1-" + hashlib.sha256(
                _canonical_json(identity).encode("utf-8")
            ).hexdigest(),
        })
    if not output:
        raise ValueError("Signal prediction cache requires at least one feature record.")
    return output


def validate_signal_cache(
    records: Iterable[dict],
    *,
    checkpoint_bundle: dict,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
    expected_feature_records: Iterable[dict],
) -> list[dict]:
    metadata = _validated_manifest_metadata(manifest_metadata)
    records = list(records)
    expected_feature_records = list(expected_feature_records)
    validated_experiment = _validate_experiment_record_digests(
        expected_experiment_provenance,
        validation_records=expected_feature_records,
    )
    model, validated_bundle = _validated_model_bundle_payload(
        checkpoint_bundle,
        manifest_metadata=metadata,
        expected_experiment_provenance=validated_experiment,
    )
    normalization = validated_bundle["normalization"]
    mean, scale = _validate_normalization(
        normalization,
        metadata,
        validated_experiment,
    )
    checkpoint_revision = validated_bundle["checkpoint_revision"]
    normalization_revision = validated_bundle["normalization_revision"]
    expected_by_variant = {}
    for index, candidate in enumerate(expected_feature_records):
        expected, _ = _validated_feature_record(candidate, index, metadata)
        variant_id = expected["variant_id"]
        if variant_id in expected_by_variant:
            raise ValueError(f"Expected signal features have duplicate variant_id {variant_id!r}.")
        expected_by_variant[variant_id] = expected
    if not expected_by_variant:
        raise ValueError("Expected signal feature records must be non-empty.")

    validated = []
    seen_variants = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Signal cache record {index} must be an object.")
        variant_id = record.get("variant_id")
        if variant_id in seen_variants:
            raise ValueError(f"Signal cache has duplicate variant_id {variant_id!r}.")
        seen_variants.add(variant_id)
        expected_feature = expected_by_variant.get(variant_id)
        if expected_feature is None:
            raise ValueError(f"Signal cache record {index} has an unexpected variant_id.")
        expected_features = np.asarray(expected_feature["features"], dtype=np.float64)
        expected_logit = float(model.logits(((expected_features - mean) / scale)[None, :])[0])
        expected_probability = 1 / (1 + math.exp(-max(-40, min(40, expected_logit))))
        expected_identity = _cache_identity(
            expected_feature,
            manifest_metadata=metadata,
            checkpoint_revision=checkpoint_revision,
            normalization_revision=normalization_revision,
            signal_logit=expected_logit,
            pred=expected_probability,
        )
        for field, value in expected_identity.items():
            if not _strict_json_equal(record.get(field), value):
                if field in {
                    "source_id",
                    "materialized_sha256",
                    "materialized_encoding",
                    "signal_logit",
                    "pred",
                }:
                    raise ValueError(f"Signal cache record {index} has incompatible {field}.")
                raise ValueError(f"Signal cache record {index} has stale {field} metadata.")
        for field in ("split", "authenticity_label", "condition_family", "severity"):
            if not _strict_json_equal(record.get(field), expected_feature.get(field)):
                raise ValueError(f"Signal cache record {index} has incompatible {field}.")
        features = _feature_matrix([record], f"Signal cache record {index}")[0]
        if not np.array_equal(features, expected_features):
            raise ValueError(f"Signal cache record {index} has incompatible features.")
        logit = record.get("signal_logit")
        probability = record.get("pred")
        if (
            isinstance(logit, bool)
            or not isinstance(logit, (int, float))
            or not math.isfinite(logit)
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"Signal cache record {index} contains an invalid logit or probability.")
        cache_key = "signal-cache-v1-" + hashlib.sha256(
            _canonical_json(expected_identity).encode("utf-8")
        ).hexdigest()
        if record.get("cache_key") != cache_key:
            raise ValueError(f"Signal cache record {index} has a stale cache_key.")
        validated.append(record)
    if seen_variants != set(expected_by_variant):
        raise ValueError("Signal cache is missing expected variants.")
    if [record.get("variant_id") for record in records] != list(expected_by_variant):
        raise ValueError("Signal cache record order is incompatible with expected features.")
    return validated


def evaluate_signal_only(records: Iterable[dict]) -> dict:
    rows = list(records)
    _reject_evaluation_only_records(rows, "Signal-only selection metrics")
    if not rows or any(row.get("split") != ALLOWED_SELECTION_SPLIT for row in rows):
        raise ValueError("Signal-only selection metrics require internal-validation observations.")
    _validate_canonical_selection_matrix(rows)
    seen_variants = set()
    for index, row in enumerate(rows):
        variant_id = row.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in seen_variants:
            raise ValueError(
                f"Signal-only selection metrics record {index} has a missing or duplicate variant_id."
            )
        seen_variants.add(variant_id)
        _binary_labels([row])
        logit = row.get("signal_logit")
        if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(logit):
            raise ValueError("Signal-only selection metrics require finite numeric logits.")
    from rgb_baseline import evaluate_internal_validation

    return evaluate_internal_validation(
        rows,
        score_field="signal_logit",
        metric_schema_version="signal-robustness-metric-v1",
    )
