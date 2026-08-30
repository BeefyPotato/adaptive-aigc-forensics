"""Deterministic low-level signal expert for AI-generated image detection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shared_observation import prepare_shared_expert_rgb


SIGNAL_REPRESENTATION_VERSION = "signal-representation-v1"
SIGNAL_ARTIFACT_VERSION = "signal-feature-logit-v1"
NORMALIZATION_SCHEMA_VERSION = "signal-normalization-v1"
CHECKPOINT_SCHEMA_VERSION = "signal-mlp-v1"
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


def decode_expert_rgb(path: Path | str, *, resolution: int = 384) -> np.ndarray:
    """Decode the shared post-corruption checkpoint crop into RGB [0, 1]."""
    return prepare_shared_expert_rgb(path, resolution=resolution).astype(np.float64) / 255.0


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


def _fourier_features(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = luminance - luminance.mean()
    window = np.outer(np.hanning(luminance.shape[0]), np.hanning(luminance.shape[1]))
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2)
    yy, xx = np.indices(spectrum.shape, dtype=np.float64)
    radius = np.sqrt((yy - (spectrum.shape[0] - 1) / 2) ** 2 + (xx - (spectrum.shape[1] - 1) / 2) ** 2)
    normalized = radius / max(float(radius.max()), np.finfo(np.float64).eps)
    bins = np.minimum((normalized * 16).astype(np.int64), 15)
    values = np.asarray([spectrum[bins == index].mean() for index in range(16)])
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
    standardized = residual / max(residual_std, np.finfo(np.float64).eps)
    sign_changes = np.concatenate([
        (residual[:, 1:] * residual[:, :-1] < 0).reshape(-1),
        (residual[1:, :] * residual[:-1, :] < 0).reshape(-1),
    ])
    residual_features = np.asarray([
        np.abs(residual).mean(), residual_std, np.mean(standardized ** 4) - 3,
        sign_changes.mean(),
    ])
    features = np.concatenate([fourier, neighbour, residual_features]).astype(np.float64)
    if features.shape != (26,) or not np.isfinite(features).all():
        raise ValueError("Signal representation must contain 26 finite values.")
    result = {"version": SIGNAL_REPRESENTATION_VERSION, "feature_names": FEATURE_NAMES, "features": features}
    if include_maps:
        result["maps"] = {"luminance": luminance, "spectrum": spectrum, "high_pass": residual, "residual": residual}
    return result


def fit_normalization(records: Iterable[dict], *, manifest_metadata: dict) -> dict:
    records = list(records)
    if not records or any(record.get("split") != ALLOWED_TRAIN_SPLIT for record in records):
        raise ValueError("Normalization may be fit only on expert-training observations.")
    source_ids = {record["source_id"] for record in records}
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 26 or not np.isfinite(features).all():
        raise ValueError("Normalization requires finite 26-value feature records.")
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "fit_split": ALLOWED_TRAIN_SPLIT,
        "source_count": len(source_ids),
        "observation_count": len(records),
        "manifest_metadata": manifest_metadata,
        "mean": features.mean(axis=0).tolist(),
        "scale": scale.tolist(),
    }


def _validate_normalization(normalization: dict, manifest_metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    if normalization.get("schema_version") != NORMALIZATION_SCHEMA_VERSION or normalization.get("fit_split") != ALLOWED_TRAIN_SPLIT:
        raise ValueError("Signal normalization schema or fit split is incompatible.")
    if normalization.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION:
        raise ValueError("Signal normalization representation version is stale.")
    if normalization.get("manifest_metadata") != manifest_metadata:
        raise ValueError("Signal normalization manifest metadata is stale.")
    mean, scale = np.asarray(normalization["mean"]), np.asarray(normalization["scale"])
    if mean.shape != (26,) or scale.shape != (26,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Signal normalization must contain 26 finite mean/positive scale values.")
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


def train_signal_mlp(training_records: Iterable[dict], validation_records: Iterable[dict], normalization: dict, *, manifest_metadata: dict, seed: int = 61, epochs: int = 200, learning_rate: float = 0.02) -> tuple[SignalMLP, dict]:
    training, validation = list(training_records), list(validation_records)
    if not training or any(row.get("split") != ALLOWED_TRAIN_SPLIT for row in training):
        raise ValueError("Signal MLP weights may use only expert-training observations.")
    if not validation or any(row.get("split") != ALLOWED_SELECTION_SPLIT for row in validation):
        raise ValueError("Signal checkpoint selection may use only internal-validation observations.")
    if {row["source_id"] for row in training} & {row["source_id"] for row in validation}:
        raise ValueError("Signal training and validation sources must be disjoint.")
    mean, scale = _validate_normalization(normalization, manifest_metadata)
    x = (np.asarray([row["features"] for row in training]) - mean) / scale
    y = np.asarray([row["authenticity_label"] for row in training], dtype=np.float64)
    vx = (np.asarray([row["features"] for row in validation]) - mean) / scale
    vy = np.asarray([row["authenticity_label"] for row in validation], dtype=np.float64)
    rng = np.random.default_rng(seed)
    model = SignalMLP(rng.normal(0, 0.1, (26, 16)), np.zeros(16), rng.normal(0, 0.1, 16), 0.0)
    best, best_loss, best_epoch = None, math.inf, -1
    for epoch in range(epochs):
        hidden = np.tanh(x @ model.input_weights + model.input_bias)
        logits = hidden @ model.output_weights + model.output_bias
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        delta = (probability - y) / len(y)
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
        validation_loss = _bce(model.logits(vx), vy)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best = SignalMLP(*(np.copy(value) if isinstance(value, np.ndarray) else value for value in (model.input_weights, model.input_bias, model.output_weights, model.output_bias)))
    metadata = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "seed": seed, "epochs": epochs, "selected_epoch": best_epoch, "validation_bce": best_loss, "training_split": ALLOWED_TRAIN_SPLIT, "selection_split": ALLOWED_SELECTION_SPLIT, "hidden_units": 16, "manifest_metadata": manifest_metadata}
    return best, metadata


def write_model_bundle(path: Path | str, model: SignalMLP, metadata: dict, normalization: dict) -> None:
    payload = {**metadata, "normalization": normalization, "weights": {"input": model.input_weights.tolist(), "input_bias": model.input_bias.tolist(), "output": model.output_weights.tolist(), "output_bias": model.output_bias}}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_model_bundle(path: Path | str, *, manifest_metadata: dict) -> tuple[SignalMLP, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or payload.get("manifest_metadata") != manifest_metadata:
        raise ValueError("Signal checkpoint schema or manifest metadata is stale.")
    _validate_normalization(payload["normalization"], manifest_metadata)
    weights = payload["weights"]
    model = SignalMLP(np.asarray(weights["input"]), np.asarray(weights["input_bias"]), np.asarray(weights["output"]), float(weights["output_bias"]))
    if model.input_weights.shape != (26, 16) or model.input_bias.shape != (16,) or model.output_weights.shape != (16,):
        raise ValueError("Signal checkpoint weight dimensions are incompatible.")
    return model, payload


def cache_signal_predictions(records: Iterable[dict], model: SignalMLP, normalization: dict, *, manifest_metadata: dict, checkpoint_revision: str) -> list[dict]:
    mean, scale = _validate_normalization(normalization, manifest_metadata)
    output = []
    for record in records:
        features = np.asarray(record["features"], dtype=np.float64)
        if features.shape != (26,) or not np.isfinite(features).all():
            raise ValueError("Signal cache requires 26 finite features.")
        logit = float(model.logits(((features - mean) / scale)[None, :])[0])
        probability = 1 / (1 + math.exp(-max(-40, min(40, logit))))
        identity = {"artifact_schema_version": SIGNAL_ARTIFACT_VERSION, "checkpoint_revision": checkpoint_revision, "manifest_metadata": manifest_metadata, "signal_representation_version": SIGNAL_REPRESENTATION_VERSION, "variant_id": record["variant_id"]}
        output.append({**identity, "source_id": record["source_id"], "features": features.tolist(), "signal_logit": logit, "pred": probability, "cache_key": "signal-cache-v1-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    return output


def validate_signal_cache(records: Iterable[dict], *, manifest_metadata: dict, checkpoint_revision: str) -> list[dict]:
    validated = []
    for index, record in enumerate(records):
        expected = {
            "artifact_schema_version": SIGNAL_ARTIFACT_VERSION,
            "checkpoint_revision": checkpoint_revision,
            "manifest_metadata": manifest_metadata,
            "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
            "variant_id": record.get("variant_id"),
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(f"Signal cache record {index} has stale {field} metadata.")
        features = np.asarray(record.get("features"), dtype=np.float64)
        if features.shape != (26,) or not np.isfinite(features).all():
            raise ValueError(f"Signal cache record {index} must contain 26 finite features.")
        if not math.isfinite(record.get("signal_logit", math.nan)) or not math.isfinite(record.get("pred", math.nan)) or not 0 <= record["pred"] <= 1:
            raise ValueError(f"Signal cache record {index} contains an invalid logit or probability.")
        identity = {field: record[field] for field in expected}
        cache_key = "signal-cache-v1-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if record.get("cache_key") != cache_key:
            raise ValueError(f"Signal cache record {index} has a stale cache_key.")
        validated.append(record)
    return validated


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean(positive[:, None] > negative[None, :]) + 0.5 * np.mean(positive[:, None] == negative[None, :]))


def evaluate_signal_only(records: Iterable[dict]) -> dict:
    rows = list(records)
    if not rows or any(row.get("split") != ALLOWED_SELECTION_SPLIT for row in rows):
        raise ValueError("Signal-only selection metrics require internal-validation observations.")
    families = sorted({row["condition_family"] for row in rows})
    metrics = {}
    for family in families:
        selected = [row for row in rows if row["condition_family"] == family]
        metrics[family] = _auroc(np.asarray([row["authenticity_label"] for row in selected]), np.asarray([row["pred"] for row in selected]))
    return {"metric_schema_version": "signal-metric-v1", "split": ALLOWED_SELECTION_SPLIT, "observation_count": len(rows), "auroc_by_corruption_family": metrics}
