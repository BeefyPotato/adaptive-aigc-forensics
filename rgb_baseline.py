"""Reproducible RGB robustness baseline contracts for Track 5."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable

from rgb_expert import load_model_metadata, predict_experiment_observations


ALLOWED_SPLITS = {"expert-training", "fusion-training", "internal-validation", "sealed-internal-test"}
METRIC_FAMILIES = ("jpeg", "blur", "resize", "noise", "color", "crop")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _auroc(records: list[dict]) -> float:
    positives = [record for record in records if record["authenticity_label"] == 1]
    negatives = [record for record in records if record["authenticity_label"] == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both authenticity classes.")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive["rgb_logit"] > negative["rgb_logit"]:
                wins += 1
            elif positive["rgb_logit"] == negative["rgb_logit"]:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _threshold_diagnostics(records: list[dict]) -> dict:
    candidates = sorted({record["rgb_logit"] for record in records}, reverse=True)
    candidates = [math.inf, *candidates, -math.inf]
    best = None
    for threshold in candidates:
        tp = sum(record["authenticity_label"] == 1 and record["rgb_logit"] >= threshold for record in records)
        tn = sum(record["authenticity_label"] == 0 and record["rgb_logit"] < threshold for record in records)
        positives = sum(record["authenticity_label"] == 1 for record in records)
        negatives = len(records) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("Threshold selection requires both authenticity classes.")
        tpr, tnr = tp / positives, tn / negatives
        candidate = (tpr + tnr - 1, -threshold, threshold, tpr, tnr)
        if best is None or candidate > best:
            best = candidate
    _, _, threshold, tpr, tnr = best
    return {
        "status": "provisional-internal-validation-only",
        "selection_rule": "maximum-youden-j",
        "threshold_logit": threshold,
        "balanced_accuracy": (tpr + tnr) / 2,
        "sensitivity": tpr,
        "specificity": tnr,
    }


def evaluate_internal_validation(cache_records: Iterable[dict]) -> dict:
    records = [record for record in cache_records if record.get("split") == "internal-validation"]
    if not records:
        raise ValueError("Internal-validation metrics require internal-validation records.")
    for index, record in enumerate(records):
        if record.get("authenticity_label") not in (0, 1):
            raise ValueError(f"Internal-validation record {index} requires a binary authenticity label.")
        _finite_number(record.get("rgb_logit"), f"Internal-validation record {index}.rgb_logit")

    by_condition = {}
    for family in ("clean", *METRIC_FAMILIES):
        family_records = [record for record in records if record.get("condition_family") == family]
        if not family_records:
            raise ValueError(f"Internal-validation records are missing condition family {family}.")
        severities = sorted({str(record.get("severity")) for record in family_records})
        by_severity = {
            severity: _auroc([record for record in family_records if str(record.get("severity")) == severity])
            for severity in severities
        }
        by_condition[family] = {
            "auroc": sum(by_severity.values()) / len(by_severity),
            "auroc_by_severity": by_severity,
        }

    clean = by_condition["clean"]["auroc"]
    corrupted = {family: by_condition[family]["auroc"] for family in METRIC_FAMILIES}
    worst = min(
        (
            (score, family, severity)
            for family in METRIC_FAMILIES
            for severity, score in by_condition[family]["auroc_by_severity"].items()
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    mean_corrupted = sum(corrupted.values()) / len(corrupted)
    return {
        "metric_schema_version": "rgb-robustness-metric-v1",
        "evaluation_split": "internal-validation",
        "clean_auroc": clean,
        "corruption_families": by_condition,
        "mean_corrupted_auroc": mean_corrupted,
        "all_condition_macro_auroc": (clean + sum(corrupted.values())) / 7,
        "worst_family_severity": {"family": worst[1], "severity": worst[2], "auroc": worst[0]},
        "degradation_drop": clean - mean_corrupted,
        "degradation_retention": mean_corrupted / clean if clean else None,
        "threshold_diagnostics": _threshold_diagnostics(records),
    }


def validate_rgb_cache(cache: dict, manifest: dict, *, resolution: int = 384) -> list[dict]:
    if cache.get("cache_schema_version") != "rgb-robustness-cache-v1":
        raise ValueError("Unsupported RGB robustness cache schema.")
    metadata = load_model_metadata()
    expected = {
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "manifest_sha256": _canonical_sha256(manifest),
        "checkpoint_revision": metadata["models"][str(resolution)]["revision"],
        "preprocessing_version": metadata["preprocessing_version"],
        "corruption_version": manifest.get("corruption", {}).get("transform_implementation_version"),
    }
    for field, value in expected.items():
        if not value or cache.get(field) != value:
            raise ValueError(
                f"RGB cache {field} is stale or incompatible: received {cache.get(field)!r}; expected {value!r}."
            )
    records = cache.get("records")
    if not isinstance(records, list):
        raise ValueError("RGB cache records must be an array.")
    return records


def compare_deterministic_subset(expected: Iterable[dict], repeated: Iterable[dict], tolerance: float) -> dict:
    expected_by_id = {record["variant_id"]: record for record in expected}
    repeated_records = list(repeated)
    maximum_difference = 0.0
    for record in repeated_records:
        prior = expected_by_id.get(record.get("variant_id"))
        if prior is None:
            raise ValueError(f"Rerun returned unknown variant {record.get('variant_id')!r}.")
        difference = abs(_finite_number(prior.get("rgb_logit"), "cached rgb_logit") - _finite_number(record.get("rgb_logit"), "rerun rgb_logit"))
        maximum_difference = max(maximum_difference, difference)
        if difference > tolerance:
            raise ValueError(
                f"RGB deterministic rerun mismatch for {record['variant_id']}: {difference} exceeds {tolerance}."
            )
    return {
        "subset_size": len(repeated_records),
        "numeric_tolerance": tolerance,
        "maximum_absolute_logit_difference": maximum_difference,
        "status": "passed",
    }


def run_rgb_baseline(
    manifest: dict,
    backend,
    *,
    dataset_root: Path,
    resolution: int = 384,
    batch_size: int = 8,
    precision: str = "float32",
    retries: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    if manifest.get("manifest_schema_version") != "track5-manifest-v1":
        raise ValueError("RGB baseline requires a track5-manifest-v1 manifest.")
    observations = []
    for index, source in enumerate(manifest.get("sources", [])):
        if source.get("split") == "organizer-demonstration":
            raise ValueError("Organizer demonstration sources cannot enter the RGB development baseline.")
        if source.get("split") not in ALLOWED_SPLITS:
            raise ValueError(f"Source {index} has unsupported split {source.get('split')!r}.")
    for index, observation in enumerate(manifest.get("observations", [])):
        split = observation.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Observation {index} has unsupported split {split!r}.")
        image_path = Path(observation.get("materialized_image_path", observation.get("image_path", "")))
        observations.append({**observation, "image_path": dataset_root / image_path})

    started = clock()
    failure_count = retry_count = 0
    try:
        predicted = predict_experiment_observations(
            observations, backend, resolution=resolution, batch_size=batch_size
        )
    except Exception:
        failure_count += 1
        if retries <= 0:
            raise
        retry_count += 1
        predicted = predict_experiment_observations(
            observations, backend, resolution=resolution, batch_size=batch_size
        )
    elapsed = max(clock() - started, 0.0)
    record_by_variant = {record["variant_id"]: record for record in observations}
    records = []
    for artifact in predicted:
        observation = record_by_variant[artifact["variant_id"]]
        record = {
            **artifact,
            "split": observation["split"],
            "condition_family": observation["condition_family"],
            "severity": observation["severity"],
            "corruption_version": manifest["corruption"]["transform_implementation_version"],
        }
        if observation["split"] != "sealed-internal-test":
            record["authenticity_label"] = observation["authenticity_label"]
        records.append(record)

    peak_memory = getattr(backend, "peak_memory_bytes", None)
    device = getattr(backend, "device", "unspecified")
    return {
        "cache_schema_version": "rgb-robustness-cache-v1",
        "manifest_schema_version": manifest["manifest_schema_version"],
        "manifest_sha256": _canonical_sha256(manifest),
        "checkpoint_revision": load_model_metadata()["models"][str(resolution)]["revision"],
        "preprocessing_version": load_model_metadata()["preprocessing_version"],
        "corruption_version": manifest["corruption"]["transform_implementation_version"],
        "profile": {
            "wall_clock_seconds": elapsed,
            "observations_per_second": len(records) / elapsed if elapsed else None,
            "batch_size": batch_size,
            "peak_gpu_memory_bytes": peak_memory,
            "device": device,
            "precision": precision,
            "failure_count": failure_count,
            "retry_count": retry_count,
        },
        "records": records,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
