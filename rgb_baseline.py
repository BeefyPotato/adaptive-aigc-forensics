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
    positives = sum(record["authenticity_label"] == 1 for record in records)
    negatives = len(records) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both authenticity classes.")
    ordered = sorted(records, key=lambda record: record["rgb_logit"])
    rank_sum = 0.0
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and ordered[end]["rgb_logit"] == ordered[offset]["rgb_logit"]:
            end += 1
        average_rank = ((offset + 1) + end) / 2
        rank_sum += average_rank * sum(
            record["authenticity_label"] == 1 for record in ordered[offset:end]
        )
        offset = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _threshold_diagnostics(records: list[dict]) -> dict:
    positives = sum(record["authenticity_label"] == 1 for record in records)
    negatives = len(records) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold selection requires both authenticity classes.")
    ordered = sorted(records, key=lambda record: record["rgb_logit"], reverse=True)
    true_positives = false_positives = 0
    best = None
    offset = 0
    while offset < len(ordered):
        threshold = ordered[offset]["rgb_logit"]
        end = offset
        while end < len(ordered) and ordered[end]["rgb_logit"] == threshold:
            true_positives += ordered[end]["authenticity_label"] == 1
            false_positives += ordered[end]["authenticity_label"] == 0
            end += 1
        tpr = true_positives / positives
        tnr = (negatives - false_positives) / negatives
        candidate = (tpr + tnr - 1, threshold, threshold, tpr, tnr)
        if best is None or candidate > best:
            best = candidate
        offset = end
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
        "rgb_preprocessing_version": metadata["preprocessing_version"],
        "shared_observation_preprocessing_version": manifest.get("materialization", {}).get(
            "shared_observation_preprocessing_version"
        ),
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
    observations = manifest.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Materialized manifest observations must be an array.")
    expected = {record.get("variant_id"): record for record in observations}
    if None in expected or len(expected) != len(observations):
        raise ValueError("Materialized manifest variant identifiers must be non-empty and unique.")
    received = {}
    for index, record in enumerate(records):
        variant_id = record.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in received:
            raise ValueError(f"RGB cache record {index} has a missing or duplicate variant identifier.")
        observation = expected.get(variant_id)
        if observation is None:
            raise ValueError(f"RGB cache contains unknown variant {variant_id}.")
        received[variant_id] = record
        _finite_number(record.get("rgb_logit"), f"RGB cache record {index}.rgb_logit")
        for field in ("source_id", "split", "condition_family", "severity", "materialized_sha256"):
            if record.get(field) != observation.get(field):
                raise ValueError(f"RGB cache record {variant_id} has incompatible {field}.")
        for field in (
            "manifest_sha256",
            "shared_observation_preprocessing_version",
            "rgb_preprocessing_version",
            "corruption_version",
            "checkpoint_revision",
        ):
            if record.get(field) != cache.get(field):
                raise ValueError(f"RGB cache record {variant_id} has incompatible {field}.")
        identity = {
            "checkpoint_revision": cache["checkpoint_revision"],
            "corruption_version": cache["corruption_version"],
            "manifest_sha256": cache["manifest_sha256"],
            "materialized_sha256": record["materialized_sha256"],
            "rgb_preprocessing_version": cache["rgb_preprocessing_version"],
            "shared_observation_preprocessing_version": cache[
                "shared_observation_preprocessing_version"
            ],
            "variant_id": variant_id,
        }
        cache_key = "rgb-robustness-record-v1-" + _canonical_sha256(identity)
        if record.get("cache_key") != cache_key:
            raise ValueError(f"RGB cache record {variant_id} has a stale or invalid cache key.")
        if observation.get("split") == "sealed-internal-test":
            if "authenticity_label" in record:
                raise ValueError(f"RGB cache record {variant_id} exposes a sealed-test label.")
        elif record.get("authenticity_label") != observation.get("authenticity_label"):
            raise ValueError(f"RGB cache record {variant_id} has an incompatible authenticity label.")
    if received.keys() != expected.keys():
        missing = sorted(expected.keys() - received.keys())
        raise ValueError(f"RGB cache is missing {len(missing)} manifest variants, including {missing[0]}.")
    return records


def compare_deterministic_subset(expected: Iterable[dict], repeated: Iterable[dict], tolerance: float) -> dict:
    expected_records = list(expected)
    repeated_records = list(repeated)
    if not expected_records or not repeated_records:
        raise ValueError("RGB deterministic rerun subset must be non-empty.")
    expected_by_id = {record.get("variant_id"): record for record in expected_records}
    repeated_by_id = {record.get("variant_id"): record for record in repeated_records}
    if None in expected_by_id or len(expected_by_id) != len(expected_records):
        raise ValueError("Cached rerun subset variant identifiers must be non-empty and unique.")
    if None in repeated_by_id or len(repeated_by_id) != len(repeated_records):
        raise ValueError("Repeated subset variant identifiers must be non-empty and unique.")
    if expected_by_id.keys() != repeated_by_id.keys():
        raise ValueError("RGB deterministic rerun must return exactly the declared subset variants.")
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
    if manifest.get("materialization_schema_version") != "track5-materialized-observations-v1":
        raise ValueError(
            "RGB baseline requires materialized Track 5 observations; recipe-only manifests are unsafe."
        )
    materialization = manifest.get("materialization")
    if not isinstance(materialization, dict):
        raise ValueError("RGB baseline requires materialization metadata.")
    shared_version = materialization.get("shared_observation_preprocessing_version")
    if shared_version != manifest.get("corruption", {}).get("preprocessing_version"):
        raise ValueError("Materialized observations have an incompatible shared preprocessing version.")
    dataset_root = dataset_root.resolve()
    observations = []
    seen_variants = set()
    source_by_id = {source.get("source_id"): source for source in manifest.get("sources", [])}
    if None in source_by_id or len(source_by_id) != len(manifest.get("sources", [])):
        raise ValueError("Materialized manifest source identifiers must be non-empty and unique.")
    for index, source in enumerate(manifest.get("sources", [])):
        if source.get("split") == "organizer-demonstration":
            raise ValueError("Organizer demonstration sources cannot enter the RGB development baseline.")
        if source.get("split") not in ALLOWED_SPLITS:
            raise ValueError(f"Source {index} has unsupported split {source.get('split')!r}.")
    for index, observation in enumerate(manifest.get("observations", [])):
        variant_id = observation.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in seen_variants:
            raise ValueError(f"Observation {index} has a missing or duplicate variant identifier.")
        seen_variants.add(variant_id)
        split = observation.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Observation {index} has unsupported split {split!r}.")
        source = source_by_id.get(observation.get("source_id"))
        if source is None or source.get("split") != split:
            raise ValueError(f"Observation {index} is missing its source or disagrees with its partition.")
        materialized_path = observation.get("materialized_image_path")
        materialized_sha256 = observation.get("materialized_sha256")
        if not isinstance(materialized_path, str) or not materialized_path or not isinstance(materialized_sha256, str):
            raise ValueError(f"Observation {index} is not materialized.")
        image_path = (dataset_root / materialized_path).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(f"Observation {index} materialized path escapes the dataset root.") from error
        if not image_path.is_file():
            raise ValueError(f"Observation {index} materialized image is missing: {materialized_path}.")
        actual_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_sha256 != materialized_sha256:
            raise ValueError(f"Observation {index} materialized image checksum mismatch.")
        observations.append({**observation, "image_path": image_path})
    if materialization.get("observation_count") != len(observations):
        raise ValueError("Materialized observation count is incompatible with the manifest.")

    reset_peak_memory = getattr(backend, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()
    started = clock()
    failure_count = retry_count = 0
    predicted = []
    for offset in range(0, len(observations), batch_size):
        batch = observations[offset : offset + batch_size]
        try:
            result = predict_experiment_observations(
                batch, backend, resolution=resolution, batch_size=batch_size
            )
        except Exception:
            failure_count += 1
            if retries <= 0:
                raise
            retry_count += 1
            result = predict_experiment_observations(
                batch, backend, resolution=resolution, batch_size=batch_size
            )
        predicted.extend(result)
    elapsed = max(clock() - started, 0.0)
    record_by_variant = {record["variant_id"]: record for record in observations}
    manifest_sha256 = _canonical_sha256(manifest)
    metadata = load_model_metadata()
    checkpoint_revision = metadata["models"][str(resolution)]["revision"]
    rgb_preprocessing_version = metadata["preprocessing_version"]
    corruption_version = manifest["corruption"]["transform_implementation_version"]
    records = []
    for artifact in predicted:
        observation = record_by_variant[artifact["variant_id"]]
        record = {
            **artifact,
            "split": observation["split"],
            "condition_family": observation["condition_family"],
            "severity": observation["severity"],
            "materialized_sha256": observation["materialized_sha256"],
            "manifest_sha256": manifest_sha256,
            "shared_observation_preprocessing_version": shared_version,
            "rgb_preprocessing_version": rgb_preprocessing_version,
            "corruption_version": corruption_version,
        }
        if observation["split"] != "sealed-internal-test":
            record["authenticity_label"] = observation["authenticity_label"]
        identity = {
            "checkpoint_revision": checkpoint_revision,
            "corruption_version": corruption_version,
            "manifest_sha256": manifest_sha256,
            "materialized_sha256": record["materialized_sha256"],
            "rgb_preprocessing_version": rgb_preprocessing_version,
            "shared_observation_preprocessing_version": shared_version,
            "variant_id": record["variant_id"],
        }
        record["cache_key"] = "rgb-robustness-record-v1-" + _canonical_sha256(identity)
        records.append(record)

    peak_memory = getattr(backend, "peak_memory_bytes", None)
    device = getattr(backend, "device", "unspecified")
    return {
        "cache_schema_version": "rgb-robustness-cache-v1",
        "manifest_schema_version": manifest["manifest_schema_version"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_revision": checkpoint_revision,
        "rgb_preprocessing_version": rgb_preprocessing_version,
        "shared_observation_preprocessing_version": shared_version,
        "corruption_version": corruption_version,
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
