"""Calibration and static-fusion contracts for frozen Track 5 experts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

from rgb_baseline import evaluate_internal_validation
from safe_output import atomic_write_bytes, resolve_output_directory


CALIBRATION_SCHEMA = "expert-calibration-v1"
MATCHED_CACHE_SCHEMA = "matched-frozen-expert-logits-v2"
MATCHED_BINDING_SCHEMA = "matched-frozen-expert-binding-v1"
CALIBRATED_CACHE_SCHEMA = "calibrated-frozen-expert-logits-v1"
BUNDLE_SCHEMA = "static-fallback-bundle-v1"
CORRECTED_BUNDLE_SCHEMA = "static-fallback-bundle-v2"
COMPLETION_SCHEMA = "static-fallback-completion-v1"
CORRECTED_COMPLETION_SCHEMA = "static-fallback-completion-v2"
APPROVED_ISSUE7_LEGACY_BINDINGS = {
    "bundle_sha256": "f205605dc2a9fbfc10bcb3ec75ba396aa3d150a7bb04196d6db2236041dd1a76",
    "bundle_revision": "static-fallback-bundle-v1-ad1009dfb06d964e1f4c5e6432e96267fbab8c47f076dce5c8203e83820048d9",
    "fusion_training_file_sha256": "bc3b161feb136ce7af1fe39e0e0c499ab3560a10e43c0493ce489fe06525d9d3",
    "internal_validation_file_sha256": "666260c84da942c54fa5eae5db2472a3bc9a434b220c963d6bfbd9f6913611a6",
    "legacy_completion_file_sha256": "ab09bcf3d7153c8d25b84bcbc67c0e93f707faaf782a69d9f4e95aa792512d95",
    "signal_model_file_sha256": "cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5",
}
FAMILIES = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
MATCHED_RECORD_PROVENANCE_FIELDS = (
    "manifest_sha256", "rgb_checkpoint_revision", "rgb_preprocessing_version",
    "signal_checkpoint_revision", "signal_normalization_revision",
    "signal_feature_extraction_version", "corruption_version",
)
SELECTION_RULE = {
    "rule_version": "static-fallback-selection-v1",
    "candidate": "learned-static-fusion",
    "reference": "calibrated-rgb-only",
    "minimum_all_condition_macro_auroc_gain": 0.005,
    "maximum_brier_worsening": 0.002,
    "require_source_bootstrap_95pct_gain_lower_bound_above_zero": True,
    "otherwise": "calibrated-rgb-only",
}


def inspect_signal_handoff_archive(path: Path) -> dict:
    required = {
        "signal-hackathon/signal-run.json", "signal-hackathon/signal-model.json",
        "signal-hackathon/signal-normalization.json", "signal-hackathon/signal-validation-logits.json",
    }
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Signal handoff archive repeats a member name.")
        for info in infos:
            pure = PurePosixPath(info.filename); mode = (info.external_attr >> 16) & 0xFFFF
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename or (pure.parts and ":" in pure.parts[0]):
                raise ValueError("Signal handoff archive contains a path traversal.")
            if stat.S_ISLNK(mode) or (mode & 0o111 and not info.is_dir()):
                raise ValueError("Signal handoff archive contains a link or executable member.")
        if not required.issubset(names) or archive.testzip() is not None:
            raise ValueError("Signal handoff archive is incomplete or corrupt.")
        run = json.loads(archive.read("signal-hackathon/signal-run.json"))
        for name, expected in run.get("artifact_sha256", {}).items():
            member = f"signal-hackathon/{name}"
            if member not in names or hashlib.sha256(archive.read(member)).hexdigest() != expected:
                raise ValueError(f"Signal handoff artifact {name} has an invalid checksum.")
        return {"member_count": len(infos), "uncompressed_bytes": sum(info.file_size for info in infos), "run": run}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number.")
    return float(value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1 / (1 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def validate_matched_records(records: Iterable[dict], *, expected_split: str) -> list[dict]:
    rows = list(records)
    if not rows:
        raise ValueError("Matched frozen-expert records must not be empty.")
    seen = set()
    labels_by_source = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Matched record {index} must be an object.")
        for field in ("source_id", "variant_id", "condition_family", "severity"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Matched record {index} requires {field}.")
        if row["variant_id"] in seen:
            raise ValueError(f"Matched records repeat variant_id {row['variant_id']!r}.")
        seen.add(row["variant_id"])
        if row.get("split") != expected_split:
            raise ValueError(f"Matched records may use only {expected_split} observations.")
        if row.get("authenticity_label") not in (0, 1):
            raise ValueError(f"Matched record {index} requires a binary authenticity label.")
        for field in ("rgb_logit", "signal_logit"):
            _finite(row.get(field), f"Matched record {index}.{field}")
        for field in ("materialized_sha256", "manifest_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Matched record {index}.{field} must be a lowercase SHA-256.")
        for field in (
            "rgb_checkpoint_revision", "rgb_preprocessing_version",
            "signal_checkpoint_revision", "signal_normalization_revision",
            "signal_feature_extraction_version", "corruption_version", "cache_key",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Matched record {index} requires {field}.")
        identity = {field: row[field] for field in (
            "variant_id", "materialized_sha256", "manifest_sha256",
            "rgb_checkpoint_revision", "rgb_preprocessing_version",
            "signal_checkpoint_revision", "signal_normalization_revision",
            "signal_feature_extraction_version", "corruption_version",
        )}
        if row["cache_key"] != "matched-frozen-expert-v1-" + _sha256(identity):
            raise ValueError(f"Matched record {index} has an invalid deterministic cache key.")
        label = row["authenticity_label"]
        prior = labels_by_source.setdefault(row["source_id"], label)
        if prior != label:
            raise ValueError(f"Source {row['source_id']!r} changes authenticity label.")
    if set(labels_by_source.values()) != {0, 1}:
        raise ValueError("Matched records require both authenticity classes.")
    revisions = {tuple(row[field] for field in (
        "manifest_sha256", "rgb_checkpoint_revision", "rgb_preprocessing_version",
        "signal_checkpoint_revision", "signal_normalization_revision",
        "signal_feature_extraction_version", "corruption_version",
    )) for row in rows}
    if len(revisions) != 1:
        raise ValueError("Matched records contain stale or substituted expert revisions.")
    return rows


def validate_matched_cache(
    cache: dict,
    *,
    expected_split: str,
    expected_provenance: dict | None = None,
    expected_binding: dict | None = None,
) -> list[dict]:
    if cache.get("cache_schema_version") != MATCHED_CACHE_SCHEMA:
        raise ValueError("Matched frozen-expert cache schema is stale or incompatible.")
    if cache.get("split") != expected_split:
        raise ValueError("Matched frozen-expert cache top-level split is relabelled or incompatible.")
    records = cache.get("records")
    if not isinstance(records, list) or cache.get("records_sha256") != _sha256(records):
        raise ValueError("Matched frozen-expert cache records or digest are stale.")
    rows = validate_matched_records(records, expected_split=expected_split)
    provenance = cache.get("provenance")
    if not isinstance(provenance, dict) or any(
        row.get(field) != provenance.get(field)
        for row in rows
        for field in MATCHED_RECORD_PROVENANCE_FIELDS
    ):
        raise ValueError("Matched frozen-expert cache provenance disagrees with its records.")
    if expected_provenance is not None and _canonical_bytes(provenance) != _canonical_bytes(expected_provenance):
        raise ValueError("Matched frozen-expert cache does not match trusted provenance.")
    binding = _matched_cache_binding(rows, expected_split=expected_split, provenance=provenance)
    if _canonical_bytes(cache.get("binding")) != _canonical_bytes(binding):
        raise ValueError("Matched frozen-expert cache binding is stale or incompatible.")
    if expected_binding is not None and _canonical_bytes(binding) != _canonical_bytes(expected_binding):
        raise ValueError("Matched frozen-expert cache does not match its trusted binding.")
    return rows


def _matched_cache_binding(records: list[dict], *, expected_split: str, provenance: dict) -> dict:
    return {
        "binding_schema_version": MATCHED_BINDING_SCHEMA,
        "split": expected_split,
        "source_count": len({row["source_id"] for row in records}),
        "observation_count": len(records),
        "source_ids_sha256": _sha256(sorted({row["source_id"] for row in records})),
        "variant_ids_sha256": _sha256([row["variant_id"] for row in records]),
        "record_cache_keys_sha256": _sha256([row["cache_key"] for row in records]),
        "records_sha256": _sha256(records),
        "provenance_sha256": _sha256(provenance),
    }


def build_matched_cache(records: Iterable[dict], *, provenance: dict, expected_split: str) -> dict:
    rows = validate_matched_records(records, expected_split=expected_split)
    cache = {
        "cache_schema_version": MATCHED_CACHE_SCHEMA,
        "split": expected_split,
        "provenance": json.loads(_canonical_bytes(provenance)),
        "binding": _matched_cache_binding(rows, expected_split=expected_split, provenance=provenance),
        "records_sha256": _sha256(rows),
        "records": rows,
    }
    validate_matched_cache(
        cache,
        expected_split=expected_split,
        expected_provenance=provenance,
        expected_binding=cache["binding"],
    )
    return cache


def migrate_legacy_matched_cache(
    legacy_cache: dict,
    *,
    expected_split: str,
    enriched_provenance: dict,
) -> dict:
    if legacy_cache.get("cache_schema_version") != "matched-frozen-expert-logits-v1":
        raise ValueError("Legacy matched frozen-expert cache schema is incompatible.")
    records = legacy_cache.get("records")
    if not isinstance(records, list) or legacy_cache.get("records_sha256") != _sha256(records):
        raise ValueError("Legacy matched frozen-expert cache records or digest are stale.")
    rows = validate_matched_records(records, expected_split=expected_split)
    legacy_provenance = legacy_cache.get("provenance")
    if not isinstance(legacy_provenance, dict) or any(
        legacy_provenance.get(field) != enriched_provenance.get(field)
        or any(row.get(field) != legacy_provenance.get(field) for row in rows)
        for field in MATCHED_RECORD_PROVENANCE_FIELDS
    ):
        raise ValueError("Legacy matched frozen-expert cache provenance is incompatible.")
    migrated = build_matched_cache(
        rows,
        provenance=enriched_provenance,
        expected_split=expected_split,
    )
    if migrated["records"] != records or migrated["records_sha256"] != legacy_cache["records_sha256"]:
        raise ValueError("Legacy matched frozen-expert migration changed raw records.")
    return migrated


def migrate_static_fallback_generation(
    legacy_generation_directory: Path | str,
    signal_model_path: Path | str,
    output_directory: Path | str,
    *,
    expected_legacy_bindings: dict | None = None,
) -> dict:
    expected_legacy_bindings = (
        APPROVED_ISSUE7_LEGACY_BINDINGS
        if expected_legacy_bindings is None
        else expected_legacy_bindings
    )
    if set(expected_legacy_bindings) != set(APPROVED_ISSUE7_LEGACY_BINDINGS):
        raise ValueError("Issue #7 expected legacy bindings are incomplete.")
    legacy_root = Path(legacy_generation_directory).absolute()
    bundle_path = legacy_root / "fallback" / "static-fallback-bundle.json"
    marker_path = legacy_root / "fallback" / "static-fallback.complete.json"
    try:
        bundle_bytes = bundle_path.read_bytes()
        marker_bytes = marker_path.read_bytes()
        marker = json.loads(marker_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Issue #7 legacy completion inputs are missing or invalid.") from error
    try:
        bundle = json.loads(bundle_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("Issue #7 legacy bundle is invalid.") from error
    if (
        marker.get("completion_schema_version") != COMPLETION_SCHEMA
        or marker.get("bundle_sha256") != hashlib.sha256(bundle_bytes).hexdigest()
        or marker.get("bundle_revision") != bundle.get("bundle_revision")
    ):
        raise ValueError("Issue #7 legacy completion marker is stale or incompatible.")
    if (
        marker["bundle_sha256"] != expected_legacy_bindings["bundle_sha256"]
        or marker["bundle_revision"] != expected_legacy_bindings["bundle_revision"]
        or hashlib.sha256(marker_bytes).hexdigest()
        != expected_legacy_bindings["legacy_completion_file_sha256"]
    ):
        raise ValueError("Issue #7 legacy bundle does not match the approved frozen revision.")
    validate_bundle(bundle)
    legacy_caches = {}
    cache_bytes = {}
    for split in ("fusion-training", "internal-validation"):
        name = f"matched-{split}-logits.json"
        path = legacy_root / name
        try:
            contents = path.read_bytes()
            cache = json.loads(contents)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Issue #7 legacy {split} cache is missing or invalid.") from error
        input_key = split.replace("-", "_")
        binding = bundle["input_cache_bindings"].get(input_key)
        if (
            not isinstance(binding, dict)
            or binding.get("file_sha256") != hashlib.sha256(contents).hexdigest()
            or binding.get("records_sha256") != cache.get("records_sha256")
        ):
            raise ValueError(f"Issue #7 legacy {split} cache binding is stale or incompatible.")
        if hashlib.sha256(contents).hexdigest() != expected_legacy_bindings[f"{input_key}_file_sha256"]:
            raise ValueError(f"Issue #7 legacy {split} cache does not match its approved frozen hash.")
        legacy_caches[split] = cache
        cache_bytes[split] = contents
    training_rows = validate_matched_records(
        legacy_caches["fusion-training"].get("records", []),
        expected_split="fusion-training",
    )
    validation_rows = validate_matched_records(
        legacy_caches["internal-validation"].get("records", []),
        expected_split="internal-validation",
    )
    _validate_issue7_partition_counts(training_rows, validation_rows)
    training_digest = legacy_caches["fusion-training"].get("records_sha256")
    if any(
        artifact.get("records_sha256") != training_digest
        for artifact in (
            bundle["rgb_calibrator"],
            bundle["signal_calibrator"],
            bundle["static_weight"],
        )
    ):
        raise ValueError("Issue #7 legacy calibrators or static weight do not bind the training records.")
    if (
        bundle["evaluation"].get("source_count") != len({row["source_id"] for row in validation_rows})
        or bundle["evaluation"].get("observation_count") != len(validation_rows)
    ):
        raise ValueError("Issue #7 legacy evaluation does not bind the validation observations.")
    signal_handoff = _validated_signal_model_handoff(
        Path(signal_model_path),
        expected_provenance=bundle["provenance"],
    )
    if signal_handoff["model_file_sha256"] != expected_legacy_bindings["signal_model_file_sha256"]:
        raise ValueError("Issue #7 signal model does not match its approved frozen hash.")
    enriched_provenance = signal_handoff["provenance"]
    if any(
        enriched_provenance.get(field) != value
        for field, value in bundle["provenance"].items()
    ):
        raise ValueError("Issue #7 signal model provenance disagrees with the legacy bundle.")
    signal_normalizer = signal_handoff["normalization"]
    if (
        signal_normalizer.get("normalization_revision")
        != enriched_provenance.get("signal_normalization_revision")
    ):
        raise ValueError("Issue #7 signal normalizer disagrees with frozen provenance.")
    rgb_normalizer = _build_rgb_normalizer(enriched_provenance)
    for field, value in (
        ("rgb_checkpoint_sha256", rgb_normalizer.get("checkpoint_sha256")),
        ("rgb_score_direction", rgb_normalizer.get("score_direction")),
    ):
        if value is not None:
            enriched_provenance = {**enriched_provenance, field: value}
    migrated_training = migrate_legacy_matched_cache(
        legacy_caches["fusion-training"],
        expected_split="fusion-training",
        enriched_provenance=enriched_provenance,
    )
    migrated_validation = migrate_legacy_matched_cache(
        legacy_caches["internal-validation"],
        expected_split="internal-validation",
        enriched_provenance=enriched_provenance,
    )
    input_bindings = {
        "fusion_training": {
            **bundle["input_cache_bindings"]["fusion_training"],
            "path": "legacy/matched-fusion-training-logits.json",
            "cache_schema_version": "matched-frozen-expert-logits-v1",
        },
        "internal_validation": {
            **bundle["input_cache_bindings"]["internal_validation"],
            "path": "legacy/matched-internal-validation-logits.json",
            "cache_schema_version": "matched-frozen-expert-logits-v1",
        },
        "legacy_bundle": {
            "path": "legacy/fallback/static-fallback-bundle.json",
            "file_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            "bundle_revision": bundle["bundle_revision"],
        },
        "legacy_completion": {
            "path": "legacy/fallback/static-fallback.complete.json",
            "file_sha256": hashlib.sha256(marker_bytes).hexdigest(),
            "bundle_revision": marker["bundle_revision"],
        },
        "signal_model": {
            "path": "upstream/signal-model.json",
            "file_sha256": signal_handoff["model_file_sha256"],
            "checkpoint_revision": enriched_provenance["signal_checkpoint_revision"],
            "normalization_revision": enriched_provenance["signal_normalization_revision"],
        },
    }
    return publish_corrected_generation(
        output_directory,
        legacy_bundle=bundle,
        fusion_training_cache=migrated_training,
        internal_validation_cache=migrated_validation,
        provenance=enriched_provenance,
        rgb_normalizer=rgb_normalizer,
        signal_normalizer=signal_normalizer,
        input_cache_bindings=input_bindings,
    )


def _validate_issue7_partition_counts(training_rows: list[dict], validation_rows: list[dict]) -> None:
    for name, rows, expected_sources, expected_observations in (
        ("fusion-training", training_rows, 2_000, 40_000),
        ("internal-validation", validation_rows, 400, 8_000),
    ):
        source_counts = Counter(row["source_id"] for row in rows)
        labels_by_source = {}
        for row in rows:
            labels_by_source.setdefault(row["source_id"], set()).add(row["authenticity_label"])
        if (
            len(source_counts) != expected_sources
            or len(rows) != expected_observations
            or any(len(labels) != 1 for labels in labels_by_source.values())
            or sum(next(iter(labels)) == 0 for labels in labels_by_source.values()) != expected_sources // 2
            or any(count != 20 for count in source_counts.values())
        ):
            raise ValueError(f"Issue #7 {name} partition counts or class balance are incompatible.")
    if {row["source_id"] for row in training_rows} & {row["source_id"] for row in validation_rows}:
        raise ValueError("Issue #7 fusion-training and validation sources must be disjoint.")


def _validated_signal_model_handoff(signal_model_path: Path, *, expected_provenance: dict) -> dict:
    from signal_expert import read_model_bundle

    try:
        model_bytes = signal_model_path.read_bytes()
        payload = json.loads(model_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Issue #7 signal model is missing or invalid.") from error
    _, bundle = read_model_bundle(
        signal_model_path,
        manifest_metadata=payload.get("manifest_metadata"),
        expected_experiment_provenance=payload.get("experiment_provenance"),
    )
    manifest = bundle["manifest_metadata"]
    experiment = bundle["experiment_provenance"]
    observed = {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "rgb_checkpoint_revision": expected_provenance.get("rgb_checkpoint_revision"),
        "rgb_preprocessing_version": expected_provenance.get("rgb_preprocessing_version"),
        "signal_checkpoint_revision": bundle.get("checkpoint_revision"),
        "signal_normalization_revision": bundle.get("normalization_revision"),
        "signal_feature_extraction_version": bundle.get("feature_extraction", {}).get("feature_extraction_version"),
        "corruption_version": manifest.get("corruption_version"),
        "signal_experiment_profile": experiment.get("experiment_profile"),
        "signal_acceptance_scope": experiment.get("acceptance_scope"),
    }
    if any(expected_provenance.get(field) != value for field, value in observed.items()):
        raise ValueError("Issue #7 signal model does not match frozen legacy provenance.")
    enriched = {
        **expected_provenance,
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "shared_observation_preprocessing_version": manifest.get("shared_observation_preprocessing_version"),
        "signal_representation_version": bundle.get("signal_representation_version"),
        "signal_resolution": experiment.get("resolution"),
    }
    return {
        "provenance": enriched,
        "normalization": bundle["normalization"],
        "model_file_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }


def _build_rgb_normalizer(expected_provenance: dict) -> dict:
    from rgb_expert import load_model_metadata

    metadata = load_model_metadata()
    model = metadata["models"]["384"]
    if (
        metadata.get("preprocessing_version") != expected_provenance.get("rgb_preprocessing_version")
        or model.get("revision") != expected_provenance.get("rgb_checkpoint_revision")
    ):
        raise ValueError("Issue #7 RGB preprocessing or checkpoint provenance is incompatible.")
    normalizer = {
        "normalization_schema_version": "rgb-imagenet-normalization-v1",
        "preprocessing_version": metadata["preprocessing_version"],
        "shared_observation_preprocessing_version": expected_provenance.get("shared_observation_preprocessing_version"),
        "channel_order": "rgb",
        "input_range": [0.0, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "scale": [0.229, 0.224, 0.225],
        "input_resolution": model["input_resolution"],
        "resize_short_edge": model["resize_short_edge"],
        "resize_interpolation": "bilinear",
        "center_crop": [model["input_resolution"], model["input_resolution"]],
        "tensor_layout": "chw",
        "tensor_dtype": "float32",
        "checkpoint_revision": model["revision"],
        "checkpoint_sha256": model["sha256"],
        "score_direction": metadata["score_direction"],
    }
    normalizer["normalization_revision"] = "rgb-normalization-v1-" + _sha256(normalizer)
    return normalizer


def build_calibrated_cache(
    matched_cache: dict,
    *,
    expected_split: str,
    input_file_sha256: str,
    rgb_calibrator: dict,
    signal_calibrator: dict,
    static_weight: dict,
    selected_fallback_type: str,
) -> dict:
    rows = validate_matched_cache(
        matched_cache,
        expected_split=expected_split,
        expected_provenance=matched_cache.get("provenance"),
        expected_binding=matched_cache.get("binding"),
    )
    if (
        not isinstance(input_file_sha256, str)
        or len(input_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in input_file_sha256)
    ):
        raise ValueError("Calibrated cache input file SHA-256 is invalid.")
    static_identity = dict(static_weight)
    static_revision = static_identity.pop("static_weight_revision", None)
    if static_revision != "static-fusion-weight-v1-" + _sha256(static_identity):
        raise ValueError("Calibrated cache static weight is stale or incompatible.")
    if selected_fallback_type not in ("learned-static-fusion", "calibrated-rgb-only"):
        raise ValueError("Calibrated cache selected fallback is incompatible.")
    logits = _candidate_logits(rows, rgb_calibrator, signal_calibrator, static_weight)
    selected_logits = logits[selected_fallback_type]
    calibrated_records = []
    for index, row in enumerate(rows):
        record = {
            **row,
            "matched_cache_key": row["cache_key"],
            "rgb_calibrated_logit": float(logits["calibrated-rgb-only"][index]),
            "signal_calibrated_logit": float(logits["calibrated-signal-only"][index]),
            "equal_50_50_calibrated_logit": float(logits["equal-50-50-calibrated-logit-fusion"][index]),
            "learned_static_logit": float(logits["learned-static-fusion"][index]),
            "selected_fallback_logit": float(selected_logits[index]),
            "selected_fallback_probability": float(_sigmoid(np.asarray([selected_logits[index]]))[0]),
        }
        record.pop("cache_key")
        record["cache_key"] = "calibrated-fusion-record-v1-" + _sha256(record)
        calibrated_records.append(record)
    return {
        "calibrated_cache_schema_version": CALIBRATED_CACHE_SCHEMA,
        "split": expected_split,
        "provenance": matched_cache["provenance"],
        "input_matched_cache": {
            "cache_schema_version": matched_cache["cache_schema_version"],
            "file_sha256": input_file_sha256,
            "records_sha256": matched_cache["records_sha256"],
            "binding_sha256": _sha256(matched_cache["binding"]),
        },
        "calibration_bindings": {
            "rgb_calibrator_revision": rgb_calibrator["calibrator_revision"],
            "signal_calibrator_revision": signal_calibrator["calibrator_revision"],
            "static_weight_revision": static_revision,
        },
        "selected_fallback_type": selected_fallback_type,
        "records_sha256": _sha256(calibrated_records),
        "records": calibrated_records,
    }


def validate_calibrated_cache(
    cache: dict,
    *,
    matched_cache: dict,
    expected_split: str,
    input_file_sha256: str,
    rgb_calibrator: dict,
    signal_calibrator: dict,
    static_weight: dict,
    selected_fallback_type: str,
) -> list[dict]:
    expected = build_calibrated_cache(
        matched_cache,
        expected_split=expected_split,
        input_file_sha256=input_file_sha256,
        rgb_calibrator=rgb_calibrator,
        signal_calibrator=signal_calibrator,
        static_weight=static_weight,
        selected_fallback_type=selected_fallback_type,
    )
    if _canonical_bytes(cache) != _canonical_bytes(expected):
        raise ValueError("Calibrated frozen-expert cache is stale or incompatible.")
    return cache["records"]


def matched_record(rgb: dict, signal: dict, provenance: dict) -> dict:
    for field in ("source_id", "variant_id", "split", "authenticity_label", "condition_family", "severity", "materialized_sha256"):
        if rgb.get(field) != signal.get(field):
            raise ValueError(f"RGB and signal records disagree on {field}.")
    row = {
        field: rgb[field] for field in
        ("source_id", "variant_id", "split", "authenticity_label", "condition_family", "severity", "materialized_sha256")
    }
    row.update({"rgb_logit": rgb["rgb_logit"], "signal_logit": signal["signal_logit"], **provenance})
    identity = {field: row[field] for field in (
        "variant_id", "materialized_sha256", "manifest_sha256",
        "rgb_checkpoint_revision", "rgb_preprocessing_version",
        "signal_checkpoint_revision", "signal_normalization_revision",
        "signal_feature_extraction_version", "corruption_version",
    )}
    row["cache_key"] = "matched-frozen-expert-v1-" + _sha256(identity)
    return row


def fit_platt_calibrator(records: Iterable[dict], *, expert: str) -> dict:
    rows = validate_matched_records(records, expected_split="fusion-training")
    field = f"{expert}_logit"
    if expert not in ("rgb", "signal"):
        raise ValueError("Calibration expert must be rgb or signal.")
    x = np.asarray([row[field] for row in rows], dtype=np.float64)
    y = np.asarray([row["authenticity_label"] for row in rows], dtype=np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    parameters = np.asarray([1.0, 0.0], dtype=np.float64)
    lower = np.asarray([0.01, -10.0]); upper = np.asarray([10.0, 10.0])
    regularization = 1e-4
    def objective(candidate: np.ndarray) -> float:
        logits = design @ candidate
        return float(np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))) + .5 * regularization * (candidate @ candidate))
    for _ in range(100):
        probability = _sigmoid(design @ parameters)
        gradient = design.T @ (probability - y) / len(y) + regularization * parameters
        weights = np.maximum(probability * (1 - probability), 1e-12)
        hessian = (design.T * weights) @ design / len(y) + regularization * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        prior = objective(parameters)
        scale_factor = 1.0
        while scale_factor >= 2 ** -20:
            candidate = np.clip(parameters - scale_factor * step, lower, upper)
            if objective(candidate) <= prior - 1e-4 * scale_factor * float(gradient @ (parameters - candidate)):
                break
            scale_factor /= 2
        change = candidate - parameters
        parameters = candidate
        if float(np.max(np.abs(change))) < 1e-12:
            break
    artifact = {
        "calibration_schema_version": CALIBRATION_SCHEMA,
        "method": "platt-affine-logit-v1",
        "expert": expert,
        "score_direction": "higher-means-fully-synthetic",
        "fit_split": "fusion-training",
        "fit_source_count": len({row["source_id"] for row in rows}),
        "fit_observation_count": len(rows),
        "slope": float(parameters[0]),
        "intercept": float(parameters[1]),
        "l2_regularization": regularization,
        "parameter_constraints": {"minimum_slope": 0.01, "maximum_slope": 10.0, "minimum_intercept": -10.0, "maximum_intercept": 10.0},
        "solver": "deterministic-damped-newton-v1",
        "records_sha256": _sha256(rows),
    }
    artifact["calibrator_revision"] = "expert-calibrator-v1-" + _sha256(artifact)
    return artifact


def calibrated_logit(raw_logit: float, calibrator: dict) -> float:
    if calibrator.get("calibration_schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("Unsupported calibrator schema.")
    identity = dict(calibrator)
    revision = identity.pop("calibrator_revision", None)
    if revision != "expert-calibrator-v1-" + _sha256(identity):
        raise ValueError("Calibrator revision is stale or incompatible.")
    return _finite(calibrator.get("slope"), "Calibrator slope") * _finite(raw_logit, "Raw logit") + _finite(calibrator.get("intercept"), "Calibrator intercept")


def _condition_balanced_brier(rows: list[dict], logits: np.ndarray) -> float:
    probabilities = _sigmoid(logits)
    labels = np.asarray([row["authenticity_label"] for row in rows])
    losses = (probabilities - labels) ** 2
    values = []
    for family in FAMILIES:
        severities = sorted({row["severity"] for row in rows if row["condition_family"] == family})
        if not severities:
            raise ValueError(f"Records are missing condition family {family}.")
        values.append(sum(float(losses[[i for i, row in enumerate(rows) if row["condition_family"] == family and row["severity"] == severity]].mean()) for severity in severities) / len(severities))
    return sum(values) / len(values)


def fit_static_weight(records: Iterable[dict], rgb_calibrator: dict, signal_calibrator: dict) -> dict:
    rows = validate_matched_records(records, expected_split="fusion-training")
    rgb = np.asarray([calibrated_logit(row["rgb_logit"], rgb_calibrator) for row in rows])
    signal = np.asarray([calibrated_logit(row["signal_logit"], signal_calibrator) for row in rows])
    best = None
    for index in range(1001):
        weight = index / 1000
        objective = _condition_balanced_brier(rows, weight * rgb + (1 - weight) * signal)
        candidate = (objective, -weight, weight)
        if best is None or candidate < best:
            best = candidate
    artifact = {
        "static_weight_schema_version": "static-fusion-weight-v1",
        "fit_split": "fusion-training",
        "objective": "condition-balanced-brier-v1",
        "rgb_weight": best[2],
        "signal_weight": 1 - best[2],
        "constraint": {"minimum_rgb_weight": 0.0, "maximum_rgb_weight": 1.0, "grid_step": 0.001},
        "tie_breaking": "largest-rgb-weight",
        "fit_source_count": len({row["source_id"] for row in rows}),
        "fit_observation_count": len(rows),
        "objective_value": best[0],
        "rgb_calibrator_revision": rgb_calibrator["calibrator_revision"],
        "signal_calibrator_revision": signal_calibrator["calibrator_revision"],
        "records_sha256": _sha256(rows),
    }
    artifact["static_weight_revision"] = "static-fusion-weight-v1-" + _sha256(artifact)
    return artifact


def _candidate_logits(rows: list[dict], rgb: dict, signal: dict, weight: dict) -> dict[str, np.ndarray]:
    raw_rgb = np.asarray([row["rgb_logit"] for row in rows])
    raw_signal = np.asarray([row["signal_logit"] for row in rows])
    calibrated_rgb = np.asarray([calibrated_logit(value, rgb) for value in raw_rgb])
    calibrated_signal = np.asarray([calibrated_logit(value, signal) for value in raw_signal])
    learned = weight["rgb_weight"] * calibrated_rgb + weight["signal_weight"] * calibrated_signal
    return {
        "raw-rgb-only": raw_rgb, "calibrated-rgb-only": calibrated_rgb,
        "raw-signal-only": raw_signal, "calibrated-signal-only": calibrated_signal,
        "equal-50-50-calibrated-logit-fusion": 0.5 * calibrated_rgb + 0.5 * calibrated_signal,
        "learned-static-fusion": learned,
    }


def _evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    scored = [{**row, "candidate_logit": float(value)} for row, value in zip(rows, logits, strict=True)]
    result = evaluate_internal_validation(scored, score_field="candidate_logit", metric_schema_version="fusion-candidate-metrics-v1")
    probabilities = _sigmoid(logits)
    labels = np.asarray([row["authenticity_label"] for row in rows])
    result["brier_score"] = float(np.mean((probabilities - labels) ** 2))
    result["condition_balanced_brier_score"] = _condition_balanced_brier(rows, logits)
    return result


def _bootstrap_gain(rows: list[dict], candidate: np.ndarray, reference: np.ndarray, *, seed: int = 71, draws: int = 1000) -> dict:
    sources = sorted({row["source_id"] for row in rows})
    indexes = {source: [i for i, row in enumerate(rows) if row["source_id"] == source] for source in sources}
    generator = np.random.default_rng(seed)
    gains = []
    for _ in range(draws):
        sampled = generator.choice(sources, size=len(sources), replace=True)
        boot_rows, boot_candidate, boot_reference = [], [], []
        for copy_index, source in enumerate(sampled):
            for index in indexes[source]:
                boot_rows.append({**rows[index], "source_id": f"bootstrap-{copy_index}"})
                boot_candidate.append(candidate[index]); boot_reference.append(reference[index])
        try:
            gain = _evaluate(boot_rows, np.asarray(boot_candidate))["all_condition_macro_auroc"] - _evaluate(boot_rows, np.asarray(boot_reference))["all_condition_macro_auroc"]
            gains.append(gain)
        except ValueError:
            continue
    if not gains:
        raise ValueError("Source bootstrap produced no valid resamples.")
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return {"method": "source-bootstrap-v1", "seed": seed, "requested_draws": draws, "valid_draws": len(gains), "lower": float(lower), "upper": float(upper)}


def evaluate_candidates(records: Iterable[dict], rgb_calibrator: dict, signal_calibrator: dict, static_weight: dict) -> dict:
    rows = validate_matched_records(records, expected_split="internal-validation")
    logits = _candidate_logits(rows, rgb_calibrator, signal_calibrator, static_weight)
    candidates = {name: _evaluate(rows, values) for name, values in logits.items()}
    bootstrap = _bootstrap_gain(rows, logits["learned-static-fusion"], logits["calibrated-rgb-only"])
    learned, reference = candidates["learned-static-fusion"], candidates["calibrated-rgb-only"]
    gain = learned["all_condition_macro_auroc"] - reference["all_condition_macro_auroc"]
    brier_change = learned["brier_score"] - reference["brier_score"]
    select_static = gain >= 0.005 and brier_change <= 0.002 and bootstrap["lower"] > 0
    rgb_errors = {row["variant_id"] for row, value in zip(rows, logits["calibrated-rgb-only"], strict=True) if (value >= 0) != bool(row["authenticity_label"])}
    signal_corrects = int(sum(variant in rgb_errors and ((value >= 0) == bool(row["authenticity_label"])) for row, value, variant in zip(rows, logits["calibrated-signal-only"], (row["variant_id"] for row in rows), strict=True)))
    return {
        "evaluation_schema_version": "static-fusion-evaluation-v1",
        "evaluation_split": "internal-validation",
        "source_count": len({row["source_id"] for row in rows}),
        "observation_count": len(rows),
        "candidates": candidates,
        "source_bootstrap_all_condition_macro_auroc_gain": bootstrap,
        "complementary_value": {"rgb_errors": len(rgb_errors), "rgb_errors_corrected_by_signal": signal_corrects, "correction_rate": signal_corrects / len(rgb_errors) if rgb_errors else 0.0},
        "selection_rule": SELECTION_RULE,
        "selection_evidence": {"all_condition_macro_auroc_gain": gain, "brier_change": brier_change},
        "selected_fallback_type": "learned-static-fusion" if select_static else "calibrated-rgb-only",
    }


def build_corrected_bundle(
    legacy_bundle: dict,
    *,
    provenance: dict,
    rgb_normalizer: dict,
    signal_normalizer: dict,
    input_cache_bindings: dict,
    artifact_bindings: dict,
) -> dict:
    validate_bundle(legacy_bundle)
    for name, value in (
        ("provenance", provenance),
        ("RGB normalizer", rgb_normalizer),
        ("signal normalizer", signal_normalizer),
        ("input cache bindings", input_cache_bindings),
        ("artifact bindings", artifact_bindings),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Corrected frozen fallback bundle requires {name}.")
    corrected = {
        key: value
        for key, value in legacy_bundle.items()
        if key not in ("bundle_revision", "bundle_schema_version")
    }
    corrected.update({
        "bundle_schema_version": CORRECTED_BUNDLE_SCHEMA,
        "legacy_bundle_revision": legacy_bundle["bundle_revision"],
        "legacy_schema_versions": json.loads(_canonical_bytes(legacy_bundle.get("schema_versions", {}))),
        "schema_versions": {
            "matched_cache": MATCHED_CACHE_SCHEMA,
            "calibrated_cache": CALIBRATED_CACHE_SCHEMA,
            "calibration": CALIBRATION_SCHEMA,
            "static_weight": "static-fusion-weight-v1",
            "evaluation": "static-fusion-evaluation-v1",
            "bundle": CORRECTED_BUNDLE_SCHEMA,
            "completion": CORRECTED_COMPLETION_SCHEMA,
        },
        "provenance": json.loads(_canonical_bytes(provenance)),
        "rgb_normalizer": json.loads(_canonical_bytes(rgb_normalizer)),
        "signal_normalizer": json.loads(_canonical_bytes(signal_normalizer)),
        "input_cache_bindings": json.loads(_canonical_bytes(input_cache_bindings)),
        "artifact_bindings": json.loads(_canonical_bytes(artifact_bindings)),
    })
    artifact_bindings_by_path = {
        binding.get("path"): binding
        for binding in artifact_bindings.values()
        if isinstance(binding, dict)
    }
    corrected["scientific_bindings"] = {
        "fusion_training_records_sha256": artifact_bindings_by_path.get(
            "matched-fusion-training-logits.json", {}
        ).get("records_sha256"),
        "internal_validation_records_sha256": artifact_bindings_by_path.get(
            "matched-internal-validation-logits.json", {}
        ).get("records_sha256"),
        "rgb_calibrator_revision": legacy_bundle["rgb_calibrator"]["calibrator_revision"],
        "signal_calibrator_revision": legacy_bundle["signal_calibrator"]["calibrator_revision"],
        "static_weight_revision": legacy_bundle["static_weight"]["static_weight_revision"],
        "evaluation_schema_version": legacy_bundle["evaluation"].get("evaluation_schema_version"),
        "selected_fallback_type": legacy_bundle["selected_fallback_type"],
    }
    corrected["bundle_revision"] = "static-fallback-bundle-v2-" + _sha256(corrected)
    return corrected


def _artifact_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _file_binding(path: str, payload: dict, contents: bytes) -> dict:
    records = payload.get("records")
    binding = {
        "path": path,
        "file_sha256": hashlib.sha256(contents).hexdigest(),
    }
    if isinstance(records, list):
        binding.update({
            "records_sha256": _sha256(records),
            "observation_count": len(records),
        })
    return binding


def publish_corrected_generation(
    output_directory: Path | str,
    *,
    legacy_bundle: dict,
    fusion_training_cache: dict,
    internal_validation_cache: dict,
    provenance: dict,
    rgb_normalizer: dict,
    signal_normalizer: dict,
    input_cache_bindings: dict,
) -> dict:
    training_rows = validate_matched_cache(
        fusion_training_cache,
        expected_split="fusion-training",
        expected_provenance=provenance,
        expected_binding=fusion_training_cache.get("binding"),
    )
    validation_rows = validate_matched_cache(
        internal_validation_cache,
        expected_split="internal-validation",
        expected_provenance=provenance,
        expected_binding=internal_validation_cache.get("binding"),
    )
    validate_bundle(legacy_bundle)
    matched_payloads = {
        "matched-fusion-training-logits.json": fusion_training_cache,
        "matched-internal-validation-logits.json": internal_validation_cache,
    }
    payloads = dict(matched_payloads)
    serialized = {name: _artifact_bytes(payload) for name, payload in payloads.items()}
    selected = legacy_bundle["selected_fallback_type"]
    calibrated_payloads = {
        "calibrated-fusion-training-logits.json": build_calibrated_cache(
            fusion_training_cache,
            expected_split="fusion-training",
            input_file_sha256=hashlib.sha256(serialized["matched-fusion-training-logits.json"]).hexdigest(),
            rgb_calibrator=legacy_bundle["rgb_calibrator"],
            signal_calibrator=legacy_bundle["signal_calibrator"],
            static_weight=legacy_bundle["static_weight"],
            selected_fallback_type=selected,
        ),
        "calibrated-internal-validation-logits.json": build_calibrated_cache(
            internal_validation_cache,
            expected_split="internal-validation",
            input_file_sha256=hashlib.sha256(serialized["matched-internal-validation-logits.json"]).hexdigest(),
            rgb_calibrator=legacy_bundle["rgb_calibrator"],
            signal_calibrator=legacy_bundle["signal_calibrator"],
            static_weight=legacy_bundle["static_weight"],
            selected_fallback_type=selected,
        ),
    }
    payloads.update(calibrated_payloads)
    serialized.update({name: _artifact_bytes(payload) for name, payload in calibrated_payloads.items()})
    artifact_bindings = {
        name.removesuffix(".json").replace("-", "_"): _file_binding(name, payloads[name], serialized[name])
        for name in sorted(payloads)
    }
    bundle = build_corrected_bundle(
        legacy_bundle,
        provenance=provenance,
        rgb_normalizer=rgb_normalizer,
        signal_normalizer=signal_normalizer,
        input_cache_bindings=input_cache_bindings,
        artifact_bindings=artifact_bindings,
    )
    validate_bundle(bundle)
    bundle_name = "static-fallback-bundle.json"
    payloads[bundle_name] = bundle
    serialized[bundle_name] = _artifact_bytes(bundle)
    completion_artifacts = {
        name: _file_binding(name, payloads[name], serialized[name])
        for name in sorted(payloads)
    }
    completion = {
        "completion_schema_version": CORRECTED_COMPLETION_SCHEMA,
        "provenance": json.loads(_canonical_bytes(provenance)),
        "bundle_revision": bundle["bundle_revision"],
        "bundle_sha256": hashlib.sha256(serialized[bundle_name]).hexdigest(),
        "artifacts": completion_artifacts,
    }
    completion["generation_revision"] = "static-fallback-generation-v2-" + _sha256(completion)
    target = Path(output_directory).absolute()
    parent = resolve_output_directory(target.parent, "Corrected static fallback generation parent")
    target = parent / target.name
    if target.exists():
        return read_static_fallback_generation(
            target,
            expected_provenance=provenance,
            expected_generation_revision=completion["generation_revision"],
        )["completion"]
    if len(training_rows) != len(fusion_training_cache["records"]) or len(validation_rows) != len(internal_validation_cache["records"]):
        raise ValueError("Corrected static fallback generation changed matched observations.")
    with tempfile.TemporaryDirectory(prefix=".static-fallback-v2-staging-", dir=parent) as temporary:
        staging = Path(temporary)
        for name in sorted(serialized):
            atomic_write_bytes(staging / name, serialized[name])
        atomic_write_bytes(staging / "static-fallback.complete.json", _artifact_bytes(completion))
        read_static_fallback_generation(
            staging,
            expected_provenance=provenance,
            expected_generation_revision=completion["generation_revision"],
        )
        os.replace(staging, target)
    read_static_fallback_generation(
        target,
        expected_provenance=provenance,
        expected_generation_revision=completion["generation_revision"],
    )
    return completion


def read_static_fallback_generation(
    output_directory: Path | str,
    *,
    expected_provenance: dict | None = None,
    expected_generation_revision: str,
) -> dict:
    root = Path(output_directory).absolute()
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("Corrected static fallback generation directory is missing.") from error
    if resolved != root or not resolved.is_dir():
        raise ValueError("Corrected static fallback generation directory is redirected or invalid.")
    marker_name = "static-fallback.complete.json"
    artifact_names = {
        "matched-fusion-training-logits.json",
        "matched-internal-validation-logits.json",
        "calibrated-fusion-training-logits.json",
        "calibrated-internal-validation-logits.json",
        "static-fallback-bundle.json",
    }
    inventory = {path.name for path in root.iterdir() if path.is_file()}
    if inventory != artifact_names | {marker_name}:
        raise ValueError("Corrected static fallback generation artifact inventory is incomplete or unexpected.")
    try:
        completion = json.loads((root / marker_name).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Corrected static fallback completion marker is missing or invalid.") from error
    if completion.get("completion_schema_version") != CORRECTED_COMPLETION_SCHEMA:
        raise ValueError("Corrected static fallback completion marker schema is incompatible.")
    completion_identity = dict(completion)
    generation_revision = completion_identity.pop("generation_revision", None)
    if generation_revision != "static-fallback-generation-v2-" + _sha256(completion_identity):
        raise ValueError("Corrected static fallback generation revision is stale or incompatible.")
    if generation_revision != expected_generation_revision:
        raise ValueError("Corrected static fallback generation does not match the trusted revision.")
    provenance = completion.get("provenance")
    if expected_provenance is not None and _canonical_bytes(provenance) != _canonical_bytes(expected_provenance):
        raise ValueError("Corrected static fallback generation does not match trusted provenance.")
    bindings = completion.get("artifacts")
    if not isinstance(bindings, dict) or set(bindings) != artifact_names:
        raise ValueError("Corrected static fallback completion artifact bindings are incomplete.")
    _validate_binding_paths(bindings)
    payloads = {}
    serialized = {}
    for name in sorted(artifact_names):
        path = root / name
        try:
            contents = path.read_bytes()
            payload = json.loads(contents)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Corrected static fallback artifact {name} is missing or invalid.") from error
        if _canonical_bytes(bindings[name]) != _canonical_bytes(_file_binding(name, payload, contents)):
            raise ValueError(f"Corrected static fallback artifact {name} has a stale binding.")
        payloads[name] = payload
        serialized[name] = contents
    bundle = validate_bundle(payloads["static-fallback-bundle.json"])
    if _canonical_bytes(completion.get("provenance")) != _canonical_bytes(bundle.get("provenance")):
        raise ValueError("Corrected static fallback completion and bundle provenance disagree.")
    if (
        completion.get("bundle_revision") != bundle["bundle_revision"]
        or completion.get("bundle_sha256") != hashlib.sha256(serialized["static-fallback-bundle.json"]).hexdigest()
    ):
        raise ValueError("Corrected static fallback completion marker does not bind the bundle.")
    bundle_bindings = bundle.get("artifact_bindings")
    by_path = {
        binding.get("path"): binding
        for binding in bundle_bindings.values()
        if isinstance(binding, dict)
    }
    expected_bound_names = artifact_names - {"static-fallback-bundle.json"}
    if set(by_path) != expected_bound_names or any(
        _canonical_bytes(by_path[name]) != _canonical_bytes(bindings[name])
        for name in expected_bound_names
    ):
        raise ValueError("Corrected static fallback bundle artifact bindings disagree with completion.")
    training = payloads["matched-fusion-training-logits.json"]
    validation = payloads["matched-internal-validation-logits.json"]
    training_rows = validate_matched_cache(
        training,
        expected_split="fusion-training",
        expected_provenance=provenance,
        expected_binding=training.get("binding"),
    )
    validation_rows = validate_matched_cache(
        validation,
        expected_split="internal-validation",
        expected_provenance=provenance,
        expected_binding=validation.get("binding"),
    )
    _validate_issue7_partition_counts(training_rows, validation_rows)
    scientific_bindings = bundle["scientific_bindings"]
    if (
        scientific_bindings.get("fusion_training_records_sha256") != training["records_sha256"]
        or scientific_bindings.get("internal_validation_records_sha256") != validation["records_sha256"]
        or bundle["rgb_calibrator"].get("records_sha256") != training["records_sha256"]
        or bundle["signal_calibrator"].get("records_sha256") != training["records_sha256"]
        or bundle["static_weight"].get("records_sha256") != training["records_sha256"]
        or bundle["rgb_calibrator"].get("fit_source_count") != len({row["source_id"] for row in training_rows})
        or bundle["rgb_calibrator"].get("fit_observation_count") != len(training_rows)
        or bundle["signal_calibrator"].get("fit_source_count") != len({row["source_id"] for row in training_rows})
        or bundle["signal_calibrator"].get("fit_observation_count") != len(training_rows)
        or bundle["evaluation"].get("source_count") != len({row["source_id"] for row in validation_rows})
        or bundle["evaluation"].get("observation_count") != len(validation_rows)
    ):
        raise ValueError("Corrected static fallback scientific bindings disagree with loaded observations.")
    for provenance_field, normalizer_field in (
        ("rgb_checkpoint_revision", "checkpoint_revision"),
        ("rgb_checkpoint_sha256", "checkpoint_sha256"),
        ("rgb_preprocessing_version", "preprocessing_version"),
        ("rgb_score_direction", "score_direction"),
        ("shared_observation_preprocessing_version", "shared_observation_preprocessing_version"),
    ):
        if (
            provenance.get(provenance_field) is not None
            and bundle["rgb_normalizer"].get(normalizer_field) != provenance[provenance_field]
        ):
            raise ValueError("Corrected static fallback RGB normalizer disagrees with bundle provenance.")
    for split, matched_name, calibrated_name in (
        ("fusion-training", "matched-fusion-training-logits.json", "calibrated-fusion-training-logits.json"),
        ("internal-validation", "matched-internal-validation-logits.json", "calibrated-internal-validation-logits.json"),
    ):
        validate_calibrated_cache(
            payloads[calibrated_name],
            matched_cache=payloads[matched_name],
            expected_split=split,
            input_file_sha256=hashlib.sha256(serialized[matched_name]).hexdigest(),
            rgb_calibrator=bundle["rgb_calibrator"],
            signal_calibrator=bundle["signal_calibrator"],
            static_weight=bundle["static_weight"],
            selected_fallback_type=bundle["selected_fallback_type"],
        )
    return {
        "completion": completion,
        "bundle": bundle,
        "fusion_training_cache": training,
        "internal_validation_cache": validation,
        "calibrated_fusion_training_cache": payloads["calibrated-fusion-training-logits.json"],
        "calibrated_internal_validation_cache": payloads["calibrated-internal-validation-logits.json"],
    }


def _validate_content_revision(artifact: dict, *, field: str, prefix: str, name: str) -> None:
    if not isinstance(artifact, dict):
        raise ValueError(f"Corrected frozen fallback bundle requires {name}.")
    identity = dict(artifact)
    revision = identity.pop(field, None)
    if revision != prefix + _sha256(identity):
        raise ValueError(f"Corrected frozen fallback bundle {name} is stale or incompatible.")


def _validate_binding_paths(bindings: dict) -> None:
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise ValueError("Corrected frozen fallback bundle artifact binding is invalid.")
        path = binding.get("path")
        if path is None:
            continue
        if not isinstance(path, str):
            raise ValueError("Corrected frozen fallback bundle artifact path is invalid.")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            raise ValueError("Corrected frozen fallback bundle contains a path traversal.")


def _validate_calibrator_contract(calibrator: dict, *, expected_expert: str) -> None:
    calibrated_logit(0.0, calibrator)
    expected = {
        "calibration_schema_version": CALIBRATION_SCHEMA,
        "method": "platt-affine-logit-v1",
        "expert": expected_expert,
        "score_direction": "higher-means-fully-synthetic",
        "fit_split": "fusion-training",
        "l2_regularization": 1e-4,
        "parameter_constraints": {
            "minimum_slope": 0.01,
            "maximum_slope": 10.0,
            "minimum_intercept": -10.0,
            "maximum_intercept": 10.0,
        },
        "solver": "deterministic-damped-newton-v1",
    }
    slope = calibrator.get("slope")
    intercept = calibrator.get("intercept")
    digest = calibrator.get("records_sha256")
    if (
        any(calibrator.get(field) != value for field, value in expected.items())
        or isinstance(slope, bool)
        or not isinstance(slope, (int, float))
        or not 0.01 <= slope <= 10.0
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not -10.0 <= intercept <= 10.0
        or type(calibrator.get("fit_source_count")) is not int
        or calibrator["fit_source_count"] <= 0
        or type(calibrator.get("fit_observation_count")) is not int
        or calibrator["fit_observation_count"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        name = "RGB" if expected_expert == "rgb" else "signal"
        raise ValueError(f"Corrected frozen fallback {name} calibrator contract is incompatible.")


def _validate_corrected_bundle(bundle: dict) -> dict:
    if bundle.get("selection_rule") != SELECTION_RULE:
        raise ValueError("Corrected frozen fallback bundle selection rule is incompatible.")
    for field in (
        "rgb_calibrator", "signal_calibrator", "static_weight", "evaluation",
        "provenance", "rgb_normalizer", "signal_normalizer",
        "input_cache_bindings", "artifact_bindings", "legacy_schema_versions",
        "schema_versions", "scientific_bindings",
    ):
        if not isinstance(bundle.get(field), dict):
            raise ValueError(f"Corrected frozen fallback bundle requires {field}.")
    if not all(
        isinstance(bundle["provenance"].get(field), str)
        and bundle["provenance"][field]
        for field in ("signal_experiment_profile", "signal_acceptance_scope")
    ):
        raise ValueError("Corrected frozen fallback bundle requires signal profile and scope provenance.")
    required_input_bindings = {
        "fusion_training", "internal_validation", "legacy_bundle",
        "legacy_completion", "signal_model",
    }
    if set(bundle["input_cache_bindings"]) != required_input_bindings:
        raise ValueError("Corrected frozen fallback input cache bindings are incomplete.")
    expected_schema_versions = {
        "matched_cache": MATCHED_CACHE_SCHEMA,
        "calibrated_cache": CALIBRATED_CACHE_SCHEMA,
        "calibration": CALIBRATION_SCHEMA,
        "static_weight": "static-fusion-weight-v1",
        "evaluation": "static-fusion-evaluation-v1",
        "bundle": CORRECTED_BUNDLE_SCHEMA,
        "completion": CORRECTED_COMPLETION_SCHEMA,
    }
    if bundle["schema_versions"] != expected_schema_versions:
        raise ValueError("Corrected frozen fallback schema versions are incompatible.")
    required_artifact_paths = {
        "matched-fusion-training-logits.json",
        "matched-internal-validation-logits.json",
        "calibrated-fusion-training-logits.json",
        "calibrated-internal-validation-logits.json",
    }
    artifact_paths = [
        binding.get("path")
        for binding in bundle["artifact_bindings"].values()
        if isinstance(binding, dict)
    ]
    if set(artifact_paths) != required_artifact_paths or len(artifact_paths) != len(required_artifact_paths):
        raise ValueError("Corrected frozen fallback artifact bindings are incomplete.")
    input_paths = {
        binding.get("path")
        for binding in bundle["input_cache_bindings"].values()
        if isinstance(binding, dict)
    }
    if input_paths & set(artifact_paths):
        raise ValueError("Corrected frozen fallback legacy input paths collide with v2 artifacts.")
    for bindings in (bundle["input_cache_bindings"], bundle["artifact_bindings"]):
        for binding in bindings.values():
            digest = binding.get("file_sha256") if isinstance(binding, dict) else None
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Corrected frozen fallback binding requires a lowercase SHA-256.")
    _validate_calibrator_contract(bundle["rgb_calibrator"], expected_expert="rgb")
    _validate_calibrator_contract(bundle["signal_calibrator"], expected_expert="signal")
    static_identity = dict(bundle["static_weight"])
    static_revision = static_identity.pop("static_weight_revision", None)
    if static_revision != "static-fusion-weight-v1-" + _sha256(static_identity):
        raise ValueError("Corrected frozen fallback bundle static weight is stale or incompatible.")
    static_weight = bundle["static_weight"]
    rgb_weight = static_weight.get("rgb_weight")
    signal_weight = static_weight.get("signal_weight")
    static_contract = {
        "static_weight_schema_version": "static-fusion-weight-v1",
        "fit_split": "fusion-training",
        "objective": "condition-balanced-brier-v1",
        "constraint": {
            "minimum_rgb_weight": 0.0,
            "maximum_rgb_weight": 1.0,
            "grid_step": 0.001,
        },
        "tie_breaking": "largest-rgb-weight",
        "rgb_calibrator_revision": bundle["rgb_calibrator"]["calibrator_revision"],
        "signal_calibrator_revision": bundle["signal_calibrator"]["calibrator_revision"],
        "records_sha256": bundle["rgb_calibrator"]["records_sha256"],
    }
    if (
        any(static_weight.get(field) != value for field, value in static_contract.items())
        or bundle["signal_calibrator"].get("records_sha256") != static_weight.get("records_sha256")
        or isinstance(rgb_weight, bool)
        or not isinstance(rgb_weight, (int, float))
        or isinstance(signal_weight, bool)
        or not isinstance(signal_weight, (int, float))
        or not math.isfinite(rgb_weight)
        or not math.isfinite(signal_weight)
        or not 0.0 <= rgb_weight <= 1.0
        or not 0.0 <= signal_weight <= 1.0
        or not math.isclose(rgb_weight + signal_weight, 1.0, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(rgb_weight * 1000, round(rgb_weight * 1000), rel_tol=0.0, abs_tol=1e-12)
        or type(static_weight.get("fit_source_count")) is not int
        or static_weight["fit_source_count"] != bundle["rgb_calibrator"].get("fit_source_count")
        or type(static_weight.get("fit_observation_count")) is not int
        or static_weight["fit_observation_count"] != bundle["rgb_calibrator"].get("fit_observation_count")
        or not math.isfinite(_finite(static_weight.get("objective_value"), "Static weight objective"))
    ):
        raise ValueError("Corrected frozen fallback static weight contract is incompatible.")
    artifact_bindings_by_path = {
        binding["path"]: binding for binding in bundle["artifact_bindings"].values()
    }
    expected_scientific_bindings = {
        "fusion_training_records_sha256": artifact_bindings_by_path[
            "matched-fusion-training-logits.json"
        ].get("records_sha256"),
        "internal_validation_records_sha256": artifact_bindings_by_path[
            "matched-internal-validation-logits.json"
        ].get("records_sha256"),
        "rgb_calibrator_revision": bundle["rgb_calibrator"]["calibrator_revision"],
        "signal_calibrator_revision": bundle["signal_calibrator"]["calibrator_revision"],
        "static_weight_revision": bundle["static_weight"]["static_weight_revision"],
        "evaluation_schema_version": bundle["evaluation"].get("evaluation_schema_version"),
        "selected_fallback_type": bundle["selected_fallback_type"],
    }
    if bundle["scientific_bindings"] != expected_scientific_bindings or any(
        not isinstance(bundle["scientific_bindings"].get(field), str)
        or not bundle["scientific_bindings"][field]
        for field in expected_scientific_bindings
    ):
        raise ValueError("Corrected frozen fallback scientific bindings are incompatible.")
    _validate_content_revision(
        bundle["rgb_normalizer"],
        field="normalization_revision",
        prefix="rgb-normalization-v1-",
        name="RGB normalizer",
    )
    rgb_normalizer_contract = {
        "normalization_schema_version": "rgb-imagenet-normalization-v1",
        "channel_order": "rgb",
        "input_range": [0.0, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "scale": [0.229, 0.224, 0.225],
    }
    if any(bundle["rgb_normalizer"].get(key) != value for key, value in rgb_normalizer_contract.items()):
        raise ValueError("Corrected frozen fallback bundle RGB normalizer is incompatible.")
    _validate_content_revision(
        bundle["signal_normalizer"],
        field="normalization_revision",
        prefix="signal-normalization-v1-",
        name="signal normalizer",
    )
    signal_normalizer = bundle["signal_normalizer"]
    feature_names = signal_normalizer.get("feature_names")
    means = signal_normalizer.get("mean")
    scales = signal_normalizer.get("scale")
    if (
        signal_normalizer.get("normalization_revision")
        != bundle["provenance"].get("signal_normalization_revision")
        or not isinstance(feature_names, list)
        or len(feature_names) != 26
        or len(set(feature_names)) != len(feature_names)
        or not isinstance(means, list)
        or not isinstance(scales, list)
        or len(means) != len(feature_names)
        or len(scales) != len(feature_names)
        or any(not math.isfinite(value) for value in means + scales)
        or any(value <= 0 for value in scales)
    ):
        raise ValueError("Corrected frozen fallback bundle signal normalizer is incompatible.")
    selected = bundle.get("selected_fallback_type")
    evaluation = bundle["evaluation"]
    if selected != evaluation.get("selected_fallback_type"):
        raise ValueError("Corrected frozen fallback selection is relabelled or incompatible.")
    candidates = evaluation.get("candidates")
    if not isinstance(candidates, dict) or selected not in candidates:
        raise ValueError("Corrected frozen fallback selected evaluation candidate is missing.")
    evidence = evaluation.get("selection_evidence")
    bootstrap = evaluation.get("source_bootstrap_all_condition_macro_auroc_gain")
    try:
        select_static = (
            _finite(evidence.get("all_condition_macro_auroc_gain"), "Selection AUROC gain")
            >= SELECTION_RULE["minimum_all_condition_macro_auroc_gain"]
            and _finite(evidence.get("brier_change"), "Selection Brier change")
            <= SELECTION_RULE["maximum_brier_worsening"]
            and _finite(bootstrap.get("lower"), "Selection bootstrap lower bound") > 0
        )
    except AttributeError as error:
        raise ValueError("Corrected frozen fallback selection evidence is incomplete.") from error
    reproduced_selection = "learned-static-fusion" if select_static else "calibrated-rgb-only"
    if selected != reproduced_selection:
        raise ValueError("Corrected frozen fallback selection does not reproduce from its evidence.")
    if bundle.get("provisional_threshold") != candidates[selected].get("threshold_diagnostics"):
        raise ValueError("Corrected frozen fallback provisional threshold is relabelled or incompatible.")
    threshold = bundle["provisional_threshold"]
    if not isinstance(threshold, dict) or set(threshold) != {
        "status", "selection_rule", "threshold_logit", "balanced_accuracy",
        "sensitivity", "specificity",
    }:
        raise ValueError("Corrected frozen fallback threshold diagnostics are incomplete.")
    sensitivity = threshold.get("sensitivity")
    specificity = threshold.get("specificity")
    balanced_accuracy = threshold.get("balanced_accuracy")
    if (
        threshold.get("status") != "provisional-internal-validation-only"
        or threshold.get("selection_rule") != "maximum-youden-j"
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (
                threshold.get("threshold_logit"),
                balanced_accuracy,
                sensitivity,
                specificity,
            )
        )
        or not 0.0 <= sensitivity <= 1.0
        or not 0.0 <= specificity <= 1.0
        or not 0.0 <= balanced_accuracy <= 1.0
        or not math.isclose(
            balanced_accuracy,
            (sensitivity + specificity) / 2,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("Corrected frozen fallback threshold diagnostics are incompatible.")
    _validate_binding_paths(bundle["input_cache_bindings"])
    _validate_binding_paths(bundle["artifact_bindings"])
    legacy_revision = bundle.get("legacy_bundle_revision")
    if not isinstance(legacy_revision, str) or not legacy_revision.startswith("static-fallback-bundle-v1-"):
        raise ValueError("Corrected frozen fallback legacy bundle revision is invalid.")
    identity = dict(bundle)
    revision = identity.pop("bundle_revision", None)
    if revision != "static-fallback-bundle-v2-" + _sha256(identity):
        raise ValueError("Corrected frozen fallback bundle revision is stale or incompatible.")
    return bundle


def validate_bundle(bundle: dict) -> dict:
    if bundle.get("bundle_schema_version") == CORRECTED_BUNDLE_SCHEMA:
        return _validate_corrected_bundle(bundle)
    if bundle.get("bundle_schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Unsupported frozen fallback bundle schema.")
    if bundle.get("selection_rule") != SELECTION_RULE:
        raise ValueError("Frozen fallback bundle selection rule is incompatible.")
    for field in ("rgb_calibrator", "signal_calibrator", "static_weight", "evaluation", "provenance", "input_cache_bindings", "numeric_tolerances"):
        if not isinstance(bundle.get(field), dict):
            raise ValueError(f"Frozen fallback bundle requires {field}.")
    for path in bundle.get("artifact_paths", []):
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            raise ValueError("Frozen fallback bundle contains a path traversal.")
    calibrated_logit(0.0, bundle["rgb_calibrator"])
    calibrated_logit(0.0, bundle["signal_calibrator"])
    static_identity = dict(bundle["static_weight"]); static_revision = static_identity.pop("static_weight_revision", None)
    if static_revision != "static-fusion-weight-v1-" + _sha256(static_identity):
        raise ValueError("Frozen fallback bundle static weight is stale or incompatible.")
    if bundle.get("selected_fallback_type") != bundle["evaluation"].get("selected_fallback_type"):
        raise ValueError("Frozen fallback selection is relabelled or incompatible.")
    identity = dict(bundle); revision = identity.pop("bundle_revision", None)
    if revision != "static-fallback-bundle-v1-" + _sha256(identity):
        raise ValueError("Frozen fallback bundle revision is stale or incompatible.")
    return bundle


def publish_bundle_atomic(output_directory: Path, bundle: dict) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(bundle)
    identity["bundle_schema_version"] = BUNDLE_SCHEMA
    identity["selection_rule"] = SELECTION_RULE
    identity["bundle_revision"] = "static-fallback-bundle-v1-" + _sha256(identity)
    validate_bundle(identity)
    payload = json.dumps(identity, indent=2, allow_nan=False) + "\n"
    temporary = output / ".static-fallback-bundle.json.tmp"
    destination = output / "static-fallback-bundle.json"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    completion = {"completion_schema_version": COMPLETION_SCHEMA, "bundle_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bundle_revision": identity["bundle_revision"]}
    marker_tmp = output / ".static-fallback.complete.json.tmp"
    marker = output / "static-fallback.complete.json"
    marker_tmp.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    marker_tmp.replace(marker)
    return completion
