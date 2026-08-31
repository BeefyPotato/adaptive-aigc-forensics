"""Bounded, leakage-safe orchestration for the signal-only experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import __version__ as PILLOW_VERSION

from safe_output import atomic_write_bytes, managed_output_path, resolve_output_directory

from signal_expert import (
    CANONICAL_VALIDATION_CONDITIONS,
    FEATURE_NAMES,
    IMPLEMENTATION_HASH_CONTRACT_VERSION,
    SIGNAL_REPRESENTATION_VERSION,
    cache_signal_predictions,
    decode_expert_rgb_bytes,
    evaluate_signal_only,
    extract_signal_representation,
    fit_normalization,
    read_model_bundle,
    train_signal_mlp,
    validate_signal_cache,
    write_model_bundle,
)


ROOT = Path(__file__).resolve().parent
SIGNAL_SHARD_CLI = ROOT / "scripts" / "materialize-signal-shard.mjs"
FEATURE_CACHE_SCHEMA_VERSION = "signal-feature-cache-v1"
FEATURE_SHARD_COMPLETION_SCHEMA_VERSION = "signal-feature-shard-completion-v1"
FEATURE_EXTRACTION_VERSION = "signal-feature-extraction-v1"
MATERIALIZATION_SCHEMA_VERSION = "track5-materialized-observations-v1"
MATERIALIZED_ENCODING = "lossless-rgb-png-v1"
JAVASCRIPT_MAX_SAFE_INTEGER = 2**53 - 1
BALANCED_TRAINING_GRANULARITY = 168
EXPERIMENT_SCOPE_BY_PROFILE = {
    "custom-v1": "non-acceptance",
    "hackathon-v1": "issue-6-timeboxed-acceptance",
    "issue-6-full-v1": "issue-6-full-acceptance",
}
EXPERIMENT_PROFILE_DEFAULTS = {
    "custom-v1": (40_320, None),
    "hackathon-v1": (8_064, 400),
    "issue-6-full-v1": (40_320, None),
}
IMPLEMENTATION_FILES = (
    "package.json",
    "package-lock.json",
    "signal_cli.py",
    "signal_expert.py",
    "signal_maps.py",
    "signal_pipeline.py",
    "shared_observation.py",
    "rgb_baseline.py",
    "rgb_expert.py",
    "requirements-signal.txt",
    "safe_output.py",
    "scripts/materialize-signal-shard.mjs",
    "src/signal-experiment-plan.js",
    "src/atomic-file.js",
    "src/managed-output.js",
    "src/balanced-sampler.js",
    "src/contract-validation.js",
    "src/contracts.js",
    "src/materialized-observations.js",
    "src/corruption-harness.js",
    "src/deterministic-random.js",
    "src/seeded-rgb-noise.js",
    "src/seeded-rgb-noise-worker.js",
    "src/track5-conditions.js",
)
FEATURE_IMPLEMENTATION_FILES = IMPLEMENTATION_FILES
FINAL_ARTIFACT_NAMES = (
    "signal-plan.json",
    "signal-training-features.json",
    "signal-validation-features.json",
    "signal-normalization.json",
    "signal-model.json",
    "signal-validation-logits.json",
    "signal-internal-validation-metrics.json",
)


def _progress(message: str) -> None:
    print(f"[signal] {message}", file=sys.stderr, flush=True)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


def resolve_signal_experiment_profile(
    experiment_profile: str,
    training_count: int | None,
    validation_source_count: int | None,
) -> tuple[int, int | None, str]:
    if experiment_profile not in EXPERIMENT_PROFILE_DEFAULTS:
        raise ValueError("Signal experiment_profile must be a supported profile.")
    default_training_count, default_validation_source_count = EXPERIMENT_PROFILE_DEFAULTS[
        experiment_profile
    ]
    resolved_training_count = (
        default_training_count if training_count is None else training_count
    )
    resolved_validation_source_count = (
        default_validation_source_count
        if validation_source_count is None and experiment_profile == "hackathon-v1"
        else validation_source_count
    )
    if experiment_profile == "hackathon-v1":
        if resolved_training_count != 8_064:
            raise ValueError(
                "Signal hackathon-v1 requires exactly 8,064 training draws."
            )
        if resolved_validation_source_count != 400:
            raise ValueError(
                "Signal hackathon-v1 requires exactly 400 validation sources."
            )
    elif experiment_profile == "issue-6-full-v1":
        if resolved_training_count != 40_320:
            raise ValueError(
                "Signal issue-6-full-v1 requires exactly 40,320 training draws."
            )
        if resolved_validation_source_count is not None:
            raise ValueError(
                "Signal issue-6-full-v1 requires the complete validation partition."
            )
    return (
        resolved_training_count,
        resolved_validation_source_count,
        EXPERIMENT_SCOPE_BY_PROFILE[experiment_profile],
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_file_sha256(path: Path) -> str:
    """Hash UTF-8 implementation text independently of checkout line endings."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Implementation file {path} must be readable UTF-8 text.") from error
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def _write_or_verify_json(path: Path, value: object, description: str) -> None:
    """Publish once, or require an existing artifact to be byte-for-byte reproducible."""
    expected = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        try:
            received = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Existing {description} at {path} is unreadable.") from error
        if received != expected:
            raise ValueError(
                f"Existing {description} at {path} differs from the freshly generated artifact."
            )
        return
    _write_json(path, value)
    if path.read_bytes() != expected:
        raise ValueError(f"Published {description} at {path} failed round-trip verification.")


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} at {path} is unreadable or invalid JSON.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} at {path} must be a JSON object.")
    return value


def _read_json_bytes(path: Path, description: str) -> tuple[bytes, dict]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} at {path} is unreadable or invalid JSON.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} at {path} must be a JSON object.")
    return raw, value


def _manifest_metadata(manifest: dict, manifest_sha256: str) -> dict:
    try:
        if manifest["manifest_schema_version"] != "track5-manifest-v1":
            raise ValueError("Signal experiment requires track5-manifest-v1.")
        if manifest["leakage_audit"]["status"] != "passed":
            raise ValueError("Signal experiment requires a passed Issue-3 leakage audit.")
        corruption = manifest["corruption"]
        return {
            "manifest_schema_version": manifest["manifest_schema_version"],
            "manifest_sha256": manifest_sha256,
            "dataset_revision": manifest["dataset"]["revision"],
            "selection_contract_version": manifest["selection_contract_version"],
            "observation_contract_version": manifest["observation_contract_version"],
            "condition_matrix_version": manifest["condition_matrix_version"],
            "sampler_contract_version": manifest["sampler_contract_version"],
            "artifact_schema_version": corruption["artifact_schema_version"],
            "corruption_version": corruption["transform_implementation_version"],
            "shared_observation_preprocessing_version": corruption["preprocessing_version"],
            "sharp_version": corruption["sharp_version"],
            "libvips_version": corruption["libvips_version"],
            "materialization_schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "materialized_encoding": MATERIALIZED_ENCODING,
            "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
            "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"Issue-3 manifest metadata is incomplete at {error}.") from error


def _run_node(command: list[str], description: str) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"{description} failed: {message or 'no diagnostic was returned'}")


def _runtime_provenance(node_binary: str) -> dict:
    completed = subprocess.run(
        [
            node_binary,
            "-e",
            (
                "const sharp=require('sharp');"
                "process.stdout.write(JSON.stringify({"
                "node:process.version,sharp:sharp.versions.sharp,libvips:sharp.versions.vips}));"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError("Signal experiment could not identify the Node.js/Sharp runtime.")
    try:
        node_runtime = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Signal experiment received invalid Node.js/Sharp runtime metadata.") from error
    if not isinstance(node_runtime, dict) or any(
        not isinstance(node_runtime.get(field), str) or not node_runtime[field]
        for field in ("node", "sharp", "libvips")
    ):
        raise ValueError("Signal experiment received incomplete Node.js/Sharp runtime metadata.")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PILLOW_VERSION,
        **node_runtime,
    }


def _implementation_provenance() -> dict:
    return {
        relative_path: _implementation_file_sha256(ROOT / relative_path)
        for relative_path in IMPLEMENTATION_FILES
    }


def _build_plan(
    manifest_path: Path,
    output_path: Path,
    *,
    experiment_profile: str,
    training_count: int,
    sampler_seed: int,
    validation_source_count: int | None,
    validation_seed: int,
    shard_raw_bytes: int,
    node_binary: str,
) -> dict:
    _run_node(
        [
            node_binary,
            str(SIGNAL_SHARD_CLI),
            "plan",
            "--manifest",
            str(manifest_path),
            "--experiment-profile",
            experiment_profile,
            "--training-count",
            str(training_count),
            "--training-seed",
            str(sampler_seed),
            "--validation-source-count",
            "all" if validation_source_count is None else str(validation_source_count),
            "--validation-seed",
            str(validation_seed),
            "--raw-byte-budget",
            str(shard_raw_bytes),
            "--output",
            str(output_path),
        ],
        "Signal observation planning",
    )
    plan = _read_json(output_path, "signal experiment plan")
    _validate_plan(
        plan,
        manifest_sha256=_file_sha256(manifest_path),
        experiment_profile=experiment_profile,
        training_count=training_count,
        sampler_seed=sampler_seed,
        validation_source_count=validation_source_count,
        validation_seed=validation_seed,
        shard_raw_bytes=shard_raw_bytes,
    )
    return plan


def _validate_plan_pin_metadata(value: dict, context: str) -> None:
    has_byte_length = "byte_length" in value
    has_exact_sha256 = "exact_sha256" in value
    if has_byte_length and not has_exact_sha256:
        raise ValueError(f"{context} byte_length requires exact_sha256.")
    if has_byte_length:
        byte_length = value["byte_length"]
        if (
            type(byte_length) is not int
            or byte_length <= 0
            or byte_length > JAVASCRIPT_MAX_SAFE_INTEGER
        ):
            raise ValueError(f"{context} byte_length must be a positive safe integer.")
    if has_exact_sha256:
        exact_sha256 = value["exact_sha256"]
        if (
            not isinstance(exact_sha256, str)
            or len(exact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in exact_sha256)
        ):
            raise ValueError(f"{context} exact_sha256 must be lowercase 64-digit hexadecimal.")


def _validate_plan(
    plan: dict,
    *,
    manifest_sha256: str,
    experiment_profile: str,
    training_count: int,
    sampler_seed: int,
    validation_source_count: int | None,
    validation_seed: int,
    shard_raw_bytes: int,
) -> None:
    resolved_training_count, resolved_validation_source_count, acceptance_scope = (
        resolve_signal_experiment_profile(
            experiment_profile,
            training_count,
            validation_source_count,
        )
    )
    if resolved_training_count != training_count:
        raise ValueError("Signal experiment plan training profile did not resolve exactly.")
    for name, value in (
        ("training_count", training_count),
        ("shard_raw_bytes", shard_raw_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Signal experiment plan expected {name} must be a positive integer.")
    for name, value in (("sampler_seed", sampler_seed), ("validation_seed", validation_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Signal experiment plan expected {name} must be a non-negative integer."
            )
    if validation_source_count is not None and (
        isinstance(validation_source_count, bool)
        or not isinstance(validation_source_count, int)
        or validation_source_count <= 0
    ):
        raise ValueError(
            "Signal experiment plan expected validation_source_count must be positive."
        )
    for name, value in (
        ("training_count", training_count),
        ("sampler_seed", sampler_seed),
        ("validation_seed", validation_seed),
        ("shard_raw_bytes", shard_raw_bytes),
    ):
        if value > JAVASCRIPT_MAX_SAFE_INTEGER:
            raise ValueError(
                f"Signal experiment plan expected {name} must be a JavaScript safe integer."
            )
    if training_count % BALANCED_TRAINING_GRANULARITY != 0:
        raise ValueError(
            "Signal experiment plan expected training_count must be divisible by 168 "
            "for the Issue-3 balanced sampler."
        )
    if plan.get("plan_schema_version") != "signal-experiment-plan-v1":
        raise ValueError("Signal experiment plan schema is stale or incompatible.")
    for field, expected in (
        ("experiment_profile", experiment_profile),
        ("acceptance_scope", acceptance_scope),
        ("parent_recipe_manifest_sha256", manifest_sha256),
        ("training_count", training_count),
        ("training_seed", sampler_seed),
        ("validation_seed", validation_seed),
        ("raw_byte_budget", shard_raw_bytes),
    ):
        if not _strict_json_equal(plan.get(field), expected):
            raise ValueError(f"Signal experiment plan has incompatible {field}.")
    plan_sha256 = plan.get("plan_sha256")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("Signal experiment plan SHA-256 is missing or malformed.")
    phases = plan.get("phases")
    if not isinstance(phases, list) or [phase.get("phase") for phase in phases] != [
        "expert-training",
        "internal-validation",
    ]:
        raise ValueError("Signal experiment plan has incompatible phases.")
    for phase in phases:
        shards = phase.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Signal experiment phase {phase.get('phase')} has no shards.")
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError("Signal experiment plan contains an invalid shard.")
            for field, expected in (
                ("parent_recipe_manifest_sha256", manifest_sha256),
                ("plan_sha256", plan_sha256),
                ("phase", phase["phase"]),
                ("index", index),
                ("count", len(shards)),
                ("raw_byte_budget", shard_raw_bytes),
            ):
                if not _strict_json_equal(shard.get(field), expected):
                    raise ValueError(f"Signal experiment shard has incompatible {field}.")
            raw_byte_estimate = shard.get("raw_byte_estimate")
            if (
                isinstance(raw_byte_estimate, bool)
                or not isinstance(raw_byte_estimate, int)
                or raw_byte_estimate <= 0
                or raw_byte_estimate > shard_raw_bytes
            ):
                raise ValueError("Signal experiment shard has an invalid raw_byte_estimate.")
            shard_identity = {
                key: value
                for key, value in shard.items()
                if key not in {"plan_sha256", "shard_sha256"}
            }
            if shard.get("shard_sha256") != _sha256(shard_identity):
                raise ValueError("Signal experiment shard content does not match its SHA-256.")
    plan_identity = {
        key: value
        for key, value in plan.items()
        if key != "plan_sha256"
    }
    plan_identity["phases"] = [
        {
            **{key: value for key, value in phase.items() if key != "shards"},
            "shards": [
                {key: value for key, value in shard.items() if key != "plan_sha256"}
                for shard in phase["shards"]
            ],
        }
        for phase in phases
    ]
    if _sha256(plan_identity) != plan_sha256:
        raise ValueError("Signal experiment plan content does not match its SHA-256.")

    records_by_phase = {}
    sources_by_phase = {}
    variants_across_phases = set()
    for phase in phases:
        phase_name = phase["phase"]
        phase_records = []
        phase_sources = set()
        phase_variants = set()
        labels = set()
        for shard in phase["shards"]:
            sources = shard.get("sources")
            records = shard.get("records")
            if not isinstance(sources, list) or not isinstance(records, list) or not records:
                raise ValueError("Signal experiment shard sources or records are missing.")
            if any(not isinstance(source, dict) for source in sources):
                raise ValueError("Signal experiment shard sources must be objects.")
            for source_index, source in enumerate(sources):
                _validate_plan_pin_metadata(
                    source,
                    f"Signal experiment plan source {source_index}",
                )
            shard_source_ids = [source.get("source_id") for source in sources]
            if (
                any(not isinstance(source_id, str) or not source_id for source_id in shard_source_ids)
                or len(set(shard_source_ids)) != len(shard_source_ids)
                or phase_sources & set(shard_source_ids)
            ):
                raise ValueError("Signal experiment sources repeat within or across shards.")
            phase_sources.update(shard_source_ids)
            source_by_id = {source["source_id"]: source for source in sources}
            record_source_ids = set()
            computed_raw_byte_estimate = 0
            for record_index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError("Signal experiment plan records must be objects.")
                _validate_plan_pin_metadata(
                    record,
                    f"Signal experiment plan record {record_index}",
                )
                source_id = record.get("source_id")
                variant_id = record.get("variant_id")
                weight = record.get("sample_weight")
                label = record.get("authenticity_label")
                if (
                    record.get("split") != phase_name
                    or source_id not in shard_source_ids
                    or not isinstance(variant_id, str)
                    or not variant_id
                    or variant_id in phase_variants
                    or variant_id in variants_across_phases
                    or isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight <= 0
                    or isinstance(label, bool)
                    or label not in (0, 1)
                ):
                    raise ValueError(
                        "Signal experiment plan has incompatible split, source, variant, weight, or label relationships."
                    )
                source = source_by_id[source_id]
                width = record.get("width")
                height = record.get("height")
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    or value > 2**53 - 1
                    for value in (width, height)
                ):
                    raise ValueError(
                        "Signal experiment plan records require positive safe-integer native dimensions."
                    )
                computed_raw_byte_estimate += width * height * 3
                if computed_raw_byte_estimate > 2**53 - 1:
                    raise ValueError(
                        "Signal experiment shard raw-byte estimate exceeds the safe-integer range."
                    )
                for field in (
                    "image_path",
                    "authenticity_label",
                    "split",
                    "width",
                    "height",
                ):
                    if field in source or field in record:
                        if not _strict_json_equal(source.get(field), record.get(field)):
                            raise ValueError(
                                f"Signal experiment plan record contradicts its source {field}."
                            )
                for field in ("byte_length", "exact_sha256"):
                    if field in record and not _strict_json_equal(
                        source.get(field), record.get(field)
                    ):
                        raise ValueError(
                            f"Signal experiment plan record contradicts its source {field}."
                        )
                phase_variants.add(variant_id)
                record_source_ids.add(source_id)
                labels.add(label)
                phase_records.append(record)
            if computed_raw_byte_estimate != shard["raw_byte_estimate"]:
                raise ValueError(
                    "Signal experiment shard raw_byte_estimate disagrees with its records."
                )
            if record_source_ids != set(shard_source_ids):
                raise ValueError("Signal experiment shard sources and records disagree.")
        if labels != {0, 1}:
            raise ValueError("Signal experiment phase requires both authenticity classes.")
        variants_across_phases.update(phase_variants)
        records_by_phase[phase_name] = phase_records
        sources_by_phase[phase_name] = phase_sources

    training_records = records_by_phase["expert-training"]
    if sum(record["sample_weight"] for record in training_records) != training_count:
        raise ValueError("Signal expert-training plan does not match requested training_count.")
    validation_records = records_by_phase["internal-validation"]
    actual_validation_source_count = len(sources_by_phase["internal-validation"])
    if not _strict_json_equal(
        plan.get("validation_source_count"), actual_validation_source_count
    ):
        raise ValueError(
            "Signal internal-validation plan has incompatible validation_source_count."
        )
    if (
        resolved_validation_source_count is not None
        and actual_validation_source_count != resolved_validation_source_count
    ):
        raise ValueError(
            "Signal internal-validation plan does not match the requested source subset."
        )
    expected_conditions = set(CANONICAL_VALIDATION_CONDITIONS)
    conditions_by_source = {source_id: set() for source_id in sources_by_phase["internal-validation"]}
    for record in validation_records:
        if record["sample_weight"] != 1:
            raise ValueError("Signal internal-validation plan sample weights must be one.")
        condition = (record.get("condition_family"), record.get("severity"))
        conditions = conditions_by_source[record["source_id"]]
        if condition not in expected_conditions or condition in conditions:
            raise ValueError("Signal internal-validation plan has an invalid or duplicate condition.")
        conditions.add(condition)
    if any(conditions != expected_conditions for conditions in conditions_by_source.values()):
        raise ValueError("Signal internal-validation plan is missing the canonical condition matrix.")
    if sources_by_phase["expert-training"] & sources_by_phase["internal-validation"]:
        raise ValueError("Signal expert-training and internal-validation plan sources must be disjoint.")


def _plan_shard_filename(shard: dict) -> str:
    return (
        f"{shard['phase']}-{shard['index']:05d}-of-{shard['count']:05d}.json"
    )


def _publish_plan(output_directory: Path, plan: dict) -> None:
    """Publish sidecars before the plan commit marker without replacing another plan."""
    plan_path = managed_output_path(
        output_directory, "signal-plan.json", "signal experiment plan"
    )
    if plan_path.exists():
        _write_or_verify_json(plan_path, plan, "signal experiment plan")
    shard_directory = managed_output_path(
        output_directory, "signal-plan.shards", "signal plan shard directory"
    )
    shard_directory.mkdir(exist_ok=True)
    shard_directory = managed_output_path(
        output_directory, "signal-plan.shards", "signal plan shard directory"
    )
    expected_sidecars = {
        _plan_shard_filename(shard): shard
        for phase in plan["phases"]
        for shard in phase["shards"]
    }
    if shard_directory.exists():
        received_names = {
            path.name for path in shard_directory.iterdir() if path.is_file()
        }
        unexpected = received_names - expected_sidecars.keys()
        if unexpected:
            raise ValueError(
                "Existing signal plan shard directory contains files from a different plan."
            )
    for name, shard in expected_sidecars.items():
        shard_path = managed_output_path(
            output_directory,
            Path("signal-plan.shards") / name,
            f"signal plan shard {name}",
        )
        _write_or_verify_json(
            shard_path,
            shard,
            f"signal plan shard {name}",
        )
    plan_path = managed_output_path(
        output_directory, "signal-plan.json", "signal experiment plan"
    )
    _write_or_verify_json(plan_path, plan, "signal experiment plan")


def _validate_existing_run_marker(
    path: Path,
    *,
    manifest_metadata: dict,
    plan_sha256: str,
    requested: dict,
    runtime_versions: dict,
    implementation_sha256: dict,
) -> None:
    if not path.exists():
        return
    path = managed_output_path(path.parent, path.name, "existing signal run marker")
    marker = _read_json(path, "existing signal run marker")
    if marker.get("run_schema_version") != "signal-experiment-run-v1":
        raise ValueError("Existing output directory has a stale signal run marker.")
    if marker.get("implementation_hash_contract_version") != IMPLEMENTATION_HASH_CONTRACT_VERSION:
        raise ValueError("Existing output directory has a stale implementation hash contract.")
    for field, expected in (
        ("manifest_metadata", manifest_metadata),
        ("plan_sha256", plan_sha256),
        ("runtime_versions", runtime_versions),
        ("implementation_sha256", implementation_sha256),
        *requested.items(),
    ):
        if not _strict_json_equal(marker.get(field), expected):
            raise ValueError(
                f"Existing output directory run marker has incompatible {field}."
            )
    artifact_sha256 = marker.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != set(
        FINAL_ARTIFACT_NAMES
    ):
        raise ValueError(
            "Existing output directory run marker has an incomplete or unexpected artifact set."
        )
    artifact_paths = {}
    for name, expected_digest in artifact_sha256.items():
        if (
            Path(name).name != name
            or Path(name).is_absolute()
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(
                "Existing output directory artifact checksums require contained basenames and "
                "lowercase SHA-256 values."
            )
        artifact_path = managed_output_path(
            path.parent, name, f"existing signal artifact {name}"
        )
        artifact_paths[name] = artifact_path
        if (
            not artifact_path.is_file()
            or _file_sha256(artifact_path) != expected_digest
        ):
            raise ValueError(
                f"Existing output directory artifact {name!r} is missing or stale."
            )
    try:
        plan = _read_json(artifact_paths["signal-plan.json"], "existing signal plan")
        training = _read_json(
            artifact_paths["signal-training-features.json"],
            "existing signal training features",
        )
        validation = _read_json(
            artifact_paths["signal-validation-features.json"],
            "existing signal validation features",
        )
        normalization = _read_json(
            artifact_paths["signal-normalization.json"],
            "existing signal normalization",
        )
        model = _read_json(
            artifact_paths["signal-model.json"], "existing signal model"
        )
        logits = _read_json(
            artifact_paths["signal-validation-logits.json"],
            "existing signal validation logits",
        )
        metrics = _read_json(
            artifact_paths["signal-internal-validation-metrics.json"],
            "existing signal metrics",
        )
        _validate_plan(
            plan,
            manifest_sha256=manifest_metadata["manifest_sha256"],
            experiment_profile=requested["experiment_profile"],
            training_count=plan["training_count"],
            sampler_seed=requested["sampler_seed"],
            validation_source_count=requested["validation_source_limit"],
            validation_seed=requested["validation_seed"],
            shard_raw_bytes=requested["shard_raw_bytes"],
        )
        phases = {phase["phase"]: phase for phase in plan["phases"]}
        expected_extraction = _feature_extraction_metadata(requested["resolution"])
        training_records = validate_signal_feature_cache(
            training,
            phases["expert-training"],
            manifest_metadata,
            expected_resolution=requested["resolution"],
            expected_feature_extraction=expected_extraction,
        )
        validation_records = validate_signal_feature_cache(
            validation,
            phases["internal-validation"],
            manifest_metadata,
            expected_resolution=requested["resolution"],
            expected_feature_extraction=expected_extraction,
        )
        experiment_provenance = {
            "experiment_profile": requested["experiment_profile"],
            "acceptance_scope": requested["acceptance_scope"],
            "training_plan_sha256": plan_sha256,
            "training_feature_records_sha256": training["records_sha256"],
            "validation_feature_records_sha256": validation["records_sha256"],
            "signal_feature_extraction_version": FEATURE_EXTRACTION_VERSION,
            "resolution": requested["resolution"],
            "feature_extraction": expected_extraction,
        }
        training_provenance = {
            field: experiment_provenance[field]
            for field in (
                "experiment_profile",
                "acceptance_scope",
                "training_plan_sha256",
                "training_feature_records_sha256",
                "signal_feature_extraction_version",
                "resolution",
                "feature_extraction",
            )
        }
        expected_normalization = fit_normalization(
            training_records,
            manifest_metadata=manifest_metadata,
            training_provenance=training_provenance,
        )
        if not _strict_json_equal(normalization, {
            "normalization_revision": expected_normalization["normalization_revision"],
            "normalization": expected_normalization,
        }):
            raise ValueError("normalization artifact does not reproduce from training records")
        _, validated_model = read_model_bundle(
            artifact_paths["signal-model.json"],
            manifest_metadata=manifest_metadata,
            expected_experiment_provenance=experiment_provenance,
        )
        if (
            not _strict_json_equal(validated_model, model)
            or not _strict_json_equal(validated_model["normalization"], expected_normalization)
            or not _strict_json_equal(validated_model.get("seed"), requested["model_seed"])
            or not _strict_json_equal(validated_model.get("epochs"), requested["epochs"])
            or not _strict_json_equal(
                validated_model.get("learning_rate"), requested["learning_rate"]
            )
            or not _strict_json_equal(
                validated_model.get("training_selection"), training["selection"]
            )
            or not _strict_json_equal(
                validated_model.get("validation_selection"), validation["selection"]
            )
        ):
            raise ValueError("checkpoint does not reproduce the completed experiment contract")
        validated_logits = validate_signal_logit_cache(
            logits,
            expected_feature_records=validation_records,
            checkpoint_bundle=validated_model,
            manifest_metadata=manifest_metadata,
            expected_experiment_provenance=experiment_provenance,
            expected_feature_cache_records_sha256=validation["records_sha256"],
        )
        if not _strict_json_equal(metrics, evaluate_signal_only(validated_logits)):
            raise ValueError("metrics do not reproduce from the validated signal logits")
        storage = {
            phase_name: {
                "shard_count": len(phase["shards"]),
                "maximum_shard_raw_byte_estimate": max(
                    shard["raw_byte_estimate"] for shard in phase["shards"]
                ),
                "total_raw_byte_estimate": sum(
                    shard["raw_byte_estimate"] for shard in phase["shards"]
                ),
            }
            for phase_name, phase in phases.items()
        }
        if (
            plan.get("plan_sha256") != plan_sha256
            or not _strict_json_equal(
                validated_model["experiment_provenance"], experiment_provenance
            )
        ):
            raise ValueError("artifact provenance relationships disagree")
        derived = {
            "training_sample_count": training["selection"]["sample_count"],
            "training_unique_observation_count": len(training_records),
            "training_source_count": len({record["source_id"] for record in training_records}),
            "validation_observation_count": len(validation_records),
            "validation_source_count": len({record["source_id"] for record in validation_records}),
            "checkpoint_revision": validated_model["checkpoint_revision"],
            "normalization_revision": validated_model["normalization_revision"],
            "total_parameter_count": 449,
            "trainable_parameter_count": 449,
            "selected_epoch": validated_model["selected_epoch"],
            "checkpoint_selection_metric": validated_model["selection_metric"],
            "checkpoint_selection_metric_version": validated_model["selection_metric_version"],
            "checkpoint_selection_score": validated_model["selection_score"],
            "bounded_materialization": storage,
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            "Existing output directory artifacts have incompatible scientific provenance."
        ) from error
    for field, expected in derived.items():
        if not _strict_json_equal(marker.get(field), expected):
            raise ValueError(
                f"Existing output directory run marker has incompatible scientific summary {field}."
            )


def _expected_phase_records(phase: dict) -> dict[str, dict]:
    expected = {}
    for shard in phase.get("shards", []):
        for record in shard.get("records", []):
            variant_id = record.get("variant_id")
            if not isinstance(variant_id, str) or not variant_id or variant_id in expected:
                raise ValueError("Signal plan has a missing or duplicate variant_id.")
            expected[variant_id] = record
    return expected


def _validate_materialized_shard(
    manifest: dict,
    root: Path,
    expected_shard: dict,
    *,
    manifest_metadata: dict,
    plan_sha256: str,
) -> list[dict]:
    if not _strict_json_equal(
        manifest.get("parent_recipe_manifest_sha256"),
        expected_shard.get("parent_recipe_manifest_sha256"),
    ):
        raise ValueError(
            "Signal materialized shard has incompatible parent_recipe_manifest_sha256."
        )
    header = expected_shard.get("recipe_manifest_header")
    if not isinstance(header, dict):
        raise ValueError("Signal shard plan is missing its recipe manifest header.")
    for field, expected_value in header.items():
        if not _strict_json_equal(manifest.get(field), expected_value):
            raise ValueError(
                f"Signal materialized shard recipe manifest header has incompatible {field}."
            )
    expected_sources = expected_shard.get("sources")
    if not isinstance(expected_sources, list) or not _strict_json_equal(
        manifest.get("sources"), expected_sources
    ):
        raise ValueError("Signal materialized shard sources are missing or incompatible.")
    if manifest.get("materialization_schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("Signal shard materialization schema is stale or incompatible.")
    materialization = manifest.get("materialization")
    if not isinstance(materialization, dict):
        raise ValueError("Signal shard materialization metadata is missing.")
    for field, expected in (
        ("encoding", MATERIALIZED_ENCODING),
        (
            "shared_observation_preprocessing_version",
            manifest_metadata["shared_observation_preprocessing_version"],
        ),
        ("corruption_version", manifest_metadata["corruption_version"]),
        ("sharp_version", manifest_metadata["sharp_version"]),
        ("libvips_version", manifest_metadata["libvips_version"]),
    ):
        if not _strict_json_equal(materialization.get(field), expected):
            raise ValueError(f"Signal shard materialization has incompatible {field}.")
    provenance = manifest.get("signal_shard_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Signal shard provenance is missing.")
    for field, expected in (
        ("parent_recipe_manifest_sha256", expected_shard.get("parent_recipe_manifest_sha256")),
        ("plan_sha256", plan_sha256),
        ("shard_sha256", expected_shard.get("shard_sha256")),
        ("phase", expected_shard.get("phase")),
        ("index", expected_shard.get("index")),
        ("count", expected_shard.get("count")),
        ("variant_set_digest", expected_shard.get("variant_set_digest")),
        ("raw_byte_budget", expected_shard.get("raw_byte_budget")),
        ("raw_byte_estimate", expected_shard.get("raw_byte_estimate")),
    ):
        if not _strict_json_equal(provenance.get(field), expected):
            raise ValueError(f"Signal materialized shard has incompatible {field}.")
    observations = manifest.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Signal materialized shard observations must be an array.")
    expected = {record["variant_id"]: record for record in expected_shard["records"]}
    if len(expected) != len(expected_shard["records"]):
        raise ValueError("Signal shard plan repeats variant identifiers.")
    received = {}
    root = root.resolve()
    for index, observation in enumerate(observations):
        variant_id = observation.get("variant_id")
        if variant_id in received or variant_id not in expected:
            raise ValueError(f"Signal materialized shard record {index} is duplicate or unexpected.")
        planned = expected[variant_id]
        for field, expected_value in planned.items():
            if not _strict_json_equal(observation.get(field), expected_value):
                raise ValueError(f"Signal materialized shard record {index} has incompatible {field}.")
        if observation.get("materialized_encoding") != MATERIALIZED_ENCODING:
            raise ValueError(f"Signal materialized shard record {index} has incompatible encoding.")
        relative_path = observation.get("materialized_image_path")
        expected_sha256 = observation.get("materialized_sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Signal materialized shard record {index} has no contained path.")
        relative_segments = relative_path.replace("\\", "/").split("/")
        relative = Path(relative_path)
        windows_relative = PureWindowsPath(relative_path)
        if (
            "\\" in relative_path
            or "\x00" in relative_path
            or any(segment in {"", ".", ".."} for segment in relative_segments)
            or relative.is_absolute()
            or relative.anchor
            or relative.drive
            or windows_relative.anchor
            or windows_relative.drive
            or relative.as_posix() != relative_path
        ):
            raise ValueError(
                f"Signal materialized shard record {index} requires a canonical relative contained path."
            )
        try:
            path = managed_output_path(
                root,
                relative_path,
                f"materialized shard record {index}",
            )
        except ValueError as error:
            raise ValueError(
                f"Signal materialized shard record {index} has no contained path."
            ) from error
        if not path.is_file():
            raise ValueError(f"Signal materialized shard record {index} file is missing.")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"Signal materialized shard record {index} checksum is malformed.")
        received[variant_id] = {**observation, "materialized_path": path}
    if received.keys() != expected.keys():
        raise ValueError("Signal materialized shard is missing planned observations.")
    if not _strict_json_equal(materialization.get("observation_count"), len(observations)):
        raise ValueError("Signal materialized shard observation count is incompatible.")
    return [received[record["variant_id"]] for record in expected_shard["records"]]


def _feature_record(observation: dict, *, resolution: int) -> dict:
    try:
        materialized_bytes = observation["materialized_path"].read_bytes()
    except (KeyError, OSError) as error:
        raise ValueError("Signal materialized observation cannot be read for feature extraction.") from error
    materialized_sha256 = hashlib.sha256(materialized_bytes).hexdigest()
    if materialized_sha256 != observation.get("materialized_sha256"):
        raise ValueError("Signal materialized observation checksum is missing or stale.")
    representation = extract_signal_representation(
        decode_expert_rgb_bytes(
            materialized_bytes,
            resolution=resolution,
            expected_width=observation["width"],
            expected_height=observation["height"],
        ),
    )
    return {
        "source_id": observation["source_id"],
        "variant_id": observation["variant_id"],
        "split": observation["split"],
        "authenticity_label": observation["authenticity_label"],
        "condition_family": observation["condition_family"],
        "severity": observation["severity"],
        "sample_weight": observation["sample_weight"],
        "materialized_sha256": observation["materialized_sha256"],
        "materialized_encoding": observation["materialized_encoding"],
        "signal_representation_version": representation["version"],
        "features": representation["features"].tolist(),
    }


def _feature_extraction_metadata(resolution: int) -> dict:
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal feature extraction resolution must be 224 or 384.")
    return {
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "resolution": resolution,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "implementation_sha256": {
            relative_path: _implementation_file_sha256(ROOT / relative_path)
            for relative_path in FEATURE_IMPLEMENTATION_FILES
        },
        "implementation_hash_contract_version": IMPLEMENTATION_HASH_CONTRACT_VERSION,
    }


def _require_current_feature_extraction_snapshot(snapshot: dict, resolution: int) -> None:
    if not _strict_json_equal(snapshot, _feature_extraction_metadata(resolution)):
        raise ValueError(
            "Signal feature implementation or runtime changed during extraction; refusing publication."
        )


def _validate_feature_shard_cache(
    cache: dict,
    expected_shard: dict,
    *,
    manifest_metadata: dict,
    plan_sha256: str,
    resolution: int,
    feature_extraction: dict | None = None,
) -> list[dict]:
    expected_extraction = feature_extraction or _feature_extraction_metadata(resolution)
    if cache.get("feature_shard_schema_version") != "signal-feature-shard-v1":
        raise ValueError("Signal feature shard cache schema is stale or incompatible.")
    for field, expected in (
        ("manifest_metadata", manifest_metadata),
        ("plan_sha256", plan_sha256),
        ("shard_sha256", expected_shard.get("shard_sha256")),
        ("phase", expected_shard.get("phase")),
        ("index", expected_shard.get("index")),
        ("count", expected_shard.get("count")),
        ("variant_set_digest", expected_shard.get("variant_set_digest")),
        ("feature_extraction", expected_extraction),
    ):
        if not _strict_json_equal(cache.get(field), expected):
            raise ValueError(f"Signal feature shard cache has incompatible {field}.")
    digest = cache.get("materialized_manifest_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Signal feature shard cache has an invalid materialized manifest digest.")
    records = cache.get("records")
    if not isinstance(records, list) or cache.get("records_sha256") != _sha256(records):
        raise ValueError("Signal feature shard cache records or digest are stale.")
    combined = {
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "manifest_metadata": manifest_metadata,
        "feature_extraction": cache["feature_extraction"],
        "selection": {
            "split": expected_shard["phase"],
            "kind": (
                "balanced-sampler"
                if expected_shard["phase"] == "expert-training"
                else "complete-condition-matrix"
            ),
            "sample_count": sum(record.get("sample_weight", 0) for record in records),
            "unique_observation_count": len(records),
            "plan_sha256": plan_sha256,
            "shard_count": 1,
        },
        "records_sha256": cache["records_sha256"],
        "records": records,
    }
    return validate_signal_feature_cache(
        combined,
        {"phase": expected_shard["phase"], "shards": [expected_shard]},
        manifest_metadata,
        expected_resolution=resolution,
        expected_feature_extraction=expected_extraction,
    )


def _feature_shard_completion_receipt(cache: dict, shard: dict) -> dict:
    return {
        "completion_schema_version": FEATURE_SHARD_COMPLETION_SCHEMA_VERSION,
        "manifest_metadata": cache["manifest_metadata"],
        "plan_sha256": cache["plan_sha256"],
        "shard_sha256": cache["shard_sha256"],
        "phase": cache["phase"],
        "index": cache["index"],
        "count": cache["count"],
        "variant_set_digest": cache["variant_set_digest"],
        "source_set_sha256": _sha256(shard["sources"]),
        "feature_extraction": cache["feature_extraction"],
        "materialized_manifest_sha256": cache["materialized_manifest_sha256"],
        "records_sha256": cache["records_sha256"],
        "feature_cache_sha256": _sha256(cache),
    }


def _validate_feature_shard_completion(
    receipt: dict,
    cache: dict,
    shard: dict,
) -> None:
    if not _strict_json_equal(receipt, _feature_shard_completion_receipt(cache, shard)):
        raise ValueError(
            "Signal feature shard cache completion receipt is stale or incompatible."
        )


def _cached_source_path(dataset_root: Path, value: object, index: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Signal cached shard source {index} has no image_path.")
    relative = Path(value)
    windows_relative = PureWindowsPath(value)
    segments = value.replace("\\", "/").split("/")
    if (
        "\\" in value
        or "\x00" in value
        or any(segment in {"", ".", ".."} for segment in segments)
        or relative.is_absolute()
        or relative.anchor
        or relative.drive
        or windows_relative.anchor
        or windows_relative.drive
        or relative.as_posix() != value
    ):
        raise ValueError(
            f"Signal cached shard source {index} requires a canonical relative image_path."
        )
    try:
        return managed_output_path(
            dataset_root,
            value,
            f"cached source {index}",
        )
    except ValueError as error:
        raise ValueError(
            f"Signal cached shard source {index} image_path is outside or redirected."
        ) from error


def _revalidate_cached_source_bytes(shard: dict, dataset_root: Path) -> bool:
    """Return whether all pins exist after verifying every source byte-for-byte."""
    sources = shard.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Signal cached shard has no source set to revalidate.")
    if any(not isinstance(source, dict) for source in sources):
        raise ValueError("Signal cached shard source set is malformed.")
    if any("exact_sha256" not in source for source in sources):
        return False
    for index, source in enumerate(sources):
        expected_sha256 = source.get("exact_sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"Signal cached shard source {index} has an invalid exact SHA-256 pin.")
        expected_byte_length = source.get("byte_length")
        if expected_byte_length is not None and (
            type(expected_byte_length) is not int or expected_byte_length <= 0
        ):
            raise ValueError(f"Signal cached shard source {index} has an invalid byte-length pin.")
        path = _cached_source_path(dataset_root, source.get("image_path"), index)
        try:
            if not path.is_file():
                raise ValueError(f"Signal cached shard source {index} file is missing.")
            if expected_byte_length is not None and path.stat().st_size != expected_byte_length:
                raise ValueError(
                    f"Signal cached shard source {index} disagrees with its pinned byte length."
                )
            received_sha256 = _file_sha256(path)
        except OSError as error:
            raise ValueError(f"Signal cached shard source {index} cannot be verified.") from error
        if received_sha256 != expected_sha256:
            raise ValueError(
                f"Signal cached shard source {index} disagrees with its pinned source SHA-256."
            )
    return True


def _extract_phase_features(
    phase: dict,
    *,
    dataset_root: Path,
    output_directory: Path,
    manifest_metadata: dict,
    plan_sha256: str,
    resolution: int,
    node_binary: str,
    feature_extraction: dict,
) -> dict:
    records = []
    work_root = managed_output_path(
        output_directory, ".signal-materialized", "materialization work directory"
    )
    work_root.mkdir(exist_ok=True)
    work_root = managed_output_path(
        output_directory, ".signal-materialized", "materialization work directory"
    )
    shard_cache_root = managed_output_path(
        output_directory, "signal-feature-shards", "feature shard cache directory"
    )
    shard_cache_root.mkdir(exist_ok=True)
    shard_cache_root = managed_output_path(
        output_directory, "signal-feature-shards", "feature shard cache directory"
    )
    for shard in phase["shards"]:
        _require_current_feature_extraction_snapshot(feature_extraction, resolution)
        stem = f"{phase['phase']}-{shard['index']:05d}"
        shard_plan_path = managed_output_path(
            output_directory,
            Path("signal-feature-shards") / f"{stem}.plan.json",
            f"signal shard plan {stem}",
        )
        feature_shard_path = managed_output_path(
            output_directory,
            Path("signal-feature-shards") / f"{stem}.features.json",
            f"signal feature shard {stem}",
        )
        completion_path = managed_output_path(
            output_directory,
            Path("signal-feature-shards") / f"{stem}.complete.json",
            f"signal feature shard completion {stem}",
        )
        materialized_relative = Path(".signal-materialized") / stem
        materialized_root = managed_output_path(
            output_directory, materialized_relative, f"materialized shard {stem}"
        )
        _write_or_verify_json(shard_plan_path, shard, f"signal shard plan {stem}")
        materialized_root = managed_output_path(
            output_directory, materialized_relative, f"materialized shard {stem} cleanup"
        )
        if materialized_root.exists():
            shutil.rmtree(materialized_root)
        existing_cache = (
            _read_json(feature_shard_path, f"signal feature shard {stem}")
            if feature_shard_path.exists()
            else None
        )
        existing_completion = (
            _read_json(completion_path, f"signal feature shard completion {stem}")
            if completion_path.exists()
            else None
        )
        if existing_cache is not None:
            cached_records = _validate_feature_shard_cache(
                existing_cache,
                shard,
                manifest_metadata=manifest_metadata,
                plan_sha256=plan_sha256,
                resolution=resolution,
                feature_extraction=feature_extraction,
            )
            if existing_completion is not None:
                _validate_feature_shard_completion(existing_completion, existing_cache, shard)
                if _revalidate_cached_source_bytes(shard, dataset_root):
                    records.extend(cached_records)
                    _progress(
                        f"reused completed feature shard {stem} after source-byte revalidation"
                    )
                    continue
        _progress(
            f"materializing shard {stem} "
            f"({shard['raw_byte_estimate']} estimated raw bytes)"
        )
        try:
            materialized_root = managed_output_path(
                output_directory, materialized_relative, f"materialized shard {stem}"
            )
            _run_node(
                [
                    node_binary,
                    str(SIGNAL_SHARD_CLI),
                    "materialize",
                    "--shard-plan",
                    str(shard_plan_path),
                    "--expected-plan-sha256",
                    plan_sha256,
                    "--expected-shard-sha256",
                    str(shard["shard_sha256"]),
                    "--phase",
                    phase["phase"],
                    "--index",
                    str(shard["index"]),
                    "--dataset-root",
                    str(dataset_root),
                    "--output-dir",
                    str(materialized_root),
                ],
                f"Signal shard {stem} materialization",
            )
            materialized_root = managed_output_path(
                output_directory, materialized_relative, f"materialized shard {stem}"
            )
            materialized_manifest_path = managed_output_path(
                output_directory,
                materialized_relative / "track5-materialized-manifest.json",
                f"materialized manifest {stem}",
            )
            materialized_manifest = _read_json(
                materialized_manifest_path,
                f"signal materialized shard {stem}",
            )
            observations = _validate_materialized_shard(
                materialized_manifest,
                materialized_root,
                shard,
                manifest_metadata=manifest_metadata,
                plan_sha256=plan_sha256,
            )
            _progress(f"extracting 26 features for {len(observations)} observations in {stem}")
            shard_records = [
                _feature_record(observation, resolution=resolution)
                for observation in observations
            ]
            _require_current_feature_extraction_snapshot(feature_extraction, resolution)
            shard_cache = {
                "feature_shard_schema_version": "signal-feature-shard-v1",
                "manifest_metadata": manifest_metadata,
                "plan_sha256": plan_sha256,
                "shard_sha256": shard["shard_sha256"],
                "phase": phase["phase"],
                "index": shard["index"],
                "count": shard["count"],
                "variant_set_digest": shard["variant_set_digest"],
                "feature_extraction": feature_extraction,
                "materialized_manifest_sha256": _sha256(materialized_manifest),
                "records_sha256": _sha256(shard_records),
                "records": shard_records,
            }
            fresh_records = _validate_feature_shard_cache(
                shard_cache,
                shard,
                manifest_metadata=manifest_metadata,
                plan_sha256=plan_sha256,
                resolution=resolution,
                feature_extraction=feature_extraction,
            )
            if existing_cache is not None:
                if not _strict_json_equal(existing_cache, shard_cache):
                    raise ValueError(
                        f"Signal feature shard cache {stem} does not match fresh extraction."
                    )
            else:
                _write_or_verify_json(
                    feature_shard_path,
                    shard_cache,
                    f"signal feature shard cache {stem}",
                )
            _write_or_verify_json(
                completion_path,
                _feature_shard_completion_receipt(shard_cache, shard),
                f"signal feature shard completion receipt {stem}",
            )
            records.extend(fresh_records)
        finally:
            materialized_root = managed_output_path(
                output_directory,
                materialized_relative,
                f"materialized shard {stem} cleanup",
            )
            if materialized_root.exists():
                shutil.rmtree(materialized_root)
        _progress(f"validated feature shard {stem}; evicted its materialized PNGs")
    work_root = managed_output_path(
        output_directory, ".signal-materialized", "materialization work directory cleanup"
    )
    if work_root.exists() and not any(work_root.iterdir()):
        work_root.rmdir()
    selection = {
        "split": phase["phase"],
        "kind": "balanced-sampler" if phase["phase"] == "expert-training" else "complete-condition-matrix",
        "sample_count": sum(record["sample_weight"] for record in records),
        "unique_observation_count": len(records),
        "plan_sha256": plan_sha256,
        "shard_count": len(phase["shards"]),
    }
    return {
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "manifest_metadata": manifest_metadata,
        "feature_extraction": feature_extraction,
        "selection": selection,
        "records_sha256": _sha256(records),
        "records": records,
    }


def validate_signal_feature_cache(
    cache: dict,
    expected_phase: dict,
    manifest_metadata: dict,
    *,
    expected_resolution: int,
    expected_feature_extraction: dict | None = None,
) -> list[dict]:
    if cache.get("feature_cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("Signal feature cache schema is stale or incompatible.")
    if not _strict_json_equal(cache.get("manifest_metadata"), manifest_metadata):
        raise ValueError("Signal feature cache manifest provenance is stale or incompatible.")
    extraction = cache.get("feature_extraction")
    expected_extraction = expected_feature_extraction or _feature_extraction_metadata(
        expected_resolution
    )
    if not _strict_json_equal(extraction, expected_extraction):
        raise ValueError("Signal feature cache order is stale or incompatible.")
    records = cache.get("records")
    if not isinstance(records, list) or cache.get("records_sha256") != _sha256(records):
        raise ValueError("Signal feature cache records or digest are stale.")
    plan_hashes = {
        shard.get("plan_sha256")
        for shard in expected_phase.get("shards", [])
    }
    if len(plan_hashes) != 1 or next(iter(plan_hashes), None) is None:
        raise ValueError("Signal feature expectation has no single plan revision.")
    plan_sha256 = next(iter(plan_hashes))
    expected = _expected_phase_records(expected_phase)
    received = {}
    for index, record in enumerate(records):
        variant_id = record.get("variant_id")
        if variant_id in received or variant_id not in expected:
            raise ValueError(f"Signal feature record {index} is duplicate or unexpected.")
        planned = expected[variant_id]
        for field in (
            "source_id",
            "split",
            "authenticity_label",
            "condition_family",
            "severity",
            "sample_weight",
        ):
            if not _strict_json_equal(record.get(field), planned.get(field)):
                raise ValueError(f"Signal feature record {index} has incompatible {field}.")
        raw_features = record.get("features")
        if (
            not isinstance(raw_features, list)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_features
            )
        ):
            raise ValueError(f"Signal feature record {index} requires strictly numeric features.")
        features = np.asarray(raw_features, dtype=np.float64)
        if features.shape != (26,) or not np.isfinite(features).all():
            raise ValueError(f"Signal feature record {index} requires 26 finite features.")
        if record.get("materialized_encoding") != MATERIALIZED_ENCODING:
            raise ValueError(f"Signal feature record {index} has incompatible materialized encoding.")
        if record.get("signal_representation_version") != SIGNAL_REPRESENTATION_VERSION:
            raise ValueError(f"Signal feature record {index} has a stale representation version.")
        materialized_sha256 = record.get("materialized_sha256", "")
        if (
            not isinstance(materialized_sha256, str)
            or len(materialized_sha256) != 64
            or any(character not in "0123456789abcdef" for character in materialized_sha256)
        ):
            raise ValueError(f"Signal feature record {index} has an invalid materialized checksum.")
        received[variant_id] = record
    if set(received) != set(expected):
        raise ValueError("Signal feature cache is missing planned observations.")
    if list(received) != list(expected):
        raise ValueError("Signal feature cache record order is incompatible with the plan.")
    selection = cache.get("selection", {})
    expected_kind = (
        "balanced-sampler"
        if expected_phase.get("phase") == "expert-training"
        else "complete-condition-matrix"
    )
    expected_selection = {
        "split": expected_phase.get("phase"),
        "kind": expected_kind,
        "sample_count": sum(record["sample_weight"] for record in records),
        "unique_observation_count": len(records),
        "plan_sha256": plan_sha256,
        "shard_count": len(expected_phase.get("shards", [])),
    }
    if not _strict_json_equal(selection, expected_selection):
        raise ValueError("Signal feature cache sample count is incompatible.")
    return [received[record["variant_id"]] for shard in expected_phase["shards"] for record in shard["records"]]


def validate_signal_logit_cache(
    cache: dict,
    *,
    expected_feature_records: list[dict],
    checkpoint_bundle: dict,
    manifest_metadata: dict,
    expected_experiment_provenance: dict,
    expected_feature_cache_records_sha256: str,
) -> list[dict]:
    if (
        not isinstance(expected_feature_cache_records_sha256, str)
        or expected_feature_cache_records_sha256
        != expected_experiment_provenance.get("validation_feature_records_sha256")
        or _sha256(expected_feature_records) != expected_feature_cache_records_sha256
    ):
        raise ValueError(
            "Signal logit cache validation features contradict checkpoint experiment provenance."
        )
    if cache.get("logit_cache_schema_version") != "signal-logit-cache-v1":
        raise ValueError("Signal logit cache schema is stale or incompatible.")
    for field, expected in (
        ("manifest_metadata", manifest_metadata),
        ("feature_cache_records_sha256", expected_feature_cache_records_sha256),
        ("checkpoint_revision", checkpoint_bundle.get("checkpoint_revision")),
        ("normalization_revision", checkpoint_bundle.get("normalization_revision")),
    ):
        if not _strict_json_equal(cache.get(field), expected):
            raise ValueError(f"Signal logit cache has stale or incompatible {field}.")
    records = cache.get("records")
    if not isinstance(records, list) or cache.get("records_sha256") != _sha256(records):
        raise ValueError("Signal logit cache records or digest are stale.")
    return validate_signal_cache(
        records,
        checkpoint_bundle=checkpoint_bundle,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=expected_experiment_provenance,
        expected_feature_records=expected_feature_records,
    )


def run_signal_experiment(
    manifest_path: Path | str,
    dataset_root: Path | str,
    output_directory: Path | str,
    *,
    experiment_profile: str = "custom-v1",
    training_count: int | None = 40_320,
    validation_source_count: int | None = None,
    validation_seed: int = 61,
    sampler_seed: int = 61,
    model_seed: int = 61,
    epochs: int = 200,
    learning_rate: float = 0.02,
    resolution: int = 384,
    shard_raw_bytes: int = 2**30,
    node_binary: str = "node",
) -> dict:
    training_count, validation_source_count, acceptance_scope = (
        resolve_signal_experiment_profile(
            experiment_profile,
            training_count,
            validation_source_count,
        )
    )
    for name, value in (
        ("training_count", training_count),
        ("epochs", epochs),
        ("shard_raw_bytes", shard_raw_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Signal {name} must be a positive integer.")
    for name, value in (
        ("sampler_seed", sampler_seed),
        ("validation_seed", validation_seed),
        ("model_seed", model_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Signal {name} must be a non-negative integer seed.")
    for name, value in (
        ("training_count", training_count),
        ("sampler_seed", sampler_seed),
        ("validation_seed", validation_seed),
        ("shard_raw_bytes", shard_raw_bytes),
    ):
        if value > JAVASCRIPT_MAX_SAFE_INTEGER:
            raise ValueError(f"Signal {name} must be a JavaScript safe integer.")
    if training_count % BALANCED_TRAINING_GRANULARITY != 0:
        raise ValueError(
            "Signal training_count must be divisible by 168 for the Issue-3 balanced sampler."
        )
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("Signal learning_rate must be positive and finite.")
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal resolution must be 224 or 384.")
    if not isinstance(node_binary, str) or not node_binary:
        raise ValueError("Signal node_binary must be a non-empty string.")
    manifest_path = Path(manifest_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_directory = resolve_output_directory(
        output_directory, "Signal experiment output directory"
    )
    manifest_bytes, manifest = _read_json_bytes(manifest_path, "Issue-3 recipe manifest")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_metadata = _manifest_metadata(manifest, manifest_sha256)
    runtime_versions = _runtime_provenance(node_binary)
    if (
        runtime_versions["sharp"] != manifest_metadata["sharp_version"]
        or runtime_versions["libvips"] != manifest_metadata["libvips_version"]
    ):
        raise ValueError(
            "Issue-3 manifest Sharp/libvips versions do not match the active materialization runtime."
        )
    implementation_sha256 = _implementation_provenance()
    feature_extraction = _feature_extraction_metadata(resolution)
    _progress("building the deterministic Issue-3 experiment plan")
    with tempfile.TemporaryDirectory(
        prefix=".signal-plan-staging-",
        dir=output_directory,
    ) as staging_directory:
        staged_manifest_path = Path(staging_directory) / "track5-manifest.json"
        staged_manifest_path.write_bytes(manifest_bytes)
        plan = _build_plan(
            staged_manifest_path,
            Path(staging_directory) / "signal-plan.json",
            experiment_profile=experiment_profile,
            training_count=training_count,
            sampler_seed=sampler_seed,
            validation_source_count=validation_source_count,
            validation_seed=validation_seed,
            shard_raw_bytes=shard_raw_bytes,
            node_binary=node_binary,
        )
    if plan.get("parent_recipe_manifest_sha256") != manifest_sha256:
        raise ValueError("Signal plan is bound to a different recipe manifest.")
    plan_sha256 = plan.get("plan_sha256")
    phases = {phase.get("phase"): phase for phase in plan.get("phases", [])}
    if set(phases) != {"expert-training", "internal-validation"}:
        raise ValueError("Signal plan must contain only expert-training and internal-validation.")
    requested_run = {
        "experiment_profile": experiment_profile,
        "acceptance_scope": acceptance_scope,
        "validation_source_limit": validation_source_count,
        "validation_seed": validation_seed,
        "sampler_seed": sampler_seed,
        "model_seed": model_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "shard_raw_bytes": shard_raw_bytes,
    }
    _validate_existing_run_marker(
        managed_output_path(
            output_directory, "signal-run.json", "signal experiment run marker"
        ),
        manifest_metadata=manifest_metadata,
        plan_sha256=plan_sha256,
        requested=requested_run,
        runtime_versions=runtime_versions,
        implementation_sha256=implementation_sha256,
    )
    _publish_plan(output_directory, plan)
    training_cache = _extract_phase_features(
        phases["expert-training"],
        dataset_root=dataset_root,
        output_directory=output_directory,
        manifest_metadata=manifest_metadata,
        plan_sha256=plan_sha256,
        resolution=resolution,
        node_binary=node_binary,
        feature_extraction=feature_extraction,
    )
    validation_cache = _extract_phase_features(
        phases["internal-validation"],
        dataset_root=dataset_root,
        output_directory=output_directory,
        manifest_metadata=manifest_metadata,
        plan_sha256=plan_sha256,
        resolution=resolution,
        node_binary=node_binary,
        feature_extraction=feature_extraction,
    )
    training_records = validate_signal_feature_cache(
        training_cache,
        phases["expert-training"],
        manifest_metadata,
        expected_resolution=resolution,
        expected_feature_extraction=feature_extraction,
    )
    validation_records = validate_signal_feature_cache(
        validation_cache,
        phases["internal-validation"],
        manifest_metadata,
        expected_resolution=resolution,
        expected_feature_extraction=feature_extraction,
    )
    _write_or_verify_json(
        managed_output_path(
            output_directory,
            "signal-training-features.json",
            "signal training feature cache",
        ),
        training_cache,
        "signal training feature cache",
    )
    _write_or_verify_json(
        managed_output_path(
            output_directory,
            "signal-validation-features.json",
            "signal validation feature cache",
        ),
        validation_cache,
        "signal validation feature cache",
    )
    _progress("fitting expert-training normalization and signal weights")
    experiment_provenance = {
        "experiment_profile": experiment_profile,
        "acceptance_scope": acceptance_scope,
        "training_plan_sha256": plan_sha256,
        "training_feature_records_sha256": training_cache["records_sha256"],
        "validation_feature_records_sha256": validation_cache["records_sha256"],
        "signal_feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "resolution": resolution,
        "feature_extraction": feature_extraction,
    }
    training_provenance = {
        field: experiment_provenance[field]
        for field in (
            "experiment_profile",
            "acceptance_scope",
            "training_plan_sha256",
            "training_feature_records_sha256",
            "signal_feature_extraction_version",
            "resolution",
            "feature_extraction",
        )
    }
    normalization = fit_normalization(
        training_records,
        manifest_metadata=manifest_metadata,
        training_provenance=training_provenance,
    )
    model, model_metadata = train_signal_mlp(
        training_records,
        validation_records,
        normalization,
        manifest_metadata=manifest_metadata,
        experiment_provenance=experiment_provenance,
        seed=model_seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model_metadata.update({
        "learning_rate": learning_rate,
        "training_selection": training_cache["selection"],
        "validation_selection": validation_cache["selection"],
        "training_feature_records_sha256": training_cache["records_sha256"],
        "validation_feature_records_sha256": validation_cache["records_sha256"],
        "feature_extraction": training_cache["feature_extraction"],
    })
    model_path = managed_output_path(
        output_directory, "signal-model.json", "signal checkpoint bundle"
    )
    with tempfile.TemporaryDirectory(
        prefix=".signal-model-staging-",
        dir=output_directory,
    ) as staging_directory:
        staged_model_path = Path(staging_directory) / "signal-model.json"
        write_model_bundle(
            staged_model_path,
            model,
            model_metadata,
            normalization,
            experiment_provenance=experiment_provenance,
        )
        _, staged_bundle = read_model_bundle(
            staged_model_path,
            manifest_metadata=manifest_metadata,
            expected_experiment_provenance=experiment_provenance,
        )
        _write_or_verify_json(
            model_path,
            staged_bundle,
            "signal checkpoint bundle",
        )
    _, bundle = read_model_bundle(
        model_path,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=experiment_provenance,
    )
    normalization_revision = bundle["normalization_revision"]
    checkpoint_revision = bundle["checkpoint_revision"]
    logits = cache_signal_predictions(
        validation_records,
        bundle,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=experiment_provenance,
    )
    validate_signal_cache(
        logits,
        checkpoint_bundle=bundle,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=experiment_provenance,
        expected_feature_records=validation_records,
    )
    logit_artifact = {
        "logit_cache_schema_version": "signal-logit-cache-v1",
        "manifest_metadata": manifest_metadata,
        "feature_cache_records_sha256": validation_cache["records_sha256"],
        "checkpoint_revision": checkpoint_revision,
        "normalization_revision": normalization_revision,
        "records_sha256": _sha256(logits),
        "records": logits,
    }
    logit_path = managed_output_path(
        output_directory, "signal-validation-logits.json", "signal validation logit cache"
    )
    _write_or_verify_json(logit_path, logit_artifact, "signal validation logit cache")
    validated_logits = validate_signal_logit_cache(
        _read_json(logit_path, "signal validation logit cache"),
        expected_feature_records=validation_records,
        checkpoint_bundle=bundle,
        manifest_metadata=manifest_metadata,
        expected_experiment_provenance=experiment_provenance,
        expected_feature_cache_records_sha256=validation_cache["records_sha256"],
    )
    from rgb_baseline import evaluate_internal_validation

    metrics = evaluate_internal_validation(
        validated_logits,
        score_field="signal_logit",
        metric_schema_version="signal-robustness-metric-v1",
    )
    artifact_values = {
        "signal-normalization.json": {
            "normalization_revision": normalization_revision,
            "normalization": normalization,
        },
        "signal-internal-validation-metrics.json": metrics,
    }
    for name, value in artifact_values.items():
        _write_or_verify_json(
            managed_output_path(output_directory, name, name), value, name
        )
    storage = {
        phase_name: {
            "shard_count": len(phase["shards"]),
            "maximum_shard_raw_byte_estimate": max(
                shard["raw_byte_estimate"] for shard in phase["shards"]
            ),
            "total_raw_byte_estimate": sum(
                shard["raw_byte_estimate"] for shard in phase["shards"]
            ),
        }
        for phase_name, phase in phases.items()
    }
    run_summary = {
        "run_schema_version": "signal-experiment-run-v1",
        "experiment_profile": experiment_profile,
        "acceptance_scope": acceptance_scope,
        "manifest_metadata": manifest_metadata,
        "plan_sha256": plan_sha256,
        "training_sample_count": training_cache["selection"]["sample_count"],
        "training_unique_observation_count": len(training_records),
        "training_source_count": len({record["source_id"] for record in training_records}),
        "validation_observation_count": len(validation_records),
        "validation_source_count": len({record["source_id"] for record in validation_records}),
        "validation_source_limit": validation_source_count,
        "validation_seed": validation_seed,
        "sampler_seed": sampler_seed,
        "model_seed": model_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "shard_raw_bytes": shard_raw_bytes,
        "checkpoint_revision": checkpoint_revision,
        "normalization_revision": normalization_revision,
        "total_parameter_count": 449,
        "trainable_parameter_count": 449,
        "selected_epoch": bundle["selected_epoch"],
        "checkpoint_selection_metric": bundle["selection_metric"],
        "checkpoint_selection_metric_version": bundle["selection_metric_version"],
        "checkpoint_selection_score": bundle["selection_score"],
        "bounded_materialization": storage,
        "runtime_versions": runtime_versions,
        "implementation_sha256": implementation_sha256,
        "implementation_hash_contract_version": IMPLEMENTATION_HASH_CONTRACT_VERSION,
        "artifact_sha256": {
            name: _file_sha256(
                managed_output_path(output_directory, name, f"signal artifact {name}")
            )
            for name in FINAL_ARTIFACT_NAMES
        },
    }
    _require_current_feature_extraction_snapshot(feature_extraction, resolution)
    if _runtime_provenance(node_binary) != runtime_versions:
        raise ValueError("Signal runtime versions changed during the experiment.")
    if _implementation_provenance() != implementation_sha256:
        raise ValueError("Signal implementation changed during the experiment.")
    if _file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("Issue-3 recipe manifest changed during the signal experiment.")
    _write_or_verify_json(
        managed_output_path(
            output_directory, "signal-run.json", "signal experiment run marker"
        ),
        run_summary,
        "signal experiment run marker",
    )
    _progress("signal-only experiment artifacts validated and complete")
    return run_summary
