"""Calibration and static-fusion contracts for frozen Track 5 experts."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

from rgb_baseline import evaluate_internal_validation


CALIBRATION_SCHEMA = "expert-calibration-v1"
MATCHED_CACHE_SCHEMA = "matched-frozen-expert-logits-v1"
BUNDLE_SCHEMA = "static-fallback-bundle-v1"
COMPLETION_SCHEMA = "static-fallback-completion-v1"
FAMILIES = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
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


def validate_bundle(bundle: dict) -> dict:
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
