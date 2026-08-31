"""Candidate-bound internal-validation evidence from a trusted frozen generation."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from fusion_pipeline import read_static_fallback_generation


CANDIDATE_SCORE_FIELDS = {
    "raw-rgb-only": "rgb_logit",
    "learned-static-fusion": "selected_fallback_logit",
}
ERROR_RANKING_VERSION = "submission-error-hash-rank-v1"
ERROR_STRATA = (
    "clean-false-positive",
    "clean-false-negative",
    "transformed-false-positive",
    "transformed-false-negative",
)
LIMITATIONS = (
    "Internal validation influenced candidate and threshold selection.",
    "These results are not organizer, sealed, official, independent-test, or unbiased estimates.",
    "Upstream checkpoint overlap cannot be disproven.",
    "The weakest-condition/FPR/FNR discussion is descriptive, not causal.",
)
EVIDENCE_FILENAME = "submission-evidence.json"
COMPLETION_FILENAME = "submission-evidence.complete.json"
COMPLETION_SCHEMA_VERSION = "submission-evidence-completion-v1"
EVIDENCE_GENERATION_PREFIX = "submission-evidence-generation-v1"
_AT_FDCWD_LINUX = -100
_AT_FDCWD_DARWIN = -2
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _artifact_bytes(value):
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Submission evidence must be finite JSON.") from error


def _finite_score(row, field):
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Internal-validation row requires finite {field}.")
    return float(value)


def _unit_interval_rate(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Submission evidence threshold {field} must be finite.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Submission evidence threshold {field} must be in [0, 1].")
    return result


def _metrics_with_false_rates(metrics):
    result = copy.deepcopy(metrics)
    threshold = result.get("threshold_diagnostics")
    if not isinstance(threshold, dict):
        raise ValueError("Submission evidence threshold diagnostics are missing or invalid.")
    sensitivity = _unit_interval_rate(threshold.get("sensitivity"), "sensitivity")
    specificity = _unit_interval_rate(threshold.get("specificity"), "specificity")
    threshold["false_positive_rate"] = 1.0 - specificity
    threshold["false_negative_rate"] = 1.0 - sensitivity
    return result


def _representative_case(row, *, score_field, threshold_logit):
    source_id = row.get("source_id")
    variant_id = row.get("variant_id")
    family = row.get("condition_family")
    severity = row.get("severity")
    label = row.get("authenticity_label")
    if (
        not all(isinstance(value, str) and value for value in (source_id, variant_id, family, severity))
        or type(label) is not int
        or label not in (0, 1)
    ):
        raise ValueError("Internal-validation rows have incompatible identity or label fields.")
    predicted_ai = _finite_score(row, score_field) >= threshold_logit
    if predicted_ai == bool(label):
        return None
    error_kind = "false-positive" if predicted_ai else "false-negative"
    stratum = f"{'clean' if family == 'clean' else 'transformed'}-{error_kind}"
    rank = hashlib.sha256(
        _canonical_bytes(
            {
                "ranking_version": ERROR_RANKING_VERSION,
                "stratum": stratum,
                "source_id": source_id,
                "variant_id": variant_id,
            }
        )
    ).hexdigest()
    rgb_prediction = _finite_score(row, "rgb_calibrated_logit") >= 0.0
    signal_prediction = _finite_score(row, "signal_calibrated_logit") >= 0.0
    rgb_correct = rgb_prediction == bool(label)
    signal_correct = signal_prediction == bool(label)
    if not rgb_correct and signal_correct:
        correction_status = "signal-corrects-rgb-error"
    elif rgb_correct and not signal_correct:
        correction_status = "rgb-corrects-signal-error"
    elif rgb_correct:
        correction_status = "both-experts-correct"
    else:
        correction_status = "both-experts-wrong"
    return stratum, {
        "source_id": source_id,
        "variant_id": variant_id,
        "condition_family": family,
        "severity": severity,
        "error_kind": error_kind,
        "expert_agreement": "agree" if rgb_prediction == signal_prediction else "disagree",
        "correction_status": correction_status,
        "rank": rank,
    }


def build_from_rows(rows, *, candidate="learned-static-fusion", threshold_logit=0.0):
    """Allocate deterministic, globally source-unique sanitized error representatives."""
    if candidate not in CANDIDATE_SCORE_FIELDS:
        raise ValueError("Submission candidate must be raw-rgb-only or learned-static-fusion.")
    if isinstance(threshold_logit, bool) or not isinstance(threshold_logit, (int, float)) or not math.isfinite(threshold_logit):
        raise ValueError("Submission evidence threshold must be finite.")
    by_source = {}
    for row in rows:
        result = _representative_case(
            row,
            score_field=CANDIDATE_SCORE_FIELDS[candidate],
            threshold_logit=float(threshold_logit),
        )
        if result is None:
            continue
        stratum, case = result
        key = (stratum, case["source_id"])
        if key not in by_source or case["rank"] < by_source[key]["rank"]:
            by_source[key] = case
    selected_sources = set()
    representatives = {}
    for stratum in ERROR_STRATA:
        candidates = sorted(
            (case for (case_stratum, _), case in by_source.items() if case_stratum == stratum),
            key=lambda case: case["rank"],
        )
        representative = next(
            (case for case in candidates if case["source_id"] not in selected_sources),
            None,
        )
        representatives[stratum] = representative
        if representative is not None:
            selected_sources.add(representative["source_id"])
    return representatives


def _evidence_from_validated_inputs(*, completion, bundle, candidate, metrics, rows):
    evidence_metrics = _metrics_with_false_rates(metrics)
    threshold = evidence_metrics["threshold_diagnostics"].get("threshold_logit")
    return {
        "schema_version": "submission-evidence-v1",
        "system_id": candidate,
        "evaluation_scope": "internal-validation",
        "bindings": {
            "generation_revision": completion["generation_revision"],
            "bundle_revision": completion["bundle_revision"],
            "bundle_sha256": completion["bundle_sha256"],
        },
        "evaluation": {
            "source_count": bundle["evaluation"]["source_count"],
            "observation_count": bundle["evaluation"]["observation_count"],
        },
        "metrics": evidence_metrics,
        "error_analysis": {
            "selection_rule": ERROR_RANKING_VERSION,
            "representative_cases": build_from_rows(
                rows, candidate=candidate, threshold_logit=threshold,
            ),
        },
        "limitations": list(LIMITATIONS),
    }


def build_submission_evidence(
    generation_directory,
    *,
    candidate,
    expected_generation_revision,
    expected_bundle_sha256,
):
    """Build public aggregate evidence after validating the frozen input generation."""
    if candidate not in CANDIDATE_SCORE_FIELDS:
        raise ValueError(
            "Submission candidate must be raw-rgb-only or learned-static-fusion."
        )
    generation = read_static_fallback_generation(
        generation_directory,
        expected_generation_revision=expected_generation_revision,
    )
    completion = generation["completion"]
    if completion["bundle_sha256"] != expected_bundle_sha256:
        raise ValueError("Submission evidence bundle SHA-256 is incompatible.")
    bundle = generation["bundle"]
    return _evidence_from_validated_inputs(
        completion=completion,
        bundle=bundle,
        candidate=candidate,
        metrics=bundle["evaluation"]["candidates"][candidate],
        rows=generation["calibrated_internal_validation_cache"]["records"],
    )


def _ordinary_directory(path, context):
    root = Path(path).absolute()
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{context} is missing or unreadable.") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        resolved != root
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse)
    ):
        raise ValueError(f"{context} is redirected or invalid.")
    return root


def _exact_inventory(root, expected, context):
    try:
        entries = list(root.iterdir())
    except OSError as error:
        raise ValueError(f"{context} cannot be read.") from error
    if {entry.name for entry in entries} != expected:
        raise ValueError(f"{context} inventory is incomplete or unexpected.")
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as error:
            raise ValueError(f"{context} contains an unreadable entry.") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse)
        ):
            raise ValueError(f"{context} contains a redirected or invalid entry.")


def _atomic_write(path, contents):
    descriptor = None
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _unsupported_rename(source, target):
    error_number = getattr(errno, "ENOTSUP", errno.EINVAL)
    return OSError(error_number, "Atomic no-replace directory rename is unsupported on this platform.", str(source), str(target))


def _rename_directory_no_replace(source, target):
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
        raise _unsupported_rename(source, target)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise _unsupported_rename(source, target) from error
    encoded_source = os.fsencode(source)
    encoded_target = os.fsencode(target)
    if sys.platform.startswith("linux"):
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise _unsupported_rename(source, target) from error
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        if renameat2(getattr(os, "AT_FDCWD", _AT_FDCWD_LINUX), encoded_source, getattr(os, "AT_FDCWD", _AT_FDCWD_LINUX), encoded_target, _RENAME_NOREPLACE) != 0:
            error_number = ctypes.get_errno() or errno.EIO
            raise OSError(error_number, os.strerror(error_number), str(source), str(target))
        return
    if sys.platform == "darwin":
        try:
            renamex_np = libc.renamex_np
        except AttributeError:
            renamex_np = None
        if renamex_np is not None:
            renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            renamex_np.restype = ctypes.c_int
            ctypes.set_errno(0)
            if renamex_np(encoded_source, encoded_target, _RENAME_EXCL) == 0:
                return
        raise _unsupported_rename(source, target)
    raise _unsupported_rename(source, target)


def _completion(evidence, evidence_bytes):
    bindings = evidence["bindings"]
    marker = {
        "completion_schema_version": COMPLETION_SCHEMA_VERSION,
        "generation_revision": bindings["generation_revision"],
        "bundle_revision": bindings["bundle_revision"],
        "bundle_sha256": bindings["bundle_sha256"],
        "system_id": evidence["system_id"],
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    marker["evidence_generation_revision"] = (
        f"{EVIDENCE_GENERATION_PREFIX}-{hashlib.sha256(_canonical_bytes(marker)).hexdigest()}"
    )
    return marker


def _validate_completed_output(root, evidence_bytes, expected_completion, context):
    _ordinary_directory(root, context)
    _exact_inventory(root, {EVIDENCE_FILENAME, COMPLETION_FILENAME}, context)
    try:
        persisted_evidence = (root / EVIDENCE_FILENAME).read_bytes()
        completion_bytes = (root / COMPLETION_FILENAME).read_bytes()
        completion = json.loads(completion_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} completion marker is invalid.") from error
    if persisted_evidence != evidence_bytes:
        raise ValueError(f"{context} file {EVIDENCE_FILENAME} is stale or mismatched.")
    if completion_bytes != _artifact_bytes(completion) or completion != expected_completion:
        raise ValueError(f"{context} completion marker is stale or mismatched.")


def _publish_directory(output_directory, evidence_bytes, completion):
    target = Path(output_directory).absolute()
    if target.exists() or target.is_symlink():
        _validate_completed_output(target, evidence_bytes, completion, "Submission evidence directory")
        return copy.deepcopy(completion)
    parent = _ordinary_directory(target.parent, "Submission evidence directory parent")
    if target.parent.absolute() != parent:
        raise ValueError("Submission evidence directory path is redirected or invalid.")
    staging = None
    owned_identity = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
        metadata = staging.lstat()
        owned_identity = (metadata.st_dev, metadata.st_ino)
        _atomic_write(staging / EVIDENCE_FILENAME, evidence_bytes)
        _atomic_write(staging / COMPLETION_FILENAME, _artifact_bytes(completion))
        _validate_completed_output(staging, evidence_bytes, completion, "Staged submission evidence directory")
        try:
            target.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ValueError("Submission evidence target cannot be checked before publication.") from error
        else:
            raise ValueError("Submission evidence target appeared before publication.")
        _rename_directory_no_replace(staging, target)
        _validate_completed_output(target, evidence_bytes, completion, "Submission evidence directory")
    except BaseException:
        if owned_identity is not None:
            for candidate in (staging, target):
                if candidate is None:
                    continue
                try:
                    metadata = candidate.lstat()
                except OSError:
                    continue
                if (
                    (metadata.st_dev, metadata.st_ino) == owned_identity
                    and stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                ):
                    shutil.rmtree(candidate)
                    break
        raise
    return copy.deepcopy(completion)


def publish_submission_evidence(
    generation_directory,
    *,
    candidate,
    expected_generation_revision,
    expected_bundle_sha256,
    output_directory,
):
    """Atomically publish exact candidate-bound public evidence and its receipt."""
    evidence = build_submission_evidence(
        generation_directory,
        candidate=candidate,
        expected_generation_revision=expected_generation_revision,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    evidence_bytes = _artifact_bytes(evidence)
    return _publish_directory(output_directory, evidence_bytes, _completion(evidence, evidence_bytes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATE_SCORE_FIELDS))
    parser.add_argument("--expected-generation-revision", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    generation_directory = Path(os.path.abspath(arguments.generation_dir))
    try:
        completion = publish_submission_evidence(
            generation_directory,
            candidate=arguments.candidate,
            expected_generation_revision=arguments.expected_generation_revision,
            expected_bundle_sha256=arguments.expected_bundle_sha256,
            output_directory=arguments.output_dir,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
