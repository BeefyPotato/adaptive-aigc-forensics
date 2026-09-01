"""Deterministic Section 5.5 rendering from completed submission evidence."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import html
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


EVIDENCE_FILENAME = "submission-evidence.json"
EVIDENCE_COMPLETION_FILENAME = "submission-evidence.complete.json"
REPORT_COMPLETION_FILENAME = "submission-report.complete.json"
REPORT_COMPLETION_SCHEMA_VERSION = "submission-report-completion-v1"
REPORT_GENERATION_PREFIX = "submission-report-generation-v1"
FAMILIES = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
SYSTEM_IDS = ("raw-rgb-only", "learned-static-fusion")
METRIC_FIELDS = {
    "metric_schema_version", "evaluation_split", "clean_auroc", "corruption_families",
    "mean_corrupted_auroc", "all_condition_macro_auroc", "worst_family_severity",
    "degradation_drop", "degradation_retention", "threshold_diagnostics", "brier_score",
    "condition_balanced_brier_score",
}
THRESHOLD_FIELDS = {
    "status", "selection_rule", "balanced_accuracy", "sensitivity", "specificity",
    "false_positive_rate", "false_negative_rate",
}
ERROR_RANKING_VERSION = "submission-error-hash-rank-v1"
THRESHOLD_STATUS = "provisional-internal-validation-only"
THRESHOLD_SELECTION_RULE = "maximum-youden-j"
ERROR_STRATA = (
    "clean-false-positive",
    "clean-false-negative",
    "transformed-false-positive",
    "transformed-false-negative",
)
CANONICAL_LIMITATIONS = (
    "Internal validation influenced candidate and threshold selection.",
    "These results are not organizer, sealed, official, independent-test, or unbiased estimates.",
    "Upstream checkpoint overlap cannot be disproven.",
    "The weakest-condition/FPR/FNR discussion is descriptive, not causal.",
)
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
        raise ValueError("Submission report artifacts must be finite JSON.") from error


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


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


def _text(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string.")
    return value


def _positive_count(value, context):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer.")
    return value


def _finite(value, context, *, unit_interval=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number.")
    result = float(value)
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be in [0, 1].")
    return result


def _metric(value, context):
    return f"{_finite(value, context, unit_interval=True):.6f}"


def _single_line_text(value, context):
    text = _text(value, context)
    if "\r" in text or "\n" in text:
        raise ValueError(f"{context} must be single-line text.")
    return text


def _markdown_text(value, context):
    text = html.escape(_single_line_text(value, context), quote=True)
    for character in "\\`|[]*_~":
        text = text.replace(character, f"\\{character}")
    return text


def _safe_body(value):
    return bool(value) and all(
        "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or character in "-_"
        for character in value
    )


def _source_identifier(value, context):
    identifier = _single_line_text(value, context)
    prefix = "sid-set:"
    if not identifier.startswith(prefix) or not _safe_body(identifier[len(prefix):]):
        raise ValueError(f"{context} must be a sid-set opaque ASCII identifier.")
    return identifier


def _variant_identifier(value, context):
    identifier = _single_line_text(value, context)
    prefix = "variant-v1-"
    digest = identifier.removeprefix(prefix)
    if not identifier.startswith(prefix) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} must be a variant-v1 SHA-256 identifier.")
    return identifier


def _svg_text(value, context):
    return html.escape(_single_line_text(value, context), quote=True)


def _read_completed_evidence(evidence_directory):
    root = _ordinary_directory(evidence_directory, "Submission evidence directory")
    _exact_inventory(root, {EVIDENCE_FILENAME, EVIDENCE_COMPLETION_FILENAME}, "Submission evidence directory")
    try:
        evidence_bytes = (root / EVIDENCE_FILENAME).read_bytes()
        completion_bytes = (root / EVIDENCE_COMPLETION_FILENAME).read_bytes()
        evidence = json.loads(evidence_bytes)
        completion = json.loads(completion_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Submission evidence directory is invalid JSON.") from error
    expected_completion_fields = {
        "completion_schema_version", "generation_revision", "bundle_revision", "bundle_sha256",
        "system_id", "evidence_sha256", "evidence_generation_revision",
    }
    if not isinstance(completion, dict) or set(completion) != expected_completion_fields:
        raise ValueError("Submission evidence completion marker is incomplete or unexpected.")
    if completion_bytes != _artifact_bytes(completion):
        raise ValueError("Submission evidence completion marker is stale or mismatched.")
    if completion["completion_schema_version"] != "submission-evidence-completion-v1":
        raise ValueError("Submission evidence completion marker has an unsupported schema.")
    for field in ("generation_revision", "bundle_revision", "bundle_sha256", "system_id", "evidence_sha256"):
        _text(completion[field], f"Submission evidence completion {field}")
    if completion["evidence_sha256"] != _sha256(evidence_bytes):
        raise ValueError("Submission evidence file is stale or mismatched.")
    revision_source = {key: completion[key] for key in expected_completion_fields - {"evidence_generation_revision"}}
    expected_revision = f"submission-evidence-generation-v1-{_sha256(_canonical_bytes(revision_source))}"
    if completion["evidence_generation_revision"] != expected_revision:
        raise ValueError("Submission evidence completion marker is stale or mismatched.")
    return _validated_evidence(evidence, completion), evidence_bytes, completion


def _validated_evidence(evidence, completion):
    top_fields = {
        "schema_version", "system_id", "evaluation_scope", "bindings", "evaluation", "metrics",
        "error_analysis", "limitations",
    }
    if not isinstance(evidence, dict) or set(evidence) != top_fields:
        raise ValueError("Submission evidence is incomplete or unexpected.")
    if evidence["schema_version"] != "submission-evidence-v1" or evidence["evaluation_scope"] != "internal-validation":
        raise ValueError("Submission evidence schema or evaluation scope is incompatible.")
    system_id = _text(evidence["system_id"], "Submission evidence system ID")
    if system_id not in SYSTEM_IDS:
        raise ValueError("Submission evidence system ID is incompatible.")
    if system_id != completion["system_id"]:
        raise ValueError("Submission evidence system ID is stale or mismatched.")
    bindings = evidence["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {"generation_revision", "bundle_revision", "bundle_sha256"}:
        raise ValueError("Submission evidence bindings are incomplete or unexpected.")
    for field, value in bindings.items():
        if _text(value, f"Submission evidence binding {field}") != completion[field]:
            raise ValueError("Submission evidence bindings are stale or mismatched.")
    evaluation = evidence["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {"source_count", "observation_count"}:
        raise ValueError("Submission evidence evaluation is incomplete or unexpected.")
    for field, value in evaluation.items():
        _positive_count(value, f"Submission evidence evaluation {field}")
    metrics = _validated_metrics(evidence["metrics"])
    errors = _validated_errors(evidence["error_analysis"])
    limitations = evidence["limitations"]
    if limitations != list(CANONICAL_LIMITATIONS):
        raise ValueError("Submission evidence limitations are incomplete or incompatible.")
    return {"system_id": system_id, "metrics": metrics, "errors": errors, "limitations": limitations}


def _validated_metrics(metrics):
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        raise ValueError("Submission evidence metrics are invalid.")
    if metrics["metric_schema_version"] != "fusion-candidate-metrics-v1":
        raise ValueError("Submission evidence metric schema is incompatible.")
    if metrics["evaluation_split"] != "internal-validation":
        raise ValueError("Submission evidence metric evaluation split is incompatible.")
    for field in ("clean_auroc", "mean_corrupted_auroc", "all_condition_macro_auroc", "brier_score"):
        _finite(metrics.get(field), f"Submission evidence {field}", unit_interval=True)
    _finite(metrics["degradation_drop"], "Submission evidence degradation drop")
    _finite(metrics["degradation_retention"], "Submission evidence degradation retention", unit_interval=True)
    _finite(metrics["condition_balanced_brier_score"], "Submission evidence condition-balanced Brier score", unit_interval=True)
    families = metrics.get("corruption_families")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise ValueError("Submission evidence corruption families are incomplete or unexpected.")
    for family in FAMILIES:
        entry = families[family]
        if not isinstance(entry, dict) or set(entry) != {"auroc", "auroc_by_severity"}:
            raise ValueError("Submission evidence corruption family is invalid.")
        _finite(entry["auroc"], f"Submission evidence {family} AUROC", unit_interval=True)
        by_severity = entry["auroc_by_severity"]
        if not isinstance(by_severity, dict) or not by_severity:
            raise ValueError("Submission evidence corruption severities are invalid.")
        for severity, value in by_severity.items():
            _text(severity, "Submission evidence severity")
            _finite(value, "Submission evidence severity AUROC", unit_interval=True)
    worst = metrics.get("worst_family_severity")
    if not isinstance(worst, dict) or set(worst) != {"family", "severity", "auroc"}:
        raise ValueError("Submission evidence worst condition is invalid.")
    if worst["family"] not in FAMILIES or worst["family"] == "clean":
        raise ValueError("Submission evidence worst condition family is invalid.")
    _single_line_text(worst["severity"], "Submission evidence worst condition severity")
    _finite(worst["auroc"], "Submission evidence worst condition AUROC", unit_interval=True)
    threshold = metrics.get("threshold_diagnostics")
    if not isinstance(threshold, dict) or set(threshold) != THRESHOLD_FIELDS:
        raise ValueError("Submission evidence threshold diagnostics are incomplete.")
    if threshold["status"] != THRESHOLD_STATUS:
        raise ValueError("Submission evidence threshold status is incompatible.")
    if threshold["selection_rule"] != THRESHOLD_SELECTION_RULE:
        raise ValueError("Submission evidence threshold selection rule is incompatible.")
    for field in ("balanced_accuracy", "sensitivity", "specificity", "false_positive_rate", "false_negative_rate"):
        _finite(threshold[field], f"Submission evidence {field}", unit_interval=True)
    return metrics


def _validated_errors(error_analysis):
    if not isinstance(error_analysis, dict) or set(error_analysis) != {"selection_rule", "representative_cases"}:
        raise ValueError("Submission evidence error analysis is incomplete or unexpected.")
    if error_analysis["selection_rule"] != ERROR_RANKING_VERSION:
        raise ValueError("Submission evidence error selection rule is incompatible.")
    cases = error_analysis["representative_cases"]
    if not isinstance(cases, dict) or set(cases) != set(ERROR_STRATA):
        raise ValueError("Submission evidence error cases are incomplete or unexpected.")
    required_case = {"source_id", "variant_id", "condition_family", "severity", "error_kind", "expert_agreement", "correction_status", "rank"}
    correction_statuses = {
        "signal-corrects-rgb-error", "rgb-corrects-signal-error",
        "both-experts-correct", "both-experts-wrong",
    }
    seen_sources = set()
    for stratum, case in cases.items():
        if case is None:
            continue
        if not isinstance(case, dict) or set(case) != required_case:
            raise ValueError("Submission evidence representative case is incomplete or unexpected.")
        source_id = _source_identifier(case["source_id"], "Submission evidence representative source ID")
        variant_id = _variant_identifier(case["variant_id"], "Submission evidence representative variant ID")
        if source_id in seen_sources:
            raise ValueError("Submission evidence representative source IDs must be globally unique.")
        seen_sources.add(source_id)
        if case["condition_family"] not in FAMILIES:
            raise ValueError("Submission evidence representative condition family is invalid.")
        _single_line_text(case["severity"], "Submission evidence representative severity")
        expected_error_kind = "false-positive" if stratum.endswith("false-positive") else "false-negative"
        if case["error_kind"] != expected_error_kind:
            raise ValueError("Submission evidence representative error kind is invalid.")
        if (stratum.startswith("clean-") and case["condition_family"] != "clean") or (
            stratum.startswith("transformed-") and case["condition_family"] == "clean"
        ):
            raise ValueError("Submission evidence representative stratum is invalid.")
        if case["expert_agreement"] not in {"agree", "disagree"}:
            raise ValueError("Submission evidence representative expert agreement is invalid.")
        if case["correction_status"] not in correction_statuses:
            raise ValueError("Submission evidence representative correction status is invalid.")
        rank = _text(case["rank"], "Submission evidence representative rank")
        if len(rank) != 64 or any(character not in "0123456789abcdef" for character in rank):
            raise ValueError("Submission evidence representative rank is invalid.")
        expected_rank = _sha256(_canonical_bytes({
            "ranking_version": ERROR_RANKING_VERSION,
            "stratum": stratum,
            "source_id": source_id,
            "variant_id": variant_id,
        }))
        if rank != expected_rank:
            raise ValueError("Submission evidence representative rank is incompatible.")
    return cases


def _markdown_report(data):
    metrics = data["metrics"]
    threshold = metrics["threshold_diagnostics"]
    lines = [
        "# Robustness and errors", "",
        "## Clean versus transformed", "",
        f"System: {_markdown_text(data['system_id'], 'Submission evidence system ID')}. This internal validation report is not an official organizer score.", "",
        "| Condition | AUROC |", "| --- | ---: |",
    ]
    labels = {"clean": "Clean", "jpeg": "JPEG", "blur": "Blur", "resize": "Resize", "noise": "Noise", "color": "Color", "crop": "Crop"}
    for family in FAMILIES:
        lines.append(f"| {labels[family]} | {_metric(metrics['corruption_families'][family]['auroc'], family + ' AUROC')} |")
    lines += [
        f"| Mean transformed | {_metric(metrics['mean_corrupted_auroc'], 'Mean transformed AUROC')} |", "",
        "## Persisted summary", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| All-condition macro AUROC | {_metric(metrics['all_condition_macro_auroc'], 'Macro AUROC')} |",
        f"| Brier score | {_metric(metrics['brier_score'], 'Brier score')} |",
        f"| Balanced accuracy | {_metric(threshold['balanced_accuracy'], 'Balanced accuracy')} |",
        f"| FPR | {_metric(threshold['false_positive_rate'], 'FPR')} |",
        f"| FNR | {_metric(threshold['false_negative_rate'], 'FNR')} |", "",
        "## Threshold trade-offs", "",
        f"The weakest persisted family/severity is {_markdown_text(metrics['worst_family_severity']['family'], 'Submission evidence worst condition family')} / {_markdown_text(metrics['worst_family_severity']['severity'], 'Submission evidence worst condition severity')} (AUROC {_metric(metrics['worst_family_severity']['auroc'], 'Worst-condition AUROC')}). The persisted threshold trade-off has balanced accuracy {_metric(threshold['balanced_accuracy'], 'Balanced accuracy')}, FPR {_metric(threshold['false_positive_rate'], 'FPR')}, and FNR {_metric(threshold['false_negative_rate'], 'FNR')}. This discussion is descriptive, not causal.", "",
        "## Error strata", "",
        "**False positives** and **False negatives** below are deterministic sanitized representatives from the frozen evaluation; they do not establish visual causes.", "",
    ]
    for stratum in ERROR_STRATA:
        title = stratum.replace("-", " ").title()
        lines += [f"### {title}", ""]
        case = data["errors"][stratum]
        if case is None:
            lines += ["No case in the frozen evaluation.", ""]
            continue
        lines += [
            "| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "| " + " | ".join(_markdown_text(case[field], f"Submission evidence representative case {field}") for field in (
                "source_id", "variant_id", "condition_family", "severity", "error_kind", "expert_agreement", "correction_status", "rank"
            )) + " |", "",
        ]
    lines += ["## Limitations", ""]
    lines.extend(f"- {_markdown_text(value, 'Submission evidence limitation')}" for value in data["limitations"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _svg(data):
    families = data["metrics"]["corruption_families"]
    metrics = data["metrics"]
    worst = metrics["worst_family_severity"]
    bars = []
    for index, family in enumerate(FAMILIES):
        value = _finite(families[family]["auroc"], f"{family} AUROC", unit_interval=True)
        y = 145 + index * 38
        bars.extend((
            f'<text x="160" y="{y + 18}" text-anchor="end" font-family="sans-serif" font-size="14">{family}</text>',
            f'<rect x="170" y="{y}" width="{value * 600:.3f}" height="24" fill="#2563eb"/>',
            f'<text x="780" y="{y + 18}" font-family="sans-serif" font-size="13">{value:.6f}</text>',
        ))
    annotations = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560" role="img">',
        '<title>Clean versus transformed AUROC</title>',
        '<desc>Persisted internal-validation robustness values and limitations.</desc>',
        '<rect width="960" height="560" fill="#fff"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">Clean versus transformed AUROC</text>',
        f'<text x="20" y="56" font-family="sans-serif" font-size="14">System: {_svg_text(data["system_id"], "Submission evidence system ID")}</text>',
        f'<text x="20" y="80" font-family="sans-serif" font-size="14">Mean transformed AUROC: {_metric(metrics["mean_corrupted_auroc"], "Mean transformed AUROC")}</text>',
        f'<text x="20" y="104" font-family="sans-serif" font-size="14">All-condition macro AUROC: {_metric(metrics["all_condition_macro_auroc"], "Macro AUROC")}</text>',
        f'<text x="20" y="128" font-family="sans-serif" font-size="14">Worst condition: {_svg_text(worst["family"], "Submission evidence worst condition family")} / {_svg_text(worst["severity"], "Submission evidence worst condition severity")} (AUROC {_metric(worst["auroc"], "Worst-condition AUROC")})</text>',
        '<line x1="170" y1="430" x2="770" y2="430" stroke="#243746"/>',
        *bars,
        '<text x="20" y="458" font-family="sans-serif" font-size="14" font-weight="bold">Limitations</text>',
        *(f'<text x="20" y="{482 + index * 22}" font-family="sans-serif" font-size="11">{_svg_text(limitation, "Submission evidence limitation")}</text>' for index, limitation in enumerate(data["limitations"])),
        '</svg>',
    ]
    return ("\n".join(annotations) + "\n").encode("utf-8")


def _atomic_write(path, contents):
    descriptor = None
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _rename_no_replace(source, target):
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
        raise OSError(getattr(errno, "ENOTSUP", errno.EINVAL), "Atomic no-replace directory rename is unsupported.")
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise OSError(getattr(errno, "ENOTSUP", errno.EINVAL), "Atomic no-replace directory rename is unsupported.") from error
    if sys.platform.startswith("linux"):
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(getattr(errno, "ENOTSUP", errno.EINVAL), "Atomic no-replace directory rename is unsupported.") from error
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        if renameat2(_AT_FDCWD_LINUX, os.fsencode(source), _AT_FDCWD_LINUX, os.fsencode(target), _RENAME_NOREPLACE) != 0:
            number = ctypes.get_errno() or errno.EIO
            raise OSError(number, os.strerror(number), str(source), str(target))
        return
    if sys.platform == "darwin":
        try:
            renamex_np = library.renamex_np
        except AttributeError:
            renamex_np = None
        if renamex_np is not None:
            renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            renamex_np.restype = ctypes.c_int
            if renamex_np(os.fsencode(source), os.fsencode(target), _RENAME_EXCL) == 0:
                return
        raise OSError(getattr(errno, "ENOTSUP", errno.EINVAL), "Atomic no-replace directory rename is unsupported.")
    raise OSError(getattr(errno, "ENOTSUP", errno.EINVAL), "Atomic no-replace directory rename is unsupported.")


def _report_completion(evidence_bytes, evidence_completion, system_id, files):
    marker = {
        "completion_schema_version": REPORT_COMPLETION_SCHEMA_VERSION,
        "evidence_generation_revision": evidence_completion["evidence_generation_revision"],
        "evidence_sha256": _sha256(evidence_bytes),
        "system_id": system_id,
        "files": {name: _sha256(files[name]) for name in sorted(files)},
    }
    marker["report_generation_revision"] = f"{REPORT_GENERATION_PREFIX}-{_sha256(_canonical_bytes(marker))}"
    return marker


def _validate_output(root, files, completion):
    _ordinary_directory(root, "Submission report directory")
    _exact_inventory(root, set(files) | {REPORT_COMPLETION_FILENAME}, "Submission report directory")
    for name, contents in files.items():
        if (root / name).read_bytes() != contents:
            raise ValueError(f"Submission report directory file {name} is stale or mismatched.")
    try:
        completion_bytes = (root / REPORT_COMPLETION_FILENAME).read_bytes()
        received = json.loads(completion_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Submission report completion marker is invalid.") from error
    if completion_bytes != _artifact_bytes(received) or received != completion:
        raise ValueError("Submission report completion marker is stale or mismatched.")


def _publish(output_directory, files, completion):
    target = Path(output_directory).absolute()
    if target.exists() or target.is_symlink():
        _validate_output(target, files, completion)
        return copy.deepcopy(completion)
    parent = _ordinary_directory(target.parent, "Submission report directory parent")
    if target.parent.absolute() != parent:
        raise ValueError("Submission report directory path is redirected or invalid.")
    staging = None
    identity = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
        metadata = staging.lstat()
        identity = (metadata.st_dev, metadata.st_ino)
        for name in sorted(files):
            _atomic_write(staging / name, files[name])
        _atomic_write(staging / REPORT_COMPLETION_FILENAME, _artifact_bytes(completion))
        _validate_output(staging, files, completion)
        if target.exists() or target.is_symlink():
            raise ValueError("Submission report target appeared before publication.")
        _rename_no_replace(staging, target)
        _validate_output(target, files, completion)
    except BaseException:
        for candidate in (staging, target):
            if candidate is None:
                continue
            try:
                metadata = candidate.lstat()
            except OSError:
                continue
            if (metadata.st_dev, metadata.st_ino) == identity and stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(candidate)
                break
        raise
    return copy.deepcopy(completion)


def render_submission_report(evidence_directory, output_directory):
    """Render a completed evidence receipt without recomputing any metrics."""
    data, evidence_bytes, evidence_completion = _read_completed_evidence(evidence_directory)
    files = {
        "robustness-and-errors.md": _markdown_report(data),
        "clean-vs-transformed.svg": _svg(data),
    }
    completion = _report_completion(evidence_bytes, evidence_completion, data["system_id"], files)
    return _publish(output_directory, files, completion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args(argv)
    try:
        completion = render_submission_report(arguments.evidence_directory, arguments.output_directory)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
