from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _artifact_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _representative_rank(stratum, source_id, variant_id):
    return hashlib.sha256(_canonical_bytes({
        "ranking_version": "submission-error-hash-rank-v1",
        "stratum": stratum,
        "source_id": source_id,
        "variant_id": variant_id,
    })).hexdigest()


def evidence_fixture() -> dict:
    families = {
        "clean": {"auroc": 0.91, "auroc_by_severity": {"clean": 0.91}},
        "jpeg": {"auroc": 0.81, "auroc_by_severity": {"quality-30": 0.81}},
        "blur": {"auroc": 0.80, "auroc_by_severity": {"sigma-2": 0.80}},
        "resize": {"auroc": 0.79, "auroc_by_severity": {"factor-0.25": 0.79}},
        "noise": {"auroc": 0.77, "auroc_by_severity": {"sigma-0.1": 0.77}},
        "color": {"auroc": 0.82, "auroc_by_severity": {"brightness-0.8": 0.82}},
        "crop": {"auroc": 0.786, "auroc_by_severity": {"center-0.8": 0.786}},
    }
    return {
        "schema_version": "submission-evidence-v1",
        "system_id": "learned-static-fusion",
        "evaluation_scope": "internal-validation",
        "bindings": {
            "generation_revision": "static-fallback-generation-v2-" + "a" * 64,
            "bundle_revision": "static-fallback-bundle-v2-" + "b" * 64,
            "bundle_sha256": "c" * 64,
        },
        "evaluation": {"source_count": 400, "observation_count": 8000},
        "metrics": {
            "metric_schema_version": "fusion-candidate-metrics-v1",
            "evaluation_split": "internal-validation",
            "clean_auroc": 0.91,
            "corruption_families": families,
            "mean_corrupted_auroc": 0.796,
            "all_condition_macro_auroc": 0.8122857142857143,
            "worst_family_severity": {"family": "noise", "severity": "sigma-0.1", "auroc": 0.77},
            "degradation_drop": 0.114,
            "degradation_retention": 0.875,
            "brier_score": 0.123456,
            "condition_balanced_brier_score": 0.125,
            "threshold_diagnostics": {
                "status": "provisional-internal-validation-only",
                "selection_rule": "maximum-youden-j",
                "balanced_accuracy": 0.75,
                "sensitivity": 0.7,
                "specificity": 0.8,
                "false_positive_rate": 0.2,
                "false_negative_rate": 0.3,
            },
        },
        "error_analysis": {
            "selection_rule": "submission-error-hash-rank-v1",
            "representative_cases": {
                "clean-false-positive": {
                    "source_id": "sid-set:source_clean_fp", "variant_id": "variant-v1-" + "1" * 64, "condition_family": "clean",
                    "severity": "clean", "error_kind": "false-positive", "expert_agreement": "agree",
                    "correction_status": "both-experts-wrong", "rank": _representative_rank(
                        "clean-false-positive", "sid-set:source_clean_fp", "variant-v1-" + "1" * 64,
                    ),
                },
                "clean-false-negative": {
                    "source_id": "sid-set:source_clean_fn", "variant_id": "variant-v1-" + "2" * 64, "condition_family": "clean",
                    "severity": "clean", "error_kind": "false-negative", "expert_agreement": "disagree",
                    "correction_status": "signal-corrects-rgb-error", "rank": _representative_rank(
                        "clean-false-negative", "sid-set:source_clean_fn", "variant-v1-" + "2" * 64,
                    ),
                },
                "transformed-false-positive": {
                    "source_id": "sid-set:source_transformed_fp", "variant_id": "variant-v1-" + "3" * 64, "condition_family": "noise",
                    "severity": "sigma-0.1", "error_kind": "false-positive", "expert_agreement": "agree",
                    "correction_status": "both-experts-wrong", "rank": _representative_rank(
                        "transformed-false-positive", "sid-set:source_transformed_fp", "variant-v1-" + "3" * 64,
                    ),
                },
                "transformed-false-negative": {
                    "source_id": "sid-set:source_transformed_fn", "variant_id": "variant-v1-" + "4" * 64, "condition_family": "noise",
                    "severity": "sigma-0.1", "error_kind": "false-negative", "expert_agreement": "disagree",
                    "correction_status": "rgb-corrects-signal-error", "rank": _representative_rank(
                        "transformed-false-negative", "sid-set:source_transformed_fn", "variant-v1-" + "4" * 64,
                    ),
                },
            },
        },
        "limitations": [
            "Internal validation influenced candidate and threshold selection.",
            "These results are not organizer, sealed, official, independent-test, or unbiased estimates.",
            "Upstream checkpoint overlap cannot be disproven.",
            "The weakest-condition/FPR/FNR discussion is descriptive, not causal.",
        ],
    }


def write_completed_evidence(directory: Path, evidence: dict) -> None:
    directory.mkdir()
    evidence_bytes = _artifact_bytes(evidence)
    completion = {
        "completion_schema_version": "submission-evidence-completion-v1",
        "generation_revision": evidence["bindings"]["generation_revision"],
        "bundle_revision": evidence["bindings"]["bundle_revision"],
        "bundle_sha256": evidence["bindings"]["bundle_sha256"],
        "system_id": evidence["system_id"],
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    completion["evidence_generation_revision"] = (
        "submission-evidence-generation-v1-" + hashlib.sha256(_canonical_bytes(completion)).hexdigest()
    )
    (directory / "submission-evidence.json").write_bytes(evidence_bytes)
    (directory / "submission-evidence.complete.json").write_bytes(_artifact_bytes(completion))


class SubmissionReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence = evidence_fixture()
        self.evidence_dir = self.root / "evidence"
        self.output_dir = self.root / "report"
        write_completed_evidence(self.evidence_dir, self.evidence)

    def tearDown(self):
        self.temporary.cleanup()

    def reset_evidence(self, evidence):
        for path in self.evidence_dir.iterdir():
            path.unlink()
        self.evidence_dir.rmdir()
        write_completed_evidence(self.evidence_dir, evidence)

    def test_report_contains_required_summary_errors_and_disclosure(self):
        from submission_report import render_submission_report

        completion = render_submission_report(self.evidence_dir, self.output_dir)
        markdown = (self.output_dir / "robustness-and-errors.md").read_text("utf-8")
        svg = ElementTree.parse(self.output_dir / "clean-vs-transformed.svg")
        self.assertIn("Clean versus transformed", markdown)
        self.assertIn("False positives", markdown)
        self.assertIn("False negatives", markdown)
        self.assertIn("internal validation", markdown)
        self.assertIn("not an official organizer score", markdown)
        self.assertIn("JPEG", markdown)
        self.assertIn("FPR", markdown)
        self.assertIn("FNR", markdown)
        self.assertEqual(svg.getroot().tag.rsplit("}", 1)[-1], "svg")
        self.assertEqual(completion["system_id"], self.evidence["system_id"])

    def test_report_accepts_trusted_namespaced_representative_identifiers(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        case = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
        case["source_id"] = "sid-set:full_synthetic_005849"
        case["variant_id"] = "variant-v1-" + "a" * 64
        case["rank"] = _representative_rank(
            "clean-false-positive", case["source_id"], case["variant_id"],
        )
        self.reset_evidence(evidence)
        render_submission_report(self.evidence_dir, self.output_dir)
        markdown = (self.output_dir / "robustness-and-errors.md").read_text("utf-8")
        self.assertIn("sid-set:full\\_synthetic\\_005849", markdown)

    def test_report_accepts_raw_rgb_only_system(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        evidence["system_id"] = "raw-rgb-only"
        self.reset_evidence(evidence)
        render_submission_report(self.evidence_dir, self.output_dir)
        self.assertIn("System: raw-rgb-only", (self.output_dir / "clean-vs-transformed.svg").read_text("utf-8"))

    def test_svg_includes_persisted_summary_and_disclosure_annotations(self):
        from submission_report import render_submission_report

        render_submission_report(self.evidence_dir, self.output_dir)
        svg = (self.output_dir / "clean-vs-transformed.svg").read_text("utf-8")
        self.assertIn("System: learned-static-fusion", svg)
        self.assertIn("Mean transformed AUROC: 0.796000", svg)
        self.assertIn("All-condition macro AUROC: 0.812286", svg)
        self.assertIn("Worst condition: noise / sigma-0.1", svg)
        for limitation in self.evidence["limitations"]:
            self.assertIn(limitation, svg)

    def test_svg_escapes_dynamic_worst_condition_severity_deterministically(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        severity = "sigma-<&>\"'"
        evidence["metrics"]["worst_family_severity"]["severity"] = severity
        self.reset_evidence(evidence)
        render_submission_report(self.evidence_dir, self.output_dir)
        first = (self.output_dir / "clean-vs-transformed.svg").read_bytes()
        render_submission_report(self.evidence_dir, self.output_dir)
        second = (self.output_dir / "clean-vs-transformed.svg").read_bytes()
        self.assertEqual(first, second)
        self.assertIn(b"sigma-&lt;&amp;&gt;&quot;&#x27;", first)
        parsed = ElementTree.parse(self.output_dir / "clean-vs-transformed.svg")
        self.assertIn(
            f"Worst condition: noise / {severity} (AUROC 0.770000)",
            [element.text for element in parsed.iter() if element.text],
        )

    def test_rejects_missing_extra_or_substituted_limitations(self):
        from submission_report import render_submission_report

        mutations = {
            "missing": lambda limitations: limitations.pop(),
            "extra": lambda limitations: limitations.append("Unexpected limitation."),
            "substituted": lambda limitations: limitations.__setitem__(0, "x" * 10_000),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                evidence = copy.deepcopy(self.evidence)
                mutate(evidence["limitations"])
                self.reset_evidence(evidence)
                with self.assertRaisesRegex(ValueError, "limitations"):
                    render_submission_report(self.evidence_dir, self.root / f"invalid-limitations-{name}")

    def test_report_cli_publishes_and_rejects_invalid_arguments(self):
        command = [sys.executable, "submission_report.py", str(self.evidence_dir), str(self.output_dir)]
        completed = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.output_dir / "submission-report.complete.json").is_file())
        self.assertEqual(json.loads(completed.stdout)["system_id"], self.evidence["system_id"])

        invalid = subprocess.run([sys.executable, "submission_report.py"], cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("usage:", invalid.stderr)

    def test_markdown_escapes_metacharacters_and_is_byte_deterministic(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        case = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
        case["source_id"] = "sid-set:source_safe"
        case["variant_id"] = "variant-v1-" + "5" * 64
        case["rank"] = _representative_rank(
            "clean-false-positive", case["source_id"], case["variant_id"],
        )
        case["severity"] = "severity-<&>\"'`|[]*_"
        self.reset_evidence(evidence)
        first = render_submission_report(self.evidence_dir, self.output_dir)
        first_bytes = {
            path.name: path.read_bytes()
            for path in self.output_dir.iterdir()
        }
        second = render_submission_report(self.evidence_dir, self.output_dir)
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, {path.name: path.read_bytes() for path in self.output_dir.iterdir()})
        markdown = first_bytes["robustness-and-errors.md"].decode("utf-8")
        self.assertIn("severity-&lt;&amp;&gt;&quot;&#x27;\\`\\|\\[\\]\\*\\_", markdown)
        self.assertNotIn("severity-<&>", markdown)

    def test_report_requires_exact_public_semantics_and_representative_ranks(self):
        from submission_report import render_submission_report

        mutations = {
            "unsupported system": lambda value: value.__setitem__("system_id", "unsupported-candidate"),
            "wrong metric schema": lambda value: value["metrics"].__setitem__("metric_schema_version", "other-v1"),
            "wrong evaluation split": lambda value: value["metrics"].__setitem__("evaluation_split", "sealed-internal-test"),
            "extra metric": lambda value: value["metrics"].__setitem__("unexpected", 1.0),
            "missing metric": lambda value: value["metrics"].pop("condition_balanced_brier_score"),
            "threshold logit": lambda value: value["metrics"]["threshold_diagnostics"].__setitem__("threshold_logit", 0.125),
            "missing threshold rate": lambda value: value["metrics"]["threshold_diagnostics"].pop("false_negative_rate"),
            "wrong threshold status": lambda value: value["metrics"]["threshold_diagnostics"].__setitem__("status", "official"),
            "wrong threshold rule": lambda value: value["metrics"]["threshold_diagnostics"].__setitem__("selection_rule", "other-rule"),
            "wrong error rule": lambda value: value["error_analysis"].__setitem__("selection_rule", "other-rule"),
            "wrong correction status": lambda value: value["error_analysis"]["representative_cases"]["clean-false-positive"].__setitem__("correction_status", "unsupported"),
            "false rank": lambda value: value["error_analysis"]["representative_cases"]["clean-false-positive"].__setitem__("rank", "0" * 64),
            "duplicate source": lambda value: value["error_analysis"]["representative_cases"]["clean-false-negative"].__setitem__("source_id", value["error_analysis"]["representative_cases"]["clean-false-positive"]["source_id"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                evidence = copy.deepcopy(self.evidence)
                mutate(evidence)
                self.reset_evidence(evidence)
                with self.assertRaises(ValueError):
                    render_submission_report(self.evidence_dir, self.root / f"invalid-semantics-{name}")

    def test_markdown_includes_verified_representative_rank(self):
        from submission_report import render_submission_report

        render_submission_report(self.evidence_dir, self.output_dir)
        markdown = (self.output_dir / "robustness-and-errors.md").read_text("utf-8")
        rank = self.evidence["error_analysis"]["representative_cases"]["clean-false-positive"]["rank"]
        self.assertIn("| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |", markdown)
        self.assertIn(rank, markdown)

    def test_report_never_publishes_the_threshold_logit(self):
        from submission_report import render_submission_report

        render_submission_report(self.evidence_dir, self.output_dir)
        outputs = b"\n".join(
            (self.output_dir / name).read_bytes()
            for name in ("robustness-and-errors.md", "clean-vs-transformed.svg", "submission-report.complete.json")
        )
        self.assertNotIn(b"threshold_logit", outputs)
        self.assertNotIn(b"0.125000", outputs)

    def test_rejects_multiline_or_non_sanitized_representative_cases(self):
        from submission_report import render_submission_report

        mutations = {
            "multiline identifier": lambda case: case.__setitem__("source_id", "source\nsecond-line"),
            "carriage-return identifier": lambda case: case.__setitem__("source_id", "source\rsecond-line"),
            "path-like identifier": lambda case: case.__setitem__("variant_id", "C:/private/image.png"),
            "drive-relative identifier": lambda case: case.__setitem__("variant_id", "C:private-report"),
            "URI-like identifier": lambda case: case.__setitem__("source_id", "file:private-report"),
            "image path": lambda case: case.__setitem__("image_path", "private/image.png"),
            "authenticity label": lambda case: case.__setitem__("authenticity_label", 1),
            "logit": lambda case: case.__setitem__("logit", 0.5),
            "probability": lambda case: case.__setitem__("probability", 0.5),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                evidence = copy.deepcopy(self.evidence)
                case = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
                mutate(case)
                self.reset_evidence(evidence)
                with self.assertRaises(ValueError):
                    render_submission_report(self.evidence_dir, self.root / f"invalid-{name}")

    def test_rejects_mutated_or_extra_evidence_and_extra_output_inventory(self):
        from submission_report import render_submission_report

        completion = render_submission_report(self.evidence_dir, self.output_dir)
        self.assertEqual(
            {entry.name for entry in self.output_dir.iterdir()},
            {"robustness-and-errors.md", "clean-vs-transformed.svg", "submission-report.complete.json"},
        )
        self.assertEqual(completion["evidence_generation_revision"], (
            json.loads((self.evidence_dir / "submission-evidence.complete.json").read_text("utf-8"))[
                "evidence_generation_revision"
            ]
        ))
        self.assertEqual(
            completion["evidence_sha256"],
            hashlib.sha256((self.evidence_dir / "submission-evidence.json").read_bytes()).hexdigest(),
        )

        (self.evidence_dir / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            render_submission_report(self.evidence_dir, self.root / "unexpected-evidence-report")
        (self.evidence_dir / "unexpected.txt").unlink()

        evidence_path = self.evidence_dir / "submission-evidence.json"
        original_evidence = evidence_path.read_bytes()
        evidence_path.write_bytes(original_evidence + b"\n")
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            render_submission_report(self.evidence_dir, self.root / "mutated-evidence-report")
        evidence_path.write_bytes(original_evidence)

        (self.output_dir / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            render_submission_report(self.evidence_dir, self.output_dir)

    def test_report_states_when_a_frozen_error_stratum_has_no_case(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        evidence["error_analysis"]["representative_cases"]["transformed-false-negative"] = None
        self.reset_evidence(evidence)
        render_submission_report(self.evidence_dir, self.output_dir)
        markdown = (self.output_dir / "robustness-and-errors.md").read_text("utf-8")
        self.assertIn("### Transformed False Negative\n\nNo case in the frozen evaluation.", markdown)
