from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _artifact_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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
            "brier_score": 0.123456,
            "threshold_diagnostics": {
                "status": "provisional-internal-validation-only",
                "selection_rule": "maximum-youden-j",
                "threshold_logit": 0.125,
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
                    "source_id": "source-clean-fp", "variant_id": "clean-fp", "condition_family": "clean",
                    "severity": "clean", "error_kind": "false-positive", "expert_agreement": "agree",
                    "correction_status": "both-experts-wrong", "rank": "1" * 64,
                },
                "clean-false-negative": {
                    "source_id": "source-clean-fn", "variant_id": "clean-fn", "condition_family": "clean",
                    "severity": "clean", "error_kind": "false-negative", "expert_agreement": "disagree",
                    "correction_status": "signal-corrects-rgb-error", "rank": "2" * 64,
                },
                "transformed-false-positive": {
                    "source_id": "source-transformed-fp", "variant_id": "noise-fp", "condition_family": "noise",
                    "severity": "sigma-0.1", "error_kind": "false-positive", "expert_agreement": "agree",
                    "correction_status": "both-experts-wrong", "rank": "3" * 64,
                },
                "transformed-false-negative": {
                    "source_id": "source-transformed-fn", "variant_id": "noise-fn", "condition_family": "noise",
                    "severity": "sigma-0.1", "error_kind": "false-negative", "expert_agreement": "disagree",
                    "correction_status": "rgb-corrects-signal-error", "rank": "4" * 64,
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

    def test_markdown_escapes_metacharacters_and_is_byte_deterministic(self):
        from submission_report import render_submission_report

        evidence = copy.deepcopy(self.evidence)
        case = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
        case["source_id"] = "source-<&>\"'`|[]*_"
        case["variant_id"] = "variant-<&>\"'`|[]*_"
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
        self.assertIn("source-&lt;&amp;&gt;&quot;&#x27;\\`\\|\\[\\]\\*\\_", markdown)
        self.assertNotIn("source-<&>", markdown)

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
