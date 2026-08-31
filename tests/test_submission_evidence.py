import unittest
from unittest.mock import patch
import tempfile
import hashlib
from pathlib import Path

TRUSTED_GENERATION_REVISION = (
    "static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181"
)
TRUSTED_BUNDLE_SHA256 = "9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2"


def completed_generation_fixture():
    candidate = {
        "clean_auroc": 0.981975,
        "mean_corrupted_auroc": 0.9567506944444445,
        "all_condition_macro_auroc": 0.9603541666666667,
        "worst_family_severity": {
            "family": "noise", "severity": "sigma-0.1", "auroc": 0.810425,
        },
        "threshold_diagnostics": {"threshold_logit": 0.0},
    }
    raw_rgb = {
        "clean_auroc": 0.97505,
        "mean_corrupted_auroc": 0.9383100694444444,
        "all_condition_macro_auroc": 0.9435586309523808,
        "worst_family_severity": {
            "family": "noise", "severity": "sigma-0.1", "auroc": 0.732425,
        },
        "threshold_diagnostics": {"threshold_logit": 0.0},
    }
    rows = [
        {
            "source_id": "source-1", "variant_id": "clean-fp", "condition_family": "clean",
            "severity": "clean", "authenticity_label": 0, "rgb_logit": 1.0,
            "rgb_calibrated_logit": 1.0, "signal_calibrated_logit": -1.0,
            "selected_fallback_logit": 1.0,
        },
        {
            "source_id": "source-2", "variant_id": "clean-fn", "condition_family": "clean",
            "severity": "clean", "authenticity_label": 1, "rgb_logit": -1.0,
            "rgb_calibrated_logit": -1.0, "signal_calibrated_logit": 1.0,
            "selected_fallback_logit": -1.0,
        },
        {
            "source_id": "source-3", "variant_id": "noise-fp", "condition_family": "noise",
            "severity": "sigma-0.1", "authenticity_label": 0, "rgb_logit": 1.0,
            "rgb_calibrated_logit": 1.0, "signal_calibrated_logit": -1.0,
            "selected_fallback_logit": 1.0,
        },
        {
            "source_id": "source-4", "variant_id": "noise-fn", "condition_family": "noise",
            "severity": "sigma-0.1", "authenticity_label": 1, "rgb_logit": -1.0,
            "rgb_calibrated_logit": -1.0, "signal_calibrated_logit": 1.0,
            "selected_fallback_logit": -1.0,
        },
    ]
    return {
        "completion": {
            "generation_revision": TRUSTED_GENERATION_REVISION,
            "bundle_revision": "static-fallback-bundle-v2-fixture",
            "bundle_sha256": TRUSTED_BUNDLE_SHA256,
        },
        "bundle": {
            "evaluation": {
                "source_count": 400,
                "observation_count": 8000,
                "candidates": {
                    "learned-static-fusion": candidate,
                    "raw-rgb-only": raw_rgb,
                },
            }
        },
        "calibrated_internal_validation_cache": {"records": rows},
    }


class SubmissionEvidenceTests(unittest.TestCase):
    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_binds_generation_bundle_and_candidate(self, reader):
        from submission_evidence import build_submission_evidence

        reader.return_value = completed_generation_fixture()
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        reader.assert_called_once_with(
            "frozen-generation",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
        )
        self.assertEqual(evidence["schema_version"], "submission-evidence-v1")
        self.assertEqual(evidence["system_id"], "learned-static-fusion")
        self.assertEqual(evidence["bindings"]["bundle_sha256"], TRUSTED_BUNDLE_SHA256)
        self.assertEqual(evidence["evaluation_scope"], "internal-validation")

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_extracts_literal_candidate_metrics(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        reader.return_value = generation
        learned = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        self.assertEqual(learned["evaluation"]["source_count"], 400)
        self.assertEqual(learned["evaluation"]["observation_count"], 8000)
        self.assertEqual(learned["metrics"]["clean_auroc"], 0.981975)
        self.assertEqual(learned["metrics"]["mean_corrupted_auroc"], 0.9567506944444445)
        self.assertEqual(learned["metrics"]["all_condition_macro_auroc"], 0.9603541666666667)
        self.assertEqual(
            learned["metrics"]["worst_family_severity"],
            {"family": "noise", "severity": "sigma-0.1", "auroc": 0.810425},
        )
        raw_rgb = build_submission_evidence(
            "frozen-generation",
            candidate="raw-rgb-only",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        self.assertEqual(
            raw_rgb["metrics"],
            generation["bundle"]["evaluation"]["candidates"]["raw-rgb-only"],
        )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_error_analysis_is_source_unique_and_order_independent(self, reader):
        from submission_evidence import build_from_rows, build_submission_evidence

        generation = completed_generation_fixture()
        rows = generation["calibrated_internal_validation_cache"]["records"]
        duplicate = dict(rows[0], variant_id="clean-fp-second")
        rows.append(duplicate)
        reader.return_value = generation
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        self.assertEqual(
            set(evidence["error_analysis"]["representative_cases"]),
            {
                "clean-false-positive",
                "clean-false-negative",
                "transformed-false-positive",
                "transformed-false-negative",
            },
        )
        self.assertEqual(
            evidence["error_analysis"]["selection_rule"],
            "submission-error-hash-rank-v1",
        )
        self.assertEqual(build_from_rows(rows), build_from_rows(list(reversed(rows))))
        for value in evidence["error_analysis"]["representative_cases"].values():
            self.assertEqual(
                set(value),
                {
                    "source_id", "variant_id", "condition_family", "severity", "error_kind",
                    "expert_agreement", "correction_status", "rank",
                },
            )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_persists_internal_validation_limitations(self, reader):
        from submission_evidence import build_submission_evidence

        reader.return_value = completed_generation_fixture()
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        limitations = " ".join(evidence["limitations"]).lower()
        self.assertIn("internal validation influenced candidate and threshold selection", limitations)
        self.assertIn("not organizer, sealed, official, independent-test, or unbiased estimates", limitations)
        self.assertIn("upstream checkpoint overlap cannot be disproven", limitations)
        self.assertIn("weakest-condition/fpr/fnr discussion is descriptive, not causal", limitations)

    @patch("submission_evidence.read_static_fallback_generation")
    def test_publish_is_atomic_and_rejects_mutated_or_extra_inventory(self, reader):
        from submission_evidence import publish_submission_evidence

        reader.return_value = completed_generation_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission-evidence"
            completion = publish_submission_evidence(
                "frozen-generation",
                candidate="learned-static-fusion",
                expected_generation_revision=TRUSTED_GENERATION_REVISION,
                expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                output_directory=output,
            )
            self.assertEqual(
                {entry.name for entry in output.iterdir()},
                {"submission-evidence.json", "submission-evidence.complete.json"},
            )
            self.assertEqual(completion["generation_revision"], TRUSTED_GENERATION_REVISION)
            self.assertEqual(completion["bundle_sha256"], TRUSTED_BUNDLE_SHA256)
            self.assertEqual(completion["system_id"], "learned-static-fusion")
            self.assertEqual(
                completion["evidence_sha256"],
                hashlib.sha256((output / "submission-evidence.json").read_bytes()).hexdigest(),
            )
            self.assertTrue(completion["evidence_generation_revision"].startswith("submission-evidence-generation-v1-"))
            self.assertEqual(
                completion,
                publish_submission_evidence(
                    "frozen-generation",
                    candidate="learned-static-fusion",
                    expected_generation_revision=TRUSTED_GENERATION_REVISION,
                    expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                    output_directory=output,
                ),
            )
            (output / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                publish_submission_evidence(
                    "frozen-generation",
                    candidate="learned-static-fusion",
                    expected_generation_revision=TRUSTED_GENERATION_REVISION,
                    expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                    output_directory=output,
                )
            (output / "unexpected.txt").unlink()
            (output / "submission-evidence.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale or mismatched"):
                publish_submission_evidence(
                    "frozen-generation",
                    candidate="learned-static-fusion",
                    expected_generation_revision=TRUSTED_GENERATION_REVISION,
                    expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                    output_directory=output,
                )


if __name__ == "__main__":
    unittest.main()
