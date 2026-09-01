import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from unittest.mock import patch
import tempfile
import hashlib
import copy
from pathlib import Path

TRUSTED_GENERATION_REVISION = (
    "static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181"
)
TRUSTED_BUNDLE_SHA256 = "9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2"


def fixture_families(clean_auroc, noise_worst_auroc):
    return {
        "clean": {"auroc": clean_auroc, "auroc_by_severity": {"clean": clean_auroc}},
        "jpeg": {"auroc": 0.93, "auroc_by_severity": {
            "quality-30": 0.9, "quality-50": 0.92, "quality-70": 0.94, "quality-90": 0.96,
        }},
        "blur": {"auroc": 0.94, "auroc_by_severity": {
            "sigma-0.5": 0.93, "sigma-1": 0.94, "sigma-2": 0.95,
        }},
        "resize": {"auroc": 0.93, "auroc_by_severity": {
            "factor-0.25": 0.91, "factor-0.5": 0.95,
        }},
        "noise": {"auroc": 0.9, "auroc_by_severity": {
            "sigma-0.02": 0.95, "sigma-0.05": 0.9, "sigma-0.1": noise_worst_auroc,
        }},
        "color": {"auroc": 0.94, "auroc_by_severity": {
            "brightness-0.8": 0.92, "brightness-1.2": 0.93,
            "contrast-0.8": 0.94, "contrast-1.2": 0.95,
            "saturation-0.8": 0.96, "saturation-1.2": 0.97,
        }},
        "crop": {"auroc": 0.93, "auroc_by_severity": {"center-0.8": 0.93}},
    }


def completed_generation_fixture():
    candidate = {
        "metric_schema_version": "fusion-candidate-metrics-v1",
        "evaluation_split": "internal-validation",
        "clean_auroc": 0.981975,
        "corruption_families": fixture_families(0.981975, 0.810425),
        "mean_corrupted_auroc": 0.9567506944444445,
        "all_condition_macro_auroc": 0.9603541666666667,
        "worst_family_severity": {
            "family": "noise", "severity": "sigma-0.1", "auroc": 0.810425,
        },
        "degradation_drop": 0.02522430555555555,
        "degradation_retention": 0.9743126805106489,
        "threshold_diagnostics": {
            "status": "provisional-internal-validation-only",
            "selection_rule": "maximum-youden-j",
            "balanced_accuracy": 0.875,
            "threshold_logit": 0.0,
            "sensitivity": 0.625,
            "specificity": 0.75,
        },
        "brier_score": 0.1,
        "condition_balanced_brier_score": 0.09,
    }
    raw_rgb = {
        "metric_schema_version": "fusion-candidate-metrics-v1",
        "evaluation_split": "internal-validation",
        "clean_auroc": 0.97505,
        "corruption_families": fixture_families(0.97505, 0.732425),
        "mean_corrupted_auroc": 0.9383100694444444,
        "all_condition_macro_auroc": 0.9435586309523808,
        "worst_family_severity": {
            "family": "noise", "severity": "sigma-0.1", "auroc": 0.732425,
        },
        "degradation_drop": 0.03673993055555558,
        "degradation_retention": 0.9623199522531608,
        "threshold_diagnostics": {
            "status": "provisional-internal-validation-only",
            "selection_rule": "maximum-youden-j",
            "balanced_accuracy": 0.85,
            "threshold_logit": 0.0,
            "sensitivity": 0.625,
            "specificity": 0.75,
        },
        "brier_score": 0.24,
        "condition_balanced_brier_score": 0.23,
    }
    rows = [
        {
            "source_id": "sid-set:source-1", "variant_id": "variant-v1-1111111111111111111111111111111111111111111111111111111111111111", "condition_family": "clean",
            "severity": "clean", "authenticity_label": 0, "rgb_logit": 1.0,
            "rgb_calibrated_logit": 1.0, "signal_calibrated_logit": -1.0,
            "selected_fallback_logit": 1.0,
        },
        {
            "source_id": "sid-set:source-2", "variant_id": "variant-v1-2222222222222222222222222222222222222222222222222222222222222222", "condition_family": "clean",
            "severity": "clean", "authenticity_label": 1, "rgb_logit": -1.0,
            "rgb_calibrated_logit": -1.0, "signal_calibrated_logit": 1.0,
            "selected_fallback_logit": -1.0,
        },
        {
            "source_id": "sid-set:source-3", "variant_id": "variant-v1-3333333333333333333333333333333333333333333333333333333333333333", "condition_family": "noise",
            "severity": "sigma-0.1", "authenticity_label": 0, "rgb_logit": 1.0,
            "rgb_calibrated_logit": 1.0, "signal_calibrated_logit": -1.0,
            "selected_fallback_logit": 1.0,
        },
        {
            "source_id": "sid-set:source-4", "variant_id": "variant-v1-4444444444444444444444444444444444444444444444444444444444444444", "condition_family": "noise",
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
                    "calibrated-rgb-only": {
                        "threshold_diagnostics": {"threshold_logit": 0.0},
                    },
                    "calibrated-signal-only": {
                        "threshold_diagnostics": {"threshold_logit": -0.2},
                    },
                },
            }
        },
        "calibrated_internal_validation_cache": {"records": rows},
    }


class SubmissionEvidenceTests(unittest.TestCase):
    def test_cli_generation_path_checks_components_before_dotdot_normalization(self):
        from submission_evidence import normalize_cli_generation_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = root / "ordinary"
            target = root / "target"
            ordinary.mkdir()
            target.mkdir()
            inspected = []
            normalized = normalize_cli_generation_directory(
                ordinary / ".." / "target", component_validator=inspected.append,
            )
            self.assertEqual(normalized, target)
            self.assertIn(ordinary, inspected)
            self.assertIn(target, inspected)

    def test_cli_generation_path_rejects_a_symlink_component_canceled_by_dotdot(self):
        from submission_evidence import normalize_cli_generation_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            redirect = root / "redirect"
            target.mkdir()
            try:
                redirect.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Physical symlink creation is unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "redirected"):
                normalize_cli_generation_directory(redirect / ".." / "target")

    @patch("submission_evidence.publish_submission_evidence")
    def test_evidence_cli_publishes_and_rejects_missing_arguments(self, publisher):
        import submission_evidence

        publisher.return_value = {"system_id": "raw-rgb-only"}
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                submission_evidence.main([
                    "--generation-dir", "..\\frozen-generation", "--candidate", "raw-rgb-only",
                    "--expected-generation-revision", TRUSTED_GENERATION_REVISION,
                    "--expected-bundle-sha256", TRUSTED_BUNDLE_SHA256, "--output-dir", "evidence-output",
                ]),
                0,
            )
        publisher.assert_called_once_with(
            Path(os.path.abspath("..\\frozen-generation")), candidate="raw-rgb-only",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256, output_directory=Path("evidence-output"),
        )
        self.assertEqual(json.loads(output.getvalue())["system_id"], "raw-rgb-only")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            submission_evidence.main([])
        self.assertNotEqual(error.exception.code, 0)

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
        expected_raw_rgb = copy.deepcopy(
            generation["bundle"]["evaluation"]["candidates"]["raw-rgb-only"]
        )
        expected_raw_rgb["threshold_diagnostics"].pop("threshold_logit")
        expected_raw_rgb["threshold_diagnostics"].update(
            {"false_positive_rate": 0.25, "false_negative_rate": 0.375}
        )
        self.assertEqual(raw_rgb["metrics"], expected_raw_rgb)

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_persists_literal_false_rates_from_threshold_diagnostics(self, reader):
        from submission_evidence import build_submission_evidence

        reader.return_value = completed_generation_fixture()
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        threshold = evidence["metrics"]["threshold_diagnostics"]
        self.assertEqual(threshold["sensitivity"], 0.625)
        self.assertEqual(threshold["specificity"], 0.75)
        self.assertEqual(threshold["false_positive_rate"], 0.25)
        self.assertEqual(threshold["false_negative_rate"], 0.375)

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_metrics_schema_is_allowlisted_and_logit_free(self, reader):
        from submission_evidence import build_submission_evidence

        reader.return_value = completed_generation_fixture()
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        self.assertEqual(
            set(evidence["metrics"]),
            {
                "metric_schema_version", "evaluation_split", "clean_auroc",
                "corruption_families", "mean_corrupted_auroc", "all_condition_macro_auroc",
                "worst_family_severity", "degradation_drop", "degradation_retention",
                "threshold_diagnostics", "brier_score", "condition_balanced_brier_score",
            },
        )
        self.assertEqual(
            set(evidence["metrics"]["threshold_diagnostics"]),
            {
                "status", "selection_rule", "balanced_accuracy", "sensitivity", "specificity",
                "false_positive_rate", "false_negative_rate",
            },
        )
        self.assertNotIn("logit", json.dumps(evidence, sort_keys=True).lower())

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_rejects_unsafe_nested_public_metrics(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        generation["bundle"]["evaluation"]["candidates"]["learned-static-fusion"]["corruption_families"]["clean"]["image_path"] = "C:/private/image.png"
        reader.return_value = generation
        with self.assertRaisesRegex(ValueError, "corruption family"):
            build_submission_evidence(
                "frozen-generation",
                candidate="learned-static-fusion",
                expected_generation_revision=TRUSTED_GENERATION_REVISION,
                expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
            )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_rejects_noncanonical_representative_identifiers(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        generation["calibrated_internal_validation_cache"]["records"][0]["source_id"] = "../private/source"
        reader.return_value = generation
        with self.assertRaisesRegex(ValueError, "source ID"):
            build_submission_evidence(
                "frozen-generation",
                candidate="learned-static-fusion",
                expected_generation_revision=TRUSTED_GENERATION_REVISION,
                expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
            )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_uses_frozen_nonzero_expert_thresholds_for_correction_status(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        generation["bundle"]["evaluation"]["candidates"]["raw-rgb-only"]["threshold_diagnostics"]["threshold_logit"] = 0.2
        row = generation["calibrated_internal_validation_cache"]["records"][0]
        row.update(
            authenticity_label=0,
            rgb_logit=0.3,
            rgb_calibrated_logit=0.3,
            signal_calibrated_logit=-0.1,
        )
        reader.return_value = generation
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="raw-rgb-only",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        case = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
        self.assertEqual(case["expert_agreement"], "agree")
        self.assertEqual(case["correction_status"], "both-experts-wrong")

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_uses_candidate_specific_nonzero_score_fields_and_thresholds(self, reader):
        from submission_evidence import build_submission_evidence

        cases = (
            ("raw-rgb-only", -0.5, {"rgb_logit": -0.4, "selected_fallback_logit": -0.9,
                                     "rgb_calibrated_logit": -0.05, "signal_calibrated_logit": -0.2}),
            ("learned-static-fusion", 0.4, {"rgb_logit": 0.1, "selected_fallback_logit": 0.5,
                                               "rgb_calibrated_logit": -0.05, "signal_calibrated_logit": -0.2}),
        )
        for candidate, threshold, scores in cases:
            with self.subTest(candidate=candidate):
                generation = completed_generation_fixture()
                candidates = generation["bundle"]["evaluation"]["candidates"]
                candidates[candidate]["threshold_diagnostics"]["threshold_logit"] = threshold
                candidates["calibrated-rgb-only"]["threshold_diagnostics"]["threshold_logit"] = -0.1
                candidates["calibrated-signal-only"]["threshold_diagnostics"]["threshold_logit"] = -0.3
                generation["calibrated_internal_validation_cache"]["records"][0].update(
                    authenticity_label=0, **scores,
                )
                reader.return_value = generation
                evidence = build_submission_evidence(
                    "frozen-generation",
                    candidate=candidate,
                    expected_generation_revision=TRUSTED_GENERATION_REVISION,
                    expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                )
                representative = evidence["error_analysis"]["representative_cases"]["clean-false-positive"]
                self.assertEqual(representative["correction_status"], "both-experts-wrong")
                self.assertEqual(representative["expert_agreement"], "agree")

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_rejects_missing_or_nonfinite_decision_thresholds(self, reader):
        from submission_evidence import build_submission_evidence

        mutations = (
            ("raw-rgb-only", "threshold_logit", None, "threshold"),
            ("calibrated-signal-only", "threshold_logit", float("nan"), "signal decision threshold"),
            ("learned-static-fusion", "threshold_logit", float("inf"), "threshold"),
        )
        for candidate_name, field, value, message in mutations:
            with self.subTest(candidate_name=candidate_name, value=value):
                generation = completed_generation_fixture()
                diagnostics = generation["bundle"]["evaluation"]["candidates"][candidate_name]["threshold_diagnostics"]
                if value is None:
                    diagnostics.pop(field)
                else:
                    diagnostics[field] = value
                reader.return_value = generation
                with self.assertRaisesRegex(ValueError, message):
                    build_submission_evidence(
                        "frozen-generation",
                        candidate="learned-static-fusion" if candidate_name == "learned-static-fusion" else "raw-rgb-only",
                        expected_generation_revision=TRUSTED_GENERATION_REVISION,
                        expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
                    )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_rejects_nonfinite_or_out_of_range_threshold_rates(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        generation["bundle"]["evaluation"]["candidates"]["learned-static-fusion"]["threshold_diagnostics"]["sensitivity"] = float("nan")
        reader.return_value = generation
        with self.assertRaisesRegex(ValueError, "sensitivity must be finite"):
            build_submission_evidence(
                "frozen-generation",
                candidate="learned-static-fusion",
                expected_generation_revision=TRUSTED_GENERATION_REVISION,
                expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
            )
        generation = completed_generation_fixture()
        generation["bundle"]["evaluation"]["candidates"]["learned-static-fusion"]["threshold_diagnostics"]["specificity"] = 1.01
        reader.return_value = generation
        with self.assertRaisesRegex(ValueError, "specificity must be in \\[0, 1\\]"):
            build_submission_evidence(
                "frozen-generation",
                candidate="learned-static-fusion",
                expected_generation_revision=TRUSTED_GENERATION_REVISION,
                expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
            )

    @patch("submission_evidence.read_static_fallback_generation")
    def test_error_analysis_is_source_unique_and_order_independent(self, reader):
        from submission_evidence import build_from_rows, build_submission_evidence

        generation = completed_generation_fixture()
        rows = generation["calibrated_internal_validation_cache"]["records"]
        duplicate = dict(rows[0], variant_id="variant-v1-5555555555555555555555555555555555555555555555555555555555555555")
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
    def test_evidence_representative_cases_are_globally_source_unique_and_order_independent(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        rows = [dict(row) for row in generation["calibrated_internal_validation_cache"]["records"]]
        rows[2]["source_id"] = "sid-set:source-1"
        rows.append(
            dict(rows[2], source_id="sid-set:source-5", variant_id="variant-v1-6666666666666666666666666666666666666666666666666666666666666666")
        )
        generation["calibrated_internal_validation_cache"]["records"] = rows
        reader.return_value = generation
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        reader.return_value = {
            **generation,
            "calibrated_internal_validation_cache": {"records": list(reversed(rows))},
        }
        reversed_evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        cases = evidence["error_analysis"]["representative_cases"]
        selected = [case for case in cases.values() if case is not None]
        self.assertEqual(len({case["source_id"] for case in selected}), len(selected))
        self.assertEqual(cases, reversed_evidence["error_analysis"]["representative_cases"])

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_maximizes_source_unique_representative_strata(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        rows = [dict(row) for row in generation["calibrated_internal_validation_cache"]["records"]]
        rows[0].update(source_id="sid-set:x", variant_id="variant-v1-7777777777777777777777777777777777777777777777777777777777777777")
        rows[2].update(source_id="sid-set:x", variant_id="variant-v1-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        rows.append(dict(rows[0], source_id="sid-set:y", variant_id="variant-v1-5555555555555555555555555555555555555555555555555555555555555555"))
        generation["calibrated_internal_validation_cache"]["records"] = rows
        reader.return_value = generation
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        reader.return_value = {
            **generation,
            "calibrated_internal_validation_cache": {"records": list(reversed(rows))},
        }
        reversed_evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        cases = evidence["error_analysis"]["representative_cases"]
        x_rank = hashlib.sha256(json.dumps({
            "ranking_version": "submission-error-hash-rank-v1",
            "stratum": "clean-false-positive",
            "source_id": "sid-set:x",
            "variant_id": "variant-v1-7777777777777777777777777777777777777777777777777777777777777777",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        y_rank = hashlib.sha256(json.dumps({
            "ranking_version": "submission-error-hash-rank-v1",
            "stratum": "clean-false-positive",
            "source_id": "sid-set:y",
            "variant_id": "variant-v1-5555555555555555555555555555555555555555555555555555555555555555",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertLess(x_rank, y_rank)
        self.assertEqual(cases["clean-false-positive"]["source_id"], "sid-set:y")
        self.assertEqual(cases["transformed-false-positive"]["source_id"], "sid-set:x")
        self.assertTrue(all(case is not None for case in cases.values()))
        self.assertEqual(cases, reversed_evidence["error_analysis"]["representative_cases"])

    @patch("submission_evidence.read_static_fallback_generation")
    def test_evidence_emits_null_only_when_maximum_matching_forces_a_stratum_empty(self, reader):
        from submission_evidence import build_submission_evidence

        generation = completed_generation_fixture()
        rows = [dict(row) for row in generation["calibrated_internal_validation_cache"]["records"]]
        rows[0].update(source_id="sid-set:shared", variant_id="variant-v1-8888888888888888888888888888888888888888888888888888888888888888")
        rows[2].update(source_id="sid-set:shared", variant_id="variant-v1-9999999999999999999999999999999999999999999999999999999999999999")
        generation["calibrated_internal_validation_cache"]["records"] = rows
        reader.return_value = generation
        evidence = build_submission_evidence(
            "frozen-generation",
            candidate="learned-static-fusion",
            expected_generation_revision=TRUSTED_GENERATION_REVISION,
            expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
        )
        cases = evidence["error_analysis"]["representative_cases"]
        self.assertIsNotNone(cases["clean-false-positive"])
        self.assertIsNone(cases["transformed-false-positive"])

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
