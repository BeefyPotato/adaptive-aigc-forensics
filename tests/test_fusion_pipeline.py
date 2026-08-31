import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

from fusion_pipeline import (
    SELECTION_RULE, calibrated_logit, evaluate_candidates, fit_platt_calibrator,
    fit_static_weight, inspect_signal_handoff_archive, matched_record, publish_bundle_atomic, validate_bundle,
    validate_matched_records,
)


PROVENANCE = {
    "manifest_sha256": "a" * 64, "rgb_checkpoint_revision": "b" * 40,
    "rgb_preprocessing_version": "rgb-v1", "signal_checkpoint_revision": "signal-model-v1",
    "signal_normalization_revision": "signal-normalization-v1",
    "signal_feature_extraction_version": "signal-feature-v1", "corruption_version": "corruption-v1",
}


def records(split, sources=20, signal_help=False):
    rows = []
    conditions = [("clean", "none"), ("jpeg", "q"), ("blur", "s"), ("resize", "r"), ("noise", "n"), ("color", "c"), ("crop", "x")]
    for source in range(sources):
        label = source % 2
        for family, severity in conditions:
            variant = f"{split}-{source}-{family}"
            base = 2.0 if label else -2.0
            if signal_help and family == "noise" and source % 5 == 0:
                rgb_logit, signal_logit = -base, base * 2
            else:
                rgb_logit, signal_logit = base, base * .5
            common = {"source_id": f"{split}-{source}", "variant_id": variant, "split": split,
                      "authenticity_label": label, "condition_family": family, "severity": severity,
                      "materialized_sha256": f"{source:064x}"[-64:]}
            rows.append(matched_record({**common, "rgb_logit": rgb_logit}, {**common, "signal_logit": signal_logit}, PROVENANCE))
    return rows


class FusionPipelineTests(unittest.TestCase):
    def test_archive_handoff_validation_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape", b"x")
            with self.assertRaisesRegex(ValueError, "traversal"):
                inspect_signal_handoff_archive(archive)

    def test_partition_enforcement_and_membership(self):
        training = records("fusion-training")
        self.assertEqual(len(validate_matched_records(training, expected_split="fusion-training")), 140)
        bad = copy.deepcopy(training); bad[0]["split"] = "internal-validation"
        with self.assertRaisesRegex(ValueError, "only fusion-training"):
            validate_matched_records(bad, expected_split="fusion-training")
        bad = copy.deepcopy(training); bad[1]["variant_id"] = bad[0]["variant_id"]
        with self.assertRaisesRegex(ValueError, "repeat variant_id"):
            validate_matched_records(bad, expected_split="fusion-training")

    def test_calibrators_and_weight_are_reproducible_and_constrained(self):
        training = records("fusion-training")
        rgb1 = fit_platt_calibrator(training, expert="rgb")
        rgb2 = fit_platt_calibrator(training, expert="rgb")
        signal = fit_platt_calibrator(training, expert="signal")
        self.assertEqual(rgb1, rgb2)
        weight1 = fit_static_weight(training, rgb1, signal)
        weight2 = fit_static_weight(training, rgb1, signal)
        self.assertEqual(weight1, weight2)
        self.assertGreaterEqual(weight1["rgb_weight"], 0); self.assertLessEqual(weight1["rgb_weight"], 1)
        self.assertAlmostEqual(weight1["rgb_weight"] + weight1["signal_weight"], 1)

    def test_equal_fusion_brier_complementarity_and_rgb_fallback(self):
        training = records("fusion-training")
        rgb = fit_platt_calibrator(training, expert="rgb")
        signal = fit_platt_calibrator(training, expert="signal")
        weight = fit_static_weight(training, rgb, signal)
        evaluation = evaluate_candidates(records("internal-validation"), rgb, signal, weight)
        self.assertIn("brier_score", evaluation["candidates"]["equal-50-50-calibrated-logit-fusion"])
        self.assertEqual(evaluation["selected_fallback_type"], "calibrated-rgb-only")
        self.assertEqual(evaluation["selection_rule"], SELECTION_RULE)
        first = records("internal-validation", sources=2)[0]
        expected = .5 * calibrated_logit(first["rgb_logit"], rgb) + .5 * calibrated_logit(first["signal_logit"], signal)
        self.assertIsInstance(expected, float)

    def test_stale_forbidden_and_nonfinite_records_fail(self):
        training = records("fusion-training")
        for field, value, message in (("split", "sealed-internal-test", "only fusion-training"), ("rgb_logit", float("nan"), "finite"), ("rgb_checkpoint_revision", "stale", "cache key")):
            bad = copy.deepcopy(training); bad[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_matched_records(bad, expected_split="fusion-training")

    def test_atomic_completion_is_written_last_and_bundle_rejects_traversal(self):
        training = records("fusion-training"); validation = records("internal-validation")
        rgb = fit_platt_calibrator(training, expert="rgb"); signal = fit_platt_calibrator(training, expert="signal")
        weight = fit_static_weight(training, rgb, signal); evaluation = evaluate_candidates(validation, rgb, signal, weight)
        bundle = {"selected_fallback_type": evaluation["selected_fallback_type"], "rgb_calibrator": rgb,
                  "signal_calibrator": signal, "static_weight": weight, "evaluation": evaluation,
                  "provenance": PROVENANCE, "input_cache_bindings": {}, "numeric_tolerances": {}, "artifact_paths": []}
        with tempfile.TemporaryDirectory() as directory:
            completion = publish_bundle_atomic(Path(directory), bundle)
            self.assertTrue((Path(directory) / "static-fallback-bundle.json").is_file())
            self.assertTrue((Path(directory) / "static-fallback.complete.json").is_file())
            published = __import__("json").loads((Path(directory) / "static-fallback-bundle.json").read_text())
            validate_bundle(published)
            published["artifact_paths"] = ["../escape"]
            with self.assertRaisesRegex(ValueError, "traversal"):
                validate_bundle(published)


if __name__ == "__main__":
    unittest.main()
