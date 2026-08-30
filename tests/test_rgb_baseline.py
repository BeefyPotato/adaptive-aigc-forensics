import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgb_baseline import (
    compare_deterministic_subset,
    evaluate_internal_validation,
    run_rgb_baseline,
    validate_rgb_cache,
)


class MeanBackend:
    device = "cpu"
    peak_memory_bytes = None

    def predict_logits(self, batch):
        return batch.mean(axis=(1, 2, 3))


def manifest_for(image_name):
    families = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
    sources = [
        {"source_id": "real", "split": "internal-validation"},
        {"source_id": "fake", "split": "internal-validation"},
        {"source_id": "sealed", "split": "sealed-internal-test"},
    ]
    observations = []
    for label, source_id in ((0, "real"), (1, "fake")):
        for family in families:
            observations.append(
                {
                    "source_id": source_id,
                    "variant_id": f"{source_id}-{family}",
                    "image_path": image_name,
                    "authenticity_label": label,
                    "split": "internal-validation",
                    "condition_family": family,
                    "severity": "clean" if family == "clean" else "1",
                }
            )
    observations.append(
        {
            "source_id": "sealed",
            "variant_id": "sealed-clean",
            "image_path": image_name,
            "authenticity_label": 1,
            "split": "sealed-internal-test",
            "condition_family": "clean",
            "severity": "clean",
        }
    )
    return {
        "manifest_schema_version": "track5-manifest-v1",
        "corruption": {"transform_implementation_version": "track5-corruption-test-v1"},
        "sources": sources,
        "observations": observations,
    }


class RgbBaselineTests(unittest.TestCase):
    def test_cache_carries_versions_profile_and_hides_sealed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            manifest = manifest_for("image.png")
            cache = run_rgb_baseline(
                manifest, MeanBackend(), dataset_root=root, resolution=224, batch_size=4
            )
            self.assertEqual(len(validate_rgb_cache(cache, manifest, resolution=224)), 15)
            self.assertNotIn("authenticity_label", cache["records"][-1])
            self.assertEqual(cache["profile"]["batch_size"], 4)
            self.assertEqual(cache["profile"]["device"], "cpu")
            self.assertEqual(cache["profile"]["failure_count"], 0)
            self.assertTrue(all("corruption_version" in record for record in cache["records"]))

            changed = copy.deepcopy(manifest)
            changed["corruption"]["transform_implementation_version"] = "stale"
            with self.assertRaisesRegex(ValueError, "stale or incompatible"):
                validate_rgb_cache(cache, changed, resolution=224)

    def test_metrics_average_severities_and_report_required_robustness_summaries(self):
        records = []
        for family in ("clean", "jpeg", "blur", "resize", "noise", "color", "crop"):
            for severity in (("clean",) if family == "clean" else ("low", "high")):
                records.extend(
                    [
                        {"split": "internal-validation", "condition_family": family, "severity": severity, "authenticity_label": 0, "rgb_logit": -1.0},
                        {"split": "internal-validation", "condition_family": family, "severity": severity, "authenticity_label": 1, "rgb_logit": 1.0},
                    ]
                )
        metrics = evaluate_internal_validation(records)
        self.assertEqual(metrics["clean_auroc"], 1.0)
        self.assertEqual(metrics["mean_corrupted_auroc"], 1.0)
        self.assertEqual(metrics["all_condition_macro_auroc"], 1.0)
        self.assertEqual(metrics["degradation_drop"], 0.0)
        self.assertEqual(metrics["degradation_retention"], 1.0)
        self.assertEqual(metrics["threshold_diagnostics"]["status"], "provisional-internal-validation-only")

    def test_rerun_comparison_uses_declared_tolerance(self):
        expected = [{"variant_id": "a", "rgb_logit": 0.25}]
        result = compare_deterministic_subset(expected, [{"variant_id": "a", "rgb_logit": 0.250001}], 0.00001)
        self.assertEqual(result["status"], "passed")
        with self.assertRaisesRegex(ValueError, "deterministic rerun mismatch"):
            compare_deterministic_subset(expected, [{"variant_id": "a", "rgb_logit": 0.3}], 0.00001)


if __name__ == "__main__":
    unittest.main()
