import copy
import hashlib
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


class OneFailureBackend(MeanBackend):
    def __init__(self):
        self.calls = 0

    def predict_logits(self, batch):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("transient batch failure")
        return super().predict_logits(batch)


def manifest_for(image_name, *, materialized=True, materialized_sha256="a" * 64):
    families = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
    sources = [
        {"source_id": "real", "split": "internal-validation"},
        {"source_id": "fake", "split": "internal-validation"},
        {"source_id": "sealed", "split": "sealed-internal-test"},
    ]
    observations = []
    for label, source_id in ((0, "real"), (1, "fake")):
        for family in families:
            observation = {
                    "source_id": source_id,
                    "variant_id": f"{source_id}-{family}",
                    "image_path": image_name,
                    "authenticity_label": label,
                    "split": "internal-validation",
                    "condition_family": family,
                    "severity": "clean" if family == "clean" else "1",
                }
            if materialized:
                observation.update(
                    materialized_image_path=image_name,
                    materialized_sha256=materialized_sha256,
                    materialized_encoding="lossless-rgb-png-v1",
                )
            observations.append(observation)
    sealed = {
            "source_id": "sealed",
            "variant_id": "sealed-clean",
            "image_path": image_name,
            "authenticity_label": 1,
            "split": "sealed-internal-test",
            "condition_family": "clean",
            "severity": "clean",
        }
    if materialized:
        sealed.update(
            materialized_image_path=image_name,
            materialized_sha256=materialized_sha256,
            materialized_encoding="lossless-rgb-png-v1",
        )
    observations.append(sealed)
    manifest = {
        "manifest_schema_version": "track5-manifest-v1",
        "corruption": {
            "preprocessing_version": "shared-preprocessing-v1",
            "transform_implementation_version": "track5-corruption-test-v1",
        },
        "sources": sources,
        "observations": observations,
    }
    if materialized:
        manifest.update(
            materialization_schema_version="track5-materialized-observations-v1",
            materialization={
                "shared_observation_preprocessing_version": "shared-preprocessing-v1",
                "corruption_version": "track5-corruption-test-v1",
                "encoding": "lossless-rgb-png-v1",
                "observation_count": len(observations),
            },
        )
    return manifest


class RgbBaselineTests(unittest.TestCase):
    def test_recipe_only_manifest_is_rejected_instead_of_scoring_clean_source_for_every_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            with self.assertRaisesRegex(ValueError, "materialized"):
                run_rgb_baseline(
                    manifest_for("image.png", materialized=False),
                    MeanBackend(),
                    dataset_root=root,
                    resolution=224,
                )

    def test_cache_carries_versions_profile_and_hides_sealed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            digest = hashlib.sha256((root / "image.png").read_bytes()).hexdigest()
            manifest = manifest_for("image.png", materialized_sha256=digest)
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
        explicit_legacy_metrics = evaluate_internal_validation(
            records,
            score_field="rgb_logit",
            metric_schema_version="rgb-robustness-metric-v1",
        )
        self.assertEqual(metrics, explicit_legacy_metrics)
        self.assertEqual(metrics["metric_schema_version"], "rgb-robustness-metric-v1")
        self.assertEqual(metrics["clean_auroc"], 1.0)
        self.assertEqual(metrics["mean_corrupted_auroc"], 1.0)
        self.assertEqual(metrics["all_condition_macro_auroc"], 1.0)
        self.assertEqual(metrics["degradation_drop"], 0.0)
        self.assertEqual(metrics["degradation_retention"], 1.0)
        self.assertEqual(metrics["threshold_diagnostics"]["status"], "provisional-internal-validation-only")

    def test_signal_logits_use_the_same_severity_first_family_macro_evaluator(self):
        records = [
            {
                "split": "internal-validation",
                "condition_family": "clean",
                "severity": "clean",
                "authenticity_label": label,
                "signal_logit": score,
            }
            for label, score in ((0, -2.0), (1, 2.0))
        ]
        for family in ("jpeg", "blur", "resize", "noise", "color", "crop"):
            records.extend(
                {
                    "split": "internal-validation",
                    "condition_family": family,
                    "severity": severity,
                    "authenticity_label": label,
                    "signal_logit": score,
                }
                for severity, scored_labels in (
                    ("low", ((0, -1.0), (1, 1.0))),
                    ("high", ((0, 1.0), (1, -1.0))),
                )
                for label, score in scored_labels
            )

        metrics = evaluate_internal_validation(
            records,
            score_field="signal_logit",
            metric_schema_version="signal-robustness-metric-v1",
        )

        self.assertEqual(metrics["metric_schema_version"], "signal-robustness-metric-v1")
        self.assertEqual(metrics["clean_auroc"], 1.0)
        self.assertEqual(metrics["corruption_families"]["jpeg"]["auroc_by_severity"], {"high": 0.0, "low": 1.0})
        self.assertEqual(metrics["corruption_families"]["jpeg"]["auroc"], 0.5)
        self.assertEqual(metrics["mean_corrupted_auroc"], 0.5)
        self.assertAlmostEqual(metrics["all_condition_macro_auroc"], 4 / 7)
        self.assertEqual(
            metrics["worst_family_severity"],
            {"family": "blur", "severity": "high", "auroc": 0.0},
        )
        self.assertEqual(
            metrics["threshold_diagnostics"]["status"],
            "provisional-internal-validation-only",
        )
        self.assertEqual(metrics["threshold_diagnostics"]["threshold_logit"], 2.0)

    def test_evaluator_checks_that_the_selected_score_field_is_finite(self):
        records = [
            {
                "split": "internal-validation",
                "condition_family": family,
                "severity": "clean" if family == "clean" else "low",
                "authenticity_label": label,
                "rgb_logit": 0.0,
                "signal_logit": float("nan") if family == "clean" and label == 0 else float(label),
            }
            for family in ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")
            for label in (0, 1)
        ]

        with self.assertRaisesRegex(ValueError, "signal_logit must be a finite number"):
            evaluate_internal_validation(
                records,
                score_field="signal_logit",
                metric_schema_version="signal-robustness-metric-v1",
            )

    def test_rerun_comparison_uses_declared_tolerance(self):
        expected = [{"variant_id": "a", "rgb_logit": 0.25}]
        result = compare_deterministic_subset(expected, [{"variant_id": "a", "rgb_logit": 0.250001}], 0.00001)
        self.assertEqual(result["status"], "passed")
        with self.assertRaisesRegex(ValueError, "deterministic rerun mismatch"):
            compare_deterministic_subset(expected, [{"variant_id": "a", "rgb_logit": 0.3}], 0.00001)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            compare_deterministic_subset([], [], 0.00001)
        with self.assertRaisesRegex(ValueError, "exactly"):
            compare_deterministic_subset(expected, [{"variant_id": "b", "rgb_logit": 0.25}], 0.00001)

    def test_cache_reader_rejects_duplicate_missing_nonfinite_and_invalid_record_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            digest = hashlib.sha256((root / "image.png").read_bytes()).hexdigest()
            manifest = manifest_for("image.png", materialized_sha256=digest)
            cache = run_rgb_baseline(manifest, MeanBackend(), dataset_root=root, resolution=224)
            for mutation, message in (
                (lambda value: value["records"].append(copy.deepcopy(value["records"][0])), "duplicate"),
                (lambda value: value["records"].pop(), "missing"),
                (lambda value: value["records"][0].update(rgb_logit=float("nan")), "finite"),
                (lambda value: value["records"][0].update(cache_key="stale"), "cache key"),
            ):
                changed = copy.deepcopy(cache)
                mutation(changed)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_rgb_cache(changed, manifest, resolution=224)

    def test_materialized_path_cannot_escape_declared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            manifest = manifest_for("image.png")
            manifest["observations"][0]["materialized_image_path"] = "../image.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                run_rgb_baseline(manifest, MeanBackend(), dataset_root=root, resolution=224)

    def test_retry_repeats_only_the_failed_batch_and_records_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), "white").save(root / "image.png")
            digest = hashlib.sha256((root / "image.png").read_bytes()).hexdigest()
            manifest = manifest_for("image.png", materialized_sha256=digest)
            backend = OneFailureBackend()
            cache = run_rgb_baseline(
                manifest,
                backend,
                dataset_root=root,
                resolution=224,
                batch_size=4,
                retries=1,
            )
            self.assertEqual(cache["profile"]["failure_count"], 1)
            self.assertEqual(cache["profile"]["retry_count"], 1)
            self.assertEqual(backend.calls, 5)


if __name__ == "__main__":
    unittest.main()
