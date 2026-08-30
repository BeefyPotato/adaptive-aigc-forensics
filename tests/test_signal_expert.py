import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_expert import (
    FEATURE_NAMES, cache_signal_predictions, evaluate_signal_only,
    extract_signal_representation, fit_normalization, read_model_bundle,
    train_signal_mlp, validate_signal_cache, write_model_bundle,
)


META = {"manifest_schema_version": "track5-manifest-v1", "dataset_revision": "fixture", "manifest_sha256": "a" * 64}


def record(source, split, label, features, family="clean"):
    return {"source_id": source, "variant_id": f"variant-{source}-{family}", "split": split, "authenticity_label": label, "condition_family": family, "features": np.asarray(features, dtype=float).tolist()}


class SignalExpertTests(unittest.TestCase):
    def test_representation_has_documented_deterministic_16_6_4_order_and_maps(self):
        yy, xx = np.indices((33, 41))
        rgb = np.stack([(xx % 7) / 6, (yy % 5) / 4, ((xx + yy) % 9) / 8], axis=2)
        first = extract_signal_representation(rgb, include_maps=True)
        second = extract_signal_representation(rgb)
        self.assertEqual(len(FEATURE_NAMES), 26)
        self.assertEqual(first["features"].shape, (26,))
        np.testing.assert_array_equal(first["features"], second["features"])
        self.assertEqual(set(first["maps"]), {"luminance", "spectrum", "high_pass", "residual"})
        self.assertTrue(np.isfinite(first["features"]).all())

    def test_normalization_rejects_non_expert_training_data_and_stale_metadata(self):
        features = np.arange(26)
        with self.assertRaisesRegex(ValueError, "only on expert-training"):
            fit_normalization([record("bad", "fusion-training", 0, features)], manifest_metadata=META)
        training = [record("a", "expert-training", 0, features), record("b", "expert-training", 1, features + 1)]
        normalization = fit_normalization(training, manifest_metadata=META)
        validation = [record("c", "internal-validation", 0, features), record("d", "internal-validation", 1, features + 1)]
        with self.assertRaisesRegex(ValueError, "stale"):
            train_signal_mlp(training, validation, normalization, manifest_metadata={**META, "dataset_revision": "other"})

    def test_training_is_reproducible_source_disjoint_and_cache_is_strict(self):
        base = np.linspace(-1, 1, 26)
        training = [record(f"train-{i}", "expert-training", i % 2, base + (i % 2) * 2 + i / 100) for i in range(8)]
        validation = [record(f"val-{i}", "internal-validation", i % 2, base + (i % 2) * 2 + i / 100) for i in range(4)]
        normalization = fit_normalization(training, manifest_metadata=META)
        first, metadata = train_signal_mlp(training, validation, normalization, manifest_metadata=META, epochs=30)
        second, _ = train_signal_mlp(training, validation, normalization, manifest_metadata=META, epochs=30)
        np.testing.assert_array_equal(first.input_weights, second.input_weights)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            write_model_bundle(path, first, metadata, normalization)
            loaded, bundle = read_model_bundle(path, manifest_metadata=META)
            cache = cache_signal_predictions(validation, loaded, bundle["normalization"], manifest_metadata=META, checkpoint_revision="fixture-checkpoint")
        self.assertEqual(len(cache), 4)
        self.assertTrue(all(len(row["features"]) == 26 and np.isfinite(row["signal_logit"]) for row in cache))
        self.assertTrue(all(row["cache_key"].startswith("signal-cache-v1-") for row in cache))
        self.assertEqual(validate_signal_cache(cache, manifest_metadata=META, checkpoint_revision="fixture-checkpoint"), cache)
        stale = json.loads(json.dumps(cache))
        stale[0]["manifest_metadata"]["dataset_revision"] = "stale"
        with self.assertRaisesRegex(ValueError, "stale manifest_metadata"):
            validate_signal_cache(stale, manifest_metadata=META, checkpoint_revision="fixture-checkpoint")

    def test_validation_metrics_are_broken_out_by_corruption_family(self):
        rows = []
        for family in ("clean", "jpeg"):
            rows.extend([
                {**record(f"{family}-real", "internal-validation", 0, np.zeros(26), family), "pred": 0.1},
                {**record(f"{family}-fake", "internal-validation", 1, np.ones(26), family), "pred": 0.9},
            ])
        metrics = evaluate_signal_only(rows)
        self.assertEqual(metrics["auroc_by_corruption_family"], {"clean": 1.0, "jpeg": 1.0})


if __name__ == "__main__":
    unittest.main()
