"""Opt-in real-artifact acceptance test for frozen submission inference.

Set SUBMISSION_REAL_BUNDLE_DIR, SUBMISSION_REAL_RGB_CHECKPOINT, and
SUBMISSION_REAL_SIGNAL_MODEL to run this gate. No model or data bytes are checked in.
"""

import json
import hashlib
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline import calibrated_logit
from rgb_expert import (
    CommunityForensicsBackend,
    discover_images,
    load_model_metadata,
    predict_experiment_observations,
)
from signal_expert import (
    decode_expert_rgb,
    extract_signal_representation,
    read_model_bundle,
)
from submission_inference import load_frozen_bundle
from submission_inference_cli import main


REAL_ENVIRONMENT = (
    os.environ.get("SUBMISSION_REAL_BUNDLE_DIR"),
    os.environ.get("SUBMISSION_REAL_RGB_CHECKPOINT"),
    os.environ.get("SUBMISSION_REAL_SIGNAL_MODEL"),
)


@unittest.skipUnless(all(REAL_ENVIRONMENT), "real frozen submission artifacts not configured")
class RealSubmissionInferenceTests(unittest.TestCase):
    def test_submission_matches_canonical_expert_path_and_repeats_on_cpu(self):
        bundle_dir, rgb_checkpoint, signal_model = map(Path, REAL_ENVIRONMENT)
        image_dir = Path(
            os.environ.get(
                "SUBMISSION_REAL_IMAGE_DIR", "fixtures/experiment/images"
            )
        ).resolve()
        paths = discover_images(image_dir)
        self.assertTrue(paths, "real parity fixture directory must contain images")

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            durations = []
            for output in (first, second):
                started = time.perf_counter()
                self.assertEqual(
                    main(
                        [
                            "--image-dir", str(image_dir),
                            "--bundle-dir", str(bundle_dir),
                            "--rgb-checkpoint", str(rgb_checkpoint),
                            "--signal-model", str(signal_model),
                            "--output", str(output),
                            "--device", "cpu",
                            "--batch-size", "2",
                        ]
                    ),
                    0,
                )
                durations.append(time.perf_counter() - started)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            first_bytes = first.read_bytes()
            actual = json.loads(first_bytes)

        names = [path.relative_to(image_dir).as_posix() for path in paths]
        self.assertEqual([row["image_path"] for row in actual], names)
        self.assertTrue(all(set(row) == {"image_path", "pred"} for row in actual))
        self.assertTrue(
            all(
                isinstance(row["pred"], float)
                and math.isfinite(row["pred"])
                and 0.0 <= row["pred"] <= 1.0
                for row in actual
            )
        )

        observations = [
            {"source_id": f"fixture-{index}", "variant_id": name, "image_path": str(path)}
            for index, (name, path) in enumerate(zip(names, paths, strict=True))
        ]
        rgb_backend = CommunityForensicsBackend(
            rgb_checkpoint, resolution=384, device="cpu"
        )
        rgb_rows = predict_experiment_observations(
            observations, rgb_backend, resolution=384, batch_size=2
        )
        bundle = load_frozen_bundle(bundle_dir)
        signal_payload = json.loads(signal_model.read_text(encoding="utf-8"))
        signal_network, validated_signal = read_model_bundle(
            signal_model,
            manifest_metadata=signal_payload["manifest_metadata"],
            expected_experiment_provenance=signal_payload["experiment_provenance"],
        )
        signal_features = np.stack(
            [
                extract_signal_representation(
                    decode_expert_rgb(path, resolution=384)
                )["features"]
                for path in paths
            ]
        )
        normalization = validated_signal["normalization"]
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        scale = np.asarray(normalization["scale"], dtype=np.float64)
        signal_logits = signal_network.logits((signal_features - mean) / scale)
        expected = []
        for row, signal_logit in zip(rgb_rows, signal_logits, strict=True):
            combined = (
                float(bundle["static_weight"]["rgb_weight"])
                * calibrated_logit(row["rgb_logit"], bundle["rgb_calibrator"])
                + float(bundle["static_weight"]["signal_weight"])
                * calibrated_logit(float(signal_logit), bundle["signal_calibrator"])
            )
            probability = (
                1.0 / (1.0 + math.exp(-combined))
                if combined >= 0
                else math.exp(combined) / (1.0 + math.exp(combined))
            )
            expected.append(probability)
        tolerance = float(load_model_metadata()["numeric_tolerance"])
        maximum_delta = max(
            abs(row["pred"] - value)
            for row, value in zip(actual, expected, strict=True)
        )
        self.assertLessEqual(maximum_delta, tolerance)
        self.assertTrue(all(duration > 0 for duration in durations))
        print(
            "[submission-parity] "
            f"images={len(actual)} max_abs_delta={maximum_delta:.17g} "
            f"output_sha256={hashlib.sha256(first_bytes).hexdigest()}"
        )


if __name__ == "__main__":
    unittest.main()
