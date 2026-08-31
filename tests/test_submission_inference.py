import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import submission_inference as subject


class ConstantBackend:
    def __init__(self, values):
        self.values = list(values)

    def predict_logits(self, batch):
        return np.asarray(self.values[: len(batch)], dtype=np.float64)


class SubmissionInferenceTests(unittest.TestCase):
    def test_frozen_policy_uses_both_calibrated_expert_logits(self):
        bundle = {
            "selected_fallback_type": "learned-static-fusion",
            "rgb_calibrator": {"slope": 0.481469358380229, "intercept": 2.725716958843671},
            "signal_calibrator": {"slope": 2.1908644107462774, "intercept": 0.002442450220552084},
            "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
        }
        actual = subject.frozen_probability(1.0, -0.5, bundle, validate_calibrators=False)
        self.assertAlmostEqual(actual, 0.8603535389227722, places=15)
        self.assertNotAlmostEqual(actual, 0.9611038172365696, places=12)  # RGB only
        self.assertNotAlmostEqual(actual, 0.7421356326475331, places=12)  # 50/50

    def test_device_selection_never_silently_falls_back(self):
        self.assertEqual(subject.resolve_device("cpu", cuda_available=True), "cpu")
        self.assertEqual(subject.resolve_device("auto", cuda_available=False), "cpu")
        self.assertEqual(subject.resolve_device("auto", cuda_available=True), "cuda")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            subject.resolve_device("cuda", cuda_available=False)

    def test_directory_output_is_sorted_relative_and_byte_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 8), "red").save(root / "z.PNG")
            Image.new("L", (8, 8), 128).save(root / "nested" / "a.jpg")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }
            output = root.parent / (root.name + ".json")
            with patch.object(subject, "load_frozen_bundle", return_value=bundle), patch.object(
                subject, "_preprocess_signal", return_value=np.zeros(26)
            ), patch.object(subject, "_preprocess_rgb", return_value=np.zeros((3, 384, 384))):
                subject.run_submission(root, root, output, ConstantBackend([0.0, 1.0]), ConstantBackend([0.0, -1.0]))
                first = output.read_bytes()
                subject.run_submission(root, root, output, ConstantBackend([0.0, 1.0]), ConstantBackend([0.0, -1.0]))
            self.assertEqual(first, output.read_bytes())
            rows = json.loads(first)
            self.assertEqual([row["image_path"] for row in rows], ["nested/a.jpg", "z.PNG"])
            self.assertTrue(all(set(row) == {"image_path", "pred"} for row in rows))

    def test_backend_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            output = root / "predictions.json"
            output.write_bytes(b"old\n")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }
            with patch.object(subject, "load_frozen_bundle", return_value=bundle), patch.object(
                subject, "_preprocess_signal", return_value=np.zeros(26)
            ), patch.object(subject, "_preprocess_rgb", return_value=np.zeros((3, 384, 384))):
                with self.assertRaisesRegex(ValueError, "one logit"):
                    subject.run_submission(root, root, output, ConstantBackend([]), ConstantBackend([0.0]))
            self.assertEqual(output.read_bytes(), b"old\n")

    def test_empty_input_writes_valid_empty_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root.parent / (root.name + "-empty.json")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }
            with patch.object(subject, "load_frozen_bundle", return_value=bundle):
                subject.run_submission(root, root, output, ConstantBackend([]), ConstantBackend([]))
            self.assertEqual(json.loads(output.read_bytes()), [])

    def test_corrupt_supported_image_fails_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.png").write_bytes(b"not an image")
            output = root.parent / (root.name + "-corrupt.json")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }
            with patch.object(subject, "load_frozen_bundle", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "unreadable or corrupt"):
                    subject.run_submission(root, root, output, ConstantBackend([0.0]), ConstantBackend([0.0]))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
