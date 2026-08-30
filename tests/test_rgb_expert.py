import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgb_expert import (
    infer_directory,
    load_model_metadata,
    predict_experiment_observations,
    preprocess_image,
    verify_checkpoint,
)
from shared_observation import decode_shared_rgb, prepare_shared_expert_rgb
from signal_expert import decode_expert_rgb


class RecordingBackend:
    def __init__(self):
        self.batches = []

    def predict_logits(self, batch):
        self.batches.append(batch.copy())
        return np.arange(len(batch), dtype=np.float32) - 0.5


class RgbExpertTests(unittest.TestCase):
    def test_rgb_and_signal_branches_decode_identical_shared_observation_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.png"
            pixels = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(10, 12, 3)
            Image.fromarray(pixels, mode="RGB").save(path)
            shared = decode_shared_rgb(path)
            shared_crop = prepare_shared_expert_rgb(path, resolution=224)
            signal = decode_expert_rgb(path, resolution=224)
            np.testing.assert_array_equal(shared, pixels)
            np.testing.assert_array_equal(np.rint(signal * 255).astype(np.uint8), shared_crop)
    def test_metadata_pins_primary_and_fallback_below_track5_limit(self):
        metadata = load_model_metadata()
        self.assertEqual(metadata["models"]["384"]["role"], "primary")
        self.assertEqual(metadata["models"]["224"]["role"], "smoke-test-and-memory-fallback")
        for model in metadata["models"].values():
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$")
            self.assertLess(model["parameter_count"], 2_000_000_000)

    def test_preprocessing_handles_orientation_grayscale_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grayscale = root / "grayscale.png"
            alpha = root / "alpha.png"
            Image.new("L", (10, 20), 128).save(grayscale)
            Image.new("RGBA", (20, 10), (128, 128, 128, 0)).save(alpha)
            first = preprocess_image(grayscale, resolution=224)
            second = preprocess_image(alpha, resolution=224)
            self.assertEqual(first.shape, (3, 224, 224))
            self.assertEqual(second.shape, (3, 224, 224))
            np.testing.assert_allclose(first, second, atol=0, rtol=0)

    def test_checkpoint_checksum_mismatch_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"not the pinned checkpoint")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_checkpoint(checkpoint, "0" * 64)

    def test_directory_and_experiment_paths_share_preprocessing_and_score_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (12, 8), "black").save(root / "b.png")
            Image.new("RGB", (8, 12), "white").save(root / "a.jpg")
            directory_backend = RecordingBackend()
            predictions = infer_directory(root, directory_backend, resolution=224, batch_size=2)
            self.assertEqual([record["image_path"] for record in predictions], ["a.jpg", "b.png"])
            self.assertLess(predictions[0]["pred"], predictions[1]["pred"])

            experiment_backend = RecordingBackend()
            cache = predict_experiment_observations(
                [
                    {"source_id": "source-a", "variant_id": "variant-a", "image_path": root / "a.jpg"},
                    {"source_id": "source-b", "variant_id": "variant-b", "image_path": root / "b.png"},
                ],
                experiment_backend,
                resolution=224,
                batch_size=2,
            )
            np.testing.assert_array_equal(directory_backend.batches[0], experiment_backend.batches[0])
            self.assertEqual([record["variant_id"] for record in cache], ["variant-a", "variant-b"])
            self.assertEqual([record["rgb_logit"] for record in cache], [-0.5, 0.5])
            self.assertTrue(all(record["cache_key"].startswith("rgb-cache-v1-") for record in cache))

    @unittest.skipUnless(os.environ.get("COMMUNITY_FORENSICS_INTEGRATION") == "1", "opt-in checkpoint smoke test")
    def test_real_checkpoints_load_and_score_fixture(self):
        from rgb_expert import CommunityForensicsBackend, download_checkpoint

        for resolution in (224, 384):
            with self.subTest(resolution=resolution):
                checkpoint = download_checkpoint(resolution=resolution)
                backend = CommunityForensicsBackend(checkpoint, resolution=resolution, device="cpu")
                predictions = infer_directory(Path("fixtures/experiment/images"), backend, resolution=resolution, batch_size=2)
                repeated = infer_directory(Path("fixtures/experiment/images"), backend, resolution=resolution, batch_size=2)
                self.assertEqual(len(predictions), 2)
                self.assertTrue(all(0 <= record["pred"] <= 1 for record in predictions))
                np.testing.assert_allclose(
                    [record["pred"] for record in predictions],
                    [record["pred"] for record in repeated],
                    atol=load_model_metadata()["numeric_tolerance"],
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
