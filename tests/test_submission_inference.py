import json
import hashlib
import math
import tempfile
import unittest
from copy import deepcopy
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
    def _deployment_fixture(self, root: Path):
        provenance = deepcopy(subject.EXPECTED_PROVENANCE)
        signal_bytes = b"trusted signal model"
        rgb_bytes = b"trusted rgb checkpoint"
        signal_sha = hashlib.sha256(signal_bytes).hexdigest()
        rgb_sha = hashlib.sha256(rgb_bytes).hexdigest()
        provenance["rgb_checkpoint_sha256"] = rgb_sha
        bundle = {
            "bundle_revision": "bundle-test",
            "selected_fallback_type": "learned-static-fusion",
            "provenance": provenance,
            "input_cache_bindings": {
                "signal_model": {
                    "path": "upstream/signal-model.json",
                    "file_sha256": signal_sha,
                    "checkpoint_revision": provenance["signal_checkpoint_revision"],
                    "normalization_revision": provenance["signal_normalization_revision"],
                }
            },
            "rgb_normalizer": {
                "checkpoint_revision": provenance["rgb_checkpoint_revision"],
                "checkpoint_sha256": rgb_sha,
                "preprocessing_version": provenance["rgb_preprocessing_version"],
                "score_direction": provenance["rgb_score_direction"],
                "shared_observation_preprocessing_version": provenance[
                    "shared_observation_preprocessing_version"
                ],
                "input_resolution": 384,
                "resize_short_edge": 440,
            },
            "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
            "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
            "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
        }
        bundle_bytes = (json.dumps(bundle, indent=2) + "\n").encode()
        bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
        artifact_bindings = {
            name: {
                "path": name,
                "file_sha256": "0" * 64,
            }
            for name in subject.EXPECTED_GENERATION_ARTIFACTS
        }
        artifact_bindings["static-fallback-bundle.json"] = {
            "path": "static-fallback-bundle.json",
            "file_sha256": bundle_sha,
        }
        receipt = {
            "completion_schema_version": "static-fallback-completion-v2",
            "provenance": provenance,
            "bundle_revision": bundle["bundle_revision"],
            "bundle_sha256": bundle_sha,
            "artifacts": artifact_bindings,
        }
        receipt_identity = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        receipt["generation_revision"] = (
            "static-fallback-generation-v2-"
            + hashlib.sha256(receipt_identity).hexdigest()
        )
        bundle_dir = root / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "static-fallback-bundle.json").write_bytes(bundle_bytes)
        (bundle_dir / "static-fallback.complete.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        rgb_path = root / "rgb.safetensors"
        signal_path = root / "signal.json"
        rgb_path.write_bytes(rgb_bytes)
        signal_path.write_bytes(signal_bytes)
        return bundle_dir, rgb_path, signal_path, bundle, receipt, rgb_sha, signal_sha

    def test_deployment_reader_needs_no_evaluation_caches_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, _, _, bundle, receipt, _, _ = self._deployment_fixture(root)
            with (
                patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                patch.object(subject, "validate_bundle", return_value=bundle),
            ):
                self.assertEqual(subject.load_frozen_bundle(bundle_dir), bundle)

            self.assertEqual(
                {path.name for path in bundle_dir.iterdir()},
                {"static-fallback-bundle.json", "static-fallback.complete.json"},
            )

    def test_deployment_reader_never_opens_poisoned_evaluation_cache_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, _, _, bundle, receipt, _, _ = self._deployment_fixture(root)
            original = Path.read_bytes

            def reject_cache_access(path):
                if "logits" in path.name:
                    raise AssertionError(f"evaluation cache accessed: {path.name}")
                return original(path)

            with (
                patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                patch.object(subject, "validate_bundle", return_value=bundle),
                patch.object(Path, "read_bytes", reject_cache_access),
            ):
                self.assertEqual(subject.load_frozen_bundle(bundle_dir), bundle)

    def test_artifact_gate_rejects_every_frozen_runtime_binding_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, rgb_path, signal_path, bundle, receipt, rgb_sha, signal_sha = (
                self._deployment_fixture(root)
            )
            metadata = {
                "preprocessing_version": bundle["provenance"]["rgb_preprocessing_version"],
                "score_direction": bundle["provenance"]["rgb_score_direction"],
                "models": {
                    "384": {
                        "revision": bundle["provenance"]["rgb_checkpoint_revision"],
                        "sha256": rgb_sha,
                        "input_resolution": 384,
                        "resize_short_edge": 440,
                    }
                },
            }
            patches = (
                patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                patch.object(subject, "RGB_CHECKPOINT_SHA256", rgb_sha),
                patch.object(subject, "SIGNAL_MODEL_SHA256", signal_sha),
                patch.object(subject, "validate_bundle", return_value=bundle),
                patch.object(subject, "load_model_metadata", return_value=metadata),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                self.assertEqual(
                    subject.validate_submission_artifacts(
                        bundle_dir, rgb_checkpoint=rgb_path, signal_model=signal_path
                    ),
                    bundle,
                )

                cases = [
                    ("signal checksum", lambda: signal_path.write_bytes(b"substitute")),
                    ("RGB checksum", lambda: rgb_path.write_bytes(b"substitute")),
                ]
                for message, mutate in cases:
                    with self.subTest(message=message):
                        rgb_path.write_bytes(b"trusted rgb checkpoint")
                        signal_path.write_bytes(b"trusted signal model")
                        mutate()
                        with self.assertRaisesRegex(ValueError, "checksum"):
                            subject.validate_submission_artifacts(
                                bundle_dir,
                                rgb_checkpoint=rgb_path,
                                signal_model=signal_path,
                            )

    def test_bundle_rejects_rgb_preprocessing_score_and_shared_geometry_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, _, _, bundle, receipt, _, _ = self._deployment_fixture(root)
            for field in (
                "rgb_preprocessing_version",
                "rgb_score_direction",
                "shared_observation_preprocessing_version",
            ):
                with self.subTest(field=field):
                    changed = deepcopy(bundle)
                    changed["provenance"][field] = "substitute"
                    with (
                        patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                        patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                        patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                        patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                        patch.object(subject, "validate_bundle", return_value=changed),
                    ):
                        with self.assertRaisesRegex(ValueError, "provenance"):
                            subject.load_frozen_bundle(bundle_dir)

    def test_artifact_gate_rejects_each_model_and_rgb_runtime_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, rgb_path, signal_path, bundle, receipt, rgb_sha, signal_sha = (
                self._deployment_fixture(root)
            )
            metadata = {
                "preprocessing_version": bundle["provenance"]["rgb_preprocessing_version"],
                "score_direction": bundle["provenance"]["rgb_score_direction"],
                "models": {
                    "384": {
                        "revision": bundle["provenance"]["rgb_checkpoint_revision"],
                        "sha256": rgb_sha,
                        "input_resolution": 384,
                        "resize_short_edge": 440,
                    }
                },
            }
            cases = [
                ("signal model", "input_cache_bindings", "file_sha256"),
                ("signal model", "input_cache_bindings", "checkpoint_revision"),
                ("signal model", "input_cache_bindings", "normalization_revision"),
                ("RGB", "rgb_normalizer", "checkpoint_revision"),
                ("RGB", "rgb_normalizer", "checkpoint_sha256"),
                ("RGB", "rgb_normalizer", "preprocessing_version"),
                ("RGB", "rgb_normalizer", "score_direction"),
                ("RGB", "rgb_normalizer", "shared_observation_preprocessing_version"),
                ("RGB", "rgb_normalizer", "input_resolution"),
                ("RGB", "rgb_normalizer", "resize_short_edge"),
            ]
            for message, section, field in cases:
                with self.subTest(section=section, field=field):
                    changed = deepcopy(bundle)
                    target = (
                        changed[section]["signal_model"]
                        if section == "input_cache_bindings"
                        else changed[section]
                    )
                    target[field] = "substitute"
                    with (
                        patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                        patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                        patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                        patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                        patch.object(subject, "RGB_CHECKPOINT_SHA256", rgb_sha),
                        patch.object(subject, "SIGNAL_MODEL_SHA256", signal_sha),
                        patch.object(subject, "validate_bundle", return_value=changed),
                        patch.object(subject, "load_model_metadata", return_value=metadata),
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            subject.validate_submission_artifacts(
                                bundle_dir,
                                rgb_checkpoint=rgb_path,
                                signal_model=signal_path,
                            )

            runtime_cases = [
                ("models", "revision"),
                ("models", "sha256"),
                ("models", "input_resolution"),
                ("models", "resize_short_edge"),
                ("top", "preprocessing_version"),
                ("top", "score_direction"),
            ]
            for section, field in runtime_cases:
                with self.subTest(runtime_section=section, field=field):
                    changed_metadata = deepcopy(metadata)
                    target = (
                        changed_metadata["models"]["384"]
                        if section == "models"
                        else changed_metadata
                    )
                    target[field] = "substitute"
                    with (
                        patch.object(subject, "BUNDLE_SHA256", receipt["bundle_sha256"]),
                        patch.object(subject, "BUNDLE_REVISION", bundle["bundle_revision"]),
                        patch.object(subject, "GENERATION_REVISION", receipt["generation_revision"]),
                        patch.object(subject, "EXPECTED_PROVENANCE", bundle["provenance"]),
                        patch.object(subject, "RGB_CHECKPOINT_SHA256", rgb_sha),
                        patch.object(subject, "SIGNAL_MODEL_SHA256", signal_sha),
                        patch.object(subject, "validate_bundle", return_value=bundle),
                        patch.object(subject, "load_model_metadata", return_value=changed_metadata),
                    ):
                        with self.assertRaisesRegex(ValueError, "RGB"):
                            subject.validate_submission_artifacts(
                                bundle_dir,
                                rgb_checkpoint=rgb_path,
                                signal_model=signal_path,
                            )

    def test_non_finite_expert_output_preserves_existing_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            output = root.parent / (root.name + "-non-finite.json")
            output.write_bytes(b"previous\n")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }
            with (
                patch.object(subject, "load_frozen_bundle", return_value=bundle),
                patch.object(subject, "_preprocess_signal", return_value=np.zeros(26)),
                patch.object(subject, "_preprocess_rgb", return_value=np.zeros((3, 384, 384))),
                self.assertRaisesRegex(ValueError, "non-finite"),
            ):
                subject.run_submission(
                    root,
                    root,
                    output,
                    ConstantBackend([math.nan]),
                    ConstantBackend([0.0]),
                )
            self.assertEqual(output.read_bytes(), b"previous\n")

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

    def test_frozen_policy_accepts_real_bundle_float_representation(self):
        bundle = {
            "selected_fallback_type": "learned-static-fusion",
            "rgb_calibrator": {"slope": 0.481469358380229, "intercept": 2.725716958843671},
            "signal_calibrator": {"slope": 2.1908644107462774, "intercept": 0.002442450220552084},
            "static_weight": {
                "rgb_weight": 0.677,
                "signal_weight": 0.32299999999999995,
            },
        }

        actual = subject.frozen_probability(1.0, -0.5, bundle, validate_calibrators=False)

        self.assertAlmostEqual(actual, 0.8603535389227722, places=15)

    def test_frozen_policy_rejects_meaningful_weight_drift(self):
        bundle = {
            "selected_fallback_type": "learned-static-fusion",
            "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
            "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
            "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323000000001},
        }

        with self.assertRaisesRegex(ValueError, "weights"):
            subject.frozen_probability(0.0, 0.0, bundle, validate_calibrators=False)

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

    def test_directory_inference_uses_production_signal_pixel_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), (128, 64, 32)).save(root / "a.jpg")
            output = root.parent / (root.name + "-scaled.json")
            bundle = {
                "selected_fallback_type": "learned-static-fusion",
                "rgb_calibrator": {"slope": 1.0, "intercept": 0.0},
                "signal_calibrator": {"slope": 1.0, "intercept": 0.0},
                "static_weight": {"rgb_weight": 0.677, "signal_weight": 0.323},
            }

            with patch.object(subject, "load_frozen_bundle", return_value=bundle):
                rows = subject.run_submission(
                    root,
                    root,
                    output,
                    ConstantBackend([0.0]),
                    ConstantBackend([0.0]),
                )

            self.assertEqual(rows, [{"image_path": "a.jpg", "pred": 0.5}])

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
