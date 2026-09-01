import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import submission_inference_cli as subject


class SubmissionInferenceCliTests(unittest.TestCase):
    def test_cli_requires_only_the_documented_artifact_inputs(self):
        parser = subject.build_parser()
        args = parser.parse_args([
            "--image-dir", "images", "--bundle-dir", "bundle",
            "--rgb-checkpoint", "rgb.safetensors", "--signal-model", "signal.json",
            "--output", "predictions.json",
        ])
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.device, "auto")

    def test_cli_rejects_unknown_device(self):
        with self.assertRaises(SystemExit):
            subject.build_parser().parse_args([
                "--image-dir", "images", "--bundle-dir", "bundle",
                "--rgb-checkpoint", "rgb.safetensors", "--signal-model", "signal.json",
                "--output", "predictions.json", "--device", "magic",
            ])

    def test_explicit_cpu_does_not_probe_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.json"
            argv = [
                "--image-dir", str(root), "--bundle-dir", str(root),
                "--rgb-checkpoint", str(root / "rgb"),
                "--signal-model", str(root / "signal"),
                "--output", str(output), "--device", "cpu",
            ]
            with (
                patch.object(subject, "run_submission_inference", return_value=[]) as run,
            ):
                self.assertEqual(subject.main(argv), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["device"], "cpu")

    def test_unavailable_explicit_cuda_fails_before_artifact_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "--image-dir", str(root), "--bundle-dir", str(root),
                "--rgb-checkpoint", str(root / "rgb"),
                "--signal-model", str(root / "signal"),
                "--output", str(root / "predictions.json"), "--device", "cuda",
            ]
            with (
                patch.object(
                    subject,
                    "run_submission_inference",
                    side_effect=ValueError("CUDA was explicitly requested but is unavailable."),
                ) as run,
                self.assertRaises(SystemExit),
            ):
                subject.main(argv)
            run.assert_called_once()
            self.assertFalse((root / "predictions.json").exists())


if __name__ == "__main__":
    unittest.main()
