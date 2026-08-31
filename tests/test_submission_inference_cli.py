import unittest

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


if __name__ == "__main__":
    unittest.main()
