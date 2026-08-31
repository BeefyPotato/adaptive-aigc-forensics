import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import signal_cli


class SignalCliTests(unittest.TestCase):
    def test_render_maps_is_bound_to_recipe_and_materialized_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "track5-manifest.json"
            materialized = root / "track5-materialized-manifest.json"
            output = root / "maps"
            with patch("signal_cli.render_signal_maps") as render:
                signal_cli.main([
                    "render-maps",
                    "--manifest",
                    str(recipe),
                    "--materialized-manifest",
                    str(materialized),
                    "--variant-id",
                    "variant-123",
                    "--output-dir",
                    str(output),
                    "--resolution",
                    "224",
                ])

        render.assert_called_once_with(
            recipe,
            materialized,
            variant_id="variant-123",
            output_directory=output,
            resolution=224,
        )

    def test_run_is_manifest_to_artifacts_public_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "track5-manifest.json"
            dataset = root / "dataset"
            output = root / "signal-run"

            with patch("signal_cli.run_signal_experiment") as run:
                signal_cli.main([
                    "run",
                    "--manifest",
                    str(manifest),
                    "--dataset-root",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--experiment-profile",
                    "issue-6-full-v1",
                    "--training-count",
                    "40320",
                    "--sampler-seed",
                    "61",
                    "--model-seed",
                    "67",
                    "--epochs",
                    "3",
                    "--learning-rate",
                    "0.01",
                    "--resolution",
                    "384",
                    "--shard-raw-bytes",
                    "1073741824",
                ])

        run.assert_called_once_with(
            manifest,
            dataset,
            output,
            experiment_profile="issue-6-full-v1",
            training_count=40_320,
            validation_source_count=None,
            validation_seed=61,
            sampler_seed=61,
            model_seed=67,
            epochs=3,
            learning_rate=0.01,
            resolution=384,
            shard_raw_bytes=1_073_741_824,
            node_binary="node",
        )

    def test_hackathon_profile_resolves_fixed_timeboxed_acceptance_counts(self):
        with patch("signal_cli.run_signal_experiment") as run:
            signal_cli.main([
                "run",
                "--manifest",
                "manifest.json",
                "--dataset-root",
                "dataset",
                "--output-dir",
                "hackathon-output",
                "--experiment-profile",
                "hackathon-v1",
            ])

        self.assertEqual(run.call_args.kwargs["experiment_profile"], "hackathon-v1")
        self.assertEqual(run.call_args.kwargs["training_count"], 8_064)
        self.assertEqual(run.call_args.kwargs["validation_source_count"], 400)
        self.assertEqual(run.call_args.kwargs["validation_seed"], 61)

    def test_hackathon_profile_rejects_count_overrides(self):
        with patch("signal_cli.run_signal_experiment") as run:
            with self.assertRaisesRegex(ValueError, "hackathon-v1.*8,064"):
                signal_cli.main([
                    "run",
                    "--manifest",
                    "manifest.json",
                    "--dataset-root",
                    "dataset",
                    "--output-dir",
                    "hackathon-output",
                    "--experiment-profile",
                    "hackathon-v1",
                    "--training-count",
                    "168",
                ])
        run.assert_not_called()

    def test_old_precomputed_feature_interface_is_not_accepted(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                signal_cli.main([
                    "--training-features",
                    "train.json",
                    "--validation-features",
                    "validation.json",
                    "--output",
                    "model.json",
                ])

    def test_run_cli_rejects_an_unsupported_resolution_before_dispatch(self):
        with patch("signal_cli.run_signal_experiment") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                signal_cli.main([
                    "run",
                    "--manifest",
                    "manifest.json",
                    "--dataset-root",
                    "dataset",
                    "--output-dir",
                    "output",
                    "--resolution",
                    "999",
                ])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
