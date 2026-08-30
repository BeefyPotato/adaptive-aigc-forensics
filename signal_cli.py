"""Command-line entry point for the leakage-safe signal-only experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from signal_maps import render_signal_maps
from signal_pipeline import run_signal_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize bounded canonical Issue-3 observation shards, extract the "
            "26 signal features, train on expert-training, and evaluate on "
            "internal-validation."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the complete manifest-to-artifacts experiment.")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--training-count", type=int, default=40_320)
    run.add_argument("--sampler-seed", type=int, default=61)
    run.add_argument("--model-seed", type=int, default=61)
    run.add_argument("--epochs", type=int, default=200)
    run.add_argument("--learning-rate", type=float, default=0.02)
    run.add_argument("--resolution", type=int, choices=(224, 384), default=384)
    run.add_argument("--shard-raw-bytes", type=int, default=2**30)
    run.add_argument("--node-binary", default="node")
    maps = commands.add_parser(
        "render-maps",
        help="Render four diagnostic maps from one verified materialized observation.",
    )
    maps.add_argument("--manifest", type=Path, required=True)
    maps.add_argument("--materialized-manifest", type=Path, required=True)
    maps.add_argument("--variant-id", required=True)
    maps.add_argument("--output-dir", type=Path, required=True)
    maps.add_argument("--resolution", type=int, choices=(224, 384), default=384)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        return run_signal_experiment(
            arguments.manifest,
            arguments.dataset_root,
            arguments.output_dir,
            training_count=arguments.training_count,
            sampler_seed=arguments.sampler_seed,
            model_seed=arguments.model_seed,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            resolution=arguments.resolution,
            shard_raw_bytes=arguments.shard_raw_bytes,
            node_binary=arguments.node_binary,
        )
    if arguments.command == "render-maps":
        return render_signal_maps(
            arguments.manifest,
            arguments.materialized_manifest,
            variant_id=arguments.variant_id,
            output_directory=arguments.output_dir,
            resolution=arguments.resolution,
        )
    raise ValueError(f"Unsupported signal command {arguments.command!r}.")


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
