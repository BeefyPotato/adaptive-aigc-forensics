"""CLI for deterministic signal feature extraction and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signal_expert import extract_signal_representation, fit_normalization, train_signal_mlp, write_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--epochs", type=int, default=200)
    arguments = parser.parse_args()
    training = json.loads(arguments.training_features.read_text(encoding="utf-8"))
    validation = json.loads(arguments.validation_features.read_text(encoding="utf-8"))
    manifest_metadata = training["manifest_metadata"]
    if validation["manifest_metadata"] != manifest_metadata:
        raise ValueError("Training and validation feature manifests are incompatible.")
    normalization = fit_normalization(training["records"], manifest_metadata=manifest_metadata)
    model, metadata = train_signal_mlp(training["records"], validation["records"], normalization, manifest_metadata=manifest_metadata, seed=arguments.seed, epochs=arguments.epochs)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_bundle(arguments.output, model, metadata, normalization)


if __name__ == "__main__":
    main()
