"""Command-line adapter for frozen Track 5 directory inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgb_expert import CommunityForensicsBackend, load_model_metadata
from signal_expert import read_model_bundle
from submission_inference import load_frozen_bundle, resolve_device, run_submission


class SignalModelBackend:
    def __init__(self, path: Path, bundle: dict):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._model, validated = read_model_bundle(
            path,
            manifest_metadata=payload.get("manifest_metadata"),
            expected_experiment_provenance=payload.get("experiment_provenance"),
        )
        provenance = bundle.get("provenance", {})
        expected = {
            "checkpoint_revision": provenance.get("signal_checkpoint_revision"),
            "normalization_revision": provenance.get("signal_normalization_revision"),
        }
        if any(expected[key] is not None and validated.get(key) != value for key, value in expected.items()):
            raise ValueError("Signal model does not match frozen bundle provenance.")
        normalization = validated["normalization"]
        self._mean = np.asarray(normalization["mean"], dtype=np.float64)
        self._scale = np.asarray(normalization["scale"], dtype=np.float64)

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        values = np.asarray(batch, dtype=np.float64)
        return self._model.logits((values - self._mean) / self._scale)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--rgb-checkpoint", required=True, type=Path)
    parser.add_argument("--signal-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch

        device = resolve_device(args.device, cuda_available=torch.cuda.is_available())
        bundle = load_frozen_bundle(args.bundle_dir)
        metadata = load_model_metadata()
        provenance = bundle.get("provenance", {})
        if provenance.get("rgb_checkpoint_revision") not in (None, metadata["models"]["384"]["revision"]):
            raise ValueError("RGB checkpoint metadata does not match frozen bundle provenance.")
        rgb = CommunityForensicsBackend(args.rgb_checkpoint, resolution=384, device=device)
        signal = SignalModelBackend(args.signal_model, bundle)
        run_submission(
            args.image_dir,
            args.bundle_dir,
            args.output,
            rgb,
            signal,
            batch_size=args.batch_size,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
