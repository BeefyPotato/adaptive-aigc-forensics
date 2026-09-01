"""Command-line adapter for frozen Track 5 directory inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission_inference import run_submission_inference


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
        run_submission_inference(
            args.image_dir,
            frozen_generation_directory=args.bundle_dir,
            rgb_checkpoint=args.rgb_checkpoint,
            signal_model=args.signal_model,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
