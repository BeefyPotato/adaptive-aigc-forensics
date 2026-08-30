#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rgb_expert import CommunityForensicsBackend, download_checkpoint, infer_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Community Forensics RGB expert.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resolution", choices=(224, 384), default=384, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint or download_checkpoint(
        resolution=arguments.resolution, cache_directory=arguments.cache_dir
    )
    backend = CommunityForensicsBackend(
        checkpoint, resolution=arguments.resolution, device=arguments.device
    )
    predictions = infer_directory(
        arguments.input_dir,
        backend,
        resolution=arguments.resolution,
        batch_size=arguments.batch_size,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} RGB predictions to {arguments.output} on {backend.device}.")


if __name__ == "__main__":
    main()
