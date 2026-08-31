"""Migrate the reviewed Issue #7 generation into its complete v2 package."""

from __future__ import annotations

import argparse
from pathlib import Path

import fusion_pipeline


def migrate_static_fallback_generation(
    legacy_generation_directory: Path,
    signal_model_path: Path,
    output_directory: Path,
) -> dict:
    return fusion_pipeline.migrate_static_fallback_generation(
        legacy_generation_directory,
        signal_model_path,
        output_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-generation-dir", required=True, type=Path)
    parser.add_argument("--signal-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    completion = migrate_static_fallback_generation(
        args.legacy_generation_dir,
        args.signal_model,
        args.output_dir,
    )
    print(f"Issue #7 v2 generation complete: {completion['generation_revision']}")


if __name__ == "__main__":
    main()
