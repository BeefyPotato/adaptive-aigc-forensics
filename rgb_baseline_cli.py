#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rgb_baseline import (
    compare_deterministic_subset,
    evaluate_internal_validation,
    run_rgb_baseline,
    validate_rgb_cache,
    write_json,
)
from rgb_expert import CommunityForensicsBackend, download_checkpoint, load_model_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen RGB Track 5 robustness baseline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resolution", choices=(224, 384), default=384, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", default="float32", choices=("float32",))
    parser.add_argument("--retry-once", action="store_true")
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    checkpoint = arguments.checkpoint or download_checkpoint(resolution=arguments.resolution)
    backend = CommunityForensicsBackend(checkpoint, resolution=arguments.resolution, device=arguments.device)
    cache = run_rgb_baseline(
        manifest,
        backend,
        dataset_root=arguments.dataset_root,
        resolution=arguments.resolution,
        batch_size=arguments.batch_size,
        precision=arguments.precision,
        retries=1 if arguments.retry_once else 0,
    )
    validate_rgb_cache(cache, manifest, resolution=arguments.resolution)
    metrics = evaluate_internal_validation(cache["records"])
    subset = cache["records"][: min(arguments.batch_size, len(cache["records"]))]
    subset_observations = [
        {**next(item for item in manifest["observations"] if item["variant_id"] == record["variant_id"]),
         "image_path": arguments.dataset_root / next(item for item in manifest["observations"] if item["variant_id"] == record["variant_id"]).get("materialized_image_path", next(item for item in manifest["observations"] if item["variant_id"] == record["variant_id"])["image_path"])}
        for record in subset
    ]
    from rgb_expert import predict_experiment_observations
    repeated = predict_experiment_observations(subset_observations, backend, resolution=arguments.resolution, batch_size=arguments.batch_size)
    rerun = compare_deterministic_subset(subset, repeated, load_model_metadata()["numeric_tolerance"])
    write_json(arguments.output_dir / "rgb-logits.json", cache)
    write_json(arguments.output_dir / "rgb-internal-validation-metrics.json", metrics)
    write_json(arguments.output_dir / "rgb-rerun-check.json", rerun)
    print(f"Wrote {len(cache['records'])} RGB logits and validation metrics to {arguments.output_dir}.")


if __name__ == "__main__":
    main()
