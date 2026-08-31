"""Build resumable matched frozen-expert logits with bounded materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from fusion_pipeline import _sha256, matched_record, validate_matched_records
from rgb_expert import CommunityForensicsBackend, load_model_metadata, predict_experiment_observations
from signal_expert import read_model_bundle
from signal_pipeline import _feature_record


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _signal_logit(feature_record: dict, model, bundle: dict) -> float:
    features = np.asarray(feature_record["features"], dtype=np.float64)
    mean = np.asarray(bundle["normalization"]["mean"], dtype=np.float64)
    scale = np.asarray(bundle["normalization"]["scale"], dtype=np.float64)
    return float(model.logits(((features - mean) / scale)[None, :])[0])


def _subset_manifest(manifest: dict, source_ids: list[str]) -> dict:
    selected = set(source_ids)
    sources = [source for source in manifest["sources"] if source["source_id"] in selected]
    observations = [
        {**observation, "sample_weight": 1}
        for observation in manifest["observations"] if observation["source_id"] in selected
    ]
    if len(sources) != len(selected) or len(observations) != 20 * len(selected):
        raise ValueError("Bounded fusion shard does not contain the complete declared source matrix.")
    subset = {**manifest, "sources": sources, "observations": observations}
    subset["selection"] = {**manifest["selection"], "source_count": len(sources)}
    return subset


def _balanced_source_order(manifest: dict, source_ids: list[str]) -> list[str]:
    labels = {source["source_id"]: source["authenticity_label"] for source in manifest["sources"]}
    by_class = {label: sorted(source_id for source_id in source_ids if labels.get(source_id) == label) for label in (0, 1)}
    if len(by_class[0]) != len(by_class[1]) or len(by_class[0]) + len(by_class[1]) != len(source_ids):
        raise ValueError("Issue #7 source selection must be exactly class-balanced.")
    return [source_id for pair in zip(by_class[0], by_class[1], strict=True) for source_id in pair]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--signal-model", required=True, type=Path)
    parser.add_argument("--signal-validation-logits", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--node-binary", required=True)
    parser.add_argument("--sources-per-shard", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--materialization-concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.sources_per_shard <= 0:
        raise ValueError("sources-per-shard must be positive.")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    signal_payload = json.loads(args.signal_model.read_text(encoding="utf-8"))
    model, bundle = read_model_bundle(
        args.signal_model,
        manifest_metadata=signal_payload["manifest_metadata"],
        expected_experiment_provenance=signal_payload["experiment_provenance"],
    )
    validation_cache = json.loads(args.signal_validation_logits.read_text(encoding="utf-8"))
    validation_signal = {row["variant_id"]: row for row in validation_cache["records"]}
    validation_sources = _balanced_source_order(manifest, list({row["source_id"] for row in validation_signal.values()}))
    fusion_sources = _balanced_source_order(manifest, [source["source_id"] for source in manifest["sources"] if source["split"] == "fusion-training"])
    if len(fusion_sources) != 2000 or len(validation_sources) != 400:
        raise ValueError("Issue #7 requires 2,000 fusion-training and 400 matched validation sources.")
    metadata = load_model_metadata(); rgb_model = metadata["models"]["384"]
    provenance = {
        "manifest_sha256": manifest_sha256,
        "rgb_checkpoint_revision": rgb_model["revision"],
        "rgb_preprocessing_version": metadata["preprocessing_version"],
        "signal_checkpoint_revision": bundle["checkpoint_revision"],
        "signal_normalization_revision": bundle["normalization_revision"],
        "signal_feature_extraction_version": bundle["feature_extraction"]["feature_extraction_version"],
        "corruption_version": manifest["corruption"]["transform_implementation_version"],
    }
    backend = CommunityForensicsBackend(args.checkpoint, resolution=384, device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for split, source_ids in (("fusion-training", fusion_sources), ("internal-validation", validation_sources)):
        shard_dir = args.output_dir / "shards" / split
        for offset in range(0, len(source_ids), args.sources_per_shard):
            shard_sources = source_ids[offset:offset + args.sources_per_shard]
            name = f"{offset // args.sources_per_shard:05d}"
            cache_path = shard_dir / f"{name}.json"
            receipt_path = shard_dir / f"{name}.complete.json"
            if cache_path.is_file() and receipt_path.is_file():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                rows = validate_matched_records(cache["records"], expected_split=split)
                if receipt.get("cache_sha256") != _file_sha256(cache_path) or receipt.get("records_sha256") != _sha256(rows):
                    raise ValueError(f"Stale Issue #7 shard receipt {split}/{name}.")
                print(f"[fusion-cache] reused {split}/{name}", flush=True)
                continue
            work = args.output_dir / "work" / f"{split}-{name}"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            recipe = work / "recipe.json"
            _write_atomic(recipe, _subset_manifest(manifest, shard_sources))
            materialized = work / "materialized"
            subprocess.run([
                args.node_binary, "scripts/materialize-track5-observations.mjs",
                "--manifest", str(recipe), "--dataset-root", str(args.dataset_root),
                "--output-dir", str(materialized), "--concurrency", str(args.materialization_concurrency),
            ], check=True)
            materialized_manifest = json.loads((materialized / "track5-materialized-manifest.json").read_text(encoding="utf-8"))
            observations = materialized_manifest["observations"]
            for observation in observations:
                observation["image_path"] = materialized / observation["materialized_image_path"]
                observation["materialized_path"] = observation["image_path"]
            rgb = {row["variant_id"]: row for row in predict_experiment_observations(observations, backend, resolution=384, batch_size=args.batch_size)}
            matched = []
            for observation in observations:
                rgb_row = {**observation, **rgb[observation["variant_id"]]}
                if split == "fusion-training":
                    feature = _feature_record(observation, resolution=384)
                    signal_row = {**observation, "signal_logit": _signal_logit(feature, model, bundle)}
                else:
                    signal_row = validation_signal.get(observation["variant_id"])
                    if signal_row is None or signal_row.get("materialized_sha256") != observation["materialized_sha256"]:
                        raise ValueError("Issue #6 validation logits do not match rematerialized observations.")
                matched.append(matched_record(rgb_row, signal_row, provenance))
            matched = validate_matched_records(matched, expected_split=split)
            cache = {"cache_schema_version": "matched-frozen-expert-logits-v1", "provenance": provenance, "records_sha256": _sha256(matched), "records": matched}
            _write_atomic(cache_path, cache)
            receipt = {"completion_schema_version": "matched-frozen-expert-shard-completion-v1", "cache_sha256": _file_sha256(cache_path), "records_sha256": cache["records_sha256"], "source_count": len(shard_sources), "observation_count": len(matched)}
            _write_atomic(receipt_path, receipt)
            shutil.rmtree(work)
            print(f"[fusion-cache] completed {split}/{name}: {len(matched)} observations", flush=True)
    for split in ("fusion-training", "internal-validation"):
        rows = []
        for path in sorted(
            path for path in (args.output_dir / "shards" / split).glob("[0-9]*.json")
            if not path.name.endswith(".complete.json")
        ):
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
        rows = validate_matched_records(rows, expected_split=split)
        expected = 40000 if split == "fusion-training" else 8000
        if len(rows) != expected:
            raise ValueError(f"Issue #7 {split} cache has {len(rows)} records; expected {expected}.")
        _write_atomic(args.output_dir / f"matched-{split}-logits.json", {"cache_schema_version": "matched-frozen-expert-logits-v1", "provenance": provenance, "records_sha256": _sha256(rows), "records": rows})
    print(json.dumps({"status": "complete", "wall_clock_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": backend.peak_memory_bytes}, sort_keys=True))


if __name__ == "__main__":
    main()
