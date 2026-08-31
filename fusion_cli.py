"""Fit and evaluate Issue #7 calibration/static fusion from matched logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fusion_pipeline import evaluate_candidates, fit_platt_calibrator, fit_static_weight, publish_bundle_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-training-cache", required=True, type=Path)
    parser.add_argument("--internal-validation-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    training = json.loads(args.fusion_training_cache.read_text(encoding="utf-8"))
    validation = json.loads(args.internal_validation_cache.read_text(encoding="utf-8"))
    rgb = fit_platt_calibrator(training["records"], expert="rgb")
    signal = fit_platt_calibrator(training["records"], expert="signal")
    weight = fit_static_weight(training["records"], rgb, signal)
    evaluation = evaluate_candidates(validation["records"], rgb, signal, weight)
    provenance = {**training["provenance"], "signal_experiment_profile": "hackathon-v1", "signal_acceptance_scope": "issue-6-timeboxed-acceptance"}
    selected = evaluation["selected_fallback_type"]
    bundle = {
        "selected_fallback_type": selected,
        "rgb_calibrator": rgb, "signal_calibrator": signal, "static_weight": weight,
        "evaluation": evaluation, "provenance": provenance,
        "provisional_threshold": evaluation["candidates"][selected]["threshold_diagnostics"],
        "input_cache_bindings": {
            "fusion_training": {"records_sha256": training["records_sha256"], "file_sha256": hashlib.sha256(args.fusion_training_cache.read_bytes()).hexdigest()},
            "internal_validation": {"records_sha256": validation["records_sha256"], "file_sha256": hashlib.sha256(args.internal_validation_cache.read_bytes()).hexdigest()},
        },
        "schema_versions": {"matched_cache": training["cache_schema_version"], "calibration": "expert-calibration-v1", "static_weight": "static-fusion-weight-v1", "evaluation": "static-fusion-evaluation-v1"},
        "numeric_tolerances": {"rgb_logit": 0.00001, "calibration_solver": 1e-12},
        "artifact_paths": [],
    }
    completion = publish_bundle_atomic(args.output_dir, bundle)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
