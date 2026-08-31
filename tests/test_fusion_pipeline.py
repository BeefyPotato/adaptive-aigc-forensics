import copy
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import fusion_cli
import fusion_pipeline
from signal_maps import FEATURE_NAMES
from fusion_pipeline import (
    SELECTION_RULE, calibrated_logit, evaluate_candidates, fit_platt_calibrator,
    fit_static_weight, inspect_signal_handoff_archive, matched_record, publish_bundle_atomic, validate_bundle,
    validate_matched_records,
)


PROVENANCE = {
    "manifest_sha256": "a" * 64, "rgb_checkpoint_revision": "b" * 40,
    "rgb_preprocessing_version": "rgb-v1", "signal_checkpoint_revision": "signal-model-v1",
    "signal_normalization_revision": "signal-normalization-v1",
    "signal_feature_extraction_version": "signal-feature-v1", "corruption_version": "corruption-v1",
}
FROZEN_PROVENANCE = {
    **PROVENANCE,
    "signal_experiment_profile": "hackathon-v1",
    "signal_acceptance_scope": "issue-6-timeboxed-acceptance",
}


def records(split, sources=20, signal_help=False, provenance=PROVENANCE):
    rows = []
    conditions = [("clean", "none"), ("jpeg", "q"), ("blur", "s"), ("resize", "r"), ("noise", "n"), ("color", "c"), ("crop", "x")]
    for source in range(sources):
        label = source % 2
        for family, severity in conditions:
            variant = f"{split}-{source}-{family}"
            base = 2.0 if label else -2.0
            if signal_help and family == "noise" and source % 5 == 0:
                rgb_logit, signal_logit = -base, base * 2
            else:
                rgb_logit, signal_logit = base, base * .5
            common = {"source_id": f"{split}-{source}", "variant_id": variant, "split": split,
                      "authenticity_label": label, "condition_family": family, "severity": severity,
                      "materialized_sha256": f"{source:064x}"[-64:]}
            rows.append(matched_record({**common, "rgb_logit": rgb_logit}, {**common, "signal_logit": signal_logit}, provenance))
    return rows


class FusionPipelineTests(unittest.TestCase):
    def test_matched_cache_reader_rejects_an_unsupported_schema(self):
        reader = getattr(fusion_pipeline, "validate_matched_cache", lambda *args, **kwargs: None)
        with self.assertRaisesRegex(ValueError, "cache schema"):
            reader({"cache_schema_version": "stale"}, expected_split="fusion-training")

    def test_matched_cache_reader_rejects_a_stale_records_digest(self):
        training = records("fusion-training")
        cache = {
            "cache_schema_version": fusion_pipeline.MATCHED_CACHE_SCHEMA,
            "split": "fusion-training",
            "provenance": PROVENANCE,
            "records_sha256": "0" * 64,
            "records": training,
        }
        with self.assertRaisesRegex(ValueError, "records or digest"):
            fusion_pipeline.validate_matched_cache(cache, expected_split="fusion-training")

    def test_matched_cache_reader_rejects_cache_to_row_provenance_mismatch(self):
        training = records("fusion-training")
        cache = {
            "cache_schema_version": fusion_pipeline.MATCHED_CACHE_SCHEMA,
            "split": "fusion-training",
            "provenance": {**PROVENANCE, "rgb_checkpoint_revision": "c" * 40},
            "records_sha256": fusion_pipeline._sha256(training),
            "records": training,
        }
        with self.assertRaisesRegex(ValueError, "provenance"):
            fusion_pipeline.validate_matched_cache(cache, expected_split="fusion-training")

    def test_matched_cache_reader_binds_provenance_to_a_trusted_expectation(self):
        training = records("fusion-training")
        cache = {
            "cache_schema_version": fusion_pipeline.MATCHED_CACHE_SCHEMA,
            "split": "fusion-training",
            "provenance": PROVENANCE,
            "records_sha256": fusion_pipeline._sha256(training),
            "records": training,
        }
        self.assertIn(
            "expected_provenance",
            inspect.signature(fusion_pipeline.validate_matched_cache).parameters,
        )
        expected = {**PROVENANCE, "rgb_checkpoint_revision": "c" * 40}
        with self.assertRaisesRegex(ValueError, "trusted provenance"):
            fusion_pipeline.validate_matched_cache(
                cache,
                expected_split="fusion-training",
                expected_provenance=expected,
            )

    def test_matched_cache_builder_round_trips_a_complete_binding(self):
        training = records("fusion-training")
        builder = getattr(fusion_pipeline, "build_matched_cache", None)
        self.assertTrue(callable(builder))
        cache = builder(training, provenance=PROVENANCE, expected_split="fusion-training")
        self.assertEqual(cache["binding"]["records_sha256"], cache["records_sha256"])
        self.assertEqual(cache["binding"]["provenance_sha256"], fusion_pipeline._sha256(PROVENANCE))
        self.assertEqual(
            fusion_pipeline.validate_matched_cache(
                cache,
                expected_split="fusion-training",
                expected_provenance=PROVENANCE,
                expected_binding=cache["binding"],
            ),
            training,
        )

    def test_matched_cache_reader_rejects_a_relabelled_top_level_split(self):
        cache = fusion_pipeline.build_matched_cache(
            records("fusion-training"),
            provenance=PROVENANCE,
            expected_split="fusion-training",
        )
        cache["split"] = "sealed-test"
        with self.assertRaisesRegex(ValueError, "top-level split"):
            fusion_pipeline.validate_matched_cache(cache, expected_split="fusion-training")

    def test_legacy_matched_cache_migration_preserves_raw_records(self):
        legacy_rows = records("fusion-training")
        legacy = {
            "cache_schema_version": "matched-frozen-expert-logits-v1",
            "provenance": PROVENANCE,
            "records_sha256": fusion_pipeline._sha256(legacy_rows),
            "records": legacy_rows,
        }
        migrator = getattr(fusion_pipeline, "migrate_legacy_matched_cache", None)
        self.assertTrue(callable(migrator))
        migrated = migrator(
            legacy,
            expected_split="fusion-training",
            enriched_provenance=FROZEN_PROVENANCE,
        )
        self.assertEqual(migrated["records"], legacy_rows)
        self.assertEqual(migrated["records_sha256"], legacy["records_sha256"])
        self.assertEqual(
            migrated["provenance"]["signal_experiment_profile"],
            "hackathon-v1",
        )
        self.assertEqual(
            fusion_pipeline.validate_matched_cache(
                migrated,
                expected_split="fusion-training",
                expected_provenance=FROZEN_PROVENANCE,
                expected_binding=migrated["binding"],
            ),
            legacy_rows,
        )

    def test_fusion_cli_is_a_migration_adapter_and_does_not_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            signal_model = root / "signal-model.json"
            output = root / "v2"
            with (
                patch.object(
                    fusion_cli,
                    "migrate_static_fallback_generation",
                    create=True,
                ) as migrate,
                patch.object(fusion_cli, "fit_platt_calibrator", create=True) as fit,
                patch.object(sys, "argv", [
                    "fusion_cli.py",
                    "--legacy-generation-dir", str(legacy),
                    "--signal-model", str(signal_model),
                    "--output-dir", str(output),
                ]),
            ):
                fusion_cli.main()
            migrate.assert_called_once_with(legacy, signal_model, output)
            fit.assert_not_called()

    def test_generation_migration_rejects_a_stale_legacy_completion_marker_first(self):
        migrator = getattr(fusion_pipeline, "migrate_static_fallback_generation", None)
        self.assertTrue(callable(migrator))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            fallback = legacy / "fallback"
            fallback.mkdir(parents=True)
            bundle = {
                "bundle_schema_version": "static-fallback-bundle-v1",
                "bundle_revision": "static-fallback-bundle-v1-" + "a" * 64,
            }
            (fallback / "static-fallback-bundle.json").write_text(
                json.dumps(bundle), encoding="utf-8"
            )
            (fallback / "static-fallback.complete.json").write_text(
                json.dumps({
                    "completion_schema_version": "static-fallback-completion-v1",
                    "bundle_sha256": "0" * 64,
                    "bundle_revision": bundle["bundle_revision"],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy completion"):
                migrator(legacy, root / "signal-model.json", root / "v2")
            self.assertFalse((root / "v2").exists())

    def test_generation_migration_preserves_legacy_science_and_uses_validated_normalizers(self):
        signal_normalizer = {
            "schema_version": "signal-normalization-v1",
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * 26,
            "scale": [1.0] * 26,
        }
        signal_normalizer["normalization_revision"] = (
            "signal-normalization-v1-" + fusion_pipeline._sha256(signal_normalizer)
        )
        legacy_provenance = {
            **PROVENANCE,
            "signal_normalization_revision": signal_normalizer["normalization_revision"],
        }
        enriched_provenance = {
            **legacy_provenance,
            "signal_experiment_profile": "hackathon-v1",
            "signal_acceptance_scope": "issue-6-timeboxed-acceptance",
            "manifest_schema_version": "track5-manifest-v1",
            "shared_observation_preprocessing_version": "shared-preprocessing-v1",
            "rgb_checkpoint_sha256": "c" * 64,
            "rgb_score_direction": "positive-logit-means-ai-generated",
            "signal_representation_version": "signal-representation-v1",
            "signal_resolution": 384,
        }
        training_rows = records("fusion-training", provenance=legacy_provenance)
        validation_rows = records("internal-validation", provenance=legacy_provenance)
        legacy_caches = {}
        for split, rows in (
            ("fusion-training", training_rows),
            ("internal-validation", validation_rows),
        ):
            legacy_caches[split] = {
                "cache_schema_version": "matched-frozen-expert-logits-v1",
                "provenance": legacy_provenance,
                "records_sha256": fusion_pipeline._sha256(rows),
                "records": rows,
            }
        rgb = fit_platt_calibrator(training_rows, expert="rgb")
        signal = fit_platt_calibrator(training_rows, expert="signal")
        weight = fit_static_weight(training_rows, rgb, signal)
        evaluation = evaluate_candidates(validation_rows, rgb, signal, weight)
        rgb_normalizer = {
            "normalization_schema_version": "rgb-imagenet-normalization-v1",
            "preprocessing_version": PROVENANCE["rgb_preprocessing_version"],
            "channel_order": "rgb",
            "input_range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "scale": [0.229, 0.224, 0.225],
            "checkpoint_revision": PROVENANCE["rgb_checkpoint_revision"],
        }
        rgb_normalizer["normalization_revision"] = (
            "rgb-normalization-v1-" + fusion_pipeline._sha256(rgb_normalizer)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            fallback = legacy / "fallback"
            fallback.mkdir(parents=True)
            cache_bytes = {}
            for split, cache in legacy_caches.items():
                name = f"matched-{split}-logits.json"
                cache_bytes[split] = fusion_pipeline._artifact_bytes(cache)
                (legacy / name).write_bytes(cache_bytes[split])
            legacy_bundle = {
                "selected_fallback_type": evaluation["selected_fallback_type"],
                "rgb_calibrator": rgb,
                "signal_calibrator": signal,
                "static_weight": weight,
                "evaluation": evaluation,
                "provenance": {
                    **legacy_provenance,
                    "signal_experiment_profile": "hackathon-v1",
                    "signal_acceptance_scope": "issue-6-timeboxed-acceptance",
                },
                "provisional_threshold": evaluation["candidates"][evaluation["selected_fallback_type"]]["threshold_diagnostics"],
                "input_cache_bindings": {
                    split.replace("-", "_"): {
                        "records_sha256": cache["records_sha256"],
                        "file_sha256": hashlib.sha256(cache_bytes[split]).hexdigest(),
                    }
                    for split, cache in legacy_caches.items()
                },
                "schema_versions": {},
                "numeric_tolerances": {},
                "artifact_paths": [],
                "bundle_schema_version": "static-fallback-bundle-v1",
                "selection_rule": SELECTION_RULE,
            }
            legacy_bundle["bundle_revision"] = (
                "static-fallback-bundle-v1-" + fusion_pipeline._sha256(legacy_bundle)
            )
            bundle_bytes = fusion_pipeline._artifact_bytes(legacy_bundle)
            (fallback / "static-fallback-bundle.json").write_bytes(bundle_bytes)
            marker = {
                "completion_schema_version": "static-fallback-completion-v1",
                "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
                "bundle_revision": legacy_bundle["bundle_revision"],
            }
            marker_bytes = fusion_pipeline._artifact_bytes(marker)
            (fallback / "static-fallback.complete.json").write_bytes(marker_bytes)
            signal_model = root / "signal-model.json"
            signal_model.write_text("{}", encoding="utf-8")
            expected_completion = {"generation_revision": "static-fallback-generation-v2-test"}
            with (
                patch.object(
                    fusion_pipeline,
                    "_validated_signal_model_handoff",
                    return_value={
                        "provenance": enriched_provenance,
                        "normalization": signal_normalizer,
                        "model_file_sha256": "d" * 64,
                    },
                    create=True,
                ),
                patch.object(
                    fusion_pipeline,
                    "_build_rgb_normalizer",
                    return_value=rgb_normalizer,
                    create=True,
                ),
                patch.object(
                    fusion_pipeline,
                    "_validate_issue7_partition_counts",
                    create=True,
                ),
                patch.object(
                    fusion_pipeline,
                    "publish_corrected_generation",
                    return_value=expected_completion,
                ) as publish,
            ):
                result = fusion_pipeline.migrate_static_fallback_generation(
                    legacy,
                    signal_model,
                    root / "v2",
                    expected_legacy_bindings={
                        "bundle_sha256": marker["bundle_sha256"],
                        "bundle_revision": legacy_bundle["bundle_revision"],
                        "fusion_training_file_sha256": hashlib.sha256(cache_bytes["fusion-training"]).hexdigest(),
                        "internal_validation_file_sha256": hashlib.sha256(cache_bytes["internal-validation"]).hexdigest(),
                        "legacy_completion_file_sha256": hashlib.sha256(marker_bytes).hexdigest(),
                        "signal_model_file_sha256": "d" * 64,
                    },
                )
            self.assertEqual(result, expected_completion)
            migrated_training = publish.call_args.kwargs["fusion_training_cache"]
            migrated_validation = publish.call_args.kwargs["internal_validation_cache"]
            self.assertEqual(migrated_training["records"], training_rows)
            self.assertEqual(migrated_validation["records"], validation_rows)
            self.assertEqual(
                publish.call_args.kwargs["legacy_bundle"],
                legacy_bundle,
            )
            self.assertEqual(
                publish.call_args.kwargs["signal_normalizer"],
                signal_normalizer,
            )
            self.assertEqual(
                publish.call_args.kwargs["input_cache_bindings"]["legacy_bundle"]["file_sha256"],
                marker["bundle_sha256"],
            )

    def test_calibrated_cache_persists_raw_and_calibrated_logits(self):
        training_rows = records("fusion-training")
        training = fusion_pipeline.build_matched_cache(
            training_rows, provenance=PROVENANCE, expected_split="fusion-training"
        )
        rgb = fit_platt_calibrator(training_rows, expert="rgb")
        signal = fit_platt_calibrator(training_rows, expert="signal")
        weight = fit_static_weight(training_rows, rgb, signal)
        builder = getattr(fusion_pipeline, "build_calibrated_cache", None)
        self.assertTrue(callable(builder))
        cache = builder(
            training,
            expected_split="fusion-training",
            input_file_sha256="d" * 64,
            rgb_calibrator=rgb,
            signal_calibrator=signal,
            static_weight=weight,
            selected_fallback_type="learned-static-fusion",
        )
        first = cache["records"][0]
        self.assertEqual(first["rgb_logit"], training_rows[0]["rgb_logit"])
        self.assertEqual(
            first["rgb_calibrated_logit"],
            calibrated_logit(training_rows[0]["rgb_logit"], rgb),
        )
        self.assertEqual(
            first["signal_calibrated_logit"],
            calibrated_logit(training_rows[0]["signal_logit"], signal),
        )
        self.assertIn("selected_fallback_probability", first)
        self.assertNotIn("selected_fallback_pred", first)

    def test_calibrated_cache_reader_round_trips_derived_records(self):
        training_rows = records("fusion-training")
        matched = fusion_pipeline.build_matched_cache(
            training_rows, provenance=PROVENANCE, expected_split="fusion-training"
        )
        rgb = fit_platt_calibrator(training_rows, expert="rgb")
        signal = fit_platt_calibrator(training_rows, expert="signal")
        weight = fit_static_weight(training_rows, rgb, signal)
        calibrated = fusion_pipeline.build_calibrated_cache(
            matched,
            expected_split="fusion-training",
            input_file_sha256="d" * 64,
            rgb_calibrator=rgb,
            signal_calibrator=signal,
            static_weight=weight,
            selected_fallback_type="learned-static-fusion",
        )
        reader = getattr(fusion_pipeline, "validate_calibrated_cache", None)
        self.assertTrue(callable(reader))
        self.assertEqual(
            reader(
                calibrated,
                matched_cache=matched,
                expected_split="fusion-training",
                input_file_sha256="d" * 64,
                rgb_calibrator=rgb,
                signal_calibrator=signal,
                static_weight=weight,
                selected_fallback_type="learned-static-fusion",
            ),
            calibrated["records"],
        )

    def test_corrected_bundle_embeds_normalizers_threshold_and_artifact_bindings(self):
        training_rows = records("fusion-training")
        validation_rows = records("internal-validation")
        rgb = fit_platt_calibrator(training_rows, expert="rgb")
        signal = fit_platt_calibrator(training_rows, expert="signal")
        weight = fit_static_weight(training_rows, rgb, signal)
        evaluation = evaluate_candidates(validation_rows, rgb, signal, weight)
        legacy = {
            "selected_fallback_type": evaluation["selected_fallback_type"],
            "rgb_calibrator": rgb,
            "signal_calibrator": signal,
            "static_weight": weight,
            "evaluation": evaluation,
            "provenance": PROVENANCE,
            "provisional_threshold": evaluation["candidates"][evaluation["selected_fallback_type"]]["threshold_diagnostics"],
            "input_cache_bindings": {},
            "schema_versions": {
                "matched_cache": "matched-frozen-expert-logits-v1",
                "calibration": "expert-calibration-v1",
                "static_weight": "static-fusion-weight-v1",
                "evaluation": "static-fusion-evaluation-v1",
            },
            "numeric_tolerances": {},
            "artifact_paths": [],
            "bundle_schema_version": "static-fallback-bundle-v1",
            "selection_rule": SELECTION_RULE,
        }
        legacy["bundle_revision"] = "static-fallback-bundle-v1-" + fusion_pipeline._sha256(legacy)
        rgb_normalizer = {
            "normalization_schema_version": "rgb-imagenet-normalization-v1",
            "channel_order": "rgb",
            "input_range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "scale": [0.229, 0.224, 0.225],
        }
        rgb_normalizer["normalization_revision"] = "rgb-normalization-v1-" + fusion_pipeline._sha256(rgb_normalizer)
        signal_normalizer = {
            "schema_version": "signal-normalization-v1",
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * 26,
            "scale": [1.0] * 26,
        }
        signal_normalizer["normalization_revision"] = "signal-normalization-v1-" + fusion_pipeline._sha256(signal_normalizer)
        corrected_provenance = {
            **FROZEN_PROVENANCE,
            "signal_normalization_revision": signal_normalizer["normalization_revision"],
        }
        input_bindings = {
            "fusion_training": {"path": "legacy/matched-fusion-training-logits.json", "file_sha256": "a" * 64, "records_sha256": fusion_pipeline._sha256(training_rows)},
            "internal_validation": {"path": "legacy/matched-internal-validation-logits.json", "file_sha256": "b" * 64, "records_sha256": fusion_pipeline._sha256(validation_rows)},
            "legacy_bundle": {"path": "legacy/fallback/static-fallback-bundle.json", "file_sha256": "c" * 64},
            "legacy_completion": {"path": "legacy/fallback/static-fallback.complete.json", "file_sha256": "d" * 64},
            "signal_model": {"path": "upstream/signal-model.json", "file_sha256": "e" * 64},
        }
        artifact_bindings = {
            "matched_fusion_training": {"path": "matched-fusion-training-logits.json", "file_sha256": "1" * 64, "records_sha256": fusion_pipeline._sha256(training_rows)},
            "matched_internal_validation": {"path": "matched-internal-validation-logits.json", "file_sha256": "2" * 64, "records_sha256": fusion_pipeline._sha256(validation_rows)},
            "calibrated_fusion_training": {"path": "calibrated-fusion-training-logits.json", "file_sha256": "3" * 64},
            "calibrated_internal_validation": {"path": "calibrated-internal-validation-logits.json", "file_sha256": "4" * 64},
        }
        builder = getattr(fusion_pipeline, "build_corrected_bundle", None)
        self.assertTrue(callable(builder))
        bundle = builder(
            legacy,
            provenance=corrected_provenance,
            rgb_normalizer=rgb_normalizer,
            signal_normalizer=signal_normalizer,
            input_cache_bindings=input_bindings,
            artifact_bindings=artifact_bindings,
        )
        self.assertEqual(bundle["bundle_schema_version"], "static-fallback-bundle-v2")
        self.assertEqual(bundle["rgb_normalizer"], rgb_normalizer)
        self.assertEqual(bundle["signal_normalizer"], signal_normalizer)
        self.assertEqual(bundle["provisional_threshold"], legacy["provisional_threshold"])
        self.assertEqual(bundle["artifact_bindings"], artifact_bindings)
        self.assertEqual(bundle["legacy_schema_versions"], legacy["schema_versions"])
        self.assertEqual(bundle["schema_versions"]["matched_cache"], fusion_pipeline.MATCHED_CACHE_SCHEMA)
        self.assertEqual(
            bundle["scientific_bindings"]["fusion_training_records_sha256"],
            fusion_pipeline._sha256(training_rows),
        )
        self.assertEqual(
            bundle["scientific_bindings"]["internal_validation_records_sha256"],
            fusion_pipeline._sha256(validation_rows),
        )
        self.assertEqual(validate_bundle(bundle), bundle)
        tampered = copy.deepcopy(bundle)
        tampered["rgb_normalizer"]["mean"][0] = 0.5
        normalizer_identity = dict(tampered["rgb_normalizer"])
        normalizer_identity.pop("normalization_revision")
        tampered["rgb_normalizer"]["normalization_revision"] = (
            "rgb-normalization-v1-" + fusion_pipeline._sha256(normalizer_identity)
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "RGB normalizer"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        tampered["signal_normalizer"]["mean"][0] = 0.5
        normalizer_identity = dict(tampered["signal_normalizer"])
        normalizer_identity.pop("normalization_revision")
        tampered["signal_normalizer"]["normalization_revision"] = (
            "signal-normalization-v1-" + fusion_pipeline._sha256(normalizer_identity)
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "signal normalizer"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        tampered["input_cache_bindings"].pop("signal_model")
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "input cache bindings are incomplete"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        tampered["rgb_calibrator"]["expert"] = "signal"
        calibrator_identity = dict(tampered["rgb_calibrator"])
        calibrator_identity.pop("calibrator_revision")
        tampered["rgb_calibrator"]["calibrator_revision"] = (
            "expert-calibrator-v1-" + fusion_pipeline._sha256(calibrator_identity)
        )
        tampered["static_weight"]["rgb_calibrator_revision"] = tampered["rgb_calibrator"]["calibrator_revision"]
        static_identity = dict(tampered["static_weight"])
        static_identity.pop("static_weight_revision")
        tampered["static_weight"]["static_weight_revision"] = (
            "static-fusion-weight-v1-" + fusion_pipeline._sha256(static_identity)
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "RGB calibrator contract"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        tampered["static_weight"]["rgb_weight"] = 0.5
        tampered["static_weight"]["signal_weight"] = 0.6
        static_identity = dict(tampered["static_weight"])
        static_identity.pop("static_weight_revision")
        tampered["static_weight"]["static_weight_revision"] = (
            "static-fusion-weight-v1-" + fusion_pipeline._sha256(static_identity)
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "static weight contract"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        selected = tampered["selected_fallback_type"]
        tampered["provisional_threshold"]["sensitivity"] = 1.2
        tampered["evaluation"]["candidates"][selected]["threshold_diagnostics"] = copy.deepcopy(
            tampered["provisional_threshold"]
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "threshold diagnostics"):
            validate_bundle(tampered)
        tampered = copy.deepcopy(bundle)
        relabelled = (
            "learned-static-fusion"
            if tampered["selected_fallback_type"] == "calibrated-rgb-only"
            else "calibrated-rgb-only"
        )
        tampered["selected_fallback_type"] = relabelled
        tampered["evaluation"]["selected_fallback_type"] = relabelled
        tampered["scientific_bindings"]["selected_fallback_type"] = relabelled
        tampered["provisional_threshold"] = copy.deepcopy(
            tampered["evaluation"]["candidates"][relabelled]["threshold_diagnostics"]
        )
        tampered_identity = dict(tampered)
        tampered_identity.pop("bundle_revision")
        tampered["bundle_revision"] = "static-fallback-bundle-v2-" + fusion_pipeline._sha256(tampered_identity)
        with self.assertRaisesRegex(ValueError, "selection does not reproduce"):
            validate_bundle(tampered)

    @patch.object(fusion_pipeline, "_validate_issue7_partition_counts")
    def test_corrected_generation_publishes_one_complete_artifact_set(self, validate_partition_counts):
        rgb_normalizer = {
            "normalization_schema_version": "rgb-imagenet-normalization-v1",
            "preprocessing_version": PROVENANCE["rgb_preprocessing_version"],
            "channel_order": "rgb",
            "input_range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "scale": [0.229, 0.224, 0.225],
            "checkpoint_revision": PROVENANCE["rgb_checkpoint_revision"],
        }
        rgb_normalizer["normalization_revision"] = "rgb-normalization-v1-" + fusion_pipeline._sha256(rgb_normalizer)
        signal_normalizer = {
            "schema_version": "signal-normalization-v1",
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * 26,
            "scale": [1.0] * 26,
        }
        signal_normalizer["normalization_revision"] = "signal-normalization-v1-" + fusion_pipeline._sha256(signal_normalizer)
        provenance = {
            **FROZEN_PROVENANCE,
            "signal_normalization_revision": signal_normalizer["normalization_revision"],
        }
        training_rows = records("fusion-training", provenance=provenance)
        validation_rows = records("internal-validation", provenance=provenance)
        training = fusion_pipeline.build_matched_cache(
            training_rows, provenance=provenance, expected_split="fusion-training"
        )
        validation = fusion_pipeline.build_matched_cache(
            validation_rows, provenance=provenance, expected_split="internal-validation"
        )
        rgb = fit_platt_calibrator(training_rows, expert="rgb")
        signal = fit_platt_calibrator(training_rows, expert="signal")
        weight = fit_static_weight(training_rows, rgb, signal)
        evaluation = evaluate_candidates(validation_rows, rgb, signal, weight)
        legacy = {
            "selected_fallback_type": evaluation["selected_fallback_type"],
            "rgb_calibrator": rgb,
            "signal_calibrator": signal,
            "static_weight": weight,
            "evaluation": evaluation,
            "provenance": PROVENANCE,
            "provisional_threshold": evaluation["candidates"][evaluation["selected_fallback_type"]]["threshold_diagnostics"],
            "input_cache_bindings": {},
            "schema_versions": {},
            "numeric_tolerances": {},
            "artifact_paths": [],
            "bundle_schema_version": "static-fallback-bundle-v1",
            "selection_rule": SELECTION_RULE,
        }
        legacy["bundle_revision"] = "static-fallback-bundle-v1-" + fusion_pipeline._sha256(legacy)
        publisher = getattr(fusion_pipeline, "publish_corrected_generation", None)
        self.assertTrue(callable(publisher))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "generation"
            completion = publisher(
                target,
                legacy_bundle=legacy,
                fusion_training_cache=training,
                internal_validation_cache=validation,
                provenance=provenance,
                rgb_normalizer=rgb_normalizer,
                signal_normalizer=signal_normalizer,
                input_cache_bindings={
                    "fusion_training": {
                        "path": "legacy/matched-fusion-training-logits.json",
                        "cache_schema_version": "matched-frozen-expert-logits-v1",
                        "file_sha256": "a" * 64,
                        "records_sha256": training["records_sha256"],
                    },
                    "internal_validation": {
                        "path": "legacy/matched-internal-validation-logits.json",
                        "cache_schema_version": "matched-frozen-expert-logits-v1",
                        "file_sha256": "b" * 64,
                        "records_sha256": validation["records_sha256"],
                    },
                    "legacy_bundle": {
                        "path": "legacy/fallback/static-fallback-bundle.json",
                        "file_sha256": "c" * 64,
                    },
                    "legacy_completion": {
                        "path": "legacy/fallback/static-fallback.complete.json",
                        "file_sha256": "d" * 64,
                    },
                    "signal_model": {
                        "path": "upstream/signal-model.json",
                        "file_sha256": "e" * 64,
                    },
                },
            )
            self.assertEqual(completion["completion_schema_version"], "static-fallback-completion-v2")
            self.assertEqual(
                {path.name for path in target.iterdir()},
                {
                    "matched-fusion-training-logits.json",
                    "matched-internal-validation-logits.json",
                    "calibrated-fusion-training-logits.json",
                    "calibrated-internal-validation-logits.json",
                    "static-fallback-bundle.json",
                    "static-fallback.complete.json",
                },
            )
            self.assertEqual(
                set(completion["artifacts"]),
                {
                    "matched-fusion-training-logits.json",
                    "matched-internal-validation-logits.json",
                    "calibrated-fusion-training-logits.json",
                    "calibrated-internal-validation-logits.json",
                    "static-fallback-bundle.json",
                },
            )
            reader = getattr(fusion_pipeline, "read_static_fallback_generation", None)
            self.assertTrue(callable(reader))
            loaded = reader(
                target,
                expected_provenance=provenance,
                expected_generation_revision=completion["generation_revision"],
            )
            self.assertEqual(loaded["completion"], completion)
            self.assertEqual(
                loaded["bundle"]["bundle_revision"],
                completion["bundle_revision"],
            )
            self.assertTrue(validate_partition_counts.called)
            self.assertEqual(
                publisher(
                    target,
                    legacy_bundle=legacy,
                    fusion_training_cache=training,
                    internal_validation_cache=validation,
                    provenance=provenance,
                    rgb_normalizer=rgb_normalizer,
                    signal_normalizer=signal_normalizer,
                    input_cache_bindings={
                        "fusion_training": {
                            "path": "legacy/matched-fusion-training-logits.json",
                            "cache_schema_version": "matched-frozen-expert-logits-v1",
                            "file_sha256": "a" * 64,
                            "records_sha256": training["records_sha256"],
                        },
                        "internal_validation": {
                            "path": "legacy/matched-internal-validation-logits.json",
                            "cache_schema_version": "matched-frozen-expert-logits-v1",
                            "file_sha256": "b" * 64,
                            "records_sha256": validation["records_sha256"],
                        },
                        "legacy_bundle": {"path": "legacy/fallback/static-fallback-bundle.json", "file_sha256": "c" * 64},
                        "legacy_completion": {"path": "legacy/fallback/static-fallback.complete.json", "file_sha256": "d" * 64},
                        "signal_model": {"path": "upstream/signal-model.json", "file_sha256": "e" * 64},
                    },
                ),
                completion,
            )
            interrupted = Path(directory) / "interrupted-generation"
            with patch.object(
                fusion_pipeline,
                "read_static_fallback_generation",
                side_effect=ValueError("staging generation is invalid"),
            ):
                with self.assertRaisesRegex(ValueError, "staging generation"):
                    publisher(
                        interrupted,
                        legacy_bundle=legacy,
                        fusion_training_cache=training,
                        internal_validation_cache=validation,
                        provenance=provenance,
                        rgb_normalizer=rgb_normalizer,
                        signal_normalizer=signal_normalizer,
                        input_cache_bindings={
                            "fusion_training": {"path": "legacy/matched-fusion-training-logits.json", "file_sha256": "a" * 64},
                            "internal_validation": {"path": "legacy/matched-internal-validation-logits.json", "file_sha256": "b" * 64},
                            "legacy_bundle": {"path": "legacy/fallback/static-fallback-bundle.json", "file_sha256": "c" * 64},
                            "legacy_completion": {"path": "legacy/fallback/static-fallback.complete.json", "file_sha256": "d" * 64},
                            "signal_model": {"path": "upstream/signal-model.json", "file_sha256": "e" * 64},
                        },
                    )
            self.assertFalse(interrupted.exists())

    def test_archive_handoff_validation_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape", b"x")
            with self.assertRaisesRegex(ValueError, "traversal"):
                inspect_signal_handoff_archive(archive)

    def test_partition_enforcement_and_membership(self):
        training = records("fusion-training")
        self.assertEqual(len(validate_matched_records(training, expected_split="fusion-training")), 140)
        bad = copy.deepcopy(training); bad[0]["split"] = "internal-validation"
        with self.assertRaisesRegex(ValueError, "only fusion-training"):
            validate_matched_records(bad, expected_split="fusion-training")
        bad = copy.deepcopy(training); bad[1]["variant_id"] = bad[0]["variant_id"]
        with self.assertRaisesRegex(ValueError, "repeat variant_id"):
            validate_matched_records(bad, expected_split="fusion-training")

    def test_calibrators_and_weight_are_reproducible_and_constrained(self):
        training = records("fusion-training")
        rgb1 = fit_platt_calibrator(training, expert="rgb")
        rgb2 = fit_platt_calibrator(training, expert="rgb")
        signal = fit_platt_calibrator(training, expert="signal")
        self.assertEqual(rgb1, rgb2)
        weight1 = fit_static_weight(training, rgb1, signal)
        weight2 = fit_static_weight(training, rgb1, signal)
        self.assertEqual(weight1, weight2)
        self.assertGreaterEqual(weight1["rgb_weight"], 0); self.assertLessEqual(weight1["rgb_weight"], 1)
        self.assertAlmostEqual(weight1["rgb_weight"] + weight1["signal_weight"], 1)

    def test_equal_fusion_brier_complementarity_and_rgb_fallback(self):
        training = records("fusion-training")
        rgb = fit_platt_calibrator(training, expert="rgb")
        signal = fit_platt_calibrator(training, expert="signal")
        weight = fit_static_weight(training, rgb, signal)
        evaluation = evaluate_candidates(records("internal-validation"), rgb, signal, weight)
        self.assertIn("brier_score", evaluation["candidates"]["equal-50-50-calibrated-logit-fusion"])
        self.assertEqual(evaluation["selected_fallback_type"], "calibrated-rgb-only")
        self.assertEqual(evaluation["selection_rule"], SELECTION_RULE)
        first = records("internal-validation", sources=2)[0]
        expected = .5 * calibrated_logit(first["rgb_logit"], rgb) + .5 * calibrated_logit(first["signal_logit"], signal)
        self.assertIsInstance(expected, float)

    def test_stale_forbidden_and_nonfinite_records_fail(self):
        training = records("fusion-training")
        for field, value, message in (("split", "sealed-internal-test", "only fusion-training"), ("rgb_logit", float("nan"), "finite"), ("rgb_checkpoint_revision", "stale", "cache key")):
            bad = copy.deepcopy(training); bad[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_matched_records(bad, expected_split="fusion-training")

    def test_atomic_completion_is_written_last_and_bundle_rejects_traversal(self):
        training = records("fusion-training"); validation = records("internal-validation")
        rgb = fit_platt_calibrator(training, expert="rgb"); signal = fit_platt_calibrator(training, expert="signal")
        weight = fit_static_weight(training, rgb, signal); evaluation = evaluate_candidates(validation, rgb, signal, weight)
        bundle = {"selected_fallback_type": evaluation["selected_fallback_type"], "rgb_calibrator": rgb,
                  "signal_calibrator": signal, "static_weight": weight, "evaluation": evaluation,
                  "provenance": PROVENANCE, "input_cache_bindings": {}, "numeric_tolerances": {}, "artifact_paths": []}
        with tempfile.TemporaryDirectory() as directory:
            completion = publish_bundle_atomic(Path(directory), bundle)
            self.assertTrue((Path(directory) / "static-fallback-bundle.json").is_file())
            self.assertTrue((Path(directory) / "static-fallback.complete.json").is_file())
            published = __import__("json").loads((Path(directory) / "static-fallback-bundle.json").read_text())
            validate_bundle(published)
            published["artifact_paths"] = ["../escape"]
            with self.assertRaisesRegex(ValueError, "traversal"):
                validate_bundle(published)


if __name__ == "__main__":
    unittest.main()
