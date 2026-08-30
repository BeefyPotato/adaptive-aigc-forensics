import copy
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import signal_pipeline
from signal_expert import FEATURE_NAMES
from signal_pipeline import (
    FEATURE_CACHE_SCHEMA_VERSION,
    FEATURE_EXTRACTION_VERSION,
    MATERIALIZED_ENCODING,
    _build_plan,
    _file_sha256,
    _implementation_file_sha256,
    _feature_record,
    _feature_extraction_metadata,
    _extract_phase_features,
    _require_current_feature_extraction_snapshot,
    _sha256,
    _validate_materialized_shard,
    _validate_existing_run_marker,
    _validate_plan,
    _write_json,
    run_signal_experiment,
    validate_signal_feature_cache,
    validate_signal_logit_cache,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _installed_sharp_versions():
    completed = subprocess.run(
        [
            "node",
            "-e",
            "const sharp=require('sharp');process.stdout.write(JSON.stringify(sharp.versions));",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


SHARP_VERSIONS = _installed_sharp_versions()


CONDITIONS = [
    ("clean", "clean", {}),
    *[("jpeg", f"quality-{quality}", {"quality": quality, "chroma_subsampling": "4:2:0"}) for quality in (90, 70, 50, 30)],
    *[("blur", f"sigma-{sigma}", {"sigma": sigma}) for sigma in (0.5, 1, 2)],
    *[("resize", f"factor-{factor}", {"factor": factor, "down_kernel": "lanczos3", "up_kernel": "cubic"}) for factor in (0.5, 0.25)],
    *[("noise", f"sigma-{sigma}", {"sigma": sigma, "color_space": "rgb-0-1"}) for sigma in (0.02, 0.05, 0.1)],
    *[("color", f"{property_}-{factor}", {"property": property_, "factor": factor}) for property_ in ("brightness", "contrast", "saturation") for factor in (0.8, 1.2)],
    ("crop", "center-0.8", {"retained_fraction": 0.8, "position": "center", "restoration_kernel": "cubic"}),
]


def _variant_id(source_id, family, severity, parameters):
    identity = {
        "artifact_schema_version": "artifact-v1",
        "condition_family": family,
        "corruption_parameters": parameters,
        "corruption_seed": 23,
        "preprocessing_version": "shared-preprocessing-v1",
        "severity": severity,
        "source_id": source_id,
        "transform_implementation_version": "track5-corruption-v1+sharp-0.35.4",
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"variant-v1-{hashlib.sha256(encoded).hexdigest()}"


def _write_fixture(root: Path) -> Path:
    sources = []
    specifications = (
        ("train-real", "expert-training", 0),
        ("train-fake", "expert-training", 1),
        ("validation-real", "internal-validation", 0),
        ("validation-fake", "internal-validation", 1),
    )
    for index, (source_id, split, label) in enumerate(specifications):
        yy, xx = np.indices((12, 16))
        rgb = np.stack([
            (xx * (index + 3) * 11) % 256,
            (yy * (index + 5) * 13) % 256,
            ((xx + yy) * (index + 7) * 5) % 256,
        ], axis=2).astype(np.uint8)
        image_path = f"images/{source_id}.png"
        destination = root / image_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(destination)
        sources.append({
            "source_id": source_id,
            "image_path": image_path,
            "authenticity_label": label,
            "split": split,
            "width": 16,
            "height": 12,
            "exact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    observations = []
    for source in sources:
        for family, severity, parameters in CONDITIONS:
            observations.append({
                "observation_schema_version": "track5-observation-v1",
                **source,
                "variant_id": _variant_id(source["source_id"], family, severity, parameters),
                "condition_family": family,
                "severity": severity,
                "corruption_parameters": parameters,
                "corruption_seed": 23,
                "transform_implementation_version": "track5-corruption-v1+sharp-0.35.4",
            })
    manifest = {
        "manifest_schema_version": "track5-manifest-v1",
        "source_contract_version": "track5-source-v1",
        "observation_contract_version": "track5-observation-v1",
        "selection_contract_version": "track5-source-selection-v2",
        "condition_matrix_version": "track5-condition-matrix-v1",
        "sampler_contract_version": "track5-balanced-sampler-v1",
        "dataset": {"name": "SID_Set", "revision": "fixture-revision"},
        "selection": {"split_seed": 17},
        "corruption": {
            "root_seed": 23,
            "preprocessing_version": "shared-preprocessing-v1",
            "artifact_schema_version": "artifact-v1",
            "transform_implementation_version": "track5-corruption-v1+sharp-0.35.4",
            "sharp_version": SHARP_VERSIONS["sharp"],
            "libvips_version": SHARP_VERSIONS["vips"],
            "condition_count_per_source": 20,
        },
        "organizer_demonstration_policy": {"usage": "evaluation-only"},
        "sources": sources,
        "observations": observations,
        "leakage_audit": {
            "audit_schema_version": "track5-leakage-audit-v1",
            "status": "passed",
        },
    }
    path = root / "track5-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _rehash_plan(value):
    for phase in value["phases"]:
        for shard in phase["shards"]:
            shard_identity = {
                key: item
                for key, item in shard.items()
                if key not in {"plan_sha256", "shard_sha256"}
            }
            shard["shard_sha256"] = _sha256(shard_identity)
    plan_identity = {
        key: item for key, item in value.items() if key != "plan_sha256"
    }
    plan_identity["phases"] = [
        {
            **{key: item for key, item in phase.items() if key != "shards"},
            "shards": [
                {key: item for key, item in shard.items() if key != "plan_sha256"}
                for shard in phase["shards"]
            ],
        }
        for phase in value["phases"]
    ]
    value["plan_sha256"] = _sha256(plan_identity)
    for phase in value["phases"]:
        for shard in phase["shards"]:
            shard["plan_sha256"] = value["plan_sha256"]
    return value


class SignalPipelineTests(unittest.TestCase):
    def test_atomic_json_writer_never_reuses_a_preplanted_predictable_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "outside-sentinel.txt"
            sentinel.write_text("must remain unchanged", encoding="utf-8")
            legacy_temporary = root / "artifact.json.tmp"
            os.link(sentinel, legacy_temporary)

            artifact = root / "artifact.json"
            _write_json(artifact, {"status": "complete"})

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "must remain unchanged"
            )
            self.assertEqual(
                legacy_temporary.read_text(encoding="utf-8"), "must remain unchanged"
            )
            self.assertEqual(
                json.loads(artifact.read_text(encoding="utf-8")),
                {"status": "complete"},
            )

    def test_feature_extraction_rejects_redirected_managed_roots_without_touching_targets(self):
        phase = {
            "phase": "expert-training",
            "shards": [
                {
                    "phase": "expert-training",
                    "index": 0,
                    "count": 1,
                    "raw_byte_estimate": 1,
                    "shard_sha256": "a" * 64,
                    "variant_set_digest": "b" * 64,
                }
            ],
        }

        for redirected_name in (".signal-materialized", "signal-feature-shards"):
            with self.subTest(redirected_name=redirected_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                outside = root / "outside"
                output.mkdir()
                outside.mkdir()
                sentinel = outside / "must-survive.txt"
                sentinel.write_text("sentinel", encoding="utf-8")
                if redirected_name == ".signal-materialized":
                    external_shard = outside / "expert-training-00000"
                    external_shard.mkdir()
                    sentinel = external_shard / "must-survive.txt"
                    sentinel.write_text("sentinel", encoding="utf-8")
                subprocess.run(
                    [
                        "node",
                        "-e",
                        (
                            "require('node:fs').symlinkSync(process.argv[1],process.argv[2],"
                            "process.platform==='win32'?'junction':'dir')"
                        ),
                        str(outside),
                        str(output / redirected_name),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                with patch("signal_pipeline._run_node") as run_node, self.assertRaisesRegex(
                    ValueError, "managed output.*redirected|outside.*output"
                ):
                    _extract_phase_features(
                        phase,
                        dataset_root=root,
                        output_directory=output,
                        manifest_metadata={},
                        plan_sha256="c" * 64,
                        resolution=224,
                        node_binary="node",
                        feature_extraction=_feature_extraction_metadata(224),
                    )
                run_node.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel")
                self.assertFalse(
                    (outside / "expert-training-00000.plan.json").exists(),
                    "managed shard plans must not be written through a redirected cache root",
                )

    def test_existing_run_marker_requires_the_exact_contained_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "signal-plan.json",
                "signal-training-features.json",
                "signal-validation-features.json",
                "signal-normalization.json",
                "signal-model.json",
                "signal-validation-logits.json",
                "signal-internal-validation-metrics.json",
            )
            for name in names:
                (root / name).write_text(name, encoding="utf-8")
            marker = {
                "run_schema_version": "signal-experiment-run-v1",
                "manifest_metadata": {"manifest": "fixture"},
                "plan_sha256": "a" * 64,
                "epochs": 1,
                "runtime_versions": {"python": "fixture"},
                "implementation_sha256": {"pipeline": "b" * 64},
                "implementation_hash_contract_version": "utf8-lf-normalized-sha256-v1",
                "artifact_sha256": {
                    name: _file_sha256(root / name) for name in names
                },
            }
            marker_path = root / "signal-run.json"

            def validate(value):
                marker_path.write_text(json.dumps(value), encoding="utf-8")
                _validate_existing_run_marker(
                    marker_path,
                    manifest_metadata={"manifest": "fixture"},
                    plan_sha256="a" * 64,
                    requested={"epochs": 1},
                    runtime_versions={"python": "fixture"},
                    implementation_sha256={"pipeline": "b" * 64},
                )

            subset = copy.deepcopy(marker)
            subset["artifact_sha256"] = {
                "signal-plan.json": subset["artifact_sha256"]["signal-plan.json"]
            }
            with self.assertRaisesRegex(ValueError, "artifact.*set|checksums"):
                validate(subset)

            outside = root.parent / f"{root.name}-outside.json"
            outside.write_text("outside", encoding="utf-8")
            traversal = copy.deepcopy(marker)
            traversal["artifact_sha256"] = {
                "../" + outside.name: _file_sha256(outside)
            }
            with self.assertRaisesRegex(ValueError, "artifact.*set|contained|checksums"):
                validate(traversal)

            malformed = copy.deepcopy(marker)
            malformed["artifact_sha256"]["signal-plan.json"] = ["a"] * 64
            with self.assertRaisesRegex(ValueError, "SHA-256|checksums"):
                validate(malformed)

            boolean_epoch = copy.deepcopy(marker)
            boolean_epoch["epochs"] = True
            with self.assertRaisesRegex(ValueError, "incompatible epochs"):
                validate(boolean_epoch)

    def test_python_plan_and_materialized_contracts_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_fixture(root)
            plan = _build_plan(
                manifest_path,
                root / "signal-plan.json",
                training_count=168,
                sampler_seed=61,
                shard_raw_bytes=20_000,
                node_binary="node",
            )
            arguments = {
                "manifest_sha256": _file_sha256(manifest_path),
                "training_count": 168,
                "sampler_seed": 61,
                "shard_raw_bytes": 20_000,
            }
            _validate_plan(plan, **arguments)
            for oversized_argument in (
                {"training_count": 2**53},
                {"sampler_seed": 2**53},
                {"shard_raw_bytes": 2**53},
            ):
                with self.subTest(oversized_argument=oversized_argument), self.assertRaisesRegex(
                    ValueError, "safe integer"
                ):
                    _validate_plan(plan, **(arguments | oversized_argument))
            for unbalanced_count in (1, 167, 169):
                with self.subTest(unbalanced_count=unbalanced_count), self.assertRaisesRegex(
                    ValueError, "divisible by 168"
                ):
                    _validate_plan(plan, **(arguments | {"training_count": unbalanced_count}))
            tampered = copy.deepcopy(plan)
            tampered["phases"][0]["shards"][0]["records"][0]["source_id"] = "wrong-source"
            with self.assertRaisesRegex(ValueError, "shard content"):
                _validate_plan(tampered, **arguments)

            wrong_count = copy.deepcopy(plan)
            wrong_count["phases"][0]["shards"][0]["records"][0]["sample_weight"] += 1
            with self.assertRaisesRegex(ValueError, "training_count"):
                _validate_plan(_rehash_plan(wrong_count), **arguments)

            incomplete_validation = copy.deepcopy(plan)
            removed_validation_record = incomplete_validation["phases"][1]["shards"][0][
                "records"
            ].pop()
            incomplete_validation["phases"][1]["shards"][0][
                "raw_byte_estimate"
            ] -= (
                removed_validation_record["width"]
                * removed_validation_record["height"]
                * 3
            )
            with self.assertRaisesRegex(ValueError, "condition matrix"):
                _validate_plan(_rehash_plan(incomplete_validation), **arguments)

            overlapping = copy.deepcopy(plan)
            training_source_id = overlapping["phases"][0]["shards"][0]["sources"][0][
                "source_id"
            ]
            validation_shard = overlapping["phases"][1]["shards"][0]
            old_source_id = validation_shard["sources"][0]["source_id"]
            validation_shard["sources"][0]["source_id"] = training_source_id
            for candidate in validation_shard["records"]:
                if candidate["source_id"] == old_source_id:
                    candidate["source_id"] = training_source_id
            with self.assertRaisesRegex(ValueError, "disjoint"):
                _validate_plan(_rehash_plan(overlapping), **arguments)

            boolean_index = copy.deepcopy(plan)
            boolean_index["phases"][0]["shards"][0]["index"] = False
            with self.assertRaisesRegex(ValueError, "index"):
                _validate_plan(_rehash_plan(boolean_index), **arguments)

            boolean_raw_estimate = copy.deepcopy(plan)
            boolean_raw_estimate["phases"][0]["shards"][0][
                "raw_byte_estimate"
            ] = True
            with self.assertRaisesRegex(ValueError, "raw_byte_estimate"):
                _validate_plan(_rehash_plan(boolean_raw_estimate), **arguments)

            understated_raw_estimate = copy.deepcopy(plan)
            understated_raw_estimate["phases"][0]["shards"][0][
                "raw_byte_estimate"
            ] -= 1
            with self.assertRaisesRegex(ValueError, "raw_byte_estimate"):
                _validate_plan(_rehash_plan(understated_raw_estimate), **arguments)
            shard = plan["phases"][0]["shards"][0]
            record = shard["records"][0]
            source = next(
                source for source in shard["sources"]
                if source["source_id"] == record["source_id"]
            )
            materialized_root = root / "materialized"
            observation_path = materialized_root / "observations" / "fixture.png"
            observation_path.parent.mkdir(parents=True)
            fixture_rgb = np.zeros((record["height"], record["width"], 3), dtype=np.uint8)
            Image.fromarray(fixture_rgb, mode="RGB").save(observation_path, format="PNG")
            observation = {
                **record,
                "materialized_image_path": "observations/fixture.png",
                "materialized_sha256": _file_sha256(observation_path),
                "materialized_encoding": MATERIALIZED_ENCODING,
            }
            manifest_metadata = {
                "shared_observation_preprocessing_version": "shared-preprocessing-v1",
                "corruption_version": "track5-corruption-v1+sharp-0.35.4",
                "sharp_version": SHARP_VERSIONS["sharp"],
                "libvips_version": SHARP_VERSIONS["vips"],
            }
            materialized = {
                **shard["recipe_manifest_header"],
                "parent_recipe_manifest_sha256": shard["parent_recipe_manifest_sha256"],
                "materialization_schema_version": "track5-materialized-observations-v1",
                "materialization": {
                    "encoding": MATERIALIZED_ENCODING,
                    "shared_observation_preprocessing_version": manifest_metadata[
                        "shared_observation_preprocessing_version"
                    ],
                    "corruption_version": manifest_metadata["corruption_version"],
                    "sharp_version": manifest_metadata["sharp_version"],
                    "libvips_version": manifest_metadata["libvips_version"],
                    "observation_count": 1,
                },
                "signal_shard_provenance": {
                    field: shard[field]
                    for field in (
                        "parent_recipe_manifest_sha256",
                        "plan_sha256",
                        "shard_sha256",
                        "phase",
                        "index",
                        "count",
                        "variant_set_digest",
                        "raw_byte_budget",
                        "raw_byte_estimate",
                    )
                },
                "sources": [source],
                "observations": [observation],
            }
            one_record_shard = {**shard, "sources": [source], "records": [record]}
            validated = _validate_materialized_shard(
                    materialized,
                    materialized_root,
                    one_record_shard,
                    manifest_metadata=manifest_metadata,
                    plan_sha256=plan["plan_sha256"],
                )
            self.assertEqual(validated[0]["variant_id"], record["variant_id"])
            self.assertEqual(
                _feature_record(validated[0], resolution=224)["materialized_sha256"],
                observation["materialized_sha256"],
            )
            stale = copy.deepcopy(materialized)
            stale["materialization"]["corruption_version"] = "stale"
            with self.assertRaisesRegex(ValueError, "corruption_version"):
                _validate_materialized_shard(
                    stale,
                    materialized_root,
                    one_record_shard,
                    manifest_metadata=manifest_metadata,
                    plan_sha256=plan["plan_sha256"],
                )

            for name, mutate, message in (
                (
                    "root parent",
                    lambda value: value.update(parent_recipe_manifest_sha256="f" * 64),
                    "parent_recipe_manifest_sha256",
                ),
                (
                    "recipe header",
                    lambda value: value.update(observation_contract_version="stale"),
                    "recipe manifest header|observation_contract_version",
                ),
                (
                    "sources",
                    lambda value: value["sources"][0].update(split="internal-validation"),
                    "source",
                ),
                (
                    "sharp runtime",
                    lambda value: value["materialization"].update(sharp_version="stale"),
                    "sharp_version",
                ),
                (
                    "absolute materialized path inside root",
                    lambda value: value["observations"][0].update(
                        materialized_image_path=str(observation_path.resolve())
                    ),
                    "relative|contained path",
                ),
                (
                    "noncanonical contained traversal",
                    lambda value: value["observations"][0].update(
                        materialized_image_path="observations/../observations/fixture.png"
                    ),
                    "relative|canonical|contained path",
                ),
            ):
                with self.subTest(name=name):
                    changed = copy.deepcopy(materialized)
                    mutate(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        _validate_materialized_shard(
                            changed,
                            materialized_root,
                            one_record_shard,
                            manifest_metadata=manifest_metadata,
                            plan_sha256=plan["plan_sha256"],
                        )

            jpeg = io.BytesIO()
            Image.fromarray(fixture_rgb, mode="RGB").save(jpeg, format="JPEG")
            observation_path.write_bytes(jpeg.getvalue())
            jpeg_observation = copy.deepcopy(validated[0])
            jpeg_observation["materialized_sha256"] = _file_sha256(observation_path)
            with self.assertRaisesRegex(ValueError, "PNG"):
                _feature_record(jpeg_observation, resolution=224)

    def test_feature_cache_fails_closed_on_relationship_and_version_mutations(self):
        planned = {
            "source_id": "source-a",
            "variant_id": "variant-a",
            "split": "expert-training",
            "authenticity_label": 0,
            "condition_family": "clean",
            "severity": "clean",
            "sample_weight": 1,
        }
        phase = {
            "phase": "expert-training",
            "shards": [{"plan_sha256": "a" * 64, "records": [planned]}],
        }
        metadata = {
            "manifest_schema_version": "track5-manifest-v1",
            "manifest_sha256": "b" * 64,
        }
        record = {
            **planned,
            "materialized_sha256": "c" * 64,
            "materialized_encoding": MATERIALIZED_ENCODING,
            "signal_representation_version": "signal-representation-v1",
            "features": [float(index) for index in range(26)],
        }
        cache = {
            "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "manifest_metadata": metadata,
            "feature_extraction": _feature_extraction_metadata(384),
            "selection": {
                "split": "expert-training",
                "kind": "balanced-sampler",
                "sample_count": 1,
                "unique_observation_count": 1,
                "plan_sha256": "a" * 64,
                "shard_count": 1,
            },
            "records": [record],
        }
        cache["records_sha256"] = _sha256(cache["records"])
        self.assertEqual(
            validate_signal_feature_cache(
                cache,
                phase,
                metadata,
                expected_resolution=384,
            ),
            [record],
        )

        mutations = {
            "duplicate": lambda value: value["records"].append(copy.deepcopy(record)),
            "missing": lambda value: value["records"].clear(),
            "wrong source join": lambda value: value["records"][0].update(source_id="source-b"),
            "stale plan": lambda value: value["selection"].update(plan_sha256="d" * 64),
            "stale extraction": lambda value: value["feature_extraction"].update(
                feature_extraction_version="stale"
            ),
            "stale representation": lambda value: value["records"][0].update(
                signal_representation_version="stale"
            ),
            "wrong resolution": lambda value: value["feature_extraction"].update(
                resolution=224
            ),
            "stale implementation": lambda value: value["feature_extraction"].update(
                implementation_sha256={"signal_expert.py": "0" * 64}
            ),
            "invalid weight": lambda value: value["records"][0].update(sample_weight=0),
            "wrong selection kind": lambda value: value["selection"].update(
                kind="complete-condition-matrix"
            ),
            "non-string materialized digest": lambda value: value["records"][0].update(
                materialized_sha256=list("c" * 64)
            ),
            "numeric string feature": lambda value: value["records"][0]["features"].__setitem__(
                0, "1.0"
            ),
            "boolean label": lambda value: value["records"][0].update(
                authenticity_label=False
            ),
            "boolean sample weight": lambda value: value["records"][0].update(
                sample_weight=True
            ),
            "boolean selection sample count": lambda value: value["selection"].update(
                sample_count=True
            ),
            "boolean selection unique count": lambda value: value["selection"].update(
                unique_observation_count=True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                mutated = copy.deepcopy(cache)
                mutate(mutated)
                mutated["records_sha256"] = _sha256(mutated["records"])
                with self.assertRaises(ValueError):
                    validate_signal_feature_cache(
                        mutated,
                        phase,
                        metadata,
                        expected_resolution=384,
                    )

    def test_feature_snapshot_covers_and_detects_transitive_implementation_changes(self):
        snapshot = _feature_extraction_metadata(384)
        implementation = snapshot["implementation_sha256"]
        for required in (
            "package-lock.json",
            "src/contract-validation.js",
            "src/contracts.js",
            "src/deterministic-random.js",
            "src/track5-conditions.js",
        ):
            self.assertIn(required, implementation)

        def changed_digest(path):
            if Path(path).name == "deterministic-random.js":
                return "0" * 64
            return _implementation_file_sha256(Path(path))

        with patch("signal_pipeline._implementation_file_sha256", side_effect=changed_digest):
            with self.assertRaisesRegex(ValueError, "implementation.*changed"):
                _require_current_feature_extraction_snapshot(snapshot, 384)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"first\nsecond\n")
            crlf.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(
                _implementation_file_sha256(lf),
                _implementation_file_sha256(crlf),
            )

    def test_run_rejects_invalid_configuration_before_reading_the_manifest(self):
        invalid_configurations = (
            {"training_count": 0},
            {"training_count": 1},
            {"training_count": 167},
            {"training_count": 169},
            {"training_count": 2**53},
            {"sampler_seed": -1},
            {"sampler_seed": 2**53},
            {"model_seed": -1},
            {"epochs": 0},
            {"learning_rate": 0},
            {"learning_rate": float("inf")},
            {"resolution": 999},
            {"resolution": 224.0},
            {"shard_raw_bytes": 0},
            {"shard_raw_bytes": 2**53},
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration), patch(
                "signal_pipeline._read_json"
            ) as read_json:
                with self.assertRaisesRegex(ValueError, "training|seed|epochs|learning|resolution|shard"):
                    run_signal_experiment(
                        "missing-manifest.json",
                        "missing-dataset",
                        "unused-output",
                        **configuration,
                    )
                read_json.assert_not_called()

    def test_run_rejects_a_redirected_output_root_before_reading_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "must-survive.txt"
            sentinel.write_text("sentinel", encoding="utf-8")
            redirected_output = root / "redirected-output"
            subprocess.run(
                [
                    "node",
                    "-e",
                    (
                        "require('node:fs').symlinkSync(process.argv[1],process.argv[2],"
                        "process.platform==='win32'?'junction':'dir')"
                    ),
                    str(outside),
                    str(redirected_output),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            with patch("signal_pipeline._read_json_bytes") as read_json, self.assertRaisesRegex(
                ValueError, "must not be a symlink|junction|reparse"
            ):
                run_signal_experiment(
                    root / "missing-manifest.json",
                    root,
                    redirected_output,
                )
            read_json.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel")

    def test_run_materializes_bounded_shards_then_emits_strict_signal_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_fixture(root)
            output = root / "signal-output"

            result = run_signal_experiment(
                manifest_path,
                root,
                output,
                training_count=168,
                sampler_seed=61,
                model_seed=61,
                epochs=2,
                resolution=224,
                shard_raw_bytes=20_000,
            )

            expected = {
                "signal-training-features.json",
                "signal-validation-features.json",
                "signal-normalization.json",
                "signal-model.json",
                "signal-validation-logits.json",
                "signal-internal-validation-metrics.json",
                "signal-run.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertFalse(list(output.rglob("*.png")), "materialized PNG shards must be evicted")
            completion_receipts = sorted(
                (output / "signal-feature-shards").glob("*.complete.json")
            )
            self.assertEqual(len(completion_receipts), 4)
            training = json.loads((output / "signal-training-features.json").read_text(encoding="utf-8"))
            validation = json.loads((output / "signal-validation-features.json").read_text(encoding="utf-8"))
            logits = json.loads((output / "signal-validation-logits.json").read_text(encoding="utf-8"))
            checkpoint = json.loads((output / "signal-model.json").read_text(encoding="utf-8"))
            experiment_provenance = {
                "training_plan_sha256": result["plan_sha256"],
                "training_feature_records_sha256": training["records_sha256"],
                "validation_feature_records_sha256": validation["records_sha256"],
                "signal_feature_extraction_version": FEATURE_EXTRACTION_VERSION,
                "resolution": 224,
                "feature_extraction": training["feature_extraction"],
            }
            self.assertEqual(
                validate_signal_logit_cache(
                    logits,
                    expected_feature_records=validation["records"],
                    checkpoint_bundle=checkpoint,
                    manifest_metadata=result["manifest_metadata"],
                    expected_experiment_provenance=experiment_provenance,
                    expected_feature_cache_records_sha256=validation["records_sha256"],
                ),
                logits["records"],
            )
            stale_logits = copy.deepcopy(logits)
            stale_logits["feature_cache_records_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                validate_signal_logit_cache(
                    stale_logits,
                    expected_feature_records=validation["records"],
                    checkpoint_bundle=checkpoint,
                    manifest_metadata=result["manifest_metadata"],
                    expected_experiment_provenance=experiment_provenance,
                    expected_feature_cache_records_sha256=validation["records_sha256"],
                )
            published_plan = (output / "signal-plan.json").read_bytes()
            published_run = (output / "signal-run.json").read_bytes()
            published_model = (output / "signal-model.json").read_bytes()
            malformed_model = json.loads(published_model)
            malformed_model["weights"]["output_bias"] = "1.0"
            (output / "signal-model.json").write_text(
                json.dumps(malformed_model, indent=2) + "\n", encoding="utf-8"
            )
            marker_with_rehashed_malformed_model = json.loads(published_run)
            marker_with_rehashed_malformed_model["artifact_sha256"][
                "signal-model.json"
            ] = _file_sha256(output / "signal-model.json")
            (output / "signal-run.json").write_text(
                json.dumps(marker_with_rehashed_malformed_model), encoding="utf-8"
            )
            with patch("signal_pipeline._extract_phase_features") as extract_features:
                with self.assertRaisesRegex(
                    ValueError, "scientific provenance|weights|checkpoint"
                ):
                    run_signal_experiment(
                        manifest_path,
                        root,
                        output,
                        training_count=168,
                        sampler_seed=61,
                        model_seed=61,
                        epochs=2,
                        resolution=224,
                        shard_raw_bytes=20_000,
                    )
                extract_features.assert_not_called()
            (output / "signal-model.json").write_bytes(published_model)
            (output / "signal-run.json").write_bytes(published_run)
            stale_summary = json.loads(published_run)
            stale_summary["checkpoint_revision"] = "f" * 64
            (output / "signal-run.json").write_text(
                json.dumps(stale_summary), encoding="utf-8"
            )
            with patch("signal_pipeline._extract_phase_features") as extract_features:
                with self.assertRaisesRegex(
                    ValueError, "scientific summary checkpoint_revision"
                ):
                    run_signal_experiment(
                        manifest_path,
                        root,
                        output,
                        training_count=168,
                        sampler_seed=61,
                        model_seed=61,
                        epochs=2,
                        resolution=224,
                        shard_raw_bytes=20_000,
                    )
                extract_features.assert_not_called()
            (output / "signal-run.json").write_bytes(published_run)
            with self.assertRaisesRegex(ValueError, "existing.*plan|different.*plan|output directory"):
                run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=62,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            self.assertEqual((output / "signal-plan.json").read_bytes(), published_plan)
            self.assertEqual((output / "signal-run.json").read_bytes(), published_run)
            feature_shard_path = next(
                (output / "signal-feature-shards").glob("expert-training-*.features.json")
            )
            original_feature_shard = feature_shard_path.read_bytes()
            feature_completion_path = feature_shard_path.with_name(
                feature_shard_path.name.replace(".features.json", ".complete.json")
            )
            original_feature_completion = feature_completion_path.read_bytes()
            boolean_index_feature_shard = json.loads(original_feature_shard)
            boolean_index_feature_shard["index"] = bool(
                boolean_index_feature_shard["index"]
            )
            feature_shard_path.write_text(
                json.dumps(boolean_index_feature_shard, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature shard cache.*index"):
                run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=61,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            feature_shard_path.write_bytes(original_feature_shard)
            reordered_feature_shard = json.loads(original_feature_shard)
            reordered_feature_shard["records"].reverse()
            reordered_feature_shard["records_sha256"] = _sha256(
                reordered_feature_shard["records"]
            )
            feature_shard_path.write_text(
                json.dumps(reordered_feature_shard, indent=2) + "\n",
                encoding="utf-8",
            )
            reordered_completion = json.loads(original_feature_completion)
            reordered_completion["records_sha256"] = reordered_feature_shard["records_sha256"]
            reordered_completion["feature_cache_sha256"] = _sha256(reordered_feature_shard)
            feature_completion_path.write_text(
                json.dumps(reordered_completion, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "order"):
                run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=61,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            feature_shard_path.write_bytes(original_feature_shard)
            feature_completion_path.write_bytes(original_feature_completion)
            tampered_feature_shard = json.loads(original_feature_shard)
            tampered_feature_shard["records"][0]["features"][0] += 1.0
            tampered_feature_shard["records"][0]["materialized_sha256"] = "f" * 64
            tampered_feature_shard["records_sha256"] = _sha256(
                tampered_feature_shard["records"]
            )
            feature_shard_path.write_text(
                json.dumps(tampered_feature_shard, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fresh extraction|feature shard cache"):
                run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=61,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            feature_shard_path.write_bytes(original_feature_shard)
            (output / "signal-run.json").unlink()
            missing_feature_shard = next(
                (output / "signal-feature-shards").glob("internal-validation-*.features.json")
            )
            missing_feature_shard.unlink()
            with patch("signal_pipeline._run_node", wraps=signal_pipeline._run_node) as run_node:
                repeated = run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=61,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            materialize_calls = [
                call
                for call in run_node.call_args_list
                if "materialize" in call.args[0]
            ]
            self.assertEqual(len(materialize_calls), 1)

            cached_source = json.loads((output / "signal-plan.json").read_text(encoding="utf-8"))[
                "phases"
            ][0]["shards"][0]["sources"][0]
            cached_source_path = root / cached_source["image_path"]
            original_source_bytes = cached_source_path.read_bytes()
            cached_source_path.write_bytes(original_source_bytes + b"stale")
            try:
                with patch("signal_pipeline._run_node", wraps=signal_pipeline._run_node) as run_node:
                    with self.assertRaisesRegex(ValueError, "source.*SHA-256|pinned source"):
                        _extract_phase_features(
                            json.loads((output / "signal-plan.json").read_text(encoding="utf-8"))[
                                "phases"
                            ][0],
                            dataset_root=root,
                            output_directory=output,
                            manifest_metadata=result["manifest_metadata"],
                            plan_sha256=result["plan_sha256"],
                            resolution=224,
                            node_binary="node",
                            feature_extraction=training["feature_extraction"],
                        )
                    run_node.assert_not_called()
            finally:
                cached_source_path.write_bytes(original_source_bytes)

            with patch("signal_pipeline._run_node", wraps=signal_pipeline._run_node) as run_node:
                fully_reused = run_signal_experiment(
                    manifest_path,
                    root,
                    output,
                    training_count=168,
                    sampler_seed=61,
                    model_seed=61,
                    epochs=2,
                    resolution=224,
                    shard_raw_bytes=20_000,
                )
            self.assertFalse(
                [call for call in run_node.call_args_list if "materialize" in call.args[0]]
            )
            cold_output = root / "signal-output-cold-rerun"
            cold_repeated = run_signal_experiment(
                manifest_path,
                root,
                cold_output,
                training_count=168,
                sampler_seed=61,
                model_seed=61,
                epochs=2,
                resolution=224,
                shard_raw_bytes=20_000,
            )
            for artifact in expected | {"signal-plan.json", "signal-run.json"}:
                self.assertEqual(
                    (output / artifact).read_bytes(),
                    (cold_output / artifact).read_bytes(),
                    f"cold rerun changed {artifact}",
                )

        self.assertEqual(result["training_sample_count"], 168)
        self.assertEqual(result["validation_observation_count"], 40)
        self.assertEqual(training["selection"]["sample_count"], 168)
        self.assertEqual(validation["selection"]["sample_count"], 40)
        self.assertTrue(all(len(record["features"]) == 26 for record in training["records"]))
        self.assertTrue(all(len(record["materialized_sha256"]) == 64 for record in validation["records"]))
        self.assertEqual(repeated, result)
        self.assertEqual(fully_reused, result)
        self.assertEqual(cold_repeated, result)


if __name__ == "__main__":
    unittest.main()
