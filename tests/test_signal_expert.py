import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from signal_expert import (
    FEATURE_NAMES, RESIDUAL_KERNEL, cache_signal_predictions, evaluate_signal_only,
    decode_expert_rgb, decode_expert_rgb_bytes, extract_signal_representation, fit_normalization, read_model_bundle,
    train_signal_mlp, validate_signal_cache, write_model_bundle,
)
from shared_observation import prepare_shared_expert_rgb


META = {
    "manifest_schema_version": "track5-manifest-v1",
    "materialization_schema_version": "track5-materialized-observations-v1",
    "manifest_sha256": "a" * 64,
    "recipe_manifest_sha256": "b" * 64,
    "corruption_version": "track5-corruption-fixture-v1",
    "shared_observation_preprocessing_version": "shared-preprocessing-v1",
    "materialized_encoding": "lossless-rgb-png-v1",
    "signal_representation_version": "signal-representation-v1",
    "feature_extraction_version": "signal-representation-v1",
}
CANONICAL_CONDITIONS = (
    ("clean", "clean"),
    *(("jpeg", f"quality-{quality}") for quality in (90, 70, 50, 30)),
    *(("blur", f"sigma-{sigma}") for sigma in (0.5, 1, 2)),
    *(("resize", f"factor-{factor}") for factor in (0.5, 0.25)),
    *(("noise", f"sigma-{sigma}") for sigma in (0.02, 0.05, 0.1)),
    *(("color", f"{property_}-{factor}") for property_ in ("brightness", "contrast", "saturation") for factor in (0.8, 1.2)),
    ("crop", "center-0.8"),
)


def record(source, split, label, features, family="clean", severity=None):
    severity = severity or ("clean" if family == "clean" else "fixture")
    variant_id = f"variant-{source}-{family}-{severity}"
    return {
        "source_id": source,
        "variant_id": variant_id,
        "split": split,
        "authenticity_label": label,
        "condition_family": family,
        "severity": severity,
        "materialized_sha256": hashlib.sha256(variant_id.encode()).hexdigest(),
        "materialized_encoding": "lossless-rgb-png-v1",
        "signal_representation_version": "signal-representation-v1",
        "features": np.asarray(features, dtype=float).tolist(),
    }


def experiment_provenance(training, validation=(), *, plan_sha256="c" * 64):
    def digest(records):
        encoded = json.dumps(
            list(records), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    feature_extraction = {
        "feature_extraction_version": "signal-representation-v1",
        "signal_representation_version": "signal-representation-v1",
        "feature_names": list(FEATURE_NAMES),
        "resolution": 384,
        "runtime_versions": {"numpy": np.__version__, "pillow": "fixture"},
        "implementation_sha256": {"signal_expert.py": "f" * 64},
        "implementation_hash_contract_version": "utf8-lf-normalized-sha256-v1",
    }
    return {
        "training_plan_sha256": plan_sha256,
        "training_feature_records_sha256": digest(training),
        "validation_feature_records_sha256": digest(validation),
        "signal_feature_extraction_version": "signal-representation-v1",
        "resolution": 384,
        "feature_extraction": feature_extraction,
    }


def training_provenance(training, *, plan_sha256="c" * 64):
    experiment = experiment_provenance(training, plan_sha256=plan_sha256)
    return {
        key: experiment[key]
        for key in (
            "training_plan_sha256",
            "training_feature_records_sha256",
            "signal_feature_extraction_version",
            "resolution",
            "feature_extraction",
        )
    }


def checkpoint_revision(payload):
    identity = {key: value for key, value in payload.items() if key != "checkpoint_revision"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "signal-checkpoint-v1-" + hashlib.sha256(encoded).hexdigest()


def normalization_revision(payload):
    identity = {key: value for key, value in payload.items() if key != "normalization_revision"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "signal-normalization-v1-" + hashlib.sha256(encoded).hexdigest()


def expanded_record(candidate, index):
    expanded = copy.deepcopy(candidate)
    expanded["source_id"] = f"{candidate['source_id']}-expanded-{index}"
    expanded["variant_id"] = f"{candidate['variant_id']}-expanded-{index}"
    expanded["materialized_sha256"] = hashlib.sha256(expanded["variant_id"].encode()).hexdigest()
    expanded["sample_weight"] = 1
    return expanded


def validation_matrix(base, prefix="val"):
    rows = []
    for condition_index, (family, severity) in enumerate(CANONICAL_CONDITIONS):
        rows.extend([
            record(
                f"{prefix}-real",
                "internal-validation",
                0,
                base - 0.5 + condition_index / 100,
                family,
                severity,
            ),
            record(
                f"{prefix}-fake",
                "internal-validation",
                1,
                base + 0.5 + condition_index / 50,
                family,
                severity,
            ),
        ])
    return rows


class SignalExpertTests(unittest.TestCase):
    def test_materialized_path_uses_the_shared_observation_geometry(self):
        yy, xx = np.indices((31, 47))
        native = np.stack([
            (xx * 11) % 256,
            (yy * 17) % 256,
            ((xx + yy) * 7) % 256,
        ], axis=2).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "materialized.png"
            Image.fromarray(native).save(path)

            signal_rgb = decode_expert_rgb(path, resolution=224)
            shared_rgb = prepare_shared_expert_rgb(path, resolution=224)

        self.assertEqual(signal_rgb.shape, (224, 224, 3))
        np.testing.assert_array_equal(signal_rgb, shared_rgb.astype(np.float64) / 255.0)

    def test_materialized_bytes_fail_closed_on_container_pixel_mode_and_native_dimensions(self):
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        png = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(png, format="PNG")
        for resolution in (224, 384):
            decoded = decode_expert_rgb_bytes(
                png.getvalue(),
                resolution=resolution,
                expected_width=16,
                expected_height=12,
            )
            self.assertEqual(decoded.shape, (resolution, resolution, 3))
            self.assertEqual(
                extract_signal_representation(decoded)["features"].shape,
                (26,),
            )

        jpeg = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(jpeg, format="JPEG")
        with self.assertRaisesRegex(ValueError, "PNG"):
            decode_expert_rgb_bytes(
                jpeg.getvalue(), resolution=224, expected_width=16, expected_height=12,
            )

        rgba = io.BytesIO()
        Image.fromarray(np.zeros((12, 16, 4), dtype=np.uint8), mode="RGBA").save(
            rgba, format="PNG",
        )
        with self.assertRaisesRegex(ValueError, "RGB"):
            decode_expert_rgb_bytes(
                rgba.getvalue(), resolution=224, expected_width=16, expected_height=12,
            )

        with self.assertRaisesRegex(ValueError, "dimensions"):
            decode_expert_rgb_bytes(
                png.getvalue(), resolution=224, expected_width=15, expected_height=12,
            )
        for invalid_dimensions in (
            {"expected_width": 16.0, "expected_height": 12},
            {"expected_width": 16, "expected_height": 12.0},
        ):
            with self.subTest(invalid_dimensions=invalid_dimensions), self.assertRaisesRegex(
                ValueError, "positive integers"
            ):
                decode_expert_rgb_bytes(
                    png.getvalue(), resolution=224, **invalid_dimensions,
                )
        with self.assertRaisesRegex(ValueError, "resolution"):
            decode_expert_rgb_bytes(
                png.getvalue(), resolution=224.0, expected_width=16, expected_height=12,
            )

    def test_constant_field_has_zero_signal_evidence_in_all_26_positions(self):
        for height, width in ((2, 2), (33, 41)):
            with self.subTest(height=height, width=width):
                representation = extract_signal_representation(
                    np.full((height, width, 3), 0.5, dtype=np.float64),
                )
                np.testing.assert_array_equal(representation["features"], np.zeros(26))

    def test_representation_has_documented_deterministic_16_6_4_order_and_maps(self):
        yy, xx = np.indices((33, 41))
        rgb = np.stack([(xx % 7) / 6, (yy % 5) / 4, ((xx + yy) % 9) / 8], axis=2)
        first = extract_signal_representation(rgb, include_maps=True)
        second = extract_signal_representation(rgb)
        expected_names = tuple(
            [f"fourier_radial_log_energy_{index:02d}" for index in range(16)]
            + [
                "neighbour_horizontal_abs_mean",
                "neighbour_horizontal_abs_std",
                "neighbour_vertical_abs_mean",
                "neighbour_vertical_abs_std",
                "neighbour_diagonal_down_abs_mean",
                "neighbour_diagonal_up_abs_mean",
                "residual_abs_mean",
                "residual_std",
                "residual_excess_kurtosis",
                "residual_sign_change_rate",
            ]
        )
        self.assertEqual(FEATURE_NAMES, expected_names)
        np.testing.assert_array_equal(
            RESIDUAL_KERNEL,
            np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0,
        )
        self.assertEqual(first["features"].shape, (26,))
        np.testing.assert_array_equal(first["features"], second["features"])
        self.assertEqual(set(first["maps"]), {"luminance", "spectrum", "high_pass", "residual"})
        luminance = first["maps"]["luminance"]
        horizontal = np.zeros_like(luminance)
        vertical = np.zeros_like(luminance)
        horizontal[:, :-1] = np.diff(luminance, axis=1)
        vertical[:-1, :] = np.diff(luminance, axis=0)
        np.testing.assert_array_equal(
            first["maps"]["high_pass"],
            np.hypot(horizontal, vertical),
        )
        self.assertFalse(
            np.array_equal(first["maps"]["high_pass"], first["maps"]["residual"]),
            "The neighbour-gradient high-pass diagnostic is not the Gaussian residual.",
        )
        self.assertTrue(np.isfinite(first["features"]).all())

    def test_residual_excess_kurtosis_uses_the_central_fourth_moment(self):
        yy, xx = np.indices((5, 7))
        rgb = np.stack([
            ((xx * 3 + yy * 11) % 17) / 16,
            ((xx + yy * 5) % 13) / 12,
            ((xx * 7 + yy * 2) % 19) / 18,
        ], axis=2)

        representation = extract_signal_representation(rgb, include_maps=True)
        residual = representation["maps"]["residual"]
        expected = np.mean(((residual - residual.mean()) / residual.std()) ** 4) - 3

        self.assertAlmostEqual(representation["features"][24], expected, places=15)

    def test_nonconstant_signal_representation_matches_the_26_value_golden_vector(self):
        yy, xx = np.indices((17, 19))
        rgb = np.stack([
            ((xx * 3 + yy * 11) % 17) / 16,
            ((xx + yy * 5) % 13) / 12,
            ((xx * 7 + yy * 2) % 19) / 18,
        ], axis=2)
        expected = np.asarray([
            0.08036443901731484,
            0.059422609016029498,
            0.058120204998308611,
            0.047705638146744778,
            0.29288435143015512,
            0.63049106407475275,
            0.79375821998201868,
            1.0456154442127836,
            0.91063008944988355,
            0.99630520529425071,
            0.53046033177901464,
            0.22965952698698711,
            0.43811140572929791,
            0.68747765414272188,
            0.45418892702182068,
            0.035619405733010201,
            0.14984347312999274,
            0.16685465597409868,
            0.36953005299707603,
            0.14050761834940531,
            0.38744722704475304,
            0.33590341917438271,
            0.1926241198507912,
            0.22119278462554648,
            -1.1506860503977221,
            0.4885245901639344,
        ])

        actual = extract_signal_representation(rgb)["features"]

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_normalization_rejects_non_expert_training_data_and_stale_metadata(self):
        features = np.arange(26)
        invalid_split = [record("bad", "fusion-training", 0, features)]
        with self.assertRaisesRegex(ValueError, "only on expert-training"):
            fit_normalization(
                invalid_split,
                manifest_metadata=META,
                training_provenance=training_provenance(invalid_split),
            )
        training = [record("a", "expert-training", 0, features), record("b", "expert-training", 1, features + 1)]
        validation = validation_matrix(features, "metadata-val")
        experiment = experiment_provenance(training, validation)
        normalization = fit_normalization(
            training,
            manifest_metadata=META,
            training_provenance=training_provenance(training),
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            train_signal_mlp(
                training,
                validation,
                normalization,
                manifest_metadata={**META, "manifest_sha256": "c" * 64},
                experiment_provenance=experiment,
            )

        self.assertRegex(
            normalization["normalization_revision"],
            r"^signal-normalization-v1-[0-9a-f]{64}$",
        )
        stale_order = json.loads(json.dumps(normalization))
        stale_order["feature_names"][0], stale_order["feature_names"][1] = (
            stale_order["feature_names"][1], stale_order["feature_names"][0],
        )
        with self.assertRaisesRegex(ValueError, "feature order"):
            train_signal_mlp(
                training,
                validation,
                stale_order,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        tampered = json.loads(json.dumps(normalization))
        tampered["mean"][0] += 0.5
        with self.assertRaisesRegex(ValueError, "normalization revision"):
            train_signal_mlp(
                training,
                validation,
                tampered,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        with self.assertRaisesRegex(ValueError, "materialization_schema_version"):
            fit_normalization(
                training,
                manifest_metadata={"manifest_schema_version": "track5-manifest-v1"},
                training_provenance=training_provenance(training),
            )

        stale_features = copy.deepcopy(training)
        stale_features[0]["signal_representation_version"] = "signal-representation-stale"
        with self.assertRaisesRegex(ValueError, "representation version"):
            fit_normalization(
                stale_features,
                manifest_metadata=META,
                training_provenance=training_provenance(stale_features),
            )

        for name, mutate in (
            (
                "float provenance resolution",
                lambda value: (
                    value.update(resolution=384.0),
                    value["feature_extraction"].update(resolution=384.0),
                ),
            ),
            (
                "float snapshot resolution",
                lambda value: value["feature_extraction"].update(resolution=384.0),
            ),
        ):
            provenance = training_provenance(training)
            mutate(provenance)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "resolution|provenance"):
                fit_normalization(
                    training,
                    manifest_metadata=META,
                    training_provenance=provenance,
                )

    def test_normalization_is_bound_to_the_exact_training_feature_records(self):
        base = np.arange(26, dtype=np.float64)
        training_a = [
            record("real", "expert-training", 0, base),
            record("fake", "expert-training", 1, base + 1),
        ]
        training_b = copy.deepcopy(training_a)
        training_b[0]["features"][0] += 0.25
        validation = validation_matrix(base, "binding-val")
        experiment_b = experiment_provenance(training_b, validation)
        normalization_a = fit_normalization(
            training_a,
            manifest_metadata=META,
            training_provenance=training_provenance(training_a),
        )

        with self.assertRaisesRegex(ValueError, "training provenance"):
            train_signal_mlp(
                training_b,
                validation,
                normalization_a,
                manifest_metadata=META,
                experiment_provenance=experiment_b,
            )

    def test_normalization_weights_equal_an_expanded_balanced_draw(self):
        zeros = record("real", "expert-training", 0, np.zeros(26))
        fours = record("fake", "expert-training", 1, np.full(26, 4.0))
        zeros["sample_weight"] = 3
        fours["sample_weight"] = 1
        expanded = [
            expanded_record(zeros, 0),
            expanded_record(zeros, 1),
            expanded_record(zeros, 2),
            expanded_record(fours, 3),
        ]

        collapsed = [zeros, fours]
        weighted = fit_normalization(
            collapsed,
            manifest_metadata=META,
            training_provenance=training_provenance(collapsed),
        )
        literal = fit_normalization(
            expanded,
            manifest_metadata=META,
            training_provenance=training_provenance(expanded),
        )

        np.testing.assert_array_equal(weighted["mean"], np.ones(26))
        np.testing.assert_allclose(weighted["scale"], np.full(26, np.sqrt(3)))
        np.testing.assert_array_equal(weighted["mean"], literal["mean"])
        np.testing.assert_array_equal(weighted["scale"], literal["scale"])
        self.assertEqual(weighted["observation_count"], 4)
        self.assertEqual(weighted["unique_observation_count"], 2)
        invalid = record("invalid", "expert-training", 0, np.zeros(26))
        invalid["sample_weight"] = True
        with self.assertRaisesRegex(ValueError, "positive integers"):
            invalid_weights = [zeros, invalid]
            fit_normalization(
                invalid_weights,
                manifest_metadata=META,
                training_provenance=training_provenance(invalid_weights),
            )

        with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
            duplicates = [zeros, copy.deepcopy(zeros)]
            fit_normalization(
                duplicates,
                manifest_metadata=META,
                training_provenance=training_provenance(duplicates),
            )
        inconsistent_source = copy.deepcopy(fours)
        inconsistent_source["source_id"] = zeros["source_id"]
        with self.assertRaisesRegex(ValueError, "changes authenticity label"):
            inconsistent = [zeros, inconsistent_source]
            fit_normalization(
                inconsistent,
                manifest_metadata=META,
                training_provenance=training_provenance(inconsistent),
            )
        missing_source = copy.deepcopy(fours)
        missing_source.pop("source_id")
        with self.assertRaisesRegex(ValueError, "requires source_id"):
            missing = [zeros, missing_source]
            fit_normalization(
                missing,
                manifest_metadata=META,
                training_provenance=training_provenance(missing),
            )

    def test_training_is_reproducible_source_disjoint_and_cache_is_strict(self):
        base = np.linspace(-1, 1, 26)
        training = [record(f"train-{i}", "expert-training", i % 2, base + (i % 2) * 2 + i / 100) for i in range(8)]
        validation = validation_matrix(base)
        experiment = experiment_provenance(training, validation)
        normalization = fit_normalization(
            training,
            manifest_metadata=META,
            training_provenance=training_provenance(training),
        )
        first, metadata = train_signal_mlp(
            training,
            validation,
            normalization,
            manifest_metadata=META,
            experiment_provenance=experiment,
            epochs=30,
        )
        second, _ = train_signal_mlp(
            training,
            validation,
            normalization,
            manifest_metadata=META,
            experiment_provenance=experiment,
            epochs=30,
        )
        metadata.update({
            "training_selection": {
                "split": "expert-training",
                "kind": "balanced-sampler",
                "sample_count": metadata["training_observation_count"],
                "unique_observation_count": metadata["training_unique_observation_count"],
                "plan_sha256": experiment["training_plan_sha256"],
                "shard_count": 1,
            },
            "validation_selection": {
                "split": "internal-validation",
                "kind": "complete-condition-matrix",
                "sample_count": metadata["validation_observation_count"],
                "unique_observation_count": metadata["validation_observation_count"],
                "plan_sha256": experiment["training_plan_sha256"],
                "shard_count": 1,
            },
            "training_feature_records_sha256": experiment[
                "training_feature_records_sha256"
            ],
            "validation_feature_records_sha256": experiment[
                "validation_feature_records_sha256"
            ],
            "feature_extraction": experiment["feature_extraction"],
        })
        np.testing.assert_array_equal(first.input_weights, second.input_weights)
        self.assertEqual(metadata["selection_metric"], "condition-balanced-validation-bce")
        self.assertEqual(metadata["selection_metric_version"], "signal-condition-balanced-bce-v1")
        self.assertEqual(metadata["validation_condition_count"], 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            write_model_bundle(
                path,
                first,
                metadata,
                normalization,
                experiment_provenance=experiment,
            )
            loaded, bundle = read_model_bundle(
                path,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
            )
            self.assertRegex(bundle["checkpoint_revision"], r"^signal-checkpoint-v1-[0-9a-f]{64}$")
            self.assertEqual(bundle["normalization_revision"], normalization["normalization_revision"])
            self.assertEqual(bundle["feature_names"], list(FEATURE_NAMES))
            self.assertEqual(bundle["signal_representation_version"], "signal-representation-v1")

            for name, mutate in (
                (
                    "training feature digest",
                    lambda value: value.update(training_feature_records_sha256="f" * 64),
                ),
                (
                    "training plan",
                    lambda value: value["training_selection"].update(plan_sha256="f" * 64),
                ),
                (
                    "feature resolution",
                    lambda value: value["feature_extraction"].update(resolution=224),
                ),
                (
                    "training sample count",
                    lambda value: value["training_selection"].update(sample_count=999),
                ),
                (
                    "boolean training sample count",
                    lambda value: value["training_selection"].update(
                        sample_count=True,
                        unique_observation_count=True,
                    ),
                ),
                (
                    "boolean validation sample count",
                    lambda value: value["validation_selection"].update(
                        sample_count=True,
                        unique_observation_count=True,
                    ),
                ),
                (
                    "selection kind",
                    lambda value: value["training_selection"].update(
                        kind="complete-condition-matrix"
                    ),
                ),
                (
                    "feature runtime",
                    lambda value: value["feature_extraction"]["runtime_versions"].update(
                        numpy="stale"
                    ),
                ),
                ("negative seed", lambda value: value.update(seed=-1)),
                ("float hidden units", lambda value: value.update(hidden_units=16.0)),
                (
                    "float validation condition count",
                    lambda value: value.update(validation_condition_count=20.0),
                ),
                ("zero learning rate", lambda value: value.update(learning_rate=0)),
                (
                    "negative mutually matching counts",
                    lambda value: (
                        value.update(
                            training_observation_count=-1,
                            training_unique_observation_count=-1,
                        ),
                        value["training_selection"].update(
                            sample_count=-1,
                            unique_observation_count=-1,
                        ),
                    ),
                ),
                (
                    "normalization count",
                    lambda value: (
                        value["normalization"].update(source_count=999),
                        value["normalization"].update(
                            normalization_revision=normalization_revision(
                                value["normalization"]
                            )
                        ),
                        value.update(
                            normalization_revision=value["normalization"][
                                "normalization_revision"
                            ]
                        ),
                    ),
                ),
                (
                    "boolean validation score",
                    lambda value: value.update(validation_bce=True, selection_score=True),
                ),
                (
                    "boolean selection score",
                    lambda value: value.update(validation_bce=1.0, selection_score=True),
                ),
                (
                    "boolean output weight",
                    lambda value: value["weights"].update(output_bias=True),
                ),
                (
                    "numeric-string output weight",
                    lambda value: value["weights"].update(output_bias="1.0"),
                ),
                (
                    "boolean normalization value",
                    lambda value: (
                        value["normalization"]["mean"].__setitem__(0, True),
                        value["normalization"].update(
                            normalization_revision=normalization_revision(
                                value["normalization"]
                            )
                        ),
                        value.update(
                            normalization_revision=value["normalization"][
                                "normalization_revision"
                            ]
                        ),
                    ),
                ),
                (
                    "numeric-string normalization value",
                    lambda value: (
                        value["normalization"]["mean"].__setitem__(0, "1.0"),
                        value["normalization"].update(
                            normalization_revision=normalization_revision(
                                value["normalization"]
                            )
                        ),
                        value.update(
                            normalization_revision=value["normalization"][
                                "normalization_revision"
                            ]
                        ),
                    ),
                ),
            ):
                contradictory = copy.deepcopy(bundle)
                mutate(contradictory)
                contradictory["checkpoint_revision"] = checkpoint_revision(contradictory)
                path.write_text(json.dumps(contradictory), encoding="utf-8")
                with self.subTest(contradiction=name):
                    with self.assertRaisesRegex(
                        ValueError,
                        "contradict|incompatible|positive|non-negative|boolean|numeric|finite",
                    ):
                        read_model_bundle(
                            path,
                            manifest_metadata=META,
                            expected_experiment_provenance=experiment,
                        )

            path.write_text(json.dumps(bundle), encoding="utf-8")
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["weights"]["input"][0][0] += 0.01
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint revision"):
                read_model_bundle(
                    path,
                    manifest_metadata=META,
                    expected_experiment_provenance=experiment,
                )

            path.write_text(json.dumps(bundle), encoding="utf-8")
            nonfinite = json.loads(path.read_text(encoding="utf-8"))
            nonfinite["weights"]["output_bias"] = float("nan")
            # Rehashing cannot make a non-finite checkpoint acceptable.
            nonfinite["checkpoint_revision"] = "signal-checkpoint-v1-" + "0" * 64
            path.write_text(json.dumps(nonfinite), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite|checkpoint revision"):
                read_model_bundle(
                    path,
                    manifest_metadata=META,
                    expected_experiment_provenance=experiment,
                )

            wrong_shape = copy.deepcopy(bundle)
            wrong_shape["weights"]["input"].pop()
            wrong_shape["checkpoint_revision"] = checkpoint_revision(wrong_shape)
            path.write_text(json.dumps(wrong_shape), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_model_bundle(
                    path,
                    manifest_metadata=META,
                    expected_experiment_provenance=experiment,
                )

            wrong_order = copy.deepcopy(bundle)
            wrong_order["feature_names"][0], wrong_order["feature_names"][1] = (
                wrong_order["feature_names"][1], wrong_order["feature_names"][0],
            )
            wrong_order["checkpoint_revision"] = checkpoint_revision(wrong_order)
            path.write_text(json.dumps(wrong_order), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature order"):
                read_model_bundle(
                    path,
                    manifest_metadata=META,
                    expected_experiment_provenance=experiment,
                )

            path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "experiment provenance"):
                read_model_bundle(
                    path,
                    manifest_metadata=META,
                    expected_experiment_provenance={
                        **experiment,
                        "training_plan_sha256": "f" * 64,
                    },
                )

            cache = cache_signal_predictions(
                validation,
                bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
            )
        self.assertEqual(len(cache), 40)
        self.assertTrue(all(len(row["features"]) == 26 and np.isfinite(row["signal_logit"]) for row in cache))
        self.assertTrue(all(row["cache_key"].startswith("signal-cache-v1-") for row in cache))
        self.assertTrue(all(row["split"] == "internal-validation" for row in cache))
        self.assertTrue(all(row["authenticity_label"] in (0, 1) for row in cache))
        self.assertTrue(all(row["materialized_encoding"] == "lossless-rgb-png-v1" for row in cache))
        self.assertTrue(all(len(row["materialized_sha256"]) == 64 for row in cache))
        self.assertTrue(all(row["normalization_revision"] == normalization["normalization_revision"] for row in cache))
        with self.assertRaisesRegex(ValueError, "validation feature records|provenance"):
            cache_signal_predictions(
                validation[:-1],
                bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
            )
        with self.assertRaisesRegex(ValueError, "validation feature records|provenance"):
            validate_signal_cache(
                cache[:-1],
                checkpoint_bundle=bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
                expected_feature_records=validation[:-1],
            )
        self.assertEqual(
            validate_signal_cache(
                cache,
                checkpoint_bundle=bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
                expected_feature_records=validation,
            ),
            cache,
        )
        stale = json.loads(json.dumps(cache))
        stale[0]["manifest_metadata"]["manifest_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "stale manifest_metadata"):
            validate_signal_cache(
                stale,
                checkpoint_bundle=bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
                expected_feature_records=validation,
            )

        for mutation, message in (
            (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate variant_id"),
            (lambda rows: rows.pop(), "missing expected variants"),
            (lambda rows: rows.reverse(), "order"),
            (lambda rows: rows[0].update(source_id="wrong-source"), "incompatible source_id"),
            (lambda rows: rows[0].update(materialized_sha256="f" * 64), "incompatible materialized_sha256"),
            (lambda rows: rows[0].update(condition_family="jpeg"), "incompatible condition_family"),
            (
                lambda rows: rows[0].update(
                    authenticity_label=bool(rows[0]["authenticity_label"])
                ),
                "incompatible authenticity_label",
            ),
        ):
            changed = copy.deepcopy(cache)
            mutation(changed)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_signal_cache(
                        changed,
                        checkpoint_bundle=bundle,
                        manifest_metadata=META,
                        expected_experiment_provenance=experiment,
                        expected_feature_records=validation,
                    )

        stale_features = copy.deepcopy(validation)
        stale_features[0]["signal_representation_version"] = "signal-representation-stale"
        with self.assertRaisesRegex(ValueError, "representation version"):
            cache_signal_predictions(
                stale_features,
                bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
            )

        for field, value in (("signal_logit", 123.0), ("pred", 0.5)):
            altered_predictions = copy.deepcopy(cache)
            altered_predictions[0][field] = value
            with self.subTest(altered_prediction=field):
                with self.assertRaisesRegex(ValueError, f"incompatible {field}|cache_key"):
                    validate_signal_cache(
                        altered_predictions,
                        checkpoint_bundle=bundle,
                        manifest_metadata=META,
                        expected_experiment_provenance=experiment,
                        expected_feature_records=validation,
                    )

        mismatched_bundle = copy.deepcopy(bundle)
        mismatched_bundle["weights"]["input"][0][0] += 0.01
        with self.assertRaisesRegex(ValueError, "checkpoint revision"):
            cache_signal_predictions(
                validation,
                mismatched_bundle,
                manifest_metadata=META,
                expected_experiment_provenance=experiment,
            )

    def test_training_weights_equal_an_expanded_balanced_draw(self):
        base = np.linspace(-1, 1, 26)
        real = record("train-real", "expert-training", 0, base - 1)
        fake = record("train-fake", "expert-training", 1, base + 1)
        real["sample_weight"] = 3
        fake["sample_weight"] = 1
        collapsed = [real, fake]
        expanded = [
            expanded_record(real, 0),
            expanded_record(real, 1),
            expanded_record(real, 2),
            expanded_record(fake, 3),
        ]
        validation = validation_matrix(base)
        collapsed_experiment = experiment_provenance(collapsed, validation)
        expanded_experiment = experiment_provenance(expanded, validation)
        collapsed_normalization = fit_normalization(
            collapsed,
            manifest_metadata=META,
            training_provenance=training_provenance(collapsed),
        )
        expanded_normalization = fit_normalization(
            expanded,
            manifest_metadata=META,
            training_provenance=training_provenance(expanded),
        )

        collapsed_model, _ = train_signal_mlp(
            collapsed,
            validation,
            collapsed_normalization,
            manifest_metadata=META,
            experiment_provenance=collapsed_experiment,
            epochs=5,
        )
        expanded_model, _ = train_signal_mlp(
            expanded,
            validation,
            expanded_normalization,
            manifest_metadata=META,
            experiment_provenance=expanded_experiment,
            epochs=5,
        )

        np.testing.assert_allclose(collapsed_model.input_weights, expanded_model.input_weights, atol=1e-15, rtol=0)
        np.testing.assert_allclose(collapsed_model.input_bias, expanded_model.input_bias, atol=1e-15, rtol=0)
        np.testing.assert_allclose(collapsed_model.output_weights, expanded_model.output_weights, atol=1e-15, rtol=0)
        self.assertAlmostEqual(collapsed_model.output_bias, expanded_model.output_bias, places=15)

    def test_checkpoint_selection_averages_severities_then_families(self):
        base = np.linspace(-1, 1, 26)
        training = [
            record("train-real", "expert-training", 0, base - 1),
            record("train-fake", "expert-training", 1, base + 1),
        ]
        validation = validation_matrix(base)
        experiment = experiment_provenance(training, validation)
        normalization = fit_normalization(
            training,
            manifest_metadata=META,
            training_provenance=training_provenance(training),
        )

        model, metadata = train_signal_mlp(
            training,
            validation,
            normalization,
            manifest_metadata=META,
            experiment_provenance=experiment,
            epochs=1,
        )

        mean = np.asarray(normalization["mean"])
        scale = np.asarray(normalization["scale"])
        features = np.asarray([row["features"] for row in validation])
        labels = np.asarray([row["authenticity_label"] for row in validation])
        logits = model.logits((features - mean) / scale)
        losses = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        family_means = []
        for family in ("clean", "jpeg", "blur", "resize", "noise", "color", "crop"):
            severity_means = []
            for severity in {row["severity"] for row in validation if row["condition_family"] == family}:
                indexes = [
                    index for index, row in enumerate(validation)
                    if row["condition_family"] == family and row["severity"] == severity
                ]
                severity_means.append(float(losses[indexes].mean()))
            family_means.append(sum(severity_means) / len(severity_means))
        expected = sum(family_means) / len(family_means)

        self.assertAlmostEqual(metadata["selection_score"], expected, places=15)
        self.assertNotAlmostEqual(metadata["selection_score"], float(losses.mean()), places=10)

    def test_training_rejects_invalid_features_labels_and_hyperparameters(self):
        features = np.arange(26, dtype=np.float64)
        training = [
            record("train-real", "expert-training", 0, features),
            record("train-fake", "expert-training", 1, features + 1),
        ]
        validation = validation_matrix(features)
        experiment = experiment_provenance(training, validation)
        normalization = fit_normalization(
            training,
            manifest_metadata=META,
            training_provenance=training_provenance(training),
        )

        invalid_label = json.loads(json.dumps(training))
        invalid_label[0]["authenticity_label"] = 2
        with self.assertRaisesRegex(ValueError, "binary labels"):
            train_signal_mlp(
                invalid_label,
                validation,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        invalid_features = json.loads(json.dumps(validation))
        invalid_features[0]["features"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite 26-value"):
            train_signal_mlp(
                training,
                invalid_features,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        for arguments in (
            {"epochs": 0},
            {"epochs": 1.5},
            {"learning_rate": 0},
            {"learning_rate": float("inf")},
            {"seed": -1},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "positive|non-negative"):
                    train_signal_mlp(
                        training,
                        validation,
                        normalization,
                        manifest_metadata=META,
                        experiment_provenance=experiment,
                        **arguments,
                    )

        with self.assertRaisesRegex(ValueError, "diverged"):
            train_signal_mlp(
                training,
                validation,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
                epochs=2,
                learning_rate=1e308,
            )

    def test_training_and_selection_partitions_are_source_disjoint_and_fail_closed(self):
        features = np.arange(26, dtype=np.float64)
        training = [
            record("train-real", "expert-training", 0, features),
            record("train-fake", "expert-training", 1, features + 1),
        ]
        validation = validation_matrix(features)
        experiment = experiment_provenance(training, validation)
        normalization = fit_normalization(
            training,
            manifest_metadata=META,
            training_provenance=training_provenance(training),
        )

        fusion_training = copy.deepcopy(training)
        fusion_training[0]["split"] = "fusion-training"
        with self.assertRaisesRegex(ValueError, "only expert-training"):
            train_signal_mlp(
                fusion_training,
                validation,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        sealed_selection = copy.deepcopy(validation)
        sealed_selection[0]["split"] = "sealed-internal-test"
        with self.assertRaisesRegex(ValueError, "only internal-validation"):
            train_signal_mlp(
                training,
                sealed_selection,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        overlapping = copy.deepcopy(validation)
        overlapping[0]["source_id"] = training[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "source.*disjoint"):
            train_signal_mlp(
                training,
                overlapping,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )

        incomplete = [row for row in validation if row["condition_family"] != "crop"]
        with self.assertRaisesRegex(ValueError, "canonical condition matrix"):
            train_signal_mlp(
                training,
                incomplete,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment_provenance(training, incomplete),
            )

        disguised_organizer = copy.deepcopy(training)
        disguised_organizer[0]["dataset"] = "organizer-demonstration"
        with self.assertRaisesRegex(ValueError, "organizer-demonstration"):
            fit_normalization(
                disguised_organizer,
                manifest_metadata=META,
                training_provenance=training_provenance(disguised_organizer),
            )

        evaluation_only = copy.deepcopy(validation)
        evaluation_only[0]["usage"] = "evaluation-only"
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            train_signal_mlp(
                training,
                evaluation_only,
                normalization,
                manifest_metadata=META,
                experiment_provenance=experiment,
            )
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            evaluate_signal_only(evaluation_only)

    def test_signal_metrics_delegate_to_canonical_severity_first_family_macro_evaluator(self):
        rows = []
        good_severities = {
            "clean",
            "quality-90",
            "sigma-0.5",
            "factor-0.5",
            "sigma-0.02",
            "brightness-0.8",
            "center-0.8",
        }
        for family, severity in CANONICAL_CONDITIONS:
            good = severity in good_severities
            rows.extend([
                {
                    **record(
                        "metric-real", "internal-validation", 0, np.zeros(26), family, severity,
                    ),
                    "signal_logit": -1.0 if good else 1.0,
                },
                {
                    **record(
                        "metric-fake", "internal-validation", 1, np.ones(26), family, severity,
                    ),
                    "signal_logit": 1.0 if good else -1.0,
                },
            ])

        metrics = evaluate_signal_only(rows)

        self.assertEqual(metrics["metric_schema_version"], "signal-robustness-metric-v1")
        self.assertEqual(metrics["clean_auroc"], 1.0)
        self.assertEqual(
            metrics["corruption_families"]["jpeg"]["auroc_by_severity"],
            {"quality-30": 0.0, "quality-50": 0.0, "quality-70": 0.0, "quality-90": 1.0},
        )
        self.assertEqual(metrics["corruption_families"]["jpeg"]["auroc"], 0.25)
        family_sum = 1 / 4 + 1 / 3 + 1 / 2 + 1 / 3 + 1 / 6 + 1
        self.assertAlmostEqual(metrics["mean_corrupted_auroc"], family_sum / 6)
        self.assertAlmostEqual(metrics["all_condition_macro_auroc"], (1 + family_sum) / 7)

        with self.assertRaisesRegex(ValueError, "canonical condition matrix"):
            evaluate_signal_only(rows[:-1])


if __name__ == "__main__":
    unittest.main()
