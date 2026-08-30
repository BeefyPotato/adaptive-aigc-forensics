import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from signal_maps import render_signal_maps


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_materialized_fixture(root: Path) -> tuple[Path, Path, str]:
    observations = root / "observations"
    observations.mkdir()
    yy, xx = np.indices((280, 300), dtype=np.uint16)
    image = np.stack(
        [
            ((xx * 7 + yy * 3) % 256).astype(np.uint8),
            (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8),
            ((xx * yy) % 256).astype(np.uint8),
        ],
        axis=2,
    )
    image_path = observations / "selected.png"
    Image.fromarray(image, mode="RGB").save(image_path, format="PNG")
    image_sha256 = sha256_file(image_path)
    variant_id = "source-1--noise--0.05"
    recipe = {
        "manifest_schema_version": "track5-manifest-v1",
        "leakage_audit": {"status": "passed"},
        "corruption": {
            "preprocessing_version": "shared-preprocessing-v1",
            "transform_implementation_version": "track5-corruption-v1+sharp-0.35.4",
            "sharp_version": "0.35.4",
            "libvips_version": "8.18.6",
        },
        "sources": [
            {
                "source_id": "source-1",
                "image_path": "source.png",
                "authenticity_label": 1,
                "split": "internal-validation",
                "width": 300,
                "height": 280,
            }
        ],
        "observations": [
            {
                "source_id": "source-1",
                "variant_id": variant_id,
                "image_path": "source.png",
                "authenticity_label": 1,
                "split": "internal-validation",
                "condition_family": "noise",
                "severity": "0.05",
                "corruption_parameters": {"sigma": 0.05},
                "corruption_seed": 23,
                "transform_implementation_version": "track5-corruption-v1+sharp-0.35.4",
                "width": 300,
                "height": 280,
            }
        ],
    }
    recipe_path = root / "track5-manifest.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    manifest = copy.deepcopy(recipe)
    manifest["materialization_schema_version"] = "track5-materialized-observations-v1"
    manifest["materialization"] = {
        "shared_observation_preprocessing_version": "shared-preprocessing-v1",
        "corruption_version": "track5-corruption-v1+sharp-0.35.4",
        "sharp_version": "0.35.4",
        "libvips_version": "8.18.6",
        "encoding": "lossless-rgb-png-v1",
        "observation_count": 1,
    }
    manifest["parent_recipe_manifest_sha256"] = sha256_file(recipe_path)
    manifest["signal_shard_provenance"] = {
        "parent_recipe_manifest_sha256": sha256_file(recipe_path),
        "plan_sha256": "a" * 64,
        "shard_sha256": "b" * 64,
        "phase": "internal-validation",
        "index": 0,
        "count": 1,
        "variant_set_digest": "c" * 64,
        "raw_byte_budget": 1_000_000,
        "raw_byte_estimate": 252_000,
    }
    manifest["observations"][0].update(
        materialized_image_path="observations/selected.png",
        materialized_sha256=image_sha256,
        materialized_encoding="lossless-rgb-png-v1",
    )
    manifest_path = root / "track5-materialized-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return recipe_path, manifest_path, variant_id


def mutate_manifest(manifest_path: Path, mutation) -> None:
    manifest = json.loads(manifest_path.read_text("utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class SignalMapTests(unittest.TestCase):
    def test_render_rejects_noninteger_resolution_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "diagnostics"
            with self.assertRaisesRegex(ValueError, "resolution"):
                render_signal_maps(
                    root / "missing-recipe.json",
                    root / "missing-materialized.json",
                    variant_id="fixture-variant",
                    output_directory=output,
                    resolution=224.0,
                )
            self.assertFalse(output.exists())

    def test_render_never_reuses_a_preplanted_predictable_temp_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
            output = root / "diagnostics"
            output.mkdir()
            sentinel = root / "outside-sentinel.txt"
            sentinel.write_text("must remain unchanged", encoding="utf-8")
            legacy_temporary = output / ".luminance.png.tmp"
            os.link(sentinel, legacy_temporary)

            render_signal_maps(
                recipe_path,
                manifest_path,
                variant_id=variant_id,
                output_directory=output,
                resolution=224,
            )

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "must remain unchanged"
            )
            self.assertEqual(
                legacy_temporary.read_text(encoding="utf-8"), "must remain unchanged"
            )

    def test_render_is_deterministic_auditable_and_keeps_distinct_diagnostic_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
            output = root / "diagnostics"

            first = render_signal_maps(
                recipe_path,
                manifest_path,
                variant_id=variant_id,
                output_directory=output,
                resolution=224,
            )
            first_bytes = {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file()
            }
            second = render_signal_maps(
                recipe_path,
                manifest_path,
                variant_id=variant_id,
                output_directory=output,
                resolution=224,
            )

            self.assertEqual(first, second)
            self.assertEqual(
                first_bytes,
                {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()},
            )
            self.assertEqual(first["schema_version"], "signal-diagnostic-maps-v1")
            self.assertEqual(first["usage"], "diagnostic-only")
            self.assertTrue(first["excluded_from_inference_cache"])
            self.assertEqual(first["variant"]["variant_id"], variant_id)
            self.assertEqual(len(first["feature_names"]), 26)
            self.assertEqual(len(first["features"]), 26)

            artifacts = first["artifacts"]
            self.assertEqual(
                set(artifacts),
                {"luminance", "fourier_log_spectrum", "neighbour_high_pass", "residual"},
            )
            for artifact in artifacts.values():
                path = output / artifact["filename"]
                self.assertEqual(artifact["artifact_sha256"], sha256_file(path))
                self.assertEqual(artifact["encoding"], "png-grayscale-uint16-v1")
                self.assertEqual(artifact["scaling"]["method"], "fixed-linear-clip-round-v1")
                with Image.open(path) as opened:
                    self.assertEqual(opened.size, (224, 224))
                    self.assertIn(opened.mode, {"I;16", "I"})
            self.assertNotEqual(
                artifacts["neighbour_high_pass"]["artifact_sha256"],
                artifacts["residual"]["artifact_sha256"],
            )

            persisted = json.loads((output / "signal-diagnostic-maps.json").read_text("utf-8"))
            self.assertEqual(persisted, first)
            self.assertEqual(
                first["input"]["materialized_manifest_sha256"], sha256_file(manifest_path)
            )
            self.assertEqual(first["input"]["parent_recipe_manifest_sha256"], sha256_file(recipe_path))

    def test_render_rejects_duplicate_or_missing_variant_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path, manifest_path, variant_id = write_materialized_fixture(root)

            with self.assertRaisesRegex(ValueError, "missing"):
                render_signal_maps(
                    recipe_path,
                    manifest_path,
                    variant_id="missing-variant",
                    output_directory=root / "missing",
                    resolution=224,
                )

            def add_unselected_duplicate(manifest):
                duplicate = copy.deepcopy(manifest["observations"][0])
                duplicate["variant_id"] = "unselected-duplicate"
                manifest["observations"].extend([duplicate, copy.deepcopy(duplicate)])
                manifest["materialization"]["observation_count"] = 3

            mutate_manifest(manifest_path, add_unselected_duplicate)
            with self.assertRaisesRegex(ValueError, "unique|duplicate"):
                render_signal_maps(
                    recipe_path,
                    manifest_path,
                    variant_id=variant_id,
                    output_directory=root / "duplicate",
                    resolution=224,
                )

    def test_render_rejects_traversal_stale_checksum_and_stale_versions(self):
        cases = [
            (
                "path traversal",
                lambda manifest: manifest["observations"][0].update(
                    materialized_image_path="../outside.png"
                ),
                "escapes",
            ),
            (
                "noncanonical contained path",
                lambda manifest: manifest["observations"][0].update(
                    materialized_image_path="observations/../observations/selected.png"
                ),
                "canonical|relative|contained",
            ),
            (
                "float observation count",
                lambda manifest: manifest["materialization"].update(observation_count=1.0),
                "observation count",
            ),
            (
                "boolean authenticity labels",
                lambda manifest: (
                    manifest["sources"][0].update(authenticity_label=True),
                    manifest["observations"][0].update(authenticity_label=True),
                ),
                "authenticity_label|label",
            ),
            (
                "float native dimensions",
                lambda manifest: (
                    manifest["sources"][0].update(width=300.0, height=280.0),
                    manifest["observations"][0].update(width=300.0, height=280.0),
                ),
                "width|height|dimensions",
            ),
            (
                "stale materialized bytes",
                lambda manifest: manifest["observations"][0].update(
                    materialized_sha256="0" * 64
                ),
                "stale SHA-256",
            ),
            (
                "stale shared preprocessing",
                lambda manifest: manifest["materialization"].update(
                    shared_observation_preprocessing_version="stale-preprocessing-v0"
                ),
                "preprocessing version is stale",
            ),
            (
                "stale corruption metadata",
                lambda manifest: manifest["materialization"].update(
                    corruption_version="stale-corruption-v0"
                ),
                "corruption version is stale",
            ),
            (
                "stale observation corruption",
                lambda manifest: manifest["observations"][0].update(
                    transform_implementation_version="stale-corruption-v0"
                ),
                "recipe transform_implementation_version",
            ),
        ]
        for name, mutation, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
                mutate_manifest(manifest_path, mutation)
                with self.assertRaisesRegex(ValueError, expected_error):
                    render_signal_maps(
                        recipe_path,
                        manifest_path,
                        variant_id=variant_id,
                        output_directory=root / "diagnostics",
                        resolution=224,
                    )
                self.assertFalse((root / "diagnostics").exists())

    def test_render_rejects_non_png_or_wrong_native_dimensions_even_with_matching_checksum(self):
        for name, image_factory, expected_error in (
            (
                "jpeg container",
                lambda: Image.fromarray(np.zeros((280, 300, 3), dtype=np.uint8), mode="RGB"),
                "PNG",
            ),
            (
                "wrong dimensions",
                lambda: Image.fromarray(np.zeros((280, 299, 3), dtype=np.uint8), mode="RGB"),
                "dimensions",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
                image_path = root / "observations" / "selected.png"
                image_factory().save(
                    image_path,
                    format="JPEG" if name == "jpeg container" else "PNG",
                )
                mutate_manifest(
                    manifest_path,
                    lambda manifest: manifest["observations"][0].update(
                        materialized_sha256=sha256_file(image_path),
                    ),
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    render_signal_maps(
                        recipe_path,
                        manifest_path,
                        variant_id=variant_id,
                        output_directory=root / "diagnostics",
                        resolution=224,
                    )
                self.assertFalse((root / "diagnostics").exists())

    def test_render_uses_recipe_as_the_independent_provenance_expectation(self):
        def stale_but_self_consistent(materialized):
            materialized["corruption"]["preprocessing_version"] = "stale-shared-v0"
            materialized["materialization"][
                "shared_observation_preprocessing_version"
            ] = "stale-shared-v0"

        cases = [
            (
                "self-consistent stale preprocessing",
                stale_but_self_consistent,
                "manifest preprocessing version is stale",
            ),
            (
                "wrong parent recipe",
                lambda materialized: (
                    materialized.update(parent_recipe_manifest_sha256="f" * 64),
                    materialized["signal_shard_provenance"].update(
                        parent_recipe_manifest_sha256="f" * 64
                    ),
                ),
                "parent recipe SHA-256",
            ),
            (
                "wrong source relationship",
                lambda materialized: materialized["observations"][0].update(
                    source_id="unrelated-source"
                ),
                "source relationship|recipe source_id",
            ),
        ]
        for name, mutation, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
                mutate_manifest(manifest_path, mutation)
                with self.assertRaisesRegex(ValueError, expected_error):
                    render_signal_maps(
                        recipe_path,
                        manifest_path,
                        variant_id=variant_id,
                        output_directory=root / "diagnostics",
                        resolution=224,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
            mutate_manifest(
                recipe_path,
                lambda recipe: recipe["leakage_audit"].update(status="failed"),
            )
            with self.assertRaisesRegex(ValueError, "passed leakage audit"):
                render_signal_maps(
                    recipe_path,
                    manifest_path,
                    variant_id=variant_id,
                    output_directory=root / "diagnostics",
                    resolution=224,
                )

    def test_render_rejects_partitions_outside_the_signal_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path, manifest_path, variant_id = write_materialized_fixture(root)
            mutate_manifest(
                recipe_path,
                lambda recipe: (
                    recipe["sources"][0].update(split="sealed-internal-test"),
                    recipe["observations"][0].update(split="sealed-internal-test"),
                ),
            )
            recipe_sha256 = sha256_file(recipe_path)
            mutate_manifest(
                manifest_path,
                lambda materialized: (
                    materialized.update(parent_recipe_manifest_sha256=recipe_sha256),
                    materialized["signal_shard_provenance"].update(
                        parent_recipe_manifest_sha256=recipe_sha256,
                        phase="sealed-internal-test",
                    ),
                    materialized["sources"][0].update(split="sealed-internal-test"),
                    materialized["observations"][0].update(split="sealed-internal-test"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "expert-training or internal-validation"):
                render_signal_maps(
                    recipe_path,
                    manifest_path,
                    variant_id=variant_id,
                    output_directory=root / "diagnostics",
                    resolution=224,
                )


if __name__ == "__main__":
    unittest.main()
