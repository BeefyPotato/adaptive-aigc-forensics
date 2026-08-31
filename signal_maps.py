"""Auditable, regenerable diagnostic maps for one materialized observation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image

from safe_output import atomic_write_bytes, managed_output_path, resolve_output_directory

from signal_expert import (
    FEATURE_NAMES,
    SIGNAL_REPRESENTATION_VERSION,
    decode_expert_rgb_bytes,
    extract_signal_representation,
)


DIAGNOSTIC_MAP_SCHEMA_VERSION = "signal-diagnostic-maps-v1"
MATERIALIZATION_SCHEMA_VERSION = "track5-materialized-observations-v1"
MATERIALIZED_ENCODING = "lossless-rgb-png-v1"
MAP_ENCODING = "png-grayscale-uint16-v1"
MAP_METADATA_FILENAME = "signal-diagnostic-maps.json"
SIGNAL_EXPERIMENT_SPLITS = frozenset({"expert-training", "internal-validation"})
_HEX = frozenset("0123456789abcdef")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    atomic_write_bytes(path, value)


def _require_object(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return value


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _contained_materialized_path(root: Path, relative_value: object) -> tuple[str, Path]:
    relative = _require_nonempty_string(relative_value, "materialized_image_path")
    relative_path = Path(relative)
    windows_relative = PureWindowsPath(relative)
    segments = relative.replace("\\", "/").split("/")
    if (
        "\\" in relative
        or "\x00" in relative
        or any(segment in {"", ".", ".."} for segment in segments)
        or relative_path.is_absolute()
        or relative_path.anchor
        or relative_path.drive
        or windows_relative.anchor
        or windows_relative.drive
        or relative_path.as_posix() != relative
    ):
        raise ValueError(
            "materialized_image_path escapes the materialized manifest root or is not a "
            "canonical relative contained path."
        )
    absolute_root = root.resolve()
    try:
        absolute_path = managed_output_path(
            absolute_root,
            relative,
            "diagnostic materialized observation",
        )
    except ValueError as error:
        raise ValueError("materialized_image_path escapes the materialized manifest root.") from error
    return relative, absolute_path


def _unique_records_by_id(records: object, field: str, context: str) -> dict[str, dict]:
    if not isinstance(records, list):
        raise ValueError(f"{context} must be an array.")
    by_id = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{context} record {index} must be an object.")
        identity = record.get(field)
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError(f"{context} {field} values must be non-empty and unique.")
        by_id[identity] = record
    return by_id


def _validated_recipe_observation(recipe: dict, variant_id: str) -> tuple[dict, dict]:
    if recipe.get("manifest_schema_version") != "track5-manifest-v1":
        raise ValueError("Diagnostic maps require a track5-manifest-v1 recipe manifest.")
    leakage_audit = _require_object(recipe.get("leakage_audit"), "recipe leakage_audit")
    if leakage_audit.get("status") != "passed":
        raise ValueError("Diagnostic maps require a recipe manifest with a passed leakage audit.")
    corruption = _require_object(recipe.get("corruption"), "recipe corruption")
    _require_nonempty_string(
        corruption.get("preprocessing_version"), "recipe corruption.preprocessing_version"
    )
    corruption_version = _require_nonempty_string(
        corruption.get("transform_implementation_version"),
        "recipe corruption.transform_implementation_version",
    )
    _require_nonempty_string(corruption.get("sharp_version"), "recipe corruption.sharp_version")
    _require_nonempty_string(
        corruption.get("libvips_version"), "recipe corruption.libvips_version"
    )
    sources = _unique_records_by_id(recipe.get("sources"), "source_id", "Recipe sources")
    observations = _unique_records_by_id(
        recipe.get("observations"), "variant_id", "Recipe observations"
    )
    observation = observations.get(variant_id)
    if observation is None:
        raise ValueError(f"Variant {variant_id!r} is missing from the recipe manifest.")
    source = sources.get(observation.get("source_id"))
    if source is None:
        raise ValueError("Selected recipe observation has no source relationship.")
    if observation.get("split") not in SIGNAL_EXPERIMENT_SPLITS:
        raise ValueError(
            "Signal diagnostic maps may use only expert-training or internal-validation observations."
        )
    for field in ("image_path", "authenticity_label", "split", "width", "height"):
        if field in source or field in observation:
            if not _strict_json_equal(source.get(field), observation.get(field)):
                raise ValueError(f"Selected recipe observation disagrees with its source {field}.")
    if observation.get("transform_implementation_version") != corruption_version:
        raise ValueError("Selected recipe observation corruption version is stale.")
    for context, record in (("source", source), ("observation", observation)):
        label = record.get("authenticity_label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Selected recipe {context} requires a binary integer authenticity_label.")
        for field in ("width", "height"):
            value = record.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Selected recipe {context} requires positive integer {field}.")
    return observation, source


def _validated_selected_observation(
    manifest: dict,
    recipe: dict,
    recipe_observation: dict,
    recipe_source: dict,
    recipe_sha256: str,
    variant_id: str,
    root: Path,
) -> tuple[dict, bytes]:
    if manifest.get("manifest_schema_version") != "track5-manifest-v1":
        raise ValueError("Diagnostic maps require track5-manifest-v1 provenance.")
    if manifest.get("materialization_schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError(f"Diagnostic maps require {MATERIALIZATION_SCHEMA_VERSION}.")
    if manifest.get("parent_recipe_manifest_sha256") != recipe_sha256:
        raise ValueError("Materialized shard has a stale parent recipe SHA-256.")
    shard_provenance = _require_object(
        manifest.get("signal_shard_provenance"), "signal_shard_provenance"
    )
    if shard_provenance.get("parent_recipe_manifest_sha256") != recipe_sha256:
        raise ValueError("Signal shard provenance has a stale parent recipe SHA-256.")
    if shard_provenance.get("phase") != recipe_observation.get("split"):
        raise ValueError("Signal shard provenance phase disagrees with the selected recipe observation.")
    corruption = _require_object(manifest.get("corruption"), "corruption")
    recipe_corruption = recipe["corruption"]
    preprocessing_version = recipe_corruption["preprocessing_version"]
    corruption_version = recipe_corruption["transform_implementation_version"]
    if corruption.get("preprocessing_version") != preprocessing_version:
        raise ValueError("Materialized manifest preprocessing version is stale.")
    if corruption.get("transform_implementation_version") != corruption_version:
        raise ValueError("Materialized manifest corruption version is stale.")
    materialization = _require_object(manifest.get("materialization"), "materialization")
    if materialization.get("shared_observation_preprocessing_version") != preprocessing_version:
        raise ValueError("Materialized shared-observation preprocessing version is stale.")
    if materialization.get("corruption_version") != corruption_version:
        raise ValueError("Materialized corruption version is stale.")
    for field in ("sharp_version", "libvips_version"):
        if materialization.get(field) != recipe_corruption[field]:
            raise ValueError(f"Materialized {field} is stale.")
    if materialization.get("encoding") != MATERIALIZED_ENCODING:
        raise ValueError("Materialized observation encoding is stale or incompatible.")

    observations = manifest.get("observations")
    materialized_sources = _unique_records_by_id(
        manifest.get("sources"), "source_id", "Materialized sources"
    )
    materialized_observations = _unique_records_by_id(
        observations, "variant_id", "Materialized observations"
    )
    observation_count = materialization.get("observation_count")
    if type(observation_count) is not int or observation_count != len(observations):
        raise ValueError("Materialized observation count disagrees with the manifest.")
    observation = materialized_observations.get(variant_id)
    if observation is None:
        raise ValueError(f"Variant {variant_id!r} must appear exactly once in the materialized manifest.")
    source = materialized_sources.get(observation.get("source_id"))
    if source is None:
        raise ValueError("Selected materialized observation has no source relationship.")
    relationship_fields = (
        "source_id",
        "image_path",
        "authenticity_label",
        "split",
        "condition_family",
        "severity",
        "corruption_parameters",
        "corruption_seed",
        "transform_implementation_version",
        "width",
        "height",
        "byte_length",
        "exact_sha256",
    )
    for field in relationship_fields:
        if field in recipe_observation or field in observation:
            if not _strict_json_equal(observation.get(field), recipe_observation.get(field)):
                raise ValueError(f"Selected materialized observation disagrees with recipe {field}.")
    for field in ("source_id", "image_path", "authenticity_label", "split", "width", "height"):
        if field in recipe_source or field in source:
            if not _strict_json_equal(source.get(field), recipe_source.get(field)):
                raise ValueError(f"Selected materialized source disagrees with recipe {field}.")
        if field != "source_id" and (field in source or field in observation):
            if not _strict_json_equal(observation.get(field), source.get(field)):
                raise ValueError(f"Selected materialized observation disagrees with its source {field}.")
    if observation.get("transform_implementation_version") != corruption_version:
        raise ValueError("Selected observation corruption version is stale.")
    if observation.get("materialized_encoding") != MATERIALIZED_ENCODING:
        raise ValueError("Selected observation encoding is stale or incompatible.")
    materialized_sha256 = _require_sha256(
        observation.get("materialized_sha256"), "materialized_sha256"
    )
    relative_path, absolute_path = _contained_materialized_path(
        root, observation.get("materialized_image_path")
    )
    try:
        materialized_bytes = absolute_path.read_bytes()
    except OSError as error:
        raise ValueError("Selected materialized observation cannot be read.") from error
    if _sha256_bytes(materialized_bytes) != materialized_sha256:
        raise ValueError("Selected materialized observation has a stale SHA-256 checksum.")
    validated = dict(observation)
    validated["materialized_image_path"] = relative_path
    return validated, materialized_bytes


def _encode_map(values: np.ndarray, *, lower: float, upper: float) -> bytes:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Diagnostic maps must be finite two-dimensional arrays.")
    normalized = (np.clip(array, lower, upper) - lower) / (upper - lower)
    quantized = np.floor(normalized * 65535.0 + 0.5).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(quantized).save(
        buffer,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return buffer.getvalue()


def render_signal_maps(
    recipe_manifest_path: Path | str,
    materialized_manifest_path: Path | str,
    *,
    variant_id: str,
    output_directory: Path | str,
    resolution: int = 384,
) -> dict:
    """Render four diagnostic-only maps for one verified materialized variant."""
    variant_id = _require_nonempty_string(variant_id, "variant_id")
    if type(resolution) is not int or resolution not in (224, 384):
        raise ValueError("Signal diagnostic resolution must be 224 or 384.")
    recipe_path = Path(recipe_manifest_path)
    try:
        recipe_bytes = recipe_path.read_bytes()
        recipe = json.loads(recipe_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Recipe manifest must be readable JSON.") from error
    recipe = _require_object(recipe, "Recipe manifest")
    recipe_observation, recipe_source = _validated_recipe_observation(recipe, variant_id)
    recipe_sha256 = _sha256_bytes(recipe_bytes)
    manifest_path = Path(materialized_manifest_path)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Materialized manifest must be readable JSON.") from error
    manifest = _require_object(manifest, "Materialized manifest")
    observation, materialized_bytes = _validated_selected_observation(
        manifest,
        recipe,
        recipe_observation,
        recipe_source,
        recipe_sha256,
        variant_id,
        manifest_path.parent,
    )

    representation = extract_signal_representation(
        decode_expert_rgb_bytes(
            materialized_bytes,
            resolution=resolution,
            expected_width=recipe_observation["width"],
            expected_height=recipe_observation["height"],
        ),
        include_maps=True,
    )
    maps = _require_object(representation.get("maps"), "Signal diagnostic maps")
    map_contracts = {
        "luminance": ("luminance.png", 0.0, 1.0),
        "fourier_log_spectrum": (
            "fourier-log-spectrum.png",
            0.0,
            math.log1p(float((resolution * resolution) ** 2)),
        ),
        "neighbour_high_pass": ("neighbour-high-pass.png", 0.0, math.sqrt(2.0)),
        "residual": ("residual.png", -1.0, 1.0),
    }
    map_values = {
        "luminance": maps["luminance"],
        "fourier_log_spectrum": maps["spectrum"],
        "neighbour_high_pass": maps["high_pass"],
        "residual": maps["residual"],
    }
    output = resolve_output_directory(
        output_directory, "Signal diagnostic map output directory"
    )
    artifacts = {}
    for name, (filename, lower, upper) in map_contracts.items():
        encoded = _encode_map(map_values[name], lower=lower, upper=upper)
        _atomic_write(
            managed_output_path(output, filename, f"signal diagnostic map {filename}"),
            encoded,
        )
        artifacts[name] = {
            "filename": filename,
            "artifact_sha256": _sha256_bytes(encoded),
            "encoding": MAP_ENCODING,
            "width": resolution,
            "height": resolution,
            "scaling": {
                "method": "fixed-linear-clip-round-v1",
                "source_min": lower,
                "source_max": upper,
                "stored_min": 0,
                "stored_max": 65535,
                "quantization": "floor(scaled_value + 0.5)",
            },
        }

    document = {
        "schema_version": DIAGNOSTIC_MAP_SCHEMA_VERSION,
        "usage": "diagnostic-only",
        "excluded_from_inference_cache": True,
        "signal_representation_version": SIGNAL_REPRESENTATION_VERSION,
        "resolution": resolution,
        "variant": {
            key: observation.get(key)
            for key in (
                "source_id",
                "variant_id",
                "authenticity_label",
                "split",
                "condition_family",
                "severity",
                "corruption_seed",
            )
        },
        "input": {
            "recipe_manifest_sha256": recipe_sha256,
            "parent_recipe_manifest_sha256": manifest["parent_recipe_manifest_sha256"],
            "manifest_schema_version": manifest["manifest_schema_version"],
            "materialization_schema_version": manifest["materialization_schema_version"],
            "materialized_manifest_sha256": _sha256_bytes(manifest_bytes),
            "corruption_version": manifest["materialization"]["corruption_version"],
            "shared_observation_preprocessing_version": manifest["materialization"][
                "shared_observation_preprocessing_version"
            ],
            "materialized_encoding": observation["materialized_encoding"],
            "materialized_image_path": observation["materialized_image_path"],
            "materialized_sha256": observation["materialized_sha256"],
            "signal_shard_provenance": manifest["signal_shard_provenance"],
        },
        "feature_names": list(FEATURE_NAMES),
        "features": np.asarray(representation["features"], dtype=np.float64).tolist(),
        "artifacts": artifacts,
    }
    _atomic_write(
        managed_output_path(
            output, MAP_METADATA_FILENAME, "signal diagnostic map metadata"
        ),
        _canonical_json_bytes(document),
    )
    return document
