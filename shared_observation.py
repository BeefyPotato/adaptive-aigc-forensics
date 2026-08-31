"""Shared lossless RGB observation decoding used before expert-specific processing."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SHARED_EXPERT_GEOMETRY = {224: 256, 384: 440}


def decode_shared_rgb(path: Path | str) -> np.ndarray:
    """Return the oriented HxWx3 uint8 pixels shared by both expert branches."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def decode_materialized_png_bytes(
    value: bytes,
    *,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    """Decode exactly the verified canonical materialized PNG bytes."""
    if (
        type(expected_width) is not int
        or expected_width <= 0
        or type(expected_height) is not int
        or expected_height <= 0
    ):
        raise ValueError("Materialized observation dimensions must be positive integers.")
    if not isinstance(value, bytes) or not value:
        raise ValueError("Materialized observation must contain PNG bytes.")
    try:
        with Image.open(io.BytesIO(value)) as opened:
            if opened.format != "PNG":
                raise ValueError("Materialized observation container must be PNG.")
            if opened.mode != "RGB":
                raise ValueError("Materialized PNG must contain exactly three RGB channels.")
            if opened.size != (expected_width, expected_height):
                raise ValueError("Materialized PNG dimensions disagree with the native observation.")
            opened.load()
            return np.asarray(opened, dtype=np.uint8).copy()
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError("Materialized observation is not a readable canonical PNG.") from error


def prepare_shared_expert_rgb_array(rgb: np.ndarray, *, resolution: int) -> np.ndarray:
    """Apply the common checkpoint geometry to an already verified RGB array."""
    if type(resolution) is not int or resolution not in SHARED_EXPERT_GEOMETRY:
        raise ValueError("Shared expert resolution must be 384 or 224.")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Shared expert input must be an HxWx3 uint8 RGB array.")
    resize_short_edge = SHARED_EXPERT_GEOMETRY[resolution]
    with Image.fromarray(array, mode="RGB") as image:
        width, height = image.size
        scale = resize_short_edge / min(width, height)
        resized = image.resize(
            (int(width * scale), int(height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
        left = (resized.width - resolution) // 2
        top = (resized.height - resolution) // 2
        cropped = resized.crop((left, top, left + resolution, top + resolution))
        return np.asarray(cropped, dtype=np.uint8).copy()


def prepare_shared_expert_rgb(path: Path | str, *, resolution: int) -> np.ndarray:
    """Apply the common checkpoint geometry and return HxWx3 RGB bytes."""
    return prepare_shared_expert_rgb_array(decode_shared_rgb(path), resolution=resolution)
