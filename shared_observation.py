"""Shared lossless RGB observation decoding used before expert-specific processing."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SHARED_EXPERT_GEOMETRY = {224: 256, 384: 440}


def decode_shared_rgb(path: Path | str) -> np.ndarray:
    """Return the oriented HxWx3 uint8 pixels shared by both expert branches."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def prepare_shared_expert_rgb(path: Path | str, *, resolution: int) -> np.ndarray:
    """Apply the common checkpoint geometry and return HxWx3 RGB bytes."""
    resize_short_edge = SHARED_EXPERT_GEOMETRY.get(resolution)
    if resize_short_edge is None:
        raise ValueError("Shared expert resolution must be 384 or 224.")
    with Image.fromarray(decode_shared_rgb(path), mode="RGB") as image:
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
