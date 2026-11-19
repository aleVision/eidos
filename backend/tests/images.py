"""Helpers that build in-memory test images (no files, no real faces)."""

import io

import numpy as np
from PIL import Image


def png_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def random_png(seed: int, size: tuple[int, int] = (150, 200)) -> bytes:
    """A random RGB image of size (rows, cols)."""
    rng = np.random.default_rng(seed)
    return png_bytes(rng.integers(0, 256, (*size, 3)))
