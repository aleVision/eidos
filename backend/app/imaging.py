"""Image I/O for the web adapter: decode uploads and encode arrays as PNG."""

import base64
import io

import numpy as np
from PIL import Image, ImageOps


def read_image(data: bytes) -> np.ndarray:
    """Decode an upload to an upright RGB uint8 array.

    Phone photos are often stored sideways with an EXIF "orientation" tag. The
    browser shows them upright and the user draws the face box on that view,
    so we apply the same rotation here before using the box.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as error:  # Pillow raises many types for bad input
        raise ValueError("Unsupported or corrupt image") from error
    return np.asarray(upright, dtype=np.uint8)


def array_png(pixels: np.ndarray) -> bytes:
    """Encode a uint8 grey or RGB array as PNG."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def to_png(vector: np.ndarray, shape: tuple[int, int], stretch: bool) -> bytes:
    """Encode a flat face vector as a grey PNG.

    stretch=True rescales min..max to 0..255, needed for eigenfaces, whose
    values are small and signed. Otherwise values are clipped to 0..255.
    """
    pixels = vector.reshape(shape).astype(np.float64)
    if stretch:
        span = pixels.max() - pixels.min()
        pixels = (pixels - pixels.min()) / (span if span > 0 else 1.0) * 255
    return array_png(np.clip(pixels, 0, 255).astype(np.uint8))


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def face_map_png(epsilon: np.ndarray, face_shape: tuple[int, int], size: tuple[int, int]) -> bytes:
    """The face map as a grey PNG of `size` (width, height): darker = more face-like.

    Each ε value belongs to a box; it is drawn at the box centre. The border,
    where no whole box fits, gets the largest (brightest) value.
    """
    rows, cols = face_shape
    finite = np.isfinite(epsilon)
    brightest = epsilon[finite].max() if finite.any() else 0.0
    epsilon = np.where(finite, epsilon, brightest)  # flat boxes (ε = ∞) show as least face-like
    canvas = np.full((epsilon.shape[0] + rows - 1, epsilon.shape[1] + cols - 1), brightest)
    canvas[rows // 2 : rows // 2 + epsilon.shape[0], cols // 2 : cols // 2 + epsilon.shape[1]] = (
        epsilon
    )
    span = canvas.max() - canvas.min()
    grey = (canvas - canvas.min()) / (span if span > 0 else 1.0) * 255
    resized = Image.fromarray(grey.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    return array_png(np.asarray(resized))
