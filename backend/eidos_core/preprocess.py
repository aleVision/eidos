"""Turn a photo plus a hand-drawn face box into the vector Eigenfaces works on.

Turk & Pentland (1991) treat an N x N grey image as a point in N²-dimensional
space. Every face must therefore have the same size and alignment: we crop the
user's box, convert to grey, and resize to one canonical shape.

The paper also notes that the background "can significantly affect the
recognition performance" and suggests multiplying the face by a 2-D Gaussian
window centred on it. That de-emphasizes background and hairstyle and keeps
the inner face, which is what humans rely on most (section "Eliminating the
Background"). `gaussian_window` implements that idea.
"""

import cv2
import numpy as np

# Rows x columns = 112 x 92, the native size of the ORL face database.
FACE_SHAPE: tuple[int, int] = (112, 92)

# (x, y, width, height) in pixels of the source image.
FaceBox = tuple[int, int, int, int]

# ITU-R BT.601 luma weights, the classic RGB -> grey conversion.
_LUMA = np.array([0.299, 0.587, 0.114])


def to_grey(image: np.ndarray) -> np.ndarray:
    """Return a float64 grey image in [0, 255] from a grey, RGB or RGBA array."""
    pixels = image.astype(np.float64)
    if pixels.ndim == 2:
        return pixels
    grey: np.ndarray = pixels[..., :3] @ _LUMA
    return grey


def crop_face(image: np.ndarray, box: FaceBox) -> np.ndarray:
    """Return the region of `image` inside `box`. The box must fit the image."""
    x, y, w, h = box
    rows, cols = image.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > cols or y + h > rows:
        raise ValueError("Face box must lie inside the image")
    return image[y : y + h, x : x + w]


def normalize(face: np.ndarray, shape: tuple[int, int] = FACE_SHAPE) -> np.ndarray:
    """Resize a face to the canonical shape (rows, cols) as float64."""
    rows, cols = shape
    resized = cv2.resize(face.astype(np.float64), (cols, rows), interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.float64)


def gaussian_window(shape: tuple[int, int] = FACE_SHAPE, sigma_frac: float = 0.35) -> np.ndarray:
    """A 2-D Gaussian peaking at 1 in the centre; sigma = sigma_frac x each dimension."""
    rows, cols = shape
    y = np.arange(rows) - (rows - 1) / 2
    x = np.arange(cols) - (cols - 1) / 2
    gy = np.exp(-(y**2) / (2 * (sigma_frac * rows) ** 2))
    gx = np.exp(-(x**2) / (2 * (sigma_frac * cols) ** 2))
    return np.outer(gy, gx)


def prepare(image: np.ndarray, box: FaceBox, use_window: bool = True) -> np.ndarray:
    """Grey -> crop -> resize -> (Gaussian window) -> flat vector Γ of length N²."""
    face = normalize(crop_face(to_grey(image), box))
    if use_window:
        face = face * gaussian_window(face.shape)
    return face.ravel()
