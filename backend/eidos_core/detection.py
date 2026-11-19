"""Locating faces with "face space" (Turk & Pentland 1991, Eqs. 10-15, Figure 7).

Faces project onto face space almost unchanged; anything else does not. So the
distance from face space, ε, computed at every position of a photo, is a "face
map": it is lowest where a face is.

Cropping every position and calling `distance_from_face_space` works, but costs
one full projection per pixel. The paper rewrites ε² so that a whole map needs
only a handful of correlations over the image:

    ε² = ‖Φ‖² − Σ_i ω_i²        (Φ_f is orthogonal to Φ − Φ_f, Eqs. 11-12)

and each term is a correlation of the image I with a fixed kernel. With g the
Gaussian background window (all ones when disabled) and ⋆ correlation:

    ω_i  = I ⋆ (g·u_i) − u_iᵀΨ
    ‖Φ‖² = I² ⋆ g² − 2·(I ⋆ (g·Ψ)) + ‖Ψ‖²

That is M' + 2 correlations per map (the paper's "L + 1", plus one for g).

Not in the 1991 paper: with `normalize=True`, each box is first standardized
(zero mean, unit standard deviation), as later detectors do. Measured on ORL, the
paper's raw ε rates flat walls and smooth gradients as more face-like than real
faces; standardizing removes brightness and contrast, so only the pattern counts.
With μ and σ the box's mean and standard deviation (also correlations):

    ω_i  = (I ⋆ (g·u_i) − μ·Σ(g·u_i)) / σ − u_iᵀΨ
    ‖Φ‖² = (I² ⋆ g² − 2μ·(I ⋆ g²) + μ²·Σg²) / σ² − 2·(I ⋆ (g·Ψ) − μ·Σ(g·Ψ)) / σ + ‖Ψ‖²

Flat boxes (σ ≈ 0) cannot be standardized and get ε = ∞.

The map only finds faces of the model's size (112 x 92). `detect` therefore
searches several sizes: it resizes the photo so each candidate face size becomes
112 x 92, computes a map for each, and keeps the best position overall. This is
the multiscale search the paper suggests for unknown head size.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from eidos_core.eigenfaces import EigenfaceModel
from eidos_core.preprocess import MIN_CONTRAST, FaceBox, gaussian_window, to_grey

# Search a copy whose longer side is at most this many pixels: faces are found
# almost as well and the search is much faster on phone photos.
_SEARCH_SIDE = 640
# Smallest face (box height) searched, in pixels of that copy.
_MIN_FACE = 40


def _correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlation at every position where the kernel fits entirely (float32 inside OpenCV)."""
    result = cv2.matchTemplate(image, kernel.astype(np.float32), cv2.TM_CCORR)
    return np.asarray(result, dtype=np.float64)


def face_map(
    model: EigenfaceModel, image: np.ndarray, use_window: bool = True, normalize: bool = False
) -> np.ndarray:
    """ε, the distance from face space, for every top-left position of a model-sized box.

    `image` is a grey image (any float or int type). The result has shape
    (H - rows + 1, W - cols + 1) with (rows, cols) = model.image_shape. With
    `normalize=True` every box is standardized first (not in the paper).
    """
    rows, cols = model.image_shape
    if image.shape[0] < rows or image.shape[1] < cols:
        raise ValueError("Image is too small to search for a face")

    g = gaussian_window((rows, cols)) if use_window else np.ones((rows, cols))
    mean = model.mean.reshape(rows, cols)
    pixels = image.astype(np.float64)

    if not normalize:  # the paper (Eqs. 10-15)
        img = pixels.astype(np.float32)
        phi_energy = (
            _correlate(img * img, g * g)  # ‖g·Γ‖²
            - 2 * _correlate(img, g * mean)  # −2 (g·Γ)ᵀΨ
            + float(model.mean @ model.mean)  # ‖Ψ‖²
        )
        explained = np.zeros_like(phi_energy)
        for u in model.eigenfaces.T:
            weight = _correlate(img, g * u.reshape(rows, cols)) - float(u @ model.mean)  # ω_i
            explained += weight * weight
        paper_distances: np.ndarray = np.sqrt(np.maximum(phi_energy - explained, 0.0))
        return paper_distances

    # Standardized boxes. Subtracting the image mean first keeps float32 precise
    # and changes nothing, because standardizing removes any constant offset.
    img = (pixels - pixels.mean()).astype(np.float32)
    n = rows * cols
    box_mean = _correlate(img, np.ones((rows, cols))) / n  # μ
    variance = _correlate(img * img, np.ones((rows, cols))) / n - box_mean**2
    flat = variance < MIN_CONTRAST**2
    sigma = np.sqrt(np.where(flat, 1.0, variance))

    g2 = g * g
    energy = (
        _correlate(img * img, g2) - 2 * box_mean * _correlate(img, g2) + box_mean**2 * g2.sum()
    ) / sigma**2  # ‖g·Γ'‖²
    cross = (_correlate(img, g * mean) - box_mean * float((g * mean).sum())) / sigma  # (g·Γ')ᵀΨ
    phi_energy = energy - 2 * cross + float(model.mean @ model.mean)
    explained = np.zeros_like(phi_energy)
    for u in model.eigenfaces.T:
        kernel = g * u.reshape(rows, cols)
        weight = (_correlate(img, kernel) - box_mean * float(kernel.sum())) / sigma
        weight -= float(u @ model.mean)  # ω_i
        explained += weight * weight

    distances: np.ndarray = np.sqrt(np.maximum(phi_energy - explained, 0.0))
    distances[flat] = np.inf
    return distances


@dataclass(frozen=True)
class Detection:
    box: FaceBox  # (x, y, w, h) in pixels of the original image, inside it
    distance: float  # ε at the best position and size
    face_map: np.ndarray  # the ε map of the best size, for display
    scale: float  # resize factor from the original image to the best size's image


def detect(
    model: EigenfaceModel,
    image: np.ndarray,
    use_window: bool = True,
    n_scales: int = 8,
    normalize: bool = False,
    min_face_fraction: float = 0.35,
) -> Detection:
    """The most face-like box in `image` (grey or RGB), searched over `n_scales` sizes.

    Box heights run geometrically from `min_face_fraction` to 100% of the image's
    shorter side, and keep the model's aspect ratio. The default searches portraits
    only: measured on ORL faces, ε is not comparable across sizes (resizing smooths
    small boxes, which then look more face-like), so letting small boxes compete
    with large ones makes the background win.
    """
    rows, cols = model.image_shape
    grey = to_grey(image)
    shrink = min(1.0, _SEARCH_SIDE / max(grey.shape))
    if shrink < 1.0:
        size = (round(grey.shape[1] * shrink), round(grey.shape[0] * shrink))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    height, width = grey.shape

    best: tuple[float, int, int, float, np.ndarray] | None = None
    for face_h in np.geomspace(min_face_fraction, 1.0, n_scales) * min(height, width):
        face_w = face_h * cols / rows
        if face_h < _MIN_FACE or face_h > height or face_w > width:
            continue
        factor = rows / face_h  # resize so this face size becomes the model's size
        size = (max(cols, round(width * factor)), max(rows, round(height * factor)))
        interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(grey, size, interpolation=interpolation)
        epsilon = face_map(model, resized, use_window, normalize)
        row, col = np.unravel_index(np.argmin(epsilon), epsilon.shape)
        if best is None or epsilon[row, col] < best[0]:
            best = (float(epsilon[row, col]), int(row), int(col), factor, epsilon)
    if best is None:
        raise ValueError("Image is too small to search for a face")

    distance, r, c, factor, epsilon = best
    if not np.isfinite(distance):
        raise ValueError("No face-like region found")
    scale = factor * shrink  # original image -> searched image
    original_h, original_w = image.shape[:2]
    w = min(round(cols / scale), original_w)
    h = min(round(rows / scale), original_h)
    x = min(max(0, round(c / scale)), original_w - w)
    y = min(max(0, round(r / scale)), original_h - h)
    return Detection(box=(x, y, w, h), distance=distance, face_map=epsilon, scale=scale)
