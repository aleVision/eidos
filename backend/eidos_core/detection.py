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
"""

import cv2
import numpy as np

from eidos_core.eigenfaces import EigenfaceModel
from eidos_core.preprocess import gaussian_window


def _correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlation at every position where the kernel fits entirely (float32 inside OpenCV)."""
    result = cv2.matchTemplate(image, kernel.astype(np.float32), cv2.TM_CCORR)
    return np.asarray(result, dtype=np.float64)


def face_map(model: EigenfaceModel, image: np.ndarray, use_window: bool = True) -> np.ndarray:
    """ε, the distance from face space, for every top-left position of a model-sized box.

    `image` is a grey image (any float or int type). The result has shape
    (H - rows + 1, W - cols + 1) with (rows, cols) = model.image_shape.
    """
    rows, cols = model.image_shape
    if image.shape[0] < rows or image.shape[1] < cols:
        raise ValueError("Image is too small to search for a face")

    g = gaussian_window((rows, cols)) if use_window else np.ones((rows, cols))
    pixels = image.astype(np.float32)
    mean = model.mean.reshape(rows, cols)

    phi_energy = (
        _correlate(pixels * pixels, g * g)  # ‖g·Γ‖²
        - 2 * _correlate(pixels, g * mean)  # −2 (g·Γ)ᵀΨ
        + float(model.mean @ model.mean)  # ‖Ψ‖²
    )
    explained = np.zeros_like(phi_energy)
    for u in model.eigenfaces.T:
        weight = _correlate(pixels, g * u.reshape(rows, cols)) - float(u @ model.mean)  # ω_i
        explained += weight * weight

    epsilon_squared = np.maximum(phi_energy - explained, 0.0)  # rounding can dip below 0
    distances: np.ndarray = np.sqrt(epsilon_squared)
    return distances
