"""Synthetic frame pairs with known motion, for testing and demonstrating optical flow."""

import cv2
import numpy as np


def _texture(rng: np.random.Generator, size: int, blur: float) -> np.ndarray:
    noise = rng.uniform(0, 1, (size, size)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), blur)
    return (smooth - smooth.min()) / (smooth.max() - smooth.min())


def _shift(image: np.ndarray, dx: float) -> np.ndarray:
    """Content moved right by dx pixels (bilinear; edges replicated)."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
    h, w = image.shape
    return np.asarray(
        cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    )


def two_surfaces(
    size: int = 128, motion: float = 1.0, noise: float = 0.0, seed: int = 1993
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The paper's test (Black & Anandan §6.1): a textured surface that stays still,
    and a textured square (the central half) moving left by `motion` pixels.

    `noise` adds uniform random values of up to ±noise·255 to every pixel of frame 2.
    The paper's "10% uniform random noise" is ambiguous; read as ±10% of the range,
    its error levels (RMS horizontal flow 0.18 quadratic vs 0.10 robust) are matched
    in magnitude and order here, while replacing 10% of the pixels is far harsher.
    Returns frame1, frame2 (float32, 0-255) and the true flow
    (u, v) indexed at frame 2's pixels, as the solver reports it: the square moves
    left, so the truth marks where the square is in frame 2.
    """
    rng = np.random.default_rng(seed)
    background = 40 + 175 * _texture(rng, size, 1.5)
    square = 40 + 175 * _texture(rng, size, 3.0)  # a different, coarser texture
    mask = np.zeros((size, size), np.float32)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0

    frame1 = background * (1 - mask) + square * mask
    moved_mask = _shift(mask, -motion)
    frame2 = background * (1 - moved_mask) + _shift(square, -motion) * moved_mask
    if noise > 0:
        amplitude = noise * 255.0
        frame2 = np.clip(frame2 + rng.uniform(-amplitude, amplitude, frame2.shape), 0, 255)

    u_true = np.where(moved_mask > 0.5, -float(motion), 0.0)
    return (
        frame1.astype(np.float32),
        frame2.astype(np.float32),
        u_true,
        np.zeros_like(u_true),
    )
