"""Tests for eidos_core.detection: the face map ε(x, y) (Turk & Pentland 1991, Eqs. 10-15)."""

import numpy as np
import pytest

from eidos_core.detection import face_map
from eidos_core.eigenfaces import EigenfaceModel, distance_from_face_space, train
from eidos_core.preprocess import gaussian_window

SMALL = (12, 10)  # rows, cols: a tiny "face" keeps the reference computation fast


def small_model(use_window: bool) -> tuple[EigenfaceModel, np.ndarray]:
    """A model trained on 4 synthetic people x 5 samples of 12 x 10 pixels."""
    rng = np.random.default_rng(0)
    prototypes = rng.uniform(0, 255, (4, *SMALL))
    window = gaussian_window(SMALL) if use_window else np.ones(SMALL)
    raw = [p + rng.normal(0, 5, SMALL) for p in prototypes for _ in range(5)]
    faces = np.array([(f * window).ravel() for f in raw])
    labels = [i // 5 for i in range(20)]
    return train(faces, labels, n_components=6, image_shape=SMALL), np.array(raw)


def reference_map(model: EigenfaceModel, image: np.ndarray, use_window: bool) -> np.ndarray:
    """The obvious, slow way: crop every position and measure ε (Eq. 9) directly."""
    rows, cols = model.image_shape
    window = gaussian_window((rows, cols)) if use_window else np.ones((rows, cols))
    out = np.empty((image.shape[0] - rows + 1, image.shape[1] - cols + 1))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            patch = image[r : r + rows, c : c + cols] * window
            out[r, c] = distance_from_face_space(model, patch.ravel())
    return out


@pytest.mark.parametrize("use_window", [True, False])
def test_face_map_matches_sliding_window_reference(use_window: bool) -> None:
    model, _ = small_model(use_window)
    image = np.random.default_rng(1).uniform(0, 255, (30, 26))
    expected = reference_map(model, image, use_window)
    np.testing.assert_allclose(
        face_map(model, image, use_window), expected, rtol=1e-3, atol=1e-3 * expected.max()
    )


def test_face_map_shape() -> None:
    model, _ = small_model(True)
    assert face_map(model, np.zeros((30, 26))).shape == (30 - 12 + 1, 26 - 10 + 1)


def test_face_map_rejects_image_smaller_than_model() -> None:
    model, _ = small_model(True)
    with pytest.raises(ValueError, match="Image is too small to search for a face"):
        face_map(model, np.zeros((11, 26)))


def test_face_map_is_lowest_where_a_training_face_is_pasted() -> None:
    model, raw_faces = small_model(True)
    image = np.full((30, 26), 128.0)
    image[9 : 9 + 12, 7 : 7 + 10] = raw_faces[0]
    epsilon = face_map(model, image)
    assert np.unravel_index(np.argmin(epsilon), epsilon.shape) == (9, 7)
