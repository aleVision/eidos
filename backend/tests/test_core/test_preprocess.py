"""Unit tests for eidos_core.preprocess."""

import numpy as np
import pytest

from eidos_core.preprocess import (
    FACE_SHAPE,
    crop_face,
    gaussian_window,
    normalize,
    prepare,
    to_grey,
)


def test_to_grey_rgb_and_rgba_match_luma() -> None:
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 255
    grey = to_grey(red)
    assert grey.dtype == np.float64
    np.testing.assert_allclose(grey, 0.299 * 255)

    rgba = np.concatenate([red, np.full((2, 2, 1), 255, dtype=np.uint8)], axis=2)
    np.testing.assert_allclose(to_grey(rgba), grey)


def test_to_grey_keeps_greyscale_values() -> None:
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(to_grey(image), image.astype(np.float64))


def test_crop_face_returns_box_region() -> None:
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    face = crop_face(image, (2, 3, 4, 5))
    assert face.shape == (5, 4)
    assert face[0, 0] == image[3, 2]


@pytest.mark.parametrize("box", [(-1, 0, 5, 5), (0, 0, 0, 5), (0, 0, 5, -1), (8, 8, 5, 5)])
def test_crop_face_rejects_bad_boxes(box: tuple[int, int, int, int]) -> None:
    image = np.zeros((10, 10))
    with pytest.raises(ValueError, match="Face box must lie inside the image"):
        crop_face(image, box)


def test_normalize_shape_and_dtype() -> None:
    face = normalize(np.random.default_rng(0).uniform(0, 255, (300, 200)))
    assert face.shape == FACE_SHAPE == (112, 92)
    assert face.dtype == np.float64


def test_gaussian_window_peak_and_symmetry() -> None:
    w = gaussian_window()
    assert w.shape == FACE_SHAPE
    assert w.max() == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_allclose(w, w[::-1, ::-1])
    assert w[0, 0] < 0.5


def test_prepare_returns_flat_vector() -> None:
    image = np.random.default_rng(1).integers(0, 256, (240, 180, 3), dtype=np.uint8)
    vector = prepare(image, (20, 30, 100, 120))
    assert vector.shape == (112 * 92,)

    plain = prepare(image, (20, 30, 100, 120), use_window=False)
    assert plain.min() >= 0 and plain.max() <= 255
    assert np.all(vector <= plain + 1e-9)
