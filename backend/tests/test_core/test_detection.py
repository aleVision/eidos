"""Tests for eidos_core.detection: the face map ε(x, y) (Turk & Pentland 1991, Eqs. 10-15)."""

import cv2
import numpy as np
import pytest

from eidos_core.detection import detect, face_map
from eidos_core.eigenfaces import EigenfaceModel, distance_from_face_space, train
from eidos_core.preprocess import FACE_SHAPE, gaussian_window

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


def reference_map(
    model: EigenfaceModel, image: np.ndarray, use_window: bool, normalize: bool = False
) -> np.ndarray:
    """The obvious, slow way: crop every position and measure ε (Eq. 9) directly."""
    rows, cols = model.image_shape
    window = gaussian_window((rows, cols)) if use_window else np.ones((rows, cols))
    out = np.empty((image.shape[0] - rows + 1, image.shape[1] - cols + 1))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            patch = image[r : r + rows, c : c + cols].astype(np.float64)
            if normalize:
                patch = (patch - patch.mean()) / patch.std()
            out[r, c] = distance_from_face_space(model, (patch * window).ravel())
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


def full_size_model(standardize: bool = False) -> tuple[EigenfaceModel, np.ndarray]:
    """A model at the real 112 x 92 face size, trained on windowed synthetic faces.

    The synthetic faces are smooth (random 14 x 12 patterns upsampled), like real
    faces. Pixel noise would stop resembling itself after any resize, so it could
    not test a search across sizes.
    """
    rng = np.random.default_rng(0)
    prototypes = np.array(
        [
            cv2.resize(rng.uniform(0, 255, (14, 12)), (92, 112), interpolation=cv2.INTER_CUBIC)
            for _ in range(4)
        ]
    )
    window = gaussian_window()
    raw = [p + rng.normal(0, 5, FACE_SHAPE) for p in prototypes for _ in range(5)]
    if standardize:
        raw = [(f - f.mean()) / f.std() for f in raw]
    faces = np.array([(f * window).ravel() for f in raw])
    return train(faces, [i // 5 for i in range(20)], n_components=10), prototypes


def test_detect_finds_pasted_face_at_scale() -> None:
    model, prototypes = full_size_model()
    face = cv2.resize(prototypes[2], (138, 168), interpolation=cv2.INTER_LINEAR)  # 1.5x
    # A textured background: these synthetic faces average to flat grey, so a flat
    # grey background would itself look like the mean face.
    image = np.random.default_rng(9).uniform(0, 255, (300, 400))
    image[40 : 40 + 168, 60 : 60 + 138] = face

    result = detect(model, image)

    tolerance = 0.15 * 168  # one step of the 8-size search grid
    for found, expected in zip(result.box, (60, 40, 138, 168), strict=True):
        assert abs(found - expected) <= tolerance
    noise = np.random.default_rng(5).uniform(0, 255, FACE_SHAPE) * gaussian_window()
    assert result.distance < distance_from_face_space(model, noise.ravel())


def test_detect_box_inside_panorama() -> None:
    model, _ = full_size_model()
    image = np.random.default_rng(2).uniform(0, 255, (120, 1200))
    x, y, w, h = detect(model, image).box
    assert x >= 0 and y >= 0 and x + w <= 1200 and y + h <= 120


def test_detect_rejects_tiny_image() -> None:
    model, _ = full_size_model()
    with pytest.raises(ValueError, match="Image is too small to search for a face"):
        detect(model, np.zeros((30, 30)))


def test_detect_accepts_rgb() -> None:
    model, prototypes = full_size_model()
    grey = np.full((200, 180), 100.0)
    grey[30:142, 40:132] = prototypes[1]
    rgb = np.repeat(grey[..., None], 3, axis=2).astype(np.uint8)
    assert detect(model, rgb).box == detect(model, grey.astype(np.uint8)).box


def test_normalized_face_map_matches_reference() -> None:
    model, _ = small_model(True)
    image = np.random.default_rng(3).uniform(0, 255, (30, 26))
    expected = reference_map(model, image, use_window=True, normalize=True)
    np.testing.assert_allclose(
        face_map(model, image, normalize=True), expected, rtol=1e-3, atol=1e-3 * expected.max()
    )


def test_normalized_face_map_ignores_brightness_and_contrast() -> None:
    model, _ = small_model(True)
    image = np.random.default_rng(3).uniform(50, 150, (30, 26))
    np.testing.assert_allclose(
        face_map(model, image * 0.5 + 40, normalize=True),
        face_map(model, image, normalize=True),
        rtol=1e-3,
    )


def test_normalized_face_map_flat_region_is_infinite() -> None:
    model, _ = small_model(True)
    image = np.random.default_rng(3).uniform(0, 255, (30, 26))
    image[:14, :12] = 90.0  # exactly one flat box at the top-left corner
    epsilon = face_map(model, image, normalize=True)
    assert np.isinf(epsilon[0, 0])
    assert np.isfinite(epsilon[5:, 5:]).all()


def test_normalized_detect_prefers_face_over_smooth_background() -> None:
    model, prototypes = full_size_model(standardize=True)
    background = np.tile(np.linspace(60, 200, 400), (300, 1))  # a smooth lighting gradient
    image = background.copy()
    image[60 : 60 + 112, 150 : 150 + 92] = prototypes[1]
    x, y, w, h = detect(model, image, normalize=True).box
    assert abs(x - 150) <= 17 and abs(y - 60) <= 17 and abs(h - 112) <= 17


def test_normalized_detect_on_blank_image_raises() -> None:
    model, _ = full_size_model(standardize=True)
    with pytest.raises(ValueError, match="No face-like region found"):
        detect(model, np.full((200, 200), 77.0), normalize=True)
