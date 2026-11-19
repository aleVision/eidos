"""Unit tests for eidos_core.persistence."""

from pathlib import Path

import numpy as np

from eidos_core.eigenfaces import train
from eidos_core.persistence import load_model, save_model

from .conftest import SHAPE


def test_save_load_round_trip(
    synthetic_faces: tuple[np.ndarray, list[int]], tmp_path: Path
) -> None:
    model = train(*synthetic_faces, n_components=4, image_shape=SHAPE)
    path = tmp_path / "model.npz"
    save_model(model, path)
    loaded = load_model(path)

    assert loaded.image_shape == SHAPE
    for field in (
        "mean",
        "eigenfaces",
        "eigenvalues",
        "all_eigenvalues",
        "class_ids",
        "class_vectors",
    ):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(model, field))
