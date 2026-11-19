"""Save and load an EigenfaceModel as a plain .npz file (no pickle)."""

from pathlib import Path

import numpy as np

from eidos_core.eigenfaces import EigenfaceModel

_ARRAYS = ("mean", "eigenfaces", "eigenvalues", "all_eigenvalues", "class_ids", "class_vectors")


def save_model(model: EigenfaceModel, path: Path) -> None:
    arrays = {name: getattr(model, name) for name in _ARRAYS}
    with path.open("wb") as file:
        np.savez_compressed(file, image_shape=np.array(model.image_shape), **arrays)


def load_model(path: Path) -> EigenfaceModel:
    with np.load(path, allow_pickle=False) as data:
        rows, cols = (int(v) for v in data["image_shape"])
        return EigenfaceModel(image_shape=(rows, cols), **{name: data[name] for name in _ARRAYS})
