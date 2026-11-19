"""Shared synthetic data for core tests: no network, no real faces."""

import numpy as np
import pytest

N_PEOPLE = 4
N_SAMPLES = 5
SHAPE = (8, 8)  # 64 "pixels" keeps the tests fast and exact


@pytest.fixture
def synthetic_faces() -> tuple[np.ndarray, list[int]]:
    """4 people x 5 samples; each person is a fixed prototype plus small noise."""
    rng = np.random.default_rng(0)
    prototypes = rng.uniform(0, 255, (N_PEOPLE, SHAPE[0] * SHAPE[1]))
    faces, labels = [], []
    for person in range(N_PEOPLE):
        for _ in range(N_SAMPLES):
            faces.append(prototypes[person] + rng.normal(0, 5, prototypes.shape[1]))
            labels.append(person + 1)
    return np.array(faces), labels
