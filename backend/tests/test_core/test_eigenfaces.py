"""Unit tests for eidos_core.eigenfaces, checked against plain SVD."""

import numpy as np
import pytest

from eidos_core.eigenfaces import (
    distance_from_face_space,
    explained_variance,
    project,
    reconstruct,
    train,
)

from .conftest import SHAPE

Faces = tuple[np.ndarray, list[int]]


def test_eigenfaces_match_svd_up_to_sign(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=5, image_shape=SHAPE)
    a = (faces - faces.mean(axis=0)).T
    svd_u, _, _ = np.linalg.svd(a, full_matrices=False)
    for k in range(5):
        assert abs(model.eigenfaces[:, k] @ svd_u[:, k]) > 0.999


def test_eigenfaces_are_orthonormal(synthetic_faces: Faces) -> None:
    model = train(*synthetic_faces, n_components=10, image_shape=SHAPE)
    u = model.eigenfaces
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-8)


def test_eigenvalues_descending_and_equal_mu_over_m(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=10, image_shape=SHAPE)
    assert np.all(np.diff(model.eigenvalues) <= 0)
    a = faces - faces.mean(axis=0)
    mu = np.sort(np.linalg.eigvalsh(a @ a.T))[::-1]
    np.testing.assert_allclose(model.eigenvalues, mu[:10] / len(faces))


def test_full_rank_reconstruction_reproduces_training_face(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=19, image_shape=SHAPE)
    rebuilt = reconstruct(model, project(model, faces[3]))
    np.testing.assert_allclose(rebuilt, faces[3], atol=1e-6)


def test_reconstruct_with_k_zero_is_the_mean(synthetic_faces: Faces) -> None:
    model = train(*synthetic_faces, n_components=5, image_shape=SHAPE)
    weights = project(model, synthetic_faces[0][0])
    np.testing.assert_allclose(reconstruct(model, weights, k=0), model.mean)


def test_noise_is_far_from_face_space(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=5, image_shape=SHAPE)
    worst_face = max(distance_from_face_space(model, f) for f in faces)
    noise = np.random.default_rng(42).uniform(0, 255, faces.shape[1])
    assert distance_from_face_space(model, noise) > 3 * worst_face


def test_class_vectors_are_mean_projections(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=5, image_shape=SHAPE)
    assert list(model.class_ids) == [1, 2, 3, 4]
    expected = np.mean([project(model, f) for f in faces[:5]], axis=0)
    np.testing.assert_allclose(model.class_vectors[0], expected)


def test_explained_variance_monotonic_ends_at_one(synthetic_faces: Faces) -> None:
    curve = explained_variance(train(*synthetic_faces, n_components=3, image_shape=SHAPE))
    assert np.all(np.diff(curve) >= 0)
    assert curve[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("n_faces", "labels"),
    [(2, [1, 2]), (5, [1, 1, 1, 1, 1])],
)
def test_train_rejects_too_few_faces_or_people(n_faces: int, labels: list[int]) -> None:
    faces = np.random.default_rng(0).uniform(0, 255, (n_faces, 64))
    with pytest.raises(ValueError):
        train(faces, labels, n_components=2, image_shape=SHAPE)


def test_n_components_clipped_to_rank(synthetic_faces: Faces) -> None:
    model = train(*synthetic_faces, n_components=500, image_shape=SHAPE)
    assert model.eigenfaces.shape[1] <= 19
