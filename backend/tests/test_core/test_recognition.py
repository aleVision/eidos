"""Unit tests for eidos_core.recognition: the paper's four outcomes."""

import numpy as np
import pytest

from eidos_core.eigenfaces import distance_from_face_space, project, train
from eidos_core.recognition import Outcome, Thresholds, classify, suggest_thresholds

from .conftest import SHAPE

Faces = tuple[np.ndarray, list[int]]


def _split(faces: np.ndarray, labels: list[int]) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Hold out the 5th sample of person 1; train on the rest."""
    held_out = faces[4]
    keep = [i for i in range(len(faces)) if i != 4]
    return faces[keep], [labels[i] for i in keep], held_out


@pytest.mark.parametrize(
    ("person_scale", "face_scale", "quadrant", "outcome"),
    [
        (2.0, 2.0, 1, Outcome.KNOWN),
        (0.5, 2.0, 2, Outcome.UNKNOWN_FACE),
        (2.0, 0.5, 3, Outcome.NOT_A_FACE),
        (0.5, 0.5, 4, Outcome.NOT_A_FACE),
    ],
)
def test_classify_all_four_quadrants(
    synthetic_faces: Faces,
    person_scale: float,
    face_scale: float,
    quadrant: int,
    outcome: Outcome,
) -> None:
    faces, labels, held_out = _split(*synthetic_faces)
    model = train(faces, labels, n_components=3, image_shape=SHAPE)
    eps = distance_from_face_space(model, held_out)
    eps_k = float(np.linalg.norm(project(model, held_out) - model.class_vectors[0]))

    thresholds = Thresholds(person=eps_k * person_scale, face_space=eps * face_scale)
    result = classify(model, held_out, thresholds)

    assert result.quadrant == quadrant
    assert result.outcome == outcome
    assert result.face_space_distance == pytest.approx(eps)
    assert result.nearest_distance == pytest.approx(eps_k)


def test_held_out_sample_recognized_as_its_person(synthetic_faces: Faces) -> None:
    faces, labels, held_out = _split(*synthetic_faces)
    model = train(faces, labels, n_components=3, image_shape=SHAPE)
    result = classify(model, held_out, suggest_thresholds(model, faces, labels))
    assert result.outcome == Outcome.KNOWN
    assert result.nearest_id == 1
    assert set(result.class_distances) == {1, 2, 3, 4}


def test_suggested_thresholds_accept_training_faces(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=3, image_shape=SHAPE)
    thresholds = suggest_thresholds(model, faces, labels)
    quadrants = [classify(model, f, thresholds).quadrant for f in faces]
    assert quadrants.count(1) / len(quadrants) >= 0.95


def test_noise_is_not_a_face_with_suggested_thresholds(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    model = train(faces, labels, n_components=3, image_shape=SHAPE)
    noise = np.random.default_rng(7).uniform(0, 255, faces.shape[1])
    result = classify(model, noise, suggest_thresholds(model, faces, labels))
    assert result.outcome == Outcome.NOT_A_FACE


def test_suggested_thresholds_work_when_model_is_full_rank(synthetic_faces: Faces) -> None:
    """With few photos M' can reach M - 1: training faces then sit exactly in face
    space (ε = 0), so thresholds must come from faces the model has not seen."""
    faces, labels, held_out = _split(*synthetic_faces)
    model = train(faces, labels, n_components=100, image_shape=SHAPE)
    result = classify(model, held_out, suggest_thresholds(model, faces, labels))
    assert result.outcome == Outcome.KNOWN


def test_suggest_thresholds_needs_four_faces(synthetic_faces: Faces) -> None:
    faces, labels = synthetic_faces
    few, few_labels = faces[[0, 1, 5]], [labels[0], labels[1], labels[5]]
    model = train(few, few_labels, n_components=2, image_shape=SHAPE)
    with pytest.raises(ValueError, match="at least 4 face samples"):
        suggest_thresholds(model, few, few_labels)
