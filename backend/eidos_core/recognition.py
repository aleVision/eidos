"""Recognition with two distances and four outcomes (Turk & Pentland, 1991).

For a new image the paper measures two things:
    ε_k  distance in face space to person k's class vector Ω_k  (Eq. 8)
    ε    distance from the image to face space itself            (Eq. 9)

With a threshold on each, there are four possibilities:
    1. near face space, near a person       -> a known person
    2. near face space, far from everyone   -> an unknown face
    3. far from face space, near a person   -> not a face (a false positive
                                               most systems would accept)
    4. far from face space, far from all    -> not a face
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from eidos_core.eigenfaces import EigenfaceModel, distance_from_face_space, project

# Margin applied on top of the training distances by `suggest_thresholds`.
_THRESHOLD_MARGIN = 1.5
_THRESHOLD_PERCENTILE = 95


class Outcome(StrEnum):
    KNOWN = "known"
    UNKNOWN_FACE = "unknown_face"
    NOT_A_FACE = "not_a_face"


@dataclass(frozen=True)
class Thresholds:
    person: float  # θ for ε_k: max distance to a person's class vector
    face_space: float  # θ for ε: max distance from face space


@dataclass(frozen=True)
class Recognition:
    weights: np.ndarray  # Ω
    face_space_distance: float  # ε
    class_distances: dict[int, float]  # ε_k for every person id
    nearest_id: int
    nearest_distance: float
    quadrant: int  # 1-4, as numbered in the paper
    outcome: Outcome


def classify(model: EigenfaceModel, face: np.ndarray, thresholds: Thresholds) -> Recognition:
    """Project `face`, measure ε and every ε_k, and pick one of the four outcomes."""
    weights = project(model, face)
    eps = distance_from_face_space(model, face)
    distances = np.linalg.norm(model.class_vectors - weights, axis=1)  # Eq. 8
    nearest = int(np.argmin(distances))

    near_face = eps <= thresholds.face_space
    near_person = distances[nearest] <= thresholds.person
    if near_face:
        quadrant, outcome = (1, Outcome.KNOWN) if near_person else (2, Outcome.UNKNOWN_FACE)
    else:
        quadrant, outcome = (3, Outcome.NOT_A_FACE) if near_person else (4, Outcome.NOT_A_FACE)

    return Recognition(
        weights=weights,
        face_space_distance=eps,
        class_distances={int(c): float(d) for c, d in zip(model.class_ids, distances, strict=True)},
        nearest_id=int(model.class_ids[nearest]),
        nearest_distance=float(distances[nearest]),
        quadrant=quadrant,
        outcome=outcome,
    )


def suggest_thresholds(
    model: EigenfaceModel, faces: np.ndarray, labels: Sequence[int]
) -> Thresholds:
    """A simple, data-driven starting point the user can tune.

    The paper leaves θ to the application. We take the 95th percentile of the
    training faces' own distances (ε_k to their own person, and ε) and add a
    50% margin, so typical enrolled faces are accepted and outliers are not.
    """
    class_index = {int(c): i for i, c in enumerate(model.class_ids)}
    own_distances, face_distances = [], []
    for face, label in zip(faces, labels, strict=True):
        weights = project(model, face)
        own_distances.append(np.linalg.norm(weights - model.class_vectors[class_index[label]]))
        face_distances.append(distance_from_face_space(model, face))
    return Thresholds(
        person=_THRESHOLD_MARGIN * float(np.percentile(own_distances, _THRESHOLD_PERCENTILE)),
        face_space=_THRESHOLD_MARGIN * float(np.percentile(face_distances, _THRESHOLD_PERCENTILE)),
    )
