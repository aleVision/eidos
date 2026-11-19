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

from eidos_core.eigenfaces import EigenfaceModel, distance_from_face_space, project, train

# `suggest_thresholds`: percentile of held-out distances, and a margin on top.
# Face space gets a generous margin, so real faces are rarely called "not a face";
# the person threshold gets none, so strangers are less often taken for someone.
_THRESHOLD_PERCENTILE = 95
_PERSON_MARGIN = 1.0
_FACE_SPACE_MARGIN = 1.5


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

    The paper leaves θ to the application. Distances must be measured on faces
    the model has not seen: with few photos, M' can reach M - 1 and every
    training face then lies exactly in face space (ε = 0). So we use
    leave-one-out: retrain without face i, then measure face i's ε and its ε_k
    to its own person. Each threshold is the 95th percentile of those
    distances, times a margin (1.0 for the person, 1.5 for face space).
    """
    label_list = list(labels)
    own_distances, face_distances = [], []
    for i, (face, label) in enumerate(zip(faces, label_list, strict=True)):
        rest = [j for j in range(len(label_list)) if j != i]
        try:
            held_out_model = train(
                faces[rest],
                [label_list[j] for j in rest],
                model.eigenfaces.shape[1],
                model.image_shape,
            )
        except ValueError:  # too little data left without face i
            continue
        face_distances.append(distance_from_face_space(held_out_model, face))
        if label in held_out_model.class_ids:
            own = held_out_model.class_vectors[list(held_out_model.class_ids).index(label)]
            own_distances.append(np.linalg.norm(project(held_out_model, face) - own))

    if not face_distances:
        raise ValueError("Setting thresholds needs at least 4 face samples")
    if not own_distances:  # everyone has a single photo: use the spacing between people
        gaps = np.linalg.norm(model.class_vectors[:, None] - model.class_vectors[None], axis=2)
        own_distances = [0.5 * float(np.median(gaps[np.triu_indices(len(gaps), 1)]))]
    return Thresholds(
        person=_PERSON_MARGIN * float(np.percentile(own_distances, _THRESHOLD_PERCENTILE)),
        face_space=_FACE_SPACE_MARGIN * float(np.percentile(face_distances, _THRESHOLD_PERCENTILE)),
    )
