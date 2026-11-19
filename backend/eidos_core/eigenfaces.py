"""Eigenfaces: learning "face space" (Turk & Pentland, 1991).

Notation follows the paper:
    Γ_n   training face n, flattened to a vector of length N²
    Ψ     the mean face
    Φ_n   Γ_n - Ψ, how face n differs from the mean
    A     the N² x M matrix [Φ_1 ... Φ_M]
    u_k   the k-th eigenface, eigenvector of the covariance C = A Aᵀ
    ω_k   weight of eigenface k in a face, Ω = [ω_1 ... ω_M']
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from eidos_core.preprocess import FACE_SHAPE

# Eigenvalues below this fraction of the largest are numerically zero.
_RANK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class EigenfaceModel:
    mean: np.ndarray  # Ψ, shape (N²,)
    eigenfaces: np.ndarray  # U = [u_1 ... u_M'], shape (N², M'), orthonormal columns
    eigenvalues: np.ndarray  # λ_k for the kept M' eigenfaces, descending
    all_eigenvalues: np.ndarray  # every non-zero λ_k, for the variance curve
    class_ids: np.ndarray  # (K,) person ids
    class_vectors: np.ndarray  # (K, M'), Ω_k = mean projection of each person
    image_shape: tuple[int, int]


def train(
    faces: np.ndarray,
    labels: Sequence[int],
    n_components: int,
    image_shape: tuple[int, int] = FACE_SHAPE,
) -> EigenfaceModel:
    """Compute face space from M training faces (rows of `faces`), Eqs. 1-6.

    The covariance C = A Aᵀ is N² x N² (10,304² for a 92 x 112 face), far too big.
    The paper's trick: if L = Aᵀ A (only M x M) has eigenvectors v_i, then
    A Aᵀ (A v_i) = μ_i (A v_i), so A v_i are the eigenvectors of C (Eqs. 4-5).
    """
    if len(faces) != len(labels):
        raise ValueError("Each face needs exactly one label")
    if len(faces) < 3:
        raise ValueError("Training needs at least 3 face samples")
    if len(set(labels)) < 2:
        raise ValueError("Training needs at least 2 different people")

    m = len(faces)
    mean = faces.mean(axis=0)  # Ψ
    a = (faces - mean).T  # A = [Φ_1 ... Φ_M], shape (N², M)

    mu, v = np.linalg.eigh(a.T @ a)  # L = Aᵀ A, eigen-decomposition of an M x M matrix
    order = np.argsort(mu)[::-1]
    mu, v = mu[order], v[:, order]
    nonzero = mu > _RANK_TOLERANCE * mu[0]
    mu, v = mu[nonzero], v[:, nonzero]

    u = a @ v  # Eq. 6: u_l = Σ_k v_lk Φ_k
    u /= np.linalg.norm(u, axis=0)
    eigenvalues = mu / m  # λ_k (Eq. 1); μ_k / M because C = (1/M) A Aᵀ (Eq. 3)

    keep = min(n_components, u.shape[1])
    eigenfaces = u[:, :keep]

    weights = eigenfaces.T @ a  # every training face projected: Ω columns (Eq. 7)
    label_array = np.asarray(labels)
    class_ids = np.unique(label_array)
    class_vectors = np.array([weights[:, label_array == c].mean(axis=1) for c in class_ids])

    return EigenfaceModel(
        mean=mean,
        eigenfaces=eigenfaces,
        eigenvalues=eigenvalues[:keep],
        all_eigenvalues=eigenvalues,
        class_ids=class_ids,
        class_vectors=class_vectors,
        image_shape=image_shape,
    )


def project(model: EigenfaceModel, face: np.ndarray) -> np.ndarray:
    """Eq. 7: ω_k = u_kᵀ (Γ - Ψ). Returns Ω, the face's coordinates in face space."""
    weights: np.ndarray = model.eigenfaces.T @ (face - model.mean)
    return weights


def reconstruct(model: EigenfaceModel, weights: np.ndarray, k: int | None = None) -> np.ndarray:
    """Ψ + Σ ω_i u_i over the first k eigenfaces (all kept ones when k is None)."""
    k = len(weights) if k is None else k
    image: np.ndarray = model.mean + model.eigenfaces[:, :k] @ weights[:k]
    return image


def distance_from_face_space(model: EigenfaceModel, face: np.ndarray) -> float:
    """Eq. 9: ε = ‖Φ - Φ_f‖, the part of the image the eigenfaces cannot explain."""
    phi = face - model.mean
    phi_f = model.eigenfaces @ project(model, face)
    return float(np.linalg.norm(phi - phi_f))


def explained_variance(model: EigenfaceModel) -> np.ndarray:
    """Cumulative fraction of the training variance explained by the first k eigenfaces."""
    total = model.all_eigenvalues.sum()
    curve: np.ndarray = np.cumsum(model.all_eigenvalues) / total
    return curve
