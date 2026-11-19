"""Total variation denoising: the ROF model (Rudin, Osher & Fatemi, 1992).

L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise removal
algorithms", Physica D 60:259-268, 1992.

The idea: of all images u close to the noisy photo f, prefer the one with the
smallest total variation, the sum of gradient magnitudes. Noise adds a lot of
variation; an edge adds only its height. So minimizing

    E(u) = TV(u) + (λ/2) · Σ (u − f)²,    TV(u) = Σ |∇u|

removes noise but keeps edges sharp, which blurring cannot do. λ sets the balance:
large λ stays close to f, small λ cleans more.

Model vs. solver: the energy E is the 1992 contribution. *How* to find its minimum
is a separate choice; Eidos uses Chambolle's dual algorithm (2004), and the tests
check it against the paper's own gradient flow.

Images are floats in [0, 1]; colour images are (H, W, C), each channel independent.
"""

from dataclasses import dataclass

import numpy as np

# Chambolle's step size; τ ≤ 1/8 guarantees convergence on a 2-D grid.
_TAU = 0.125


def gradient(u: np.ndarray) -> np.ndarray:
    """Forward differences, shape (2, H, W): [downward, rightward].

    The difference across the last row/column is 0 (Neumann boundary).
    """
    g = np.zeros((2, *u.shape), dtype=u.dtype)
    g[0, :-1, :] = u[1:, :] - u[:-1, :]
    g[1, :, :-1] = u[:, 1:] - u[:, :-1]
    return g


def divergence(p: np.ndarray) -> np.ndarray:
    """The negative adjoint of `gradient`: Σ ∇u·p = −Σ u·div p for every u and p."""
    py = p[0].copy()
    py[-1, :] = 0.0  # gradient never produces a value there
    px = p[1].copy()
    px[:, -1] = 0.0
    div: np.ndarray = py + px
    div[1:, :] -= py[:-1, :]
    div[:, 1:] -= px[:, :-1]
    return div


def _total_variation(u: np.ndarray) -> float:
    g = gradient(u)
    return float(np.sum(np.sqrt(g[0] ** 2 + g[1] ** 2)))


def tv_energy(u: np.ndarray, f: np.ndarray, lam: float) -> float:
    """E(u) = TV(u) + (λ/2)·Σ(u − f)². Colour images sum TV over channels."""
    channels = [u] if u.ndim == 2 else [u[..., c] for c in range(u.shape[2])]
    tv = sum(_total_variation(channel) for channel in channels)
    fidelity = 0.5 * lam * float(np.sum((u - f) ** 2))
    return tv + fidelity


@dataclass(frozen=True)
class DenoiseResult:
    image: np.ndarray  # same shape as the input, values in [0, 1]
    energies: list[float]  # E(u) after each iteration (summed over channels)


def _chambolle(f: np.ndarray, lam: float, n_iter: int) -> tuple[np.ndarray, list[float]]:
    """Minimize TV(u) + (λ/2)·Σ(u − f)² for one channel (Chambolle 2004).

    With θ = 1/λ the minimizer is u = f − θ·div p, where the dual field p solves
    p ← (p + τ·g) / (1 + τ·|g|),  g = ∇(div p − f/θ).

    Since div p − f/θ = −u/θ, g is just −∇u/θ: each iteration needs one divergence
    and one gradient, and the energy E(u) comes almost free from the same |∇u|.
    """
    theta = 1.0 / lam
    step = _TAU / theta
    p = np.zeros((2, *f.shape), dtype=np.float32)
    energies = []
    for i in range(n_iter + 1):
        u = f - theta * divergence(p)
        grad_u = gradient(u)
        magnitude = np.sqrt(grad_u[0] ** 2 + grad_u[1] ** 2)
        if i > 0:  # energy of the image produced by the previous update
            fidelity = 0.5 * lam * float(np.sum((u - f) ** 2))
            energies.append(float(magnitude.sum()) + fidelity)
        if i < n_iter:
            p = (p - step * grad_u) / (1.0 + step * magnitude)
    return u, energies


def denoise_tv(image: np.ndarray, lam: float, n_iter: int = 100) -> DenoiseResult:
    """Total variation denoising of a grey (H, W) or colour (H, W, C) image in [0, 1].

    Each colour channel is denoised independently, in float32 to save memory.
    """
    if lam <= 0:
        raise ValueError("Strength must be positive")
    f = image.astype(np.float32)
    channels = [f] if f.ndim == 2 else [f[..., c] for c in range(f.shape[2])]
    total = np.zeros(n_iter)
    cleaned = []
    for channel in channels:
        u, energies = _chambolle(channel, lam, n_iter)
        cleaned.append(u)
        total += energies
    result = cleaned[0] if f.ndim == 2 else np.stack(cleaned, axis=-1)
    return DenoiseResult(image=np.clip(result, 0.0, 1.0), energies=total.tolist())
