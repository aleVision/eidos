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

import numpy as np


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
