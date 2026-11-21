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
is a separate choice; Eidos uses Chambolle & Pock's accelerated primal-dual
algorithm (2011), and the tests check it against the paper's own gradient flow.
(Eidos 2.0.0 used Chambolle's 2004 dual algorithm: correct, but too slow to finish
in 100 iterations at strong settings, which showed as blur.)

Images are floats in [0, 1]; colour images are (H, W, C), each channel independent.
"""

import math
from dataclasses import dataclass

import numpy as np

# Supported strengths. Far outside this range float32 arithmetic overflows or loses
# all precision (NaN or a meaningless image); such λ make no practical sense anyway.
LAM_MIN = 1e-6
LAM_MAX = 1e9


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


def _chambolle_pock(f: np.ndarray, lam: float, n_iter: int) -> tuple[np.ndarray, list[float]]:
    """Minimize TV(u) + (λ/2)·Σ(u − f)² for one channel (Chambolle & Pock 2011, Alg. 2).

    A primal-dual method: the dual field p (one 2-vector per pixel, |p| ≤ 1) and
    the image u take turns, each a cheap closed-form step:

        p ← project(p + σ·∇ū),   u ← (u + τ·div p + τλ·f) / (1 + τλ),   ū = u + θ·(u − u_old)

    Because the fidelity term is strongly convex, the steps adapt every iteration
    (τ shrinks, σ grows), which makes it far faster than Chambolle 2004 for strong
    settings (small λ). Since ∇ is linear, ∇ū comes from ∇u and ∇u_old, so each
    iteration needs one gradient and one divergence; E(u) reuses the same ∇u.
    """
    # Python floats (not numpy scalars), so the arrays stay float32.
    tau = sigma = 1.0 / math.sqrt(8.0)  # τ·σ·‖∇‖² ≤ 1, with ‖∇‖² ≤ 8 on a 2-D grid
    gamma = 0.7 * lam  # acceleration rate; the paper's choice for ROF
    u = f.copy()
    grad_u = gradient(u)
    grad_bar = grad_u
    p = np.zeros((2, *f.shape), dtype=np.float32)
    energies = []
    for _ in range(n_iter):
        p = p + sigma * grad_bar
        p /= np.maximum(1.0, np.sqrt(p[0] ** 2 + p[1] ** 2))  # back into |p| ≤ 1
        grad_old = grad_u
        u = (u + tau * divergence(p) + tau * lam * f) / (1.0 + tau * lam)
        grad_u = gradient(u)
        tv = float(np.sum(np.sqrt(grad_u[0] ** 2 + grad_u[1] ** 2)))
        energies.append(tv + 0.5 * lam * float(np.sum((u - f) ** 2)))
        theta = 1.0 / math.sqrt(1.0 + 2.0 * gamma * tau)
        tau, sigma = theta * tau, sigma / theta
        grad_bar = grad_u + theta * (grad_u - grad_old)
    return u, energies


def default_iterations(lam: float) -> int:
    """100 iterations for λ ≥ 4, more for stronger settings, at most 200.

    The accelerated solver's error falls like 1/(λ·k²) after k iterations, so
    k ∝ 1/√λ keeps the leftover error about the same at every strength.
    """
    return min(200, max(100, math.ceil(100 * math.sqrt(4 / lam))))


def denoise_tv(image: np.ndarray, lam: float, n_iter: int | None = None) -> DenoiseResult:
    """Total variation denoising of a grey (H, W) or colour (H, W, C) image in [0, 1].

    Each colour channel is denoised independently, in float32 to save memory.
    `n_iter` defaults to `default_iterations(lam)`.
    """
    if not LAM_MIN <= lam <= LAM_MAX:  # also rejects NaN and infinity
        raise ValueError(f"Strength must be between {LAM_MIN:g} and {LAM_MAX:g}")
    if n_iter is None:
        n_iter = default_iterations(lam)
    f = image.astype(np.float32)
    channels = [f] if f.ndim == 2 else [f[..., c] for c in range(f.shape[2])]
    total = np.zeros(n_iter)
    cleaned = []
    for channel in channels:
        u, energies = _chambolle_pock(channel, lam, n_iter)
        cleaned.append(u)
        total += energies
    result = cleaned[0] if f.ndim == 2 else np.stack(cleaned, axis=-1)
    return DenoiseResult(image=np.clip(result, 0.0, 1.0), energies=total.tolist())
