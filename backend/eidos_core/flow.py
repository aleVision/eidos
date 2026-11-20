"""Robust optical flow (Black & Anandan, 1993).

M. J. Black, P. Anandan, "A framework for the robust estimation of optical flow",
ICCV 1993, pp. 231-236; expanded in "The robust estimation of multiple motions:
parametric and piecewise-smooth flow fields", CVIU 63(1):75-104, 1996 (the
equations and parameters below follow the journal version).

Optical flow (u, v) is the motion of every pixel between two frames. Horn & Schunck
(1981) find it by minimizing squared errors of two assumptions:

    brightness constancy:  I_x·u + I_y·v + I_t ≈ 0       (the data term)
    smoothness:            neighbours move alike         (the spatial term)

Squares trust every measurement, so one bad pixel, or the border of an object that
moves differently, pulls the whole solution. Black & Anandan replace the squares
with a robust penalty ρ whose influence ψ = ρ' falls back towards zero for large
errors: outliers stop counting, and motion boundaries stay sharp.

    E(u, v) = Σ_s λ_D·ρ(I_x u_s + I_y v_s + I_t, σ_D)
              + λ_S·Σ_(s,t) [ρ(u_s − u_t, σ_S) + ρ(v_s − v_t, σ_S)]

where (s, t) runs over each pair of 4-connected neighbours once. Frames are grey,
with intensities 0-255 (the paper's parameter values are on that scale).
"""

from collections.abc import Callable

import cv2
import numpy as np

Penalty = Callable[[np.ndarray, float], np.ndarray]

LAMBDA_D = 5.0  # weight of the data term (paper §6)


# --- Penalties: ρ and its influence ψ = ∂ρ/∂x --------------------------------


def lorentzian(x: np.ndarray, sigma: float) -> np.ndarray:
    """ρ(x, σ) = log(1 + ½(x/σ)²): grows ever more slowly, so outliers count little."""
    return np.log1p(0.5 * (x / sigma) ** 2)


def lorentzian_psi(x: np.ndarray, sigma: float) -> np.ndarray:
    """ψ(x, σ) = 2x / (2σ² + x²): largest at |x| = √2·σ (the outlier threshold), then falls."""
    return 2.0 * x / (2.0 * sigma**2 + x**2)


def quadratic(x: np.ndarray, sigma: float) -> np.ndarray:
    """ρ(x, σ) = x²/σ²: least squares (Horn & Schunck), for comparison."""
    return (x / sigma) ** 2


def quadratic_psi(x: np.ndarray, sigma: float) -> np.ndarray:
    """ψ(x, σ) = 2x/σ²: influence grows without bound, so one outlier can dominate."""
    return 2.0 * x / sigma**2


# --- Energy ------------------------------------------------------------------


def flow_energy(
    u: np.ndarray,
    v: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    it: np.ndarray,
    sigma_d: float,
    sigma_s: float,
    lam_s: float,
    robust: bool = True,
) -> float:
    """E(u, v) as in the module docstring; `robust=False` uses the quadratic penalty."""
    rho: Penalty = lorentzian if robust else quadratic
    data = LAMBDA_D * float(np.sum(rho(ix * u + iy * v + it, sigma_d)))
    smooth = 0.0
    for field in (u, v):
        smooth += float(np.sum(rho(np.diff(field, axis=0), sigma_s)))
        smooth += float(np.sum(rho(np.diff(field, axis=1), sigma_s)))
    return data + lam_s * smooth


def _neighbour_influence(field: np.ndarray, psi: Penalty, sigma: float) -> np.ndarray:
    """Σ over the 4 neighbours n of ψ(field_s − field_n, σ); no neighbour beyond the border."""
    total = np.zeros_like(field)
    down = psi(field[:-1, :] - field[1:, :], sigma)  # site above minus site below
    right = psi(field[:, :-1] - field[:, 1:], sigma)
    total[:-1, :] += down
    total[1:, :] -= down  # ψ is odd: ψ(b − a) = −ψ(a − b)
    total[:, :-1] += right
    total[:, 1:] -= right
    return total


def _energy_gradient(
    u: np.ndarray,
    v: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    it: np.ndarray,
    sigma_d: float,
    sigma_s: float,
    lam_s: float,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """(∂E/∂u, ∂E/∂v) at every pixel."""
    psi: Penalty = lorentzian_psi if robust else quadratic_psi
    data = LAMBDA_D * psi(ix * u + iy * v + it, sigma_d)
    grad_u = ix * data + lam_s * _neighbour_influence(u, psi, sigma_s)
    grad_v = iy * data + lam_s * _neighbour_influence(v, psi, sigma_s)
    return grad_u, grad_v


# --- Showing and scoring a flow field ----------------------------------------


def flow_to_rgb(u: np.ndarray, v: np.ndarray, max_magnitude: float | None = None) -> np.ndarray:
    """Colour wheel: hue = direction, saturation = speed / max_magnitude; no motion = white."""
    magnitude = np.hypot(u, v)
    top = max_magnitude if max_magnitude else float(magnitude.max()) or 1.0
    angle = (np.degrees(np.arctan2(v, u)) % 360.0) / 2.0  # OpenCV hue range is 0-180
    hsv = np.empty((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.round(angle).astype(np.uint8) % 180
    hsv[..., 1] = np.round(np.clip(magnitude / top, 0.0, 1.0) * 255).astype(np.uint8)
    hsv[..., 2] = 255
    return np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def endpoint_error(u: np.ndarray, v: np.ndarray, u_true: np.ndarray, v_true: np.ndarray) -> float:
    """Mean length of the difference vector, in pixels."""
    return float(np.mean(np.hypot(u - u_true, v - v_true)))


def angular_error(u: np.ndarray, v: np.ndarray, u_true: np.ndarray, v_true: np.ndarray) -> float:
    """Mean angle in degrees between (u, v, 1) and (u_true, v_true, 1) (Barron et al. 1994)."""
    dot = u * u_true + v * v_true + 1.0
    norms = np.sqrt(u**2 + v**2 + 1.0) * np.sqrt(u_true**2 + v_true**2 + 1.0)
    return float(np.mean(np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))))
