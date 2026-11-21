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

import math
from collections.abc import Callable
from dataclasses import dataclass

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


# --- The solver (paper §4.1-4.2 and §6) ---------------------------------------

# Over-relaxation. The paper uses 1.995; measured here, with 20 sweeps per level that
# value oscillates and the quadratic solve diverges (translation error 7.8 px).
# 1.8 converges for both penalties (error ≤ 0.01 px on a 1.5 px translation).
OMEGA = 1.8
STAGES = 6  # graduated non-convexity: σ lowered linearly over 6 stages (paper §6)
LEVELS = 3  # pyramid levels (paper §6)
# SOR sweeps per level and stage. The paper uses 20; measured on the two-surface
# scene (240 px, smoothness 1), the robust flow is then 0.17 px from the truth against
# 0.07 px after 200 sweeps. 40 sweeps reach 0.13 px in ~2 s at 240 px.
ITERATIONS = 40
SIGMA_D = (18 / math.sqrt(2), 5 / math.sqrt(2))  # data term σ: first and last stage
SIGMA_S = (3 / math.sqrt(2), 0.03 / math.sqrt(2))  # smoothness σ: first and last stage
PRE_BLUR = 1.0  # Gaussian σ (px) applied to both frames so derivatives are reliable
# Motion discontinuity threshold (px). The paper's eq. 28 sets it "experimentally";
# √2·σ_S at the last stage (0.03 px) marks every ripple. Measured on the two-surface
# scene: 0.2 px puts 82% of the marks on the square's border and finds 2/3 of it.
DISCONTINUITY = 0.2


@dataclass(frozen=True)
class FlowStage:
    sigma_d: float
    sigma_s: float
    energy_start: float  # E at the start of the stage's finest-level solve
    energy_end: float  # ...and at its end (fixed linearization: never larger)
    outlier_fraction: float  # data residuals with |r| ≥ √2·σ_D at the end of the stage


@dataclass(frozen=True)
class FlowResult:
    # Motion in pixels, indexed at frame 2's pixels: frame 1 warped by (u, v), i.e.
    # I1(x − u, y − v), matches frame 2, so (u, v) says where each frame-2 pixel's
    # content came from (paper eq. 29's backward warp).
    u: np.ndarray  # rightward
    v: np.ndarray  # downward
    data_outliers: np.ndarray  # bool: brightness constancy fails here (final stage)
    discontinuities: np.ndarray  # bool: the flow jumps here (paper eq. 28)
    stages: list[FlowStage]


def warp(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """I(x − u, y − v), bilinear (paper eq. 29): the image's content moved by (u, v)."""
    h, w = image.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xs - u).astype(np.float32)
    map_y = (ys - v).astype(np.float32)
    return np.asarray(
        cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    )


def _sor_sweep(
    u: np.ndarray,
    v: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    it: np.ndarray,
    sigma_d: float,
    sigma_s: float,
    lam_s: float,
    robust: bool,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One over-relaxation sweep, u_s ← u_s − ω·(∂E/∂u_s)/T(u_s) (paper eqs. 21-22, 27).

    Red-black (checkerboard) order: same-coloured pixels are never neighbours, so
    each colour updates in one numpy step and the result equals visiting the pixels
    one by one in that order, the setting in which 0 < ω < 2 converges.
    T bounds ∂²E/∂u_s²: the penalty's largest curvature is 1/σ² (Lorentzian) or 2/σ²
    (quadratic), times λ_D·I_x² for the data term and 4·λ_S for the 4 neighbours.
    """
    u, v = u.copy(), v.copy()
    curvature = 1.0 if robust else 2.0
    t_s = curvature * 4.0 * lam_s / sigma_s**2
    t_u = curvature * LAMBDA_D * ix**2 / sigma_d**2 + t_s
    t_v = curvature * LAMBDA_D * iy**2 / sigma_d**2 + t_s
    h, w = u.shape
    checker = (np.add.outer(np.arange(h), np.arange(w)) % 2).astype(bool)
    params = (sigma_d, sigma_s, lam_s, robust)
    for colour in (~checker, checker):
        grad_u, _ = _energy_gradient(u, v, ix, iy, it, *params)
        u[colour] -= omega * grad_u[colour] / t_u[colour]
        _, grad_v = _energy_gradient(u, v, ix, iy, it, *params)
        v[colour] -= omega * grad_v[colour] / t_v[colour]
    return u, v


def _linearize(
    frame1: np.ndarray, frame2: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives around the current flow, so that the data residual of the total flow
    (u + du, v + dv) is I_x·u_total + I_y·v_total + it (it absorbs the current flow).

    Frame 1 is warped towards frame 2 (paper eq. 29). I_x, I_y: central differences of
    the average of warped frame 1 and frame 2; I_t = frame 2 − warped frame 1.
    """
    warped = warp(frame1, u, v)
    iy, ix = np.gradient(0.5 * (warped + frame2))
    it = frame2 - warped - ix * u - iy * v
    # No reliable measurement (1) where the warp reads from outside frame 1, e.g.
    # content entering the frame, and (2) on the outer pixel ring, where the derivatives
    # are one-sided. Switch the data term off there, so the smoothness term fills the
    # flow in from the neighbours. Measured without this: up to 59 px of flow at the
    # border for a 2 px motion, from invented border values and false matches.
    h, w = frame1.shape
    ys, xs = np.mgrid[0:h, 0:w]
    outside = (xs - u < 0) | (xs - u > w - 1) | (ys - v < 0) | (ys - v > h - 1)
    outside |= (ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)
    for derivative in (ix, iy, it):
        derivative[outside] = 0.0
    return ix.astype(np.float32), iy.astype(np.float32), it.astype(np.float32)


def _to_grey(frame: np.ndarray) -> np.ndarray:
    grey = frame.astype(np.float32)
    if grey.ndim == 3:
        grey = np.asarray(cv2.cvtColor(grey[..., :3], cv2.COLOR_RGB2GRAY), dtype=np.float32)
    return np.asarray(cv2.GaussianBlur(grey, (0, 0), PRE_BLUR))


def _resize_flow(
    u: np.ndarray, v: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a flow field to `shape`, scaling the vectors with the image."""
    h, w = shape
    sy, sx = h / u.shape[0], w / u.shape[1]
    resize = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    return np.asarray(resize(u)) * sx, np.asarray(resize(v)) * sy


def _discontinuities(u: np.ndarray, v: np.ndarray, threshold: float) -> np.ndarray:
    """Paper eq. 28: a pixel whose u or v differs from a neighbour's by ≥ threshold."""
    marked = np.zeros(u.shape, dtype=bool)
    for field in (u, v):
        down = np.abs(np.diff(field, axis=0)) >= threshold
        right = np.abs(np.diff(field, axis=1)) >= threshold
        marked[:-1, :] |= down
        marked[1:, :] |= down
        marked[:, :-1] |= right
        marked[:, 1:] |= right
    return marked


def estimate_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    smoothness: float = 1.0,
    robust: bool = True,
) -> FlowResult:
    """Dense flow from frame 1 to frame 2 (grey or colour, intensities 0-255),
    indexed at frame 2's pixels (see FlowResult).

    Robust: 6 graduated non-convexity stages, from a convex start (σ large, no
    outliers, much like least squares) down to the paper's final σ. Stage 1 runs
    coarse-to-fine over a 3-level pyramid to catch motions larger than a pixel;
    later stages refine at full resolution, re-warping first (the paper repeats the
    whole coarse-to-fine pass per stage; measured here, that was no better, and worse
    at low smoothness, so the simpler schedule is kept). Quadratic (Horn &
    Schunck): the same schedule at the first stage's σ, since it is already convex.
    `smoothness` scales λ_S (the paper's λ_S = 1).
    """
    if frame1.shape[:2] != frame2.shape[:2]:
        raise ValueError("Frames must have the same size")
    if not smoothness > 0:
        raise ValueError("Smoothness must be positive")
    if min(frame1.shape[:2]) < 8:
        raise ValueError("Frames must be at least 8 pixels on each side")
    lam_s = float(smoothness)
    pyramid = [(_to_grey(frame1), _to_grey(frame2))]
    for _ in range(LEVELS - 1):
        f1, f2 = pyramid[-1]
        if min(f1.shape) < 8:
            break
        pyramid.append((np.asarray(cv2.pyrDown(f1)), np.asarray(cv2.pyrDown(f2))))

    def sigmas(stage: int) -> tuple[float, float]:
        if not robust:
            return SIGMA_D[0], SIGMA_S[0]
        t = stage / (STAGES - 1)
        return (
            SIGMA_D[0] + t * (SIGMA_D[1] - SIGMA_D[0]),
            SIGMA_S[0] + t * (SIGMA_S[1] - SIGMA_S[0]),
        )

    coarsest = pyramid[-1][0].shape
    u = np.zeros(coarsest, np.float32)
    v = np.zeros(coarsest, np.float32)
    stages = []
    for stage in range(STAGES):
        sigma_d, sigma_s = sigmas(stage)
        params = (sigma_d, sigma_s, lam_s, robust)
        levels = reversed(pyramid) if stage == 0 else [pyramid[0]]
        for f1, f2 in levels:
            u, v = _resize_flow(u, v, f1.shape) if u.shape != f1.shape else (u, v)
            ix, iy, it = _linearize(f1, f2, u, v)
            energy_start = flow_energy(u, v, ix, iy, it, *params)
            for _ in range(ITERATIONS):
                u, v = _sor_sweep(u, v, ix, iy, it, *params, omega=OMEGA)
        residual = ix * u + iy * v + it
        outliers = np.abs(residual) >= math.sqrt(2) * sigma_d
        stages.append(
            FlowStage(
                sigma_d=sigma_d,
                sigma_s=sigma_s,
                energy_start=energy_start,
                energy_end=flow_energy(u, v, ix, iy, it, *params),
                outlier_fraction=float(outliers.mean()),
            )
        )
    return FlowResult(
        u=u,
        v=v,
        data_outliers=outliers,
        discontinuities=_discontinuities(u, v, DISCONTINUITY),
        stages=stages,
    )
