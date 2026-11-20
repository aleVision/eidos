"""Tests for eidos_core.flow: robust optical flow (Black & Anandan, 1993)."""

import math

import numpy as np
import pytest

from eidos_core.flow import (
    _energy_gradient,
    angular_error,
    endpoint_error,
    flow_energy,
    flow_to_rgb,
    lorentzian,
    lorentzian_psi,
    quadratic,
    quadratic_psi,
)

# --- Penalties ---------------------------------------------------------------


def test_lorentzian_closed_form() -> None:
    sigma = 3.0
    assert lorentzian(np.array(0.0), sigma) == pytest.approx(0.0)
    assert lorentzian(np.array(math.sqrt(2) * sigma), sigma) == pytest.approx(math.log(2))


@pytest.mark.parametrize(("rho", "psi"), [(lorentzian, lorentzian_psi), (quadratic, quadratic_psi)])
def test_psi_is_derivative_of_rho(rho, psi) -> None:  # type: ignore[no-untyped-def]
    x = np.linspace(-20, 20, 81)
    h, sigma = 1e-5, 4.0
    numeric = (rho(x + h, sigma) - rho(x - h, sigma)) / (2 * h)
    np.testing.assert_allclose(psi(x, sigma), numeric, rtol=1e-5, atol=1e-8)


def test_lorentzian_influence_peaks_at_outlier_threshold() -> None:
    """Beyond |x| = √2·σ a residual's influence falls: it is treated as an outlier."""
    sigma = 2.0
    x = np.linspace(0, 20, 20001)
    peak = x[np.argmax(lorentzian_psi(x, sigma))]
    assert peak == pytest.approx(math.sqrt(2) * sigma, abs=1e-3)


def test_quadratic_influence_grows_without_bound() -> None:
    x = np.array([1.0, 10.0, 100.0])
    assert np.all(np.diff(quadratic_psi(x, 1.0)) > 0)


# --- Energy ------------------------------------------------------------------


def _random_problem(seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    shape = (6, 7)
    return tuple(rng.normal(size=shape) for _ in range(5))  # u, v, ix, iy, it


@pytest.mark.parametrize("robust", [True, False])
def test_energy_gradient_matches_numeric(robust: bool) -> None:
    u, v, ix, iy, it = _random_problem()
    params = {"sigma_d": 1.5, "sigma_s": 0.7, "lam_s": 1.3, "robust": robust}
    grad_u, grad_v = _energy_gradient(u, v, ix, iy, it, **params)
    h = 1e-6
    for field, grad in ((u, grad_u), (v, grad_v)):
        numeric = np.zeros_like(field)
        for index in np.ndindex(field.shape):
            original = field[index]
            field[index] = original + h
            plus = flow_energy(u, v, ix, iy, it, **params)
            field[index] = original - h
            minus = flow_energy(u, v, ix, iy, it, **params)
            field[index] = original
            numeric[index] = (plus - minus) / (2 * h)
        np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-6)


def test_energy_of_zero_flow_on_static_scene_is_zero() -> None:
    zeros = np.zeros((5, 5))
    ix = np.random.default_rng(1).normal(size=(5, 5))
    assert flow_energy(zeros, zeros, ix, ix, zeros, sigma_d=1.0, sigma_s=1.0, lam_s=1.0) == 0.0


def test_energy_counts_each_neighbour_pair_once() -> None:
    """A single vertical flow edge of height 1 across 3 rows: 3 pairs, quadratic σ = 1."""
    u = np.zeros((3, 4))
    u[:, 2:] = 1.0
    zeros = np.zeros_like(u)
    energy = flow_energy(u, zeros, zeros, zeros, zeros, 1.0, 1.0, lam_s=1.0, robust=False)
    assert energy == pytest.approx(3.0)


# --- Colour wheel and errors -------------------------------------------------


def test_flow_to_rgb_zero_is_white_and_directions_differ() -> None:
    u = np.array([[0.0, 1.0, -1.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0, 1.0]])
    rgb = flow_to_rgb(u, v)
    assert rgb.shape == (1, 4, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])
    colours = {tuple(rgb[0, i]) for i in range(1, 4)}
    assert len(colours) == 3


def test_flow_to_rgb_saturation_follows_speed() -> None:
    u = np.array([[1.0, 2.0]])
    rgb = flow_to_rgb(u, np.zeros_like(u), max_magnitude=2.0).astype(int)
    assert rgb[0, 1].min() < rgb[0, 0].min()  # faster = more saturated = further from white


def test_endpoint_error_known() -> None:
    u, v = np.full((2, 2), 3.0), np.full((2, 2), 4.0)
    zeros = np.zeros((2, 2))
    assert endpoint_error(u, v, zeros, zeros) == pytest.approx(5.0)


def test_angular_error_zero_for_exact_and_known_value() -> None:
    u, v = np.ones((2, 2)), np.zeros((2, 2))
    # arccos is ill-conditioned near 1: rounding of ~1e-16 becomes ~1e-6 degrees.
    assert angular_error(u, v, u, v) == pytest.approx(0.0, abs=1e-4)
    assert angular_error(u, v, np.zeros((2, 2)), v) == pytest.approx(45.0)


# --- The solver (paper §4: SOR + graduated non-convexity + coarse-to-fine) --------

from eidos_core.flow import (  # noqa: E402
    FlowResult,
    _sor_sweep,
    estimate_flow,
    warp,
)
from eidos_core.flow import lorentzian_psi as _psi  # noqa: E402
from eidos_core.synthetic_flow import two_surfaces  # noqa: E402


def textured(size: int = 96, seed: int = 0) -> np.ndarray:
    """Smooth random texture, 0-255: something to track."""
    import cv2

    noise = np.random.default_rng(seed).uniform(0, 255, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.0)
    return (texture - texture.min()) / (texture.max() - texture.min()) * 255


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The image's content moved by (dx, dy): result(x, y) = image(x − dx, y − dy)."""
    return warp(image, np.full(image.shape, dx, np.float32), np.full(image.shape, dy, np.float32))


def test_warp_sign_convention() -> None:
    """warp(I, u, v)(x, y) = I(x − u, y − v): content moves by (+u, +v)."""
    image = textured()
    moved = shifted(image, 2.0, 0.0)
    np.testing.assert_allclose(moved[10:-10, 12:-10], image[10:-10, 10:-12], atol=1e-3)


def _sequential_sweep(u, v, ix, iy, it, sigma_d, sigma_s, lam_s, omega):  # type: ignore[no-untyped-def]
    """Tests-only reference: visit red sites then black, one pixel at a time (Lorentzian)."""
    u, v = u.copy(), v.copy()
    h, w = u.shape
    t_s = 4 * lam_s / sigma_s**2
    for colour in (0, 1):
        for y in range(h):
            for x in range(w):
                if (y + x) % 2 != colour:
                    continue
                for field, grad_image in ((u, ix), (v, iy)):
                    r = ix[y, x] * u[y, x] + iy[y, x] * v[y, x] + it[y, x]
                    grad = 5.0 * grad_image[y, x] * _psi(np.array(r), sigma_d)
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            grad = grad + lam_s * _psi(
                                np.array(field[y, x] - field[ny, nx]), sigma_s
                            )
                    bound = 5.0 * grad_image[y, x] ** 2 / sigma_d**2 + t_s
                    field[y, x] -= omega * grad / bound
    return u, v


def test_red_black_sweep_equals_sequential_reference() -> None:
    rng = np.random.default_rng(3)
    u, v, ix, iy, it = (rng.normal(size=(5, 6)) for _ in range(5))
    args = {"sigma_d": 2.0, "sigma_s": 0.8, "lam_s": 1.0}
    fast = _sor_sweep(u, v, ix, iy, it, **args, robust=True, omega=1.5)
    slow = _sequential_sweep(u, v, ix, iy, it, **args, omega=1.5)
    np.testing.assert_allclose(fast[0], slow[0], rtol=1e-10)
    np.testing.assert_allclose(fast[1], slow[1], rtol=1e-10)


def _interior(a: np.ndarray, margin: int = 8) -> np.ndarray:
    return a[margin:-margin, margin:-margin]


def test_zero_motion_gives_zero_flow() -> None:
    image = textured()
    result = estimate_flow(image, image)
    assert np.abs(result.u).max() < 0.01 and np.abs(result.v).max() < 0.01


@pytest.mark.parametrize("robust", [True, False])
def test_recovers_translation(robust: bool) -> None:
    image = textured()
    result = estimate_flow(image, shifted(image, 1.5, -0.5), robust=robust)
    error = np.hypot(_interior(result.u) - 1.5, _interior(result.v) + 0.5).mean()
    assert error < 0.15


def test_robust_beats_quadratic_on_two_surfaces_with_noise() -> None:
    """The paper's experiment (§6.1, Table 1): a textured square moving over a still
    one, 10% noise. Paper: RMS horizontal error 0.18 (quadratic) vs 0.10 (robust)."""
    frame1, frame2, u_true, v_true = two_surfaces(noise=0.1)
    errors = {}
    for robust in (True, False):
        result = estimate_flow(frame1, frame2, robust=robust)
        errors[robust] = np.sqrt(np.mean((result.u - u_true) ** 2))
    assert errors[True] < 0.8 * errors[False]


def test_discontinuities_lie_on_the_square_border() -> None:
    frame1, frame2, u_true, _ = two_surfaces()
    result = estimate_flow(frame1, frame2)
    import cv2

    square = (u_true != 0).astype(np.uint8)
    border = cv2.morphologyEx(square, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    marked = result.discontinuities
    assert (marked & border).sum() / marked.sum() > 0.75  # most marks are on the border...
    assert (marked & border).sum() / border.sum() > 0.5  # ...and most of the border is marked


def test_energy_falls_within_each_stage() -> None:
    frame1, frame2, _, _ = two_surfaces(noise=0.1)
    for stage in estimate_flow(frame1, frame2).stages:
        assert stage.energy_end <= stage.energy_start * (1 + 1e-6)


def test_uniform_frames_no_nan() -> None:
    flat = np.full((40, 50), 128.0)
    result = estimate_flow(flat, flat)
    assert np.isfinite(result.u).all() and np.isfinite(result.v).all()
    assert np.abs(result.u).max() < 1e-6


def test_large_motion_is_finite_and_flags_outliers() -> None:
    """A documented limit: 3 pyramid levels cannot catch a 20 px jump."""
    image = textured(128)
    small = estimate_flow(image, shifted(image, 1.0, 0.0))
    large = estimate_flow(image, shifted(image, 20.0, 0.0))
    assert np.isfinite(large.u).all()
    assert large.stages[-1].outlier_fraction > small.stages[-1].outlier_fraction


def test_rejects_mismatched_sizes_and_bad_smoothness() -> None:
    with pytest.raises(ValueError, match="same size"):
        estimate_flow(np.zeros((10, 10)), np.zeros((10, 12)))
    with pytest.raises(ValueError, match="Smoothness must be positive"):
        estimate_flow(np.zeros((10, 10)), np.zeros((10, 10)), smoothness=0.0)


def test_robust_lowers_sigma_quadratic_keeps_it() -> None:
    image = textured(64)
    robust = estimate_flow(image, image).stages
    quadratic = estimate_flow(image, image, robust=False).stages
    assert len(robust) == 6 and robust[0].sigma_d > robust[-1].sigma_d
    assert len({(s.sigma_d, s.sigma_s) for s in quadratic}) == 1


def test_colour_frames_are_accepted() -> None:
    image = textured(48)
    colour = np.stack([image] * 3, axis=-1)
    assert isinstance(estimate_flow(colour, colour), FlowResult)
