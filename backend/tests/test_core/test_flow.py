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
