"""Tests for eidos_core.denoise: total variation denoising (Rudin, Osher & Fatemi, 1992)."""

import numpy as np
import pytest

from eidos_core.denoise import divergence, gradient, tv_energy


def test_gradient_of_ramp() -> None:
    ramp = np.tile(np.arange(5, dtype=np.float64), (3, 1))  # increases by 1 to the right
    g = gradient(ramp)
    assert g.shape == (2, 3, 5)
    np.testing.assert_array_equal(g[0], 0.0)  # no change downwards
    np.testing.assert_array_equal(g[1][:, :-1], 1.0)
    np.testing.assert_array_equal(g[1][:, -1], 0.0)  # Neumann boundary


def test_divergence_is_negative_adjoint_of_gradient() -> None:
    rng = np.random.default_rng(0)
    u = rng.normal(size=(7, 9))
    p = rng.normal(size=(2, 7, 9))
    assert np.sum(gradient(u) * p) == pytest.approx(-np.sum(u * divergence(p)), abs=1e-10)


def test_tv_energy_of_constant_image_is_fidelity_only() -> None:
    u = np.full((4, 4), 0.5)
    f = np.full((4, 4), 0.25)
    assert tv_energy(u, f, lam=2.0) == pytest.approx(0.5 * 2.0 * 16 * 0.25**2)


def test_tv_energy_of_step() -> None:
    step = np.zeros((6, 8))
    step[:, 4:] = 1.0  # a vertical edge of height 1 crossing all 6 rows
    assert tv_energy(step, step, lam=1.0) == pytest.approx(6.0)


def test_tv_energy_sums_colour_channels() -> None:
    step = np.zeros((6, 8, 3))
    step[:, 4:, :] = 1.0
    assert tv_energy(step, step, lam=1.0) == pytest.approx(18.0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_operators_handle_tiny_images(shape: tuple[int, int]) -> None:
    u = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    assert gradient(u).shape == (2, *shape)
    assert divergence(gradient(u)).shape == shape
    assert np.isfinite(tv_energy(u, u, lam=1.0))
