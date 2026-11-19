"""Tests for eidos_core.denoise: total variation denoising (Rudin, Osher & Fatemi, 1992)."""

import cv2
import numpy as np
import pytest

from eidos_core.denoise import denoise_tv, divergence, gradient, tv_energy


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


# --- The solver -------------------------------------------------------------


def noisy_step(size: int = 24, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """A vertical edge from 0.2 to 0.8 plus Gaussian noise: the classic ROF test."""
    clean = np.full((size, size), 0.2)
    clean[:, size // 2 :] = 0.8
    return clean + np.random.default_rng(seed).normal(0, sigma, clean.shape)


def gradient_flow_reference(f: np.ndarray, lam: float) -> np.ndarray:
    """The paper's own method, the slow and obvious way (tests only).

    Time-march the Euler-Lagrange equation u_t = div(∇u / |∇u|_ε) − λ(u − f), with
    a small ε so flat regions do not divide by zero, and small, stable time steps.
    """
    eps, dt = 1e-2, 2e-3
    u = f.copy()
    for _ in range(20_000):
        g = gradient(u)
        norm = np.sqrt(g[0] ** 2 + g[1] ** 2 + eps**2)
        u = u + dt * (divergence(g / norm) - lam * (u - f))
    return u


def test_matches_gradient_flow_energy() -> None:
    f = noisy_step()
    fast = tv_energy(denoise_tv(f, lam=8.0, n_iter=300).image, f, 8.0)
    reference = tv_energy(gradient_flow_reference(f, lam=8.0), f, 8.0)
    assert fast <= 1.01 * reference  # Chambolle reaches the minimum...
    assert reference <= 1.05 * fast  # ...and the reference really is near it too


def test_energy_does_not_increase() -> None:
    energies = denoise_tv(noisy_step(), lam=8.0).energies
    assert len(energies) == 100
    for before, after in zip(energies, energies[1:], strict=False):
        assert after <= before * (1 + 1e-6)


def test_constant_image_unchanged() -> None:
    image = np.full((10, 12), 0.4)
    np.testing.assert_allclose(denoise_tv(image, lam=5.0).image, image, atol=1e-6)


def test_huge_strength_returns_input() -> None:
    f = np.clip(noisy_step(), 0, 1)
    np.testing.assert_allclose(denoise_tv(f, lam=1e6).image, f, atol=1e-3)


def test_tiny_strength_flattens() -> None:
    """A very strong setting flattens the image. Chambolle converges slowly there
    (measured: ~3,000 iterations at λ = 0.001), so this tests the model, run to
    convergence; the product keeps λ where 100 iterations suffice."""
    f = np.clip(noisy_step(), 0, 1)
    assert denoise_tv(f, lam=1e-3, n_iter=3000).image.std() < 0.1 * f.std()


def _edge_width(image: np.ndarray) -> float:
    """10-90% rise width (pixels) of the average row profile across the edge."""
    profile = image.mean(axis=0)
    low, high = profile[:4].mean(), profile[-4:].mean()
    lo, hi = low + 0.1 * (high - low), low + 0.9 * (high - low)
    x = np.arange(len(profile), dtype=np.float64)
    return float(np.interp(hi, profile, x) - np.interp(lo, profile, x))


def test_keeps_edges_sharper_than_gaussian_blur() -> None:
    """At the same noise reduction in flat areas, TV keeps a narrower edge than blur."""
    f = noisy_step(size=64, sigma=0.1, seed=3)
    tv = denoise_tv(f, lam=10.0, n_iter=300).image
    flat = (slice(None), slice(4, 24))  # columns far from the edge
    tv_noise = tv[flat].std()

    sigmas = np.linspace(0.3, 8.0, 78)
    blurs = [cv2.GaussianBlur(f, (0, 0), s) for s in sigmas]
    blur = min(blurs, key=lambda b: abs(b[flat].std() - tv_noise))

    assert _edge_width(tv) < 0.5 * _edge_width(blur)


def test_colour_shape_and_range() -> None:
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 1, (12, 10, 3))
    result = denoise_tv(image, lam=4.0)
    assert result.image.shape == (12, 10, 3)
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0


@pytest.mark.parametrize("lam", [0.0, -1.0, 1e-40, 1e300, float("inf"), float("nan")])
def test_rejects_strength_outside_supported_range(lam: float) -> None:
    """Outside 1e-6..1e9 the float32 solver would overflow and return NaN."""
    with pytest.raises(ValueError, match="Strength must be between"):
        denoise_tv(np.zeros((4, 4)), lam=lam)


@pytest.mark.parametrize("lam", [1e-6, 1e9])
def test_no_nan_on_extremes(lam: float) -> None:
    result = denoise_tv(np.random.default_rng(2).uniform(0, 1, (16, 16, 3)), lam=lam)
    assert np.isfinite(result.image).all()
    assert np.isfinite(result.energies).all()
