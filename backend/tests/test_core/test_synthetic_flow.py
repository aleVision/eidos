"""Tests for eidos_core.synthetic_flow: scenes with known motion."""

import numpy as np

from eidos_core.synthetic_flow import two_surfaces


def test_two_surfaces_is_deterministic_and_truth_matches() -> None:
    a = two_surfaces(size=64, motion=2.0, noise=0.1)
    b = two_surfaces(size=64, motion=2.0, noise=0.1)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    frame1, frame2, u_true, v_true = a
    assert frame1.shape == frame2.shape == u_true.shape == v_true.shape == (64, 64)
    assert set(np.unique(u_true)) == {-2.0, 0.0}  # the square moves left, the rest is still
    assert not v_true.any()
    assert 0 <= frame1.min() and frame1.max() <= 255


def test_noise_is_uniform_within_the_requested_fraction_of_the_range() -> None:
    clean = two_surfaces(size=100, noise=0.0)[1]
    noisy = two_surfaces(size=100, noise=0.1)[1]
    difference = noisy - clean
    assert np.abs(difference).max() <= 25.5 + 1e-3
    assert np.abs(difference).mean() > 10  # uniform on ±25.5 has mean |x| ≈ 12.75
