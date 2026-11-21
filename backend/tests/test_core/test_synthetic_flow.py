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


def test_truth_is_indexed_at_frame_2_pixels() -> None:
    """The solver's flow at pixel x says where frame 2's content came from (frame 1
    warped by I1(x − u) matches I2), so the truth marks the square where it is in
    frame 2: shifted left by `motion`."""
    size, motion = 64, 8.0
    _, _, u_true, _ = two_surfaces(size=size, motion=motion)
    columns = np.flatnonzero(u_true.any(axis=0))
    assert columns.min() == size // 4 - int(motion)
    assert columns.max() == 3 * size // 4 - 1 - int(motion)
