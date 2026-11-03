# I used Claude Code to help implement and package tests for grid.py.

"""Tests for neuralphysics.grid."""

import numpy as np
import pytest
import torch

from neuralphysics.grid import (
    CylinderGrid,
    SmoothingPolicy,
    cartesian_to_circular,
    grid_at,
    levels,
    smooth_grid,
)

LEVELS = [9, 17, 33, 65]


# ---- properties of the normalisation ----------------------------------------


@pytest.mark.parametrize("norm", ["physical"])
@pytest.mark.parametrize("n", [65])
def test_mean_rim_sits_at_one_half(n, norm):
    """The normalisation makes a shell radius represent the same limb fraction at
    every resolution and fixes the mean of the four concatenated edge traces.
    Corners are counted once per incident edge, matching `CylinderGrid`.
    """
    r = grid_at(n).radius_field(norm)
    rim = np.concatenate([r[0, :], r[-1, :], r[:, 0], r[:, -1]])
    assert abs(rim.mean() - 0.5) < 1e-12


@pytest.mark.parametrize("n", [9, 65])
def test_rim_is_circular_only_to_about_one_h(n):
    """The mean is fixed, the spread is not. The rim is not an exact circle, and
    the size of that departure is a property of the mesh worth pinning as a
    bound rather than leaving unstated."""
    stats = grid_at(n).rim_stats("physical")
    assert 0.0 < stats["spread_in_h"] < 2.0


@pytest.mark.parametrize("norm", ["physical"])
def test_physical_shell_radius_inverts_the_normalisation(norm):
    """r_norm -> physical -> r_norm is the identity."""
    g = grid_at(17)
    for R in (0.12, 0.36, 0.44, 0.5):
        assert (
            abs(R - 0.5 * g.physical_shell_radius(R, norm) / g.rim_mean(norm))
            < 1e-14
        )


def test_disc_shrinks_relative_to_the_reference_square_as_n_grows():
    """The disc is a different physical size at every resolution. This is the
    reason absolute shell radii would make each multigrid level a slightly
    different limb, and the reason the inversion runs on the generation grid.
    """
    means = [grid_at(n).rim_mean("physical") for n in LEVELS]
    assert means == sorted(means, reverse=True)
    assert means[0] > means[-1] + 1e-3


# ---- the unmapped limit -----------------------------------------------------


def test_unmapped_grid_is_exactly_the_reference_lattice():
    """With the map bypassed the coordinates are the reference nodes themselves.
    A missing or spurious factor of h in the map would show here first."""
    n = 9
    g = CylinderGrid(n, mapped=False)
    ref = np.linspace(0.0, (n + 1) / (n - 1), n + 2)[1:-1]
    XI, ETA = np.meshgrid(ref, ref, indexing="ij")
    X, Y = g.coords
    assert np.array_equal(X, XI)
    assert np.array_equal(Y, ETA)


# ---- the map itself ---------------------------------------------------------


def test_projection_sends_linf_level_sets_to_exact_circles():
    """Before smoothing, a node at max-norm radius s lands at Euclidean radius s,
    so each square ring maps to a circle exactly. The smoothing perturbs this;
    the projection alone does not."""
    n, h = 17, 1.0 / 16
    ix = np.linspace(0, (n + 1) * h, n + 2)
    xg, yg = np.meshgrid(ix, ix, indexing="ij")
    X, Y = cartesian_to_circular(n, n, xg, yg, h, h)
    o = h * (n + 1) / 2.0
    linf = np.maximum(np.abs(xg - o), np.abs(yg - o)).T
    assert np.allclose(np.hypot(X - o, Y - o), linf, atol=1e-14)


def test_smoothing_pins_the_outer_ring():
    """The boundary is excluded from the sweep, so the rim is exactly where the
    projection put it however long the smoothing runs."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((11, 11)), rng.standard_normal((11, 11))
    xs, ys = smooth_grid(x.copy(), y.copy(), iterations=25)
    for a, b in ((x, xs), (y, ys)):
        assert np.array_equal(a[0, :], b[0, :]) and np.array_equal(
            a[-1, :], b[-1, :]
        )
        assert np.array_equal(a[:, 0], b[:, 0]) and np.array_equal(
            a[:, -1], b[:, -1]
        )


def test_smoothing_preserves_an_affine_field():
    """A discrete harmonic field is a fixed point of the sweep, so an affine node
    layout is returned unchanged. This is the sharpest available check that the
    5-point stencil and its weighting are right."""
    u = np.arange(9, dtype=float)
    x, y = np.meshgrid(u, u, indexing="ij")
    xs, ys = smooth_grid(x.copy(), y.copy(), iterations=50)
    assert np.allclose(xs, x, atol=1e-13) and np.allclose(ys, y, atol=1e-13)


# ---- construction and caching -----------------------------------------------


def test_grid_at_returns_the_same_object_twice():
    """Generating a grid runs the smoothing loop; the cache is what keeps that
    from happening once per caller."""
    assert grid_at(9) is grid_at(9)


def test_grids_differing_in_any_argument_are_distinct():
    assert grid_at(9) is not grid_at(17)
    assert grid_at(9) is not grid_at(
        9, smoothing=SmoothingPolicy(iterations=10)
    )
    assert grid_at(9) is not grid_at(9, mapped=False)


def test_centre_is_computed_not_one_half():
    """The disc centre drifts with resolution; hard-coding 0.5 would misplace
    every shell interface, by more at coarse levels than fine."""
    for n in LEVELS:
        g = grid_at(n)
        assert g.centre == pytest.approx(g.h * (n + 1) / 2.0)
    assert grid_at(65).centre != pytest.approx(grid_at(9).centre)


# ---- the domain extent ------------------------------------------------------


@pytest.mark.parametrize("L", [2.0])
@pytest.mark.parametrize("n", [9])
def test_physical_lengths_scale_with_the_domain_extent(configured, n, L):
    """The square-to-disc projection rescales each node's offset from the centre by
    a ratio of norms, and the smoothing is a linear average, so the whole map is
    positively homogeneous in the coordinates. Every physical length therefore
    scales exactly."""
    one, scaled = grid_at(n), grid_at(n, L=L)
    assert scaled.h == pytest.approx(L * one.h, rel=1e-15)
    assert scaled.centre == pytest.approx(L * one.centre, rel=1e-15)
    assert scaled.rim_mean("physical") == pytest.approx(
        L * one.rim_mean("physical"), rel=1e-14
    )
    for a, b in zip(one.coords, scaled.coords):
        assert np.allclose(b, L * a, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("L", [2.0, 1e3])
def test_dimensionless_quantities_do_not_move(configured, L):
    """The Jacobian is a quotient of two things that both scale, so it is invariant
    -- but only because the differencing denominator is this grid's h. Leaving it
    at an assumed 1/(n-1) would make the metric scale with L, and the operator
    would make the longitudinal metric scale spuriously and break the intended
    operator scaling.

    The comparison below allows only floating-point round-off.
    """
    n = 9
    one, scaled = grid_at(n), grid_at(n, L=L)
    # Judged against each field's own magnitude, not elementwise: the cross term
    # passes through zero, and a relative test there measures nothing.
    for a, b in zip(one.metric, scaled.metric):
        assert float((b - a).abs().max()) < 1e-13 * float(a.abs().max())
    r1, r2 = one.radius_field("physical"), scaled.radius_field("physical")
    assert np.abs(r2 - r1).max() < 1e-13 * np.abs(r1).max()
    assert float((scaled.det_J - one.det_J).abs().max()) < 1e-13 * float(
        one.det_J.abs().max()
    )


def test_the_operator_scales_as_one_over_L_squared(configured):
    """The consequence that matters: the metric is invariant and the operator
    carries 1/h^2, so A(L) = A(1)/L^2. This is the check that L reaches the
    physics rather than only the coordinates."""
    from neuralphysics.operators import face_coefficients, laplacian_metric_3D
    from neuralphysics.tissue import LIMB

    n, L = 9, 2.5
    one, scaled = grid_at(n), grid_at(n, L=L)
    sT, sL = LIMB.sigma_fields(one)
    faces = face_coefficients(one.metric, sT, sL)
    phi = torch.as_tensor(np.random.default_rng(0).standard_normal((n, n, n)))
    A1 = laplacian_metric_3D(phi, one.h, faces, one.metric, sT)
    AL = laplacian_metric_3D(phi, scaled.h, faces, scaled.metric, sT)
    assert torch.allclose(AL, A1 / L**2, rtol=1e-12, atol=0.0)


def test_the_shells_are_unaffected_by_the_extent(configured):
    """The radius field is normalised so its sampled edge mean sits at 0.5 and epsilon is a
    fraction of the extent, so the shells describe the limb rather than fix a set of
    lengths. A limb twice the size is the same limb: not merely the same tissue at
    each node, but the same membership weight at each node.

    Stated on the weights rather than on their argmax, which holds under a weaker
    condition -- the crossover between two neighbouring weights sits at the interface
    radius whatever the transition width -- and so would pass even if the width were
    the wrong share of the limb. allclose rather than equal because the two grids
    reach the same normalised width through different rim_mean values, so they agree
    only to round-off.
    """
    from neuralphysics.tissue import LIMB

    n, L = 9, 2.5
    a = LIMB.layer_weights(grid_at(n))
    b = LIMB.layer_weights(grid_at(n, L=L))
    assert torch.allclose(a, b, rtol=0.0, atol=1e-12)


# ---- the level hierarchy ----------------------------------------------------


def test_levels_terminates_at_the_base_and_never_below():
    """The production `levels` helper is the source of the default coarsening floor;
    this test checks that every supported fine size reaches it without going below.
    """
    for n in (17, 33, 65, 129):
        sizes = levels(n)
        assert sizes[0] == n
        assert sizes[-1] == 9
        assert all(b - 1 == (a - 1) // 2 for a, b in zip(sizes, sizes[1:]))


@pytest.mark.parametrize("n", [64])
def test_resolutions_that_are_not_a_power_of_two_plus_one_are_rejected(n):
    """Vertex-centred nesting needs every coarse node to sit on a fine node."""
    with pytest.raises(ValueError):
        levels(n)
