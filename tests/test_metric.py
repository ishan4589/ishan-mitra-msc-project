# I used Claude Code to help implement and package tests for metric.py.

"""Tests for neuralphysics.metric."""

import numpy as np
import pytest
import torch

from neuralphysics.grid import CylinderGrid, grid_at
from neuralphysics.metric import (
    identity_metric,
    jacobian_num,
    metric_from_jacobian,
)


def synthetic_J(n, a, b, c, d):
    """A constant Jacobian field, padded, for checking the algebra in isolation."""
    f = lambda v: torch.full((n + 2, n + 2), float(v), dtype=torch.float64)
    return torch.stack([torch.stack([f(a), f(b)]), torch.stack([f(c), f(d)])])


# ---- the identity oracle ----------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_undistorted_map_gives_the_identity_jacobian_exactly(configured, n):
    """For x = xi the derivative dx/dxi is exactly 1, which the central difference
    returns only when divided by the reference spacing. A wrong denominator shows
    here as a uniform rescale of every entry."""
    J = CylinderGrid(n, mapped=False).jacobian
    assert torch.allclose(J[0, 0], torch.ones_like(J[0, 0]), atol=1e-12)
    assert torch.allclose(J[1, 1], torch.ones_like(J[1, 1]), atol=1e-12)
    assert float(J[0, 1].abs().max()) < 1e-12
    assert float(J[1, 0].abs().max()) < 1e-12


def test_distorted_grid_does_not_collapse_to_the_identity(configured):
    """Negative control: if the map were silently bypassed the metric would be
    the identity and every geometric result would be vacuous."""
    n = 9
    assert not all(
        torch.allclose(m, i)
        for m, i in zip(grid_at(n).metric, identity_metric(n))
    )


# ---- the algebra ------------------------------------------------------------


def test_metric_algebra_against_a_hand_computed_case(configured):
    """a=2, b=1, c=0, d=3: det = 6, mxx = (1+9)/6, myy = (4+0)/6, mxy = -(2+0)/6."""
    mxx, mxy, myy, mL = metric_from_jacobian(synthetic_J(5, 2, 1, 0, 3), 5)
    assert float(mxx[0, 0, 0]) == pytest.approx(10.0 / 6.0)
    assert float(myy[0, 0, 0]) == pytest.approx(4.0 / 6.0)
    assert float(mxy[0, 0, 0]) == pytest.approx(-2.0 / 6.0)
    assert float(mL[0, 0, 0]) == pytest.approx(6.0)


def test_z_zeta_scales_the_two_groups_oppositely(configured):
    """M_T is proportional to z_zeta and m_L to its reciprocal; this test checks
    those transformation factors independently."""
    one = metric_from_jacobian(synthetic_J(5, 2, 1, 0, 3), 5, z_zeta=1.0)
    two = metric_from_jacobian(synthetic_J(5, 2, 1, 0, 3), 5, z_zeta=2.0)
    for i in (0, 1, 2):
        assert torch.allclose(two[i], 2.0 * one[i], atol=1e-12)
    assert torch.allclose(two[3], 0.5 * one[3], atol=1e-12)


def test_transverse_metric_is_positive_definite(configured):
    """mxx > 0 and mxx*myy - mxy^2 > 0 pointwise: the transverse block must stay
    elliptic or the operator is not a Laplacian at all."""
    mxx, mxy, myy, _ = grid_at(17).metric
    assert float(mxx.min()) > 0.0
    assert float((mxx * myy - mxy * mxy).min()) > 0.0


# ---- properties on the real grid --------------------------------------------


@pytest.mark.parametrize("n", [65])
def test_the_finest_mesh_is_valid(configured, n):
    """The sampled Jacobian determinant is positive on the finest mapped grid."""
    assert float(grid_at(n).det_J.min()) > 0.0


@pytest.mark.parametrize("n", [9])
def test_metric_is_constant_along_the_axis(configured, n):
    """The extrusion is straight, so every z-level carries the same disc and the
    metric cannot depend on z."""
    for field in grid_at(n).metric:
        assert torch.allclose(field[:, :, 0], field[:, :, -1], atol=0.0)


def test_jacobian_precision_survives_a_float32_default(configured):
    """The accumulators must follow the input, not the global default. Under the
    old allocation this returned float32 and the loss was silent."""
    g = CylinderGrid(9, mapped=False)
    X, Y = g.coords_haloed
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    assert Xt.dtype is torch.float64

    torch.set_default_dtype(torch.float32)
    try:
        J = jacobian_num(Xt, Yt, 9, 9)
    finally:
        torch.set_default_dtype(torch.float64)

    assert J.dtype is torch.float64


def test_jacobian_honours_an_explicit_spacing(configured):
    """Halving h_xi doubles the xi-derivatives and leaves the eta ones alone."""
    g = CylinderGrid(9, mapped=False)
    X, Y = g.coords_haloed
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    a = jacobian_num(Xt, Yt, 9, 9)
    b = jacobian_num(Xt, Yt, 9, 9, h_xi=g.h / 2.0)
    assert torch.allclose(b[0, 0], 2.0 * a[0, 0], atol=1e-10)
    assert torch.allclose(b[1, 1], a[1, 1], atol=1e-12)


def test_jacobian_boundary_ring_is_replicated_not_measured(configured):
    """The returned outer shell replicates the nearest computed Jacobian values.
    `metric_from_jacobian` crops this shell before constructing operator fields."""
    J = grid_at(9).jacobian
    for p in (0, 1):
        for q in (0, 1):
            assert torch.equal(J[p, q, 0, :], J[p, q, 1, :])
            assert torch.equal(J[p, q, -1, :], J[p, q, -2, :])


def test_coords_are_the_interior_of_the_haloed_coords(configured):
    """The two accessors must not drift apart: one is a view of the other."""
    g = grid_at(9)
    Xh, Yh = g.coords_haloed
    X, Y = g.coords
    assert np.array_equal(X, Xh[1:-1, 1:-1].T)
    assert np.array_equal(Y, Yh[1:-1, 1:-1].T)
