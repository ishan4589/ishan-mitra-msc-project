# I used Claude Code to help implement and package tests for multigrid.py.

"""Tests for neuralphysics.multigrid.

Property-based throughout. The grid transfers and the smoother have no pinned
values of their own -- what pins them is the V-cycle they compose into, which the
forward solve exercises. So what is checked here is the algebra each
one has to satisfy for the cycle to converge at all.
"""

import numpy as np
import pytest
import torch

from neuralphysics.grid import grid_at, levels
from neuralphysics.multigrid import (
    jacobi,
    probe_diagonal,
    prolong,
    restrict,
    v_cycle,
)
from neuralphysics.operators import face_coefficients, laplacian_metric_3D
from neuralphysics.projections import trapezoid_weight_field
from neuralphysics.tissue import LIMB
from neuralphysics.verification import assemble_dense

N_BASE = levels(65)[-1]  # the floor, from the one place that states it


def hierarchy(n_fine):
    """{level -> metric}, {level -> faces}, {level -> sigma_T} at the ground truth.

    The real thing arrives as CoefficientHierarchy; this is the minimum
    the V-cycle needs to be exercised now.
    """
    sizes = levels(n_fine)
    metric, faces, sig_T = {}, {}, {}
    for n in sizes:
        g = grid_at(n)
        sT, sL = LIMB.sigma_fields(g)
        metric[n], faces[n], sig_T[n] = (
            g.metric,
            face_coefficients(g.metric, sT, sL),
            sT,
        )
    return metric, faces, sig_T


def dipole(n, seed=0):
    """An admissible source: net zero, so the singular system is solvable."""
    rng = np.random.default_rng(seed)
    I = torch.as_tensor(rng.standard_normal((n, n, n)))
    W = trapezoid_weight_field(n)
    return I - (W * I).sum() / W.sum()


# ---- restriction ------------------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_restriction_preserves_constants(configured, n):
    """The 27-point full-weighting stencil is a tensor product of (1/4, 1/2, 1/4)
    and so sums to 1. If it did not, the coarse problem would see a different
    source than the fine residual it is meant to represent."""
    nc = (n + 1) // 2
    assert torch.allclose(
        restrict(torch.ones(n, n, n)), torch.ones(nc, nc, nc), atol=1e-14
    )


# ---- prolongation -----------------------------------------------------------


@pytest.mark.parametrize("nc", [5])
def test_prolongation_reproduces_coarse_values_at_even_indices(configured, nc):
    """align_corners=True: coarse node J lands exactly on fine node 2J. Anything
    else and the correction would be applied at shifted positions."""
    e = torch.as_tensor(np.random.default_rng(1).standard_normal((nc, nc, nc)))
    assert torch.allclose(prolong(e)[::2, ::2, ::2], e, atol=1e-14)


def test_prolongation_averages_at_odd_indices(configured):
    """Trilinear interpolation on a vertex-centred grid: an odd fine node is the
    mean of its two coarse neighbours along each axis."""
    nc = 5
    e = torch.as_tensor(np.random.default_rng(2).standard_normal((nc, nc, nc)))
    f = prolong(e)
    assert torch.allclose(
        f[1::2, ::2, ::2], 0.5 * (e[:-1] + e[1:]), atol=1e-14
    )


# ---- the transfer pair ------------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_transfers_are_adjoint_in_the_weighted_inner_product(configured, n):
    """<R x, y>_{W_coarse} = (1/8) <x, P y>_{W_fine}.

    The weighted form is the one that holds. The plain Euclidean relation
    P = 8 R^T fails on the boundary rows -- the reflect padding folds a mirrored
    neighbour back onto a face node, which restriction sees and interpolation does
    not -- so a test written in the Euclidean form fails, and 'fixing' it means
    breaking the transfer pair. The trapezoidal weights are exactly the discrete
    measure that reconciles the two; they are also the operator's left null vector,
    so the solver, the transfers and the compatibility projection all agree about
    what the discrete integral is.
    """
    nc = (n + 1) // 2
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, n, n, generator=g)
    y = torch.randn(nc, nc, nc, generator=g)
    lhs = (trapezoid_weight_field(nc) * restrict(x) * y).sum()
    rhs = (trapezoid_weight_field(n) * x * prolong(y)).sum() / 8.0
    assert lhs == pytest.approx(float(rhs), rel=1e-12)


def test_the_plain_euclidean_adjointness_does_not_hold(configured):
    """Negative control showing that the corresponding unweighted Euclidean
    adjoint relation does not hold at the boundary."""
    n, nc = 9, 5
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, n, n, generator=g)
    y = torch.randn(nc, nc, nc, generator=g)
    lhs = (restrict(x) * y).sum()
    rhs = (x * prolong(y)).sum() / 8.0
    assert abs(float(lhs) - float(rhs)) > 1e-3 * abs(float(rhs))


# ---- the diagonal -----------------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_probed_diagonal_is_the_true_diagonal(configured, n):
    """Against the assembled matrix, which shares none of the probe's machinery.
    An analytic sum of face coefficients would be wrong here: it misses the 1/v row
    scaling on the boundary rows and misses the cross term, which does reach the
    centre node."""
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    probed = probe_diagonal(faces[n], metric[n], h, sig_T[n]).reshape(-1)
    dense = torch.diagonal(assemble_dense(n, h, faces[n], metric[n], sig_T[n]))
    assert torch.allclose(probed, dense, atol=1e-10)


def test_the_probe_is_detached_from_the_graph(configured):
    """D^-1 is a frozen preconditioner: sigma-dependent, but not differentiated
    through. This looks like a bug to anyone who has not read the reasoning, so it
    is pinned as behaviour rather than left to a comment."""
    n = 9
    g = grid_at(n)
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    sT, sL = LIMB.sigma_fields(g)
    faces = face_coefficients(g.metric, s * sT, s * sL)
    d = probe_diagonal(faces, g.metric, 1.0 / (n - 1), s * sT)
    assert not d.requires_grad


# ---- the smoother -----------------------------------------------------------


def test_gradients_survive_the_smoother(configured):
    """The frozen preconditioner must not freeze the gradient. Every operator
    application inside jacobi, and the residual, stay in the graph."""
    n = 9
    g = grid_at(n)
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    sT, sL = LIMB.sigma_fields(g)
    faces = face_coefficients(g.metric, s * sT, s * sL)
    out = jacobi(
        torch.zeros(n, n, n),
        dipole(n),
        1.0 / (n - 1),
        faces,
        g.metric,
        s * sT,
        num_iters=2,
    )
    out.pow(2).sum().backward()
    assert s.grad is not None and float(s.grad.abs()) > 0.0


def test_a_positive_diagonal_is_refused(configured):
    """The assertion has to be able to fire, or it is decoration. Negating the face
    coefficients flips the diagonal's sign, which is what a sign error in the face
    assembly would do."""
    n = 9
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    with pytest.raises(AssertionError, match="not strictly negative"):
        jacobi(
            torch.zeros(n, n, n), dipole(n), h, -faces[n], metric[n], sig_T[n]
        )


@pytest.mark.parametrize("omega", [0.0, -0.5, 1.5])
def test_undamped_or_negative_jacobi_weights_are_rejected(configured, omega):
    """Outside (0, 1] the update is no longer the supported damped-Jacobi scheme;
    such choices may amplify rather than smooth high-frequency error."""
    n = 9
    metric, faces, sig_T = hierarchy(n)
    with pytest.raises(ValueError, match="omega"):
        jacobi(
            torch.zeros(n, n, n),
            dipole(n),
            1.0 / (n - 1),
            faces[n],
            metric[n],
            sig_T[n],
            omega=omega,
        )


def test_smoothing_reduces_the_residual(configured):
    """Weighted Jacobi at omega = 2/3 is a smoother, not a solver, but it must at
    least not amplify."""
    n = 17
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    I = dipole(n)
    phi = torch.zeros(n, n, n)
    res = [
        float(
            (
                I - laplacian_metric_3D(phi, h, faces[n], metric[n], sig_T[n])
            ).norm()
        )
    ]
    for _ in range(4):
        phi = jacobi(phi, I, h, faces[n], metric[n], sig_T[n], num_iters=2)
        res.append(
            float(
                (
                    I
                    - laplacian_metric_3D(
                        phi, h, faces[n], metric[n], sig_T[n]
                    )
                ).norm()
            )
        )
    assert all(b < a for a, b in zip(res, res[1:])), res


# ---- the cycle --------------------------------------------------------------


def test_v_cycle_reduces_the_residual_monotonically(configured):
    """Rediscretised coarse levels provide an approximate correction for the fine
    problem; repeated cycles must still contract this test residual."""
    n = 33
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    I = dipole(n)
    diag_by_n = {}
    phi = torch.zeros(n, n, n)
    res = [
        float(
            (
                I - laplacian_metric_3D(phi, h, faces[n], metric[n], sig_T[n])
            ).norm()
        )
    ]
    for _ in range(4):
        phi = v_cycle(
            phi, I, h, faces, sig_T, metric, N_BASE, diag_by_n=diag_by_n
        )
        res.append(
            float(
                (
                    I
                    - laplacian_metric_3D(
                        phi, h, faces[n], metric[n], sig_T[n]
                    )
                ).norm()
            )
        )
    assert all(b < a for a, b in zip(res, res[1:])), res
    assert res[-1] < 0.1 * res[0]


def test_caching_the_diagonal_changes_nothing(configured):
    """Two cycles sharing a diag_by_n must agree bit-for-bit with two that probe
    afresh: the cache is an optimisation, and the coefficients do not change
    between cycles."""
    n = 17
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    I = dipole(n)
    cached = {}
    a = b = torch.zeros(n, n, n)
    for _ in range(2):
        a = v_cycle(a, I, h, faces, sig_T, metric, N_BASE, diag_by_n=cached)
        b = v_cycle(b, I, h, faces, sig_T, metric, N_BASE)
    assert torch.equal(a, b)


def test_the_cycle_is_linear_in_the_source(configured):
    """From a zero initial guess the cycle is a linear operator: the diagonal is
    fixed by the coefficients, not by the iterate. Nonlinearity here would mean the
    smoother was reading something it should not."""
    n = 17
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    z = torch.zeros(n, n, n)
    I1, I2 = dipole(n, seed=5), dipole(n, seed=6)
    V = lambda I: v_cycle(z, I, h, faces, sig_T, metric, N_BASE)
    assert torch.allclose(
        V(2.0 * I1 - 3.0 * I2), 2.0 * V(I1) - 3.0 * V(I2), atol=1e-10
    )


def test_the_base_level_is_where_the_recursion_stops(configured):
    """A cycle whose fine grid already is the base level must not recurse: the
    coarsening floor is a property of the hierarchy, not of the call."""
    n = N_BASE
    h = 1.0 / (n - 1)
    metric, faces, sig_T = hierarchy(n)
    assert list(metric) == [n]
    out = v_cycle(
        torch.zeros(n, n, n), dipole(n), h, faces, sig_T, metric, N_BASE
    )
    assert out.shape == (n, n, n)
