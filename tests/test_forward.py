# I used Claude Code to help implement and package tests for forward.py.

"""Tests for neuralphysics.forward."""

import numpy as np
import pytest
import torch

from neuralphysics.forward import forward
from neuralphysics.hierarchy import CoefficientHierarchy
from neuralphysics.operators import laplacian_metric_3D
from neuralphysics.projections import trapezoid_weight_field
from neuralphysics.tissue import LIMB


def truth(n):
    return CoefficientHierarchy.from_vectors(LIMB, n)


def smooth_source(n):
    """A smooth analytic source; `forward` applies its compatibility projection."""
    u = torch.linspace(0.0, 1.0, n)
    U, V, W = torch.meshgrid(u, u, u, indexing="ij")
    return (
        torch.sin(2 * np.pi * U) * torch.cos(np.pi * V) * torch.cos(np.pi * W)
    )


def residual_norm(phi, I, hierarchy):
    n = hierarchy.n_fine
    h = hierarchy.L / (n - 1)
    from neuralphysics.projections import impose_net_zero_CSD

    b = impose_net_zero_CSD(I)
    return float(
        (
            b
            - laplacian_metric_3D(
                phi,
                h,
                hierarchy.faces[n],
                hierarchy.metric[n],
                hierarchy.sigma_T[n],
            )
        ).norm()
    )


# ---- the gauge --------------------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_the_returned_field_is_gauge_fixed(configured, n):
    """The operator annihilates constants, so phi is determined only up to one. The
    returned field is the representative orthogonal to ker(A) in the volume-weighted
    inner product -- the weighted mean, not the plain one, because a boundary node
    owns less volume than an interior one."""
    W = trapezoid_weight_field(n)
    phi = forward(smooth_source(n), truth(n))
    assert abs(float((W * phi).sum() / W.sum())) < 1e-14 * float(
        phi.abs().max()
    )


def test_the_plain_mean_is_not_what_is_removed(configured):
    """Negative control for the test above: the two means are different quantities,
    and it is the weighted one that is zeroed.

    The source must be asymmetric for this to bite. The smooth source above is
    antisymmetric about the mid-plane, so the field it induces can make both means
    vanish and would leave the control vacuous.
    """
    n = 9
    I = torch.as_tensor(np.random.default_rng(1).standard_normal((n, n, n)))
    phi = forward(
        I,
        truth(n),
    )
    assert abs(float(phi.mean())) > 1e-6 * float(phi.abs().max())


@pytest.mark.parametrize("n", [9])
def test_an_inadmissible_source_is_projected(configured, n):
    """Adding a constant to the source adds an inadmissible component that no
    conductivity can explain. It is projected out, so the answer must not move.
    """
    I = smooth_source(n)
    a = forward(I, truth(n))
    b = forward(I + 3.0, truth(n))
    assert torch.allclose(a, b, atol=1e-14)


# ---- convergence ------------------------------------------------------------


@pytest.mark.parametrize("n,bound", [(9, 1e-5), (17, 5e-2), (33, 2e-2)])
def test_the_configured_cycle_count_leaves_a_measurable_residual(
    configured, n, bound
):
    """Eight V-cycles leave a nonzero residual. These resolution-specific bounds
    document the accuracy of the configured finite-cycle solve without treating it
    as a converged linear-system solution. At n = 9 the hierarchy has only its base
    level; finer cases also use rediscretised coarse corrections.
    """
    I = smooth_source(n)
    h = truth(n)
    from neuralphysics.projections import impose_net_zero_CSD

    rel = residual_norm(forward(I, h), I, h) / float(
        impose_net_zero_CSD(I).norm()
    )
    assert rel < bound


def test_more_cycles_reduce_the_residual_further(configured):
    """A fixed cycle count gives the unrolled solver the same graph shape every
    evaluation. That only makes sense if the count controls
    accuracy, which this checks.

    The part that has broken before is the count reaching the loop bound: it was once
    dropped between the argument and the loop, and every solve ran the default eight
    cycles while nothing complained."""
    n = 17
    I = smooth_source(n)
    prev = float("inf")
    for cycles in (1, 2, 4, 8):
        phi = forward(I, truth(n), num_cycles=cycles)
        r = residual_norm(phi, I, truth(n))
        assert r < prev
        prev = r


# ---- linearity and determinism ----------------------------------------------


def test_the_solve_is_deterministic(configured):
    """Bit-for-bit repeatable for identical inputs in the configured test environment."""
    n = 9
    I = smooth_source(n)
    assert torch.equal(forward(I, truth(n)), forward(I, truth(n)))


# ---- differentiability ------------------------------------------------------


def test_gradients_reach_the_conductivities_through_the_solver(configured):
    """The whole method in one assertion: the gradient of a functional of phi with
    respect to the five conductivities, obtained by backpropagating through eight
    unrolled V-cycles."""
    n = 9
    rho = torch.log(
        torch.tensor([0.02, 0.09, 0.40, 0.04, 0.05], dtype=torch.float64)
    ).requires_grad_(True)
    s = rho.exp()
    sT_vec = torch.stack([s[0], s[1], s[3], s[4]])
    sL_vec = torch.stack([s[0], s[2], s[3], s[4]])
    h = CoefficientHierarchy.from_vectors(LIMB, n, sT_vec, sL_vec)
    forward(smooth_source(n), h).pow(2).sum().backward()
    assert (rho.grad.abs() > 0).all(), rho.grad


def test_the_ground_truth_solve_carries_no_graph(configured):
    n = 9
    assert not forward(smooth_source(n), truth(n)).requires_grad


# ---- guards -----------------------------------------------------------------


def test_the_coarsening_floor_comes_from_the_hierarchy(configured):
    """A hierarchy that stops early must not be descended past its own coarsest
    level, whatever floor the caller of the solve would otherwise have used."""
    h = CoefficientHierarchy.from_vectors(LIMB, 17, n_base=17)
    assert h.levels == (17,)
    forward(smooth_source(17), h)
