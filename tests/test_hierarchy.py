# I used Claude Code to help implement and package tests for hierarchy.py.

"""Tests for neuralphysics.hierarchy."""

import pytest
import torch

from neuralphysics.grid import grid_at, levels
from neuralphysics.hierarchy import CoefficientHierarchy
from neuralphysics.multigrid import v_cycle
from neuralphysics.operators import face_coefficients
from neuralphysics.tissue import LIMB


def unpack(rho):
    """(5,) log-parameters -> the two length-4 vectors ordered like the layers.
    bone, fat and skin appear in both (isotropic); muscle splits into T and L.
    """
    s = rho.exp()
    return (
        torch.stack([s[0], s[1], s[3], s[4]]),
        torch.stack([s[0], s[2], s[3], s[4]]),
    )


# the five recovered conductivities (bone, muscle_T, muscle_L, fat, skin), read
# off the model rather than typed out: a copy here would go stale the next time
# the tissue table is re-sourced. The dtype is pinned because this is built at
# import, before the `configured` fixture has set float64 as the default.
TRUE_RHO = torch.log(
    torch.tensor(
        [
            LIMB.sigma_layers["bone"][0],
            LIMB.sigma_layers["muscle"][0],
            LIMB.sigma_layers["muscle"][1],
            LIMB.sigma_layers["fat"][0],
            LIMB.sigma_layers["skin"][0],
        ],
        dtype=torch.float64,
    )
)


def test_the_log_parameterisation_does_not_round_trip_exactly(configured):
    """exp(log(sigma)) is not sigma bit-for-bit. The optimiser works in log space,
    so the conductivities it evaluates at the true values differ from the table in
    the last bits, and any comparison between the inversion path and the
    data-generation path has to be a tolerance rather than an equality. Measured
    here: about 1 ULP on the coefficients, roughly 1e-16 relative.
    """
    exact_T, exact_L = LIMB.sigma_vectors()
    round_T, round_L = unpack(TRUE_RHO)
    assert not torch.equal(exact_T, round_T) or not torch.equal(
        exact_L, round_L
    )
    assert torch.allclose(exact_T, round_T, rtol=1e-15, atol=0.0)
    assert torch.allclose(exact_L, round_L, rtol=1e-15, atol=0.0)

    n = 9
    a = CoefficientHierarchy.from_vectors(LIMB, n, exact_T, exact_L)
    b = CoefficientHierarchy.from_vectors(LIMB, n, round_T, round_L)
    assert torch.allclose(a.faces[n], b.faces[n], rtol=1e-14, atol=0.0)


# ---- constant conductivity --------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_equal_layer_conductivities_give_a_homogeneous_medium(configured, n):
    """The weights are a partition of unity, so equal per-layer values must leave no
    trace of the shells at all -- the membership has to cancel exactly, not to
    within the transition width."""
    ones, fives = torch.ones(LIMB.n_layers), 5.0 * torch.ones(LIMB.n_layers)
    h = CoefficientHierarchy.from_vectors(LIMB, n, ones, fives)
    expected = face_coefficients(
        grid_at(n).metric,
        torch.full((n, n, n), 1.0),
        torch.full((n, n, n), 5.0),
    )
    assert torch.allclose(h.faces[n], expected, rtol=0.0, atol=1e-14)
    assert torch.allclose(
        h.sigma_T[n], torch.ones(n, n, n), rtol=0.0, atol=1e-14
    )


# ---- variable interfaces ----------------------------------------------------


def test_the_default_geometry_is_the_tissue_models_own(configured):
    """Omitting the radii and the width must reproduce passing them explicitly, at
    every level. The per-level conversion of epsilon into normalised radius units is
    the thing this would catch if the default path bypassed it."""
    sT_vec, sL_vec = LIMB.sigma_vectors()
    implicit = CoefficientHierarchy.from_vectors(LIMB, 17, sT_vec, sL_vec)
    explicit = CoefficientHierarchy.from_vectors(
        LIMB,
        17,
        sT_vec,
        sL_vec,
        interfaces=LIMB.interfaces,
        epsilon=LIMB.epsilon,
    )
    assert all(
        torch.equal(implicit.faces[n], explicit.faces[n])
        for n in implicit.levels
    )


def test_gradients_reach_the_interface_radii(configured):
    """The radii must influence the face coefficients through soft membership and
    harmonic averaging. They also influence the separately stored transverse field
    used by the operator's cross term."""
    sT_vec, sL_vec = LIMB.sigma_vectors()
    R = torch.tensor(LIMB.interfaces, requires_grad=True)
    h = CoefficientHierarchy.from_vectors(
        LIMB,
        17,
        sT_vec,
        sL_vec,
        interfaces=R,
        epsilon=0.25 / 48.0,
    )
    sum(h.faces[n].sum() for n in h.levels).backward()
    assert R.grad is not None and bool((R.grad != 0).all())


# ---- the level-consistency invariant ----------------------------------------


@pytest.mark.parametrize("n", [9])
def test_the_three_coefficient_dicts_cover_the_same_levels(configured, n):
    h = CoefficientHierarchy.from_vectors(LIMB, n)
    assert h.levels == tuple(levels(n))
    assert set(h.metric) == set(h.faces) == set(h.sigma_T) == set(h.levels)
    assert h.n_fine == n


def test_the_hierarchy_stops_at_the_coarsening_floor(configured):
    assert CoefficientHierarchy.from_vectors(LIMB, 33).levels == (33, 17, 9)
    assert CoefficientHierarchy.from_vectors(LIMB, 33, n_base=17).levels == (
        33,
        17,
    )


# ---- differentiability ------------------------------------------------------


def test_gradcheck_from_the_five_scalars_to_the_face_coefficients(configured):
    """The single most important differentiability property in the codebase.

    The gradient has to survive the soft weighted sum that builds nodal fields and
    the harmonic-mean face averaging, which is nonlinear in conductivity. Finite
    differences against autograd in float64 check the complete path.
    """

    def functional(rho):
        sT_vec, sL_vec = unpack(rho)
        h = CoefficientHierarchy.from_vectors(LIMB, 9, sT_vec, sL_vec)
        return (h.faces[9] ** 2).sum() + (h.sigma_T[9] ** 3).sum()

    rho = TRUE_RHO.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        functional, (rho,), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_every_parameter_receives_a_gradient(configured):
    """No layer is empty at any level, so a zero gradient in any component would
    mean a parameter had been dropped rather than that it does not matter."""
    rho = TRUE_RHO.clone().requires_grad_(True)
    sT_vec, sL_vec = unpack(rho)
    h = CoefficientHierarchy.from_vectors(LIMB, 17, sT_vec, sL_vec)
    sum(h.faces[n].sum() for n in h.levels).backward()
    assert (rho.grad.abs() > 0).all(), rho.grad


def test_gradients_reach_every_level(configured):
    """The coarse levels are rediscretised from the same parameters, so they are in
    the graph too. A coarse level built from detached values would make the solver
    differentiable but wrong."""
    rho = TRUE_RHO.clone().requires_grad_(True)
    sT_vec, sL_vec = unpack(rho)
    h = CoefficientHierarchy.from_vectors(LIMB, 17, sT_vec, sL_vec)
    coarse = h.levels[-1]
    h.faces[coarse].sum().backward()
    assert (rho.grad.abs() > 0).all()


# ---- the diagonal cache -----------------------------------------------------


def test_the_cache_fills_as_the_cycle_descends(configured):
    """The hierarchy stores one probed diagonal for each level reached by the cycle."""
    n = 17
    h = CoefficientHierarchy.from_vectors(LIMB, n)
    v_cycle(
        torch.zeros(n, n, n),
        torch.zeros(n, n, n),
        1.0 / (n - 1),
        h.faces,
        h.sigma_T,
        h.metric,
        h.levels[-1],
        diag_by_n=h.diagonals,
    )
    assert set(h.diagonals) == set(h.levels)
    assert all(h.diagonals[level].shape == (level,) * 3 for level in h.levels)


def test_two_hierarchies_do_not_share_a_cache(configured):
    a = CoefficientHierarchy.from_vectors(LIMB, 9)
    b = CoefficientHierarchy.from_vectors(LIMB, 9)
    a.diagonals[9] = "sentinel"
    assert b.diagonals == {}
