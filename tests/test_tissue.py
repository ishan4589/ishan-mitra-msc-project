# I used Claude Code to help implement and package tests for tissue.py.

"""Tests for neuralphysics.tissue."""

import dataclasses

import numpy as np
import pytest
import torch

from neuralphysics.grid import grid_at, levels
from neuralphysics.tissue import (
    LIMB,
    TissueModel,
    interfaces_from_logits,
    logits_from_interfaces,
)


def dominant(grid, tissue=None):
    """The tissue holding the largest membership at each node of one z-slice.

    The solve never sees this -- it sees the smooth weights. It exists so the
    geometric claims below (which shell reaches the rim, where each boundary lands,
    that the shells nest) can be stated about discrete labels, which is the form
    those claims naturally take.
    """
    t = LIMB if tissue is None else tissue
    return t.layer_weights(grid).argmax(0)[:, :, 0].numpy()


def interior_mask(r, margin):
    """Nodes further than `margin` in normalised radius from every interface."""
    return np.stack([np.abs(r - R) > margin for R in LIMB.interfaces]).all(0)


# ---- the properties the layering must have ----------------------------------


@pytest.mark.parametrize("n", [9])
def test_conductivity_is_z_invariant(configured, n):
    """The extrusion is straight. This is not cosmetic: it is what makes every
    zeta-face join two nodes of the same tissue, so the harmonic mean along zeta
    is an identity and one code path serves all six faces."""
    sT, sL = LIMB.sigma_fields(grid_at(n))
    assert bool((sT[:, :, 0:1] == sT).all())
    assert bool((sL[:, :, 0:1] == sL).all())


@pytest.mark.parametrize("n", [33, 65])
def test_the_outermost_layer_reaches_the_rim(configured, n):
    """The insulating boundary IS the skin-air interface. A node on the rim
    labelled anything but skin would mean the Neumann condition was being applied
    at the wrong material.

    This assertion is limited to n >= 33. On coarser rediscretised levels, the thin
    skin and non-circular sampled rim can leave some rim nodes with fat as their
    dominant membership; those levels are used only for coarse correction."""
    idx = dominant(grid_at(n))
    ring = np.concatenate([idx[0, :], idx[-1, :], idx[:, 0], idx[:, -1]])
    assert (ring == LIMB.n_layers - 1).all()


@pytest.mark.parametrize("n", [33])
def test_each_interface_lands_within_one_cell_of_its_target(configured, n):
    """The membership crossover sits exactly at R_k, but the nodes do not: the
    dominant label changes at sampled nodes around that crossover. On this mapped
    grid, the tested tolerance is one reference spacing h."""
    g = grid_at(n)
    r = g.radius_field(LIMB.shell_norm)
    idx = dominant(g)
    for k, R in enumerate(LIMB.interfaces, start=1):
        realised = r[idx >= k].min()
        assert (
            abs(realised - R) < g.h
        ), f"interface {k} at {realised:.5f}, intended {R}"


def test_no_tissue_is_empty_at_any_level(configured):
    """N_BASE = 9 rather than 3 because muscle empties at n = 3 and fat has four
    nodes per slice at n = 5. A level with an empty tissue solves a different
    medium, so this is what justifies the coarsening floor."""
    for n in levels(65):
        idx = dominant(grid_at(n))
        counts = [int((idx == k).sum()) for k in range(LIMB.n_layers)]
        assert min(counts) > 0, (
            f"a tissue is empty at n = {n}: "
            f"{dict(zip(LIMB.layer_names, counts))}"
        )


def test_layers_are_nested_annuli_ordered_outwards(configured):
    """The label must increase monotonically with radius: a node cannot be in a
    more inward tissue than one closer to the axis."""
    g = grid_at(17)
    r = g.radius_field(LIMB.shell_norm).reshape(-1)
    idx = dominant(g).reshape(-1)
    order = np.argsort(r, kind="stable")
    assert (np.diff(idx[order]) >= 0).all()


# ---- the differentiable path ------------------------------------------------


def test_supplied_vectors_override_the_ground_truth(configured):
    """Away from the interfaces the membership saturates, so a node deep inside a
    shell carries that shell's supplied value and almost nothing of its neighbours'.
    Eight transition widths out the logistic is below 3.4e-4, which with layer values
    an integer apart bounds the contamination at the tolerance below."""
    g = grid_at(33)
    sT_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sT, _ = LIMB.sigma_fields(g, sT_vec=sT_vec, sL_vec=sT_vec)
    idx = dominant(g)
    r = g.radius_field(LIMB.shell_norm)
    deep = interior_mask(r, margin=8 * LIMB.epsilon_norm(g))
    assert (
        deep.sum() > 0.4 * deep.size
    ), "the margin has eaten most of the disc"
    got = sT[:, :, 0].numpy()[deep]
    assert np.allclose(got, sT_vec.numpy()[idx[deep]], rtol=0.0, atol=1e-3)


def test_gradients_reach_the_supplied_vectors(configured):
    """The weighted sum of layer values carries gradients from the conductivity
    fields back to every supplied layer scalar.
    """
    v = torch.tensor([0.02, 0.09, 0.04, 0.05], requires_grad=True)
    sT, sL = LIMB.sigma_fields(grid_at(9), sT_vec=v, sL_vec=v)
    (sT.sum() + sL.sum()).backward()
    assert v.grad is not None
    assert (
        v.grad > 0
    ).all(), "every layer is non-empty, so every parameter must get a gradient"


# ---- soft membership --------------------------------------------------------

EPS = (
    1.0 / 128.0
)  # deliberately not the model's own width, so the argument is exercised


@pytest.mark.parametrize("n", [9, 17, 33])
def test_soft_weights_are_a_partition_of_unity(configured, n):
    """Non-negative and summing to one is what makes the blend a conductivity and
    not an arbitrary combination: sigma then lies in the convex hull of the layer
    values and cannot go negative, whatever the radii do during the inversion.
    """
    w = LIMB.layer_weights(grid_at(n), LIMB.interfaces, EPS)
    assert w.shape == (LIMB.n_layers, n, n, n)
    assert bool((w >= 0).all())
    assert float((w.sum(0) - 1).abs().max()) < 1e-14


def test_a_vanishing_width_recovers_the_piecewise_constant_medium(configured):
    """The soft model is a relaxation of the piecewise-constant one, not a different
    medium. At a width far below one cell the logistics saturate, so every node
    carries its nearest layer value to machine precision for this sampled grid."""
    g = grid_at(17)
    sT, _ = LIMB.sigma_fields(g, epsilon=1e-8)
    values = torch.as_tensor(
        [LIMB.sigma_layers[nm][0] for nm in LIMB.layer_names]
    )
    nearest = (
        (sT.reshape(-1, 1) - values.reshape(1, -1)).abs().min(dim=1).values
    )
    assert float(nearest.max()) < 1e-14


def test_the_radii_receive_a_gradient(configured):
    """The point of the whole construction: the radii are tensors carrying a
    gradient, and the sign follows the contrast each interface separates."""
    g = grid_at(17)
    R = torch.tensor(LIMB.interfaces, requires_grad=True)
    sT, _ = LIMB.sigma_fields(g, interfaces=R, epsilon=EPS)
    sT.sum().backward()
    assert R.grad is not None and bool((R.grad != 0).all())
    # bone < muscle, so pushing R_1 out replaces muscle by bone and lowers sigma_T;
    # muscle > fat, so pushing R_2 out replaces fat by muscle and raises it.
    assert R.grad[0] < 0 < R.grad[1]


# ---- the softmax parameterisation -------------------------------------------


def test_the_logit_parameterisation_round_trips(configured):
    """An initial guess is natural to state as radii and has to be handed to the
    optimiser as logits, so the two directions must agree."""
    R = LIMB.interfaces
    back = interfaces_from_logits(logits_from_interfaces(R), R[-1])
    assert torch.allclose(back, torch.tensor(R), atol=1e-15)
    assert (
        float(back[-1]) == R[-1]
    )  # the fixed radius is carried, not reconstructed


def test_any_logits_give_an_ordered_geometry(configured):
    """What buys the constraint-free optimisation: L-BFGS-B may propose anything, and
    the result is still a limb whose layers increase outwards and fit inside the skin.
    """
    torch.manual_seed(0)
    R_outer = LIMB.interfaces[-1]
    for u in (
        torch.randn(200, len(LIMB.interfaces) - 1, dtype=torch.float64) * 8.0
    ):
        R = interfaces_from_logits(u, R_outer)
        assert bool((torch.diff(R) > 0).all()) and float(R[0]) > 0
        assert float(R[-1]) == R_outer


# ---- validation -------------------------------------------------------------


def _model(**kw):
    base = dict(
        layer_names=("a", "b"),
        sigma_layers={"a": (1.0, 1.0), "b": (2.0, 2.0)},
        interfaces=(0.25,),
    )
    base.update(kw)
    return TissueModel(**base)


def test_the_outermost_interface_must_leave_room_for_the_rim(configured):
    """An interface at or beyond 0.5 would put the outermost layer outside the
    domain, and the boundary condition would then be applied at the wrong material.
    """
    with pytest.raises(ValueError):
        _model(interfaces=(0.5,))


def test_non_positive_conductivity_is_rejected(configured):
    """Zero conductivity destroys ellipticity, while negative conductivity can
    reverse operator signs; neither is valid for the solver or harmonic mean."""
    with pytest.raises(ValueError):
        _model(sigma_layers={"a": (1.0, 1.0), "b": (0.0, 1.0)})


def test_model_rejects_mutation(configured):
    """The frozen dataclass prevents rebinding a declared model attribute."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _model().shell_norm = "reference"
