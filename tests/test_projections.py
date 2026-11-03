# I used Claude Code to help implement and package tests for projections.py.

"""Tests for neuralphysics.projections."""

import numpy as np
import pytest
import torch

from neuralphysics.projections import (
    impose_electric_potential_uniqueness,
    impose_net_zero_CSD,
    trapezoid_weight_field,
)

# One projection: the two projection functions are the same arithmetic,
# the identity test at the foot of this file establishes this.
ONE_PROJECTION = [impose_net_zero_CSD]


def w_mean(f):
    """Volume-weighted mean, computed independently of the module under test."""
    n = f.shape[0]
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    W = np.einsum("i,j,k->ijk", w, w, w)
    return float((W * f).sum() / W.sum())


# ---- the weight field -------------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_weight_field_is_the_trapezoidal_quadrature_rule(configured, n):
    """sum(W) = (n-1)^3 exactly, which is what makes h^3 sum(W) equal the domain
    volume and W*f summed the trapezoidal estimate of the integral of f."""
    W = trapezoid_weight_field(n)
    assert W.sum().item() == float((n - 1) ** 3)


@pytest.mark.parametrize("n", [9])
def test_weight_field_entries_by_codimension(configured, n):
    """1 in the interior, 1/2 on faces, 1/4 on edges, 1/8 at corners."""
    W = trapezoid_weight_field(n)
    assert W[n // 2, n // 2, n // 2].item() == 1.0
    assert W[0, n // 2, n // 2].item() == 0.5
    assert W[0, 0, n // 2].item() == 0.25
    assert W[0, 0, 0].item() == 0.125
    assert W[-1, -1, -1].item() == 0.125


# ---- both projections -------------------------------------------------------


@pytest.mark.parametrize("proj", ONE_PROJECTION)
@pytest.mark.parametrize("n", [9])
def test_projection_removes_the_weighted_mean(configured, proj, n):
    """The property the removed in-function assertions used to check on every
    call: the output has zero volume-weighted mean to machine precision."""
    f = torch.randn(n, n, n)
    assert abs(w_mean(proj(f).numpy())) < 1e-14 * float(f.abs().max())


@pytest.mark.parametrize("proj", ONE_PROJECTION)
def test_projection_is_invariant_under_an_added_constant(configured, proj):
    """P(f + c) = P(f): the projection removes any added constant component."""
    f = torch.randn(9, 9, 9)
    assert torch.allclose(proj(f + 2.5), proj(f), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("proj", ONE_PROJECTION)
def test_projection_passes_gradients(configured, proj):
    """Both projections sit inside the differentiated path -- one on the source,
    one on the returned field, so neither may detach."""
    f = torch.randn(9, 9, 9, requires_grad=True)
    proj(f).pow(2).sum().backward()
    assert f.grad is not None and float(f.grad.abs().max()) > 0.0


def test_the_two_projections_are_arithmetically_identical(configured):
    """They compute the same thing and are kept apart as it is easier to
    gain an intuition of what is going on.
    One is a precondition on the source (mandatory), the other a postcondition
    on the field (design choice).
    This test records the identity so that a future divergence is a deliberate act rather than an accident.
    """
    f = torch.randn(9, 9, 9)
    assert torch.equal(
        impose_net_zero_CSD(f), impose_electric_potential_uniqueness(f)
    )
