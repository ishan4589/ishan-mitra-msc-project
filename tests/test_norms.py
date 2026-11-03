# I used Claude Code to help implement and package tests for norms.py.

"""Tests for neuralphysics.verification.norms."""

import numpy as np
import pytest
import torch

from neuralphysics.projections import trapezoid_weight_field
from neuralphysics.verification import l2_w, linf


@pytest.mark.parametrize("n", [5, 9, 17])
def test_weighted_l2_of_a_constant_is_that_constant(configured, n):
    """h^3 times the trapezoidal weights integrate to the volume of the unit cube,
    so a constant field c must return |c| at every n. This is what makes the norm
    comparable across resolutions: without the h^3 it would grow like n^(3/2).
    """
    W = trapezoid_weight_field(n)
    assert l2_w(torch.full((n, n, n), 3.0), W) == pytest.approx(3.0, rel=1e-12)
    assert l2_w(torch.full((n, n, n), -3.0), W) == pytest.approx(
        3.0, rel=1e-12
    )


@pytest.mark.parametrize("n", [9])
def test_weighted_l2_matches_a_naive_reference(configured, n):
    """Written out as an explicit loop, so the vectorised form is checked against
    something with no shared machinery."""
    W = trapezoid_weight_field(n)
    f = torch.as_tensor(np.random.default_rng(31).standard_normal((n, n, n)))
    total = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += float(W[i, j, k]) * float(f[i, j, k]) ** 2
    assert l2_w(f, W) == pytest.approx(
        (total / (n - 1) ** 3) ** 0.5, rel=1e-12
    )


def test_weighted_l2_is_not_the_unweighted_l2(configured):
    """Negative control: if the weights were being ignored the constant-field test
    above would still pass on a suitably scaled weight field, but this would not.
    """
    n = 9
    W = trapezoid_weight_field(n)
    f = torch.as_tensor(np.random.default_rng(32).standard_normal((n, n, n)))
    assert abs(l2_w(f, W) - float(f.pow(2).sum().sqrt())) > 1e-3


def test_linf_respects_a_mask(configured):
    """The mask must select, not merely scale: a large value outside it is ignored."""
    f = torch.zeros(5, 5, 5)
    f[0, 0, 0] = 100.0
    f[2, 2, 2] = 7.0
    mask = torch.zeros(5, 5, 5, dtype=torch.bool)
    mask[2, 2, 2] = True
    assert linf(f) == pytest.approx(100.0)
    assert linf(f, mask) == pytest.approx(7.0)


def test_weighted_l2_respects_a_mask(configured):
    n = 5
    W = trapezoid_weight_field(n)
    f = torch.as_tensor(np.random.default_rng(34).standard_normal((n, n, n)))
    mask = torch.zeros(n, n, n, dtype=torch.bool)
    mask[1:-1, 1:-1, 1:-1] = True
    h3 = (1.0 / (n - 1)) ** 3
    assert l2_w(f, W, mask) == pytest.approx(
        float(torch.sqrt(h3 * (W[mask] * f[mask] ** 2).sum())), rel=1e-12
    )
    assert l2_w(f, W, mask) < l2_w(f, W)


def test_empty_mask_returns_zero_rather_than_raising(configured):
    """A caller sweeping a family of masks should not have to special-case the
    empty one; max() over nothing has no value, so 0.0 is the convention."""
    f = torch.ones(5, 5, 5)
    empty = torch.zeros(5, 5, 5, dtype=torch.bool)
    assert linf(f, empty) == 0.0
    assert l2_w(f, trapezoid_weight_field(5), empty) == 0.0
