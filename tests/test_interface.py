# I used Claude Code to help implement and package the planar-interface tests.

"""Tests for the planar-interface checkpoint.

This is the suite's direct transmission-condition check for harmonic face averaging.
The manufactured-solution study uses a smooth coefficient and a smooth analytic
map, so it certifies the stencil and nothing about the interface treatment; the
structural properties certify structure, which holds for a right and a wrong
coefficient alike.

Two negative controls demonstrate sensitivity to plausible alternatives:
arithmetic averaging of the product and placement of the jump on a node rather
than at the midpoint between adjacent samples.
"""

import pytest

from neuralphysics.verification import planar_interface_residual

# A representative muscle-like contrast, the homogeneous case and a thousandfold contrast.
SIGMA_PAIRS = [(0.09, 0.40), (1.0, 1.0), (0.01, 10.0)]


@pytest.mark.parametrize("sig_lo,sig_hi", SIGMA_PAIRS)
def test_exact_piecewise_linear_solution_is_reproduced(configured, sig_lo, sig_hi):
    """With the jump on a face midpoint the harmonic mean is the transmission
    condition, so the exact solution is reproduced to machine precision rather
    than to second order."""
    assert planar_interface_residual(17, sig_lo, sig_hi) < 1e-12


@pytest.mark.parametrize("n", [9, 17, 33])
def test_exactness_does_not_depend_on_resolution(configured, n):
    """Ensure exactness: the residual does not converge as h falls, it is already
    at round-off. A residual that shrank with n would mean the interface treatment
    was an approximation after all."""
    assert planar_interface_residual(n, 0.09, 0.40) < 1e-12


def test_exactness_is_symmetric_in_the_jump_direction(configured):
    """Nothing in the face averaging distinguishes low-to-high from high-to-low."""
    assert planar_interface_residual(17, 0.40, 0.09) < 1e-12


def test_negative_control_arithmetic_mean_of_the_product_fails(configured):
    """Averaging sigma*m arithmetically instead of averaging sigma harmonically.
    This is the plausible wrong implementation, and it must be caught."""
    assert planar_interface_residual(17, 0.09, 0.40, arithmetic=True) > 1e-3


def test_negative_control_jump_on_a_node_fails(configured):
    """The midpoint harmonic mean is exact for equal half-cell lengths on either
    side of the jump. A node-aligned jump does not have that sampled geometry."""
    assert planar_interface_residual(17, 0.09, 0.40, at_node=True) > 1e-3


def test_the_two_controls_fail_for_different_reasons(configured):
    """Both controls are large, but they are distinct defects and should not
    coincide numerically."""
    a = planar_interface_residual(17, 0.09, 0.40, arithmetic=True)
    b = planar_interface_residual(17, 0.09, 0.40, at_node=True)
    assert abs(a - b) > 1e-3
