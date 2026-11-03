# I used Claude Code to help implement and package the Neumann-system projections.

"""Quadrature weights and the two projections that make the singular Neumann
system solvable and its solution unique.

The operator has a one-dimensional right null space (constants) and a
one-dimensional left null space (the trapezoidal weight field W). Both hold for
any positive coefficient field, so neither projection is affected by the tissue
layering or by the coordinate map.
"""

import torch

__all__ = [
    "trapezoid_weight_field",
    "impose_net_zero_CSD",
    "impose_electric_potential_uniqueness",
]


def trapezoid_weight_field(n, dtype=None, device=None):
    """3D vertex-centred trapezoidal quadrature weights: 1 in the interior, 1/2 on faces,
    1/4 on edges, 1/8 at corners.

    Two roles, one object. As a quadrature rule, sum(W) = (n-1)^3. Thus
    h^3*sum(W) is the cube volume when h is its uniform nodal spacing, and
    h^3*sum(W*f) is the trapezoidal integral estimate. As linear algebra, W is
    the discrete left null vector of A (A^T W = 0), so the same field tests source
    compatibility and defines the volume-weighted gauge.

    Both roles hold for any positive coefficient field, so layering does not change W.
    """
    w_axis = torch.ones(n, dtype=dtype, device=device)
    w_axis[0] = w_axis[-1] = 0.5
    return torch.einsum("i,j,k->ijk", w_axis, w_axis, w_axis)


def impose_net_zero_CSD(I):  # current source density (CSD)
    """
    Project a 3D source field v (shape (N, N, N)) onto the zero-(weighted-)mean
    subspace so that the discrete system A phi = v is solvable.

    Theory:
    Integrating the PDE over the domain Omega with the homogeneous Neumann BC gives
    the compatibility condition ∫_Ω I dV = 0. The discrete analogue is the
    solvability condition w·v = 0, where w is the left null vector of the
    reflect-padded Laplacian A (i.e. A^T w = 0). On this vertex-centred grid that
    left null vector is exactly the 3D trapezoidal quadrature weight field, so w·v
    is the trapezoidal estimate of ∫_Ω I dV. Subtracting the trapezoidal-weighted
    mean makes any input field admissible.
    """
    n = I.shape[0]

    assert I.shape == (n, n, n)  # require a cubic 3-D field

    # W is the discrete left null vector of A and the trapezoidal quadrature rule at once, so
    # h^3 W·I is the trapezoidal integral estimate on a grid of spacing h. Corners
    # get 1/8, edges 1/4, faces 1/2 and interior nodes 1.
    W = trapezoid_weight_field(n, dtype=I.dtype, device=I.device)

    # Trapezoidal-weighted mean of v: (W·v)/(W·1). The common h^3 volume factor
    # cancels top and bottom, so this equals (∫v dV)/(∫1 dV) on the grid.
    weighted_mean = (W * I).sum() / W.sum()   # scalar constant to remove

    # Subtract that constant from every node.
    # Then W·(v - weighted_mean) = W·v - weighted_mean*(W·1) = 0,
    # so the projected field satisfies the
    # discrete compatibility condition exactly and A phi = v is now solvable.
    return I - weighted_mean


# Gauge choice motivated by Maksymenko et al. (2023), who constrain the potentials
# by a zero-sum condition. Here the trapezoidal-weighted sum is used instead, as the
# discrete analogue of a zero volume mean on a vertex-centred grid.
#
# The arithmetic here is identical to impose_net_zero_CSD -- both remove the W-weighted mean --
# but the two are kept as separate names because they do different jobs.
# impose_net_zero_CSD enforces solvability of A phi = b.
# This function selects a unique representative of phi.
# One acts on the source, the other on the field.
# Both remain exact under layering, since ker(A) = span{1} and ker(A^T) = span{W} hold for any
# positive coefficient field.
def impose_electric_potential_uniqueness(phi):
    """
    Impose uniqueness of phi by fixing the additive constant in the family phi + c.

    The constant is chosen so that phi is orthogonal to ker(A) = span{1} in the volume-weighted
    inner product <u, v>_W = sum W_ijk u_ijk v_ijk, i.e. the volume-weighted mean is removed rather
    than the plain arithmetic mean. This selects the minimum-W-norm representative of the coset and
    is the same projection impose_net_zero_CSD applies to the source, so field and source are gauged
    consistently. Maksymenko et al. (2023) constrain a plain zero sum; on this vertex-centred
    trapezoidal grid the weighted sum is the discrete analogue of a zero mean potential over the
    domain.
    """
    n = phi.shape[0]

    # The continuous gauge condition we are discretising is the zero mean potential
    #     (1/|Omega|) * integral_Omega phi dV = 0.
    # On a vertex-centred grid each node owns a box of side h, but a node ON a face owns only
    # half of one, an edge node a quarter, a corner node an eighth; W records exactly that.
    W = trapezoid_weight_field(n, dtype=phi.dtype, device=phi.device)

    # Discrete analogue of the continuous mean. The numerator h^3 * (W*phi).sum() is the
    # trapezoidal estimate of integral_Omega phi dV, and the denominator h^3 * W.sum() is |Omega|;
    # the common h^3 cancels, leaving the volume-weighted mean below. Using phi.mean() instead
    # would over-count the boundary nodes, which own less volume than the interior ones.
    weighted_mean = (W * phi).sum() / W.sum()

    # Subtracting a constant is legitimate precisely because constants span ker(A): A(phi - c) =
    # A phi, so phi - c solves the same system. This choice of c is the one making the result
    # W-orthogonal to 1, since W . (phi - c) = W . phi - c * W . 1 = 0 by construction.
    return phi - weighted_mean
