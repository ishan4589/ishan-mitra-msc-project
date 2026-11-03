# I used Claude Code to help implement and package the metric calculations.

r"""The numerical Jacobian of the square-to-disc map, and the metric fields the
operator consumes.

The map is known only as node positions, so its derivatives are measured by
central differences rather than differentiated in closed form. Those four
derivative fields are then combined into the three transverse coefficients and one
longitudinal scalar that carry the geometry into the reference-cube operator.
"""

import functools

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["jacobian_num", "metric_from_jacobian", "identity_metric"]

_PAD_2D = (1, 1, 1, 1)


# Adapted and refactored from Sun (2025).
def jacobian_num(x_cir, y_cir, nx, ny, device='cpu', h_xi=None, h_eta=None):
    r"""
    Numerically estimate the in-plane Jacobian J2 = d(x,y)/d(xi,eta) of the
    square-to-disc map, node-by-node, by central finite differences.

    The map Phi:(xi,eta) -> (x,y) is stored only as node-coordinate arrays
    (x_cir, y_cir); the smoothed square-to-disc composite has no closed form, so
    its derivatives must be approximated from neighbouring node positions. This
    function returns the four partials that make up

            J2 = [[a, b],      a = dx/dxi,  b = dx/deta,
                  [c, d]],      c = dy/dxi,  d = dy/deta,

    i.e. exactly the (a,b,c,d) the metric fields downstream are built from:
        m_xx = (z_zeta/det J2)(b^2 + d^2),   m_yy = (z_zeta/det J2)(a^2 + c^2),
        m_xy = -(z_zeta/det J2)(ab + cd),    m_L  = det J2 / z_zeta.
    The sampled determinant is used as a local mesh-validity check and as the
    in-plane physical-to-reference area scale.

    Discretisation
    --------------
    Each entry is the central-difference stencil f'(t) ~ (f(t+h) - f(t-h))/(2h),
    which is why xi-derivatives step the column index j +/- 1 and eta-derivatives
    step the row index i +/- 1. The loop covers the interior and reaches into a
    one-node halo (i+1, j+1), so x_cir/y_cir must carry that halo; the outer
    shell of J is filled afterwards by replicate padding (see note below).

    Denominator convention
    ----------------------
    A finite-difference derivative is a coordinate difference divided by the
    spacing of the variable being differentiated. By default this routine assumes
    unit-length reference axes, so h_xi = 1/(nx-1) and h_eta = 1/(ny-1). A caller
    using another extent must pass its actual spacings explicitly, as
    `CylinderGrid` does.

    Using a denominator inconsistent with the coordinates uniformly rescales J
    and desynchronises the metric from the operator grid. The clean check is the
    identity map: for x = xi, dx/dxi must be one when the coordinate differences
    and `h_xi` use the same scale.

    A uniform rescale J -> cJ cancels in the three transverse metric fields, which
    are ratios with det J2 in the denominator, but not in m_L, which carries a
    factor c^2. So the shear/compression geometry is insensitive to this choice
    and the longitudinal coefficient is not.

    Parameters
    ----------
    x_cir, y_cir : node-coordinate arrays for the physical (x,y) positions,
                   indexed [row = eta, col = xi] with a one-node halo so central
                   differences at the grid edge can read a neighbour on both sides.
    nx, ny       : number of interior reference nodes in xi, eta.
    device       : torch device for the output tensors.

    Precision
    ---------
    The four accumulators take their dtype from the input coordinate arrays rather
    than from the global default. Allocating at the default would silently downcast
    float64 node positions to float32 whenever this is called before the global
    configuration is applied, and every metric field would inherit that loss with
    no error raised anywhere. The package normally uses float64 for verification
    and differentiated solves, so preserving an explicitly supplied float64 input
    is important.
    
    h_xi, h_eta  : reference-coordinate spacings to differentiate against;
                   default to 1/(nx-1), 1/(ny-1).

    Returns
    -------
    J : tensor of shape (2, 2, ny+2, nx+2). J[p, q] is the field of
        d(coord_p)/d(ref_q); e.g. J[0,0] = a = dx/dxi. The trailing +2 in each
        spatial axis is the replicate-padded boundary shell.
    """
    # Default to unit-extent reference spacings; non-unit grids pass theirs in.
    if h_xi is None:  h_xi  = 1.0 / (nx - 1)
    if h_eta is None: h_eta = 1.0 / (ny - 1)

    # The four partials of J2, each a full (ny, nx) field (one scalar per node).
    j_11 = torch.zeros((ny, nx), device=device, dtype=x_cir.dtype)   # a = dx/dxi
    j_12 = torch.zeros((ny, nx), device=device, dtype=x_cir.dtype)   # b = dx/deta
    j_21 = torch.zeros((ny, nx), device=device, dtype=y_cir.dtype)   # c = dy/dxi
    j_22 = torch.zeros((ny, nx), device=device, dtype=y_cir.dtype)   # d = dy/deta

    # Central differences at every interior node. j +/- 1 steps along xi
    # (columns), i +/- 1 along eta (rows); the outer samples lie in the halo.
    for i in range(1, ny+1):
        for j in range(1, nx+1):
            j_11[i-1,j-1] = (x_cir[i, j+1] - x_cir[i, j-1]) / (2 * h_xi)   # dx/dxi
            j_12[i-1,j-1] = (x_cir[i+1, j] - x_cir[i-1, j]) / (2 * h_eta)  # dx/deta

            j_21[i-1,j-1] = (y_cir[i, j+1] - y_cir[i, j-1]) / (2 * h_xi)   # dy/dxi
            j_22[i-1,j-1] = (y_cir[i+1, j] - y_cir[i-1, j]) / (2 * h_eta)  # dy/deta

    # Pad the returned field by replicating its nearest computed values. This shell
    # is an extrapolation, not another derivative measurement. The production
    # metric constructor immediately crops it; it is retained in the return shape
    # for callers that expect the historical haloed layout.
    j_11 = F.pad(j_11.unsqueeze(0), pad=_PAD_2D, mode='replicate').squeeze(0)
    j_12 = F.pad(j_12.unsqueeze(0), pad=_PAD_2D, mode='replicate').squeeze(0)
    j_21 = F.pad(j_21.unsqueeze(0), pad=_PAD_2D, mode='replicate').squeeze(0)
    j_22 = F.pad(j_22.unsqueeze(0), pad=_PAD_2D, mode='replicate').squeeze(0)

    # Pack the four fields into a single 2x2 (block) x grid tensor:
    # J[p,q] holds d(coord_p)/d(ref_q), so J[0,0]=a, J[0,1]=b, J[1,0]=c, J[1,1]=d.
    J = torch.stack([
        torch.stack([j_11, j_12], dim=0),
        torch.stack([j_21, j_22], dim=0)
    ], dim=0)

    return J


def metric_from_jacobian(J2, N, z_zeta=1.0):
    """
    Turn the in-plane 2x2 Jacobian field J2 (shape (2,2,N+2,N+2), padded) into the four coefficient
    fields the operator consumes: M_T = (mxx, mxy, myy) and m_L, each (N, N, N). Uses
    M_T = (z_zeta/det J2) [[b^2+d^2, -(ab+cd)], [-(ab+cd), a^2+c^2]], m_L = det(J2) / z_zeta.
    jacobian_num lays fields out as [row = eta, col = xi]; the operator grid is [xi, eta, zeta], so we
    transpose the in-plane axes and tile along z (the metric is z-independent for a straight extrusion).
    """
    a = J2[0, 0, 1:-1, 1:-1]     # x_xi
    b = J2[0, 1, 1:-1, 1:-1]     # x_eta
    c = J2[1, 0, 1:-1, 1:-1]     # y_xi
    d = J2[1, 1, 1:-1, 1:-1]     # y_eta
    det = a * d - b * c          # det(J2)
    M00 =  z_zeta * (b * b + d * d) / det        # xi-xi   coefficient (mxx)
    M11 =  z_zeta * (a * a + c * c) / det        # eta-eta coefficient (myy)
    M01 = -z_zeta * (a * b + c * d) / det        # xi-eta  cross coefficient (mxy)
    mL  =  det / z_zeta                          # longitudinal scalar
    to3d = lambda f: f.T.contiguous().unsqueeze(-1).repeat(1, 1, N)   # [eta,xi] -> [xi,eta] -> tile z
    return (to3d(M00), to3d(M01), to3d(M11), to3d(mL))


def identity_metric(n):
    """The metric of the undistorted reference cube: J = I, so M_T = (1, 0, 1) and
    m_L = 1, and the operator collapses to the plain Cartesian Laplacian.

    This is what every piece of the mapped machinery is checked against.
    """
    one, zero = torch.ones(n, n, n), torch.zeros(n, n, n)
    return (one, zero, one, one)                       # J = I  ->  M_T = (1, 0, 1), m_L = 1
