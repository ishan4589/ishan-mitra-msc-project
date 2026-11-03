# I used Claude Code to help implement and package the structural verification.

r"""The structural properties of the discrete operator, which we measure.

Six properties are checked over the positive coefficient fields used here. Flux
pairing establishes the null-space and weighted-symmetry identities; positivity
and grid connectivity are also needed for the one-dimensional-null-space claims.

    1. A 1 = 0                 constants span the right null space
    2. A^T W = 0               the trapezoidal weights span the left null space
    3. A^T 1 != 0              the two null spaces are distinct
    4. rank deficiency = 1     the singularity is exactly one-dimensional
    5. diag(W) A symmetric     the operator is self-adjoint in the W inner product
    6. cos(u_min, W) = 1       the smallest left-singular direction is W itself

A failure of the algebraic identities localises to the operator or coefficient
assembly rather than to truncation error.

The interface residual is the one check that exercises the harmonic face
averaging. The manufactured-solution study uses a smooth coefficient and a smooth
analytic map, so it certifies the stencil and says nothing about the interface
treatment; the structural properties certify structure, which holds for a right
and a wrong coefficient alike.
"""

import numpy as np
import torch

from ..metric import identity_metric
from ..operators import face_coefficients, laplacian_metric_3D
from ..projections import trapezoid_weight_field

__all__ = [
    "assemble_dense",
    "structural_metrics",
    "matrix_free_null_checks",
    "planar_interface_residual",
]


def assemble_dense(n, h, faces, metric, sigma_T):
    """A as an (n^3, n^3) matrix, column by column, by applying the operator to unit basis fields."""
    N3 = n ** 3
    M  = torch.zeros(N3, N3)
    E  = torch.eye(N3)
    for k in range(N3):
        M[:, k] = laplacian_metric_3D(E[k].view(n, n, n), h, faces, metric, sigma_T).reshape(-1)
    return M


def structural_metrics(n, metric, sigma_T, sigma_L):
    """The six structural numbers for one (grid, metric, coefficient) configuration.

    Returns a dict with `scale`, `r_null`, `l_null`, `l_ones`, `rank_deficiency`,
    `symmetry` and `cosine`. `scale` is the largest entry of A in modulus and is
    what the residual-like quantities should be judged against: the operator
    carries a 1/h^2, so an absolute tolerance would mean something different at
    every resolution.

    Dense assembly is n^3 operator applications building an (n^3, n^3) matrix, so
    this is for n of 9 or so and no more. Use `matrix_free_null_checks` at
    production size.
    """
    h_n   = 1.0 / (n - 1)
    faces = face_coefficients(metric, sigma_T, sigma_L)
    M     = assemble_dense(n, h_n, faces, metric, sigma_T)

    W    = trapezoid_weight_field(n).reshape(-1)
    one  = torch.ones(n ** 3)
    D    = torch.diag(W)
    sc   = M.abs().max().item()                    # operator scale, for relative tolerances

    # Use NumPy's LAPACK SVD path, which has been more robust for this assembled
    # matrix in the supported test environments. Only its left singular vectors
    # are needed below.
    U = torch.from_numpy(np.linalg.svd(M.numpy())[0])

    return {
        "scale":           sc,
        "r_null":          (M @ one).abs().max().item(),   # 1. right null space = constants
        "l_null":          (M.T @ W).abs().max().item(),   # 2. left null space  = trapezoidal weights
        "l_ones":          (M.T @ one).abs().max().item(), # 3. constants NOT in the left null space
        "rank_deficiency": n ** 3 - int(torch.linalg.matrix_rank(M, tol=1e-9 * sc)),
        "symmetry":        (D @ M - (D @ M).T).abs().max().item(),   # 5. W-self-adjointness
        "cosine":          abs(torch.dot(U[:, -1], W / W.norm())).item(),  # 6. smallest left sing. dir.
    }


def matrix_free_null_checks(n, h, faces, metric, sigma_T):
    """Both null-space residuals at production size, using operator applications only.

    Dense assembly at N = 65 would be a 274,625 x 274,625 matrix. The right null
    residual is one forward application. The left one uses
    (A^T W)_j = d/dphi_j <W, A phi>, so a single backward pass gives the adjoint
    product without ever forming A.

    Returns a dict with `r_null` and `l_null`.
    """
    W = trapezoid_weight_field(n)

    r_null = laplacian_metric_3D(
        torch.ones(n, n, n), h, faces, metric, sigma_T).abs().max().item()

    x = torch.zeros(n, n, n, requires_grad=True)
    (laplacian_metric_3D(x, h, faces, metric, sigma_T) * W).sum().backward()
    l_null = x.grad.abs().max().item()

    return {"r_null": r_null, "l_null": l_null}


def planar_interface_residual(n, sig_lo, sig_hi, at_node=False, arithmetic=False):
    """Interior residual of the exact piecewise-linear solution across a plane interface.

    Model problem: sigma piecewise constant along xi with the jump on a face midpoint, and phi the
    exact piecewise-linear solution with continuous normal current, sigma_L * slope_L = sigma_R *
    slope_R. Then div(sigma grad phi) = 0 exactly, and the flux form should reproduce that at every
    interior node - better than second order, because the harmonic mean is the transmission
    condition rather than an approximation to it.

    Boundary rows are excluded: they carry a nonzero row for any non-constant phi, which is the
    insulating condition being imposed, not an error.

    The two flags are negative controls, and the check is only evidence because they
    exist. `arithmetic=True` averages the product sigma*m arithmetically instead of
    averaging sigma harmonically; `at_node=True` places the jump on a node rather
    than a face midpoint. Both should return a residual many orders larger.
    """
    h_n   = 1.0 / (n - 1)
    x     = torch.arange(n, dtype=torch.get_default_dtype()) * h_n
    m     = n // 2
    x_int = (m * h_n) if at_node else ((m + 0.5) * h_n)      # node vs face midpoint

    slope_lo = 1.0
    slope_hi = sig_lo * slope_lo / sig_hi                     # continuity of normal current
    phi1d = torch.where(x <= x_int, slope_lo * (x - x_int), slope_hi * (x - x_int))
    phi   = phi1d.view(-1, 1, 1).expand(n, n, n).contiguous()

    cut = (x < x_int) if at_node else (x <= x_int)
    sig = torch.where(cut, torch.tensor(sig_lo), torch.tensor(sig_hi)) \
               .view(-1, 1, 1).expand(n, n, n).contiguous()

    met   = identity_metric(n)
    faces = face_coefficients(met, sig, sig)
    if arithmetic:                                            # deliberately wrong, to show the test bites
        mxx, _, myy, mL = met
        wrong = []
        for m_f, s_f, ax in ((mxx, sig, 0), (myy, sig, 1), (mL, sig, 2)):
            k = m_f.shape[ax]
            f = (0.5 * (s_f.narrow(ax, 0, k - 1) + s_f.narrow(ax, 1, k - 1))
                 * 0.5 * (m_f.narrow(ax, 0, k - 1) + m_f.narrow(ax, 1, k - 1)))
            wrong += [torch.cat([f, f.narrow(ax, k - 2, 1)], dim=ax),
                      torch.cat([f.narrow(ax, 0, 1), f], dim=ax)]
        faces = torch.stack(wrong, 0)

    r = laplacian_metric_3D(phi, h_n, faces, met, sig)
    return r[1:-1].abs().max().item()                         # interior rows only
