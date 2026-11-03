# I used Claude Code to help implement and package the multigrid solver.

r"""The multigrid solver: grid transfers, the weighted-Jacobi smoother, and the V-cycle.

The operator is singular -- A 1 = 0 and A^T W = 0 -- so every level's problem is
singular too, and the solver controls its constant component during the cycle
rather than only at the end. The fine-grid residual r = I - A u is compatible when
W^T I = 0 because W^T A = 0. After restriction it is projected again so that the
coarse right-hand side satisfies the corresponding coarse-grid condition despite
transfer and round-off effects.

The coarse operator is a rediscretisation of the same shell geometry at the
coarser grid, so each level supplies its own coefficients rather than restricted ones.
Because the coarse disc is a different size and its interfaces re-staircase,
the coarse levels resolve slightly different geometry -- a coarse solve of a nearby problem,
which is what a preconditioner is.
"""

import torch
import torch.nn.functional as F

from .operators import laplacian_metric_3D
from .projections import (
    impose_electric_potential_uniqueness,
    impose_net_zero_CSD,
)

__all__ = ["restrict", "prolong", "probe_diagonal", "jacobi", "v_cycle"]


def restrict(r_fine):
    """
    Full-weighting restriction (fine -> coarse), 3D.

    Input  r_fine:   (N,   N,   N  ) with N = 2^k + 1
    Output r_coarse: (N_c, N_c, N_c) with N_c = (N + 1) // 2

    The 1D full-weighting stencil (1/4, 1/2, 1/4) becomes its 3x3x3 tensor product
    (a 27-point weighted average summing to 1, so constants are preserved). Implemented
    as a reflect-padded, stride-2 conv3d: with pad=1 the output at coarse index J samples
    fine indices (2J-1, 2J, 2J+1), and reflect padding supplies the missing
    neighbour at each endpoint using the same mirror convention as the operator.
    """
    n = r_fine.shape[0]
    assert r_fine.shape == (n, n, n)
    assert (n - 1) % 2 == 0, "need N = 2^k + 1 for vertex-centred coarsening"

    w = torch.tensor([0.25, 0.5, 0.25])  # 1D full-weighting weights
    kernel = torch.einsum("i,j,k->ijk", w, w, w).view(1, 1, 3, 3, 3)

    r_in = r_fine.view(1, 1, n, n, n)
    r_pad = F.pad(
        r_in, (1, 1, 1, 1, 1, 1), mode="reflect"
    )  # Neumann mirror at the six faces
    r_coarse = F.conv3d(
        r_pad, kernel, stride=2
    )  # stride 2: keep every other node

    n_c = (n + 1) // 2
    return r_coarse.view(n_c, n_c, n_c)


def prolong(e_coarse):
    """
    Trilinear-interpolation prolongation (coarse -> fine), 3D.

    Input  e_coarse: (N_c, N_c, N_c)
    Output e_fine:   (N,   N,   N  ) with N = 2 * N_c - 1

    align_corners=True so the coarse endpoints land exactly on the fine endpoints:
    coarse node J maps to fine node 2J (exact copy) and each odd fine node is the
    linear average of its two coarse neighbours.
    """
    n_c = e_coarse.shape[0]
    assert e_coarse.shape == (n_c, n_c, n_c)
    n = 2 * n_c - 1

    e_in = e_coarse.view(1, 1, n_c, n_c, n_c)
    e_fine = F.interpolate(
        e_in, size=(n, n, n), mode="trilinear", align_corners=True
    )
    return e_fine.view(n, n, n)


def probe_diagonal(faces, metric, h, sigma_T):
    """
    True diagonal of A, by 27-colour probe. The flux-form stencil reaches one node per axis
    (nine-point in plane, three-point in zeta), so nodes of the same colour, three apart, never
    interact and each probe reads a clean diagonal entry.

    It is probed rather than derived: an analytic sum of face coefficients would miss the 1/v row
    scaling on the boundary rows, and would miss the cross term, which does reach the centre node
    - always at an in-plane boundary node, where the node borders fewer than four cells so the
    alternating cell contributions cannot cancel, and at interior nodes through the cell-to-cell
    variation of mxy.

    Runs under no_grad deliberately. D^-1 is a frozen preconditioner: sigma-dependent, but not
    differentiated through. Gradients still flow through every operator application in jacobi
    and through the residual, so the parameters are reached -- but not exactly. Detaching the
    diagonal means autograd differentiates a slightly different iteration from the one that
    ran: the smoother's update carries a term in dD^-1/dsigma that is discarded here. The
    discrepancy shrinks as the solve converges, because that term is multiplied by the residual
    and vanishes with it, and it does not reach zero at a finite cycle budget.

    The old scheme probed two geometry-only diagonals once per level and recombined them as
    sigma_T*dT + sigma_L*dL. That is gone: harmonic face means are not additive, so no such split
    exists. Worse, the probes that produced it are not even computable now - they were taken at
    sigma = (1, 0) and (0, 1), and harm(0, 0) = 0/0 = nan.

    Pure: no cache. The diagonal depends on the face tensor, whose identity changes at every
    optimiser step, so the cache belongs to whatever owns the face tensors - see the
    `diag_by_n` argument of `v_cycle`.
    """
    n = metric[0].shape[0]
    d = torch.zeros(n, n, n, dtype=sigma_T.dtype, device=sigma_T.device)
    with torch.no_grad():
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    e = torch.zeros_like(d)
                    e[a::3, b::3, c::3] = 1.0
                    d[a::3, b::3, c::3] = laplacian_metric_3D(
                        e, h, faces, metric, sigma_T
                    )[a::3, b::3, c::3]
    return d


def jacobi(
    phi, I, h, faces, metric, sigma_T, omega=2.0 / 3.0, num_iters=1, diag=None
):
    """
    Weighted Jacobi with the true diagonal (probed, not derived). Every node is updated,
    including the boundary shell: under the Neumann operator a boundary node is a genuine
    unknown, not a pinned value.

    `diag` is the probed diagonal for this level. Omitted, it is probed here; supplied, the
    probe is skipped. Probing is 27 operator applications, so a caller that smooths repeatedly
    on the same coefficients should probe once and pass it in.

    omega is checked here rather than by the callers. Every route into the smoother -- a bare
    call, a V-cycle, a forward solve -- arrives through this function, so one statement covers
    all three. Outside (0, 1] the iteration is no longer a damped average of the current
    iterate and the residual correction: at omega <= 0 it steps the wrong way, and above 1 it
    over-relaxes past the point where the high-frequency error is damped, which is the one
    property the cycle needs from it.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError(
            f"damped-Jacobi omega must lie in (0, 1], got {omega}"
        )

    if diag is None:
        diag = probe_diagonal(faces, metric, h, sigma_T)

    # Strict negativity does not follow from mxx, myy, mL > 0: the cross term reaches the
    # diagonal and is not sign-definite. It is checked, because a sign flip here silently
    # turns the smoother into an amplifier at that node rather than raising anything.
    assert bool((diag < 0).all()), (
        f"Jacobi diagonal is not strictly negative at n = {metric[0].shape[0]} "
        f"(max {diag.max().item():.3e}): the cross term has overwhelmed the face terms at "
        f"some node. Check mesh skew (|mxy| against sqrt(mxx*myy)) there."
    )

    d_inv = 1.0 / diag
    for _ in range(num_iters):
        r = I - laplacian_metric_3D(phi, h, faces, metric, sigma_T)
        phi = phi + omega * d_inv * r
    return phi


def v_cycle(
    phi,
    I,
    h,
    faces_by_n,
    sigma_T_by_n,
    metric_levels,
    n_base,
    omega=2.0 / 3.0,
    nu_pre=2,
    nu_post=2,
    base_iters=50,
    diag_by_n=None,
):
    """
    Recursive V-cycle on the mapped operator. Metric, face coefficients and sigma_T threaded
    per level.

    The coarse correction and the base solve are gauge-fixed so the constant they carry is not
    injected into the fine field.

    `n_base` is required rather than defaulted: the floor is a property of the coefficient
    hierarchy being descended, whose coarsest level is the only one this cycle can reach, so a
    default here would be a second statement of something the caller already knows. Where that
    floor comes from in the first place is `grid.levels`.

    `diag_by_n` is a {level -> diagonal} cache. Omitted, one is created for this cycle, so each
    level is probed once per call; supplied and retained by the caller, each level is probed
    once per set of coefficients. It is keyed on the level because the caller already guarantees
    the coefficients are fixed for the lifetime of the dict -- keying on id(faces) instead would
    be a cache whose correctness depends on an object staying alive.
    """
    n = phi.shape[0]
    metric = metric_levels[n]
    faces = faces_by_n[n]
    sig_T = sigma_T_by_n[n]

    if diag_by_n is None:
        diag_by_n = {}
    if n not in diag_by_n:
        diag_by_n[n] = probe_diagonal(faces, metric, h, sig_T)
    diag = diag_by_n[n]

    if n <= n_base:  # base level: use a larger fixed number of smoothing sweeps
        return impose_electric_potential_uniqueness(
            jacobi(
                phi, I, h, faces, metric, sig_T, omega, base_iters, diag=diag
            )
        )

    u = jacobi(
        phi, I, h, faces, metric, sig_T, omega, nu_pre, diag=diag
    )  # pre-smooth

    r_fine = I - laplacian_metric_3D(u, h, faces, metric, sig_T)  # residual

    r_coarse = impose_net_zero_CSD(restrict(r_fine))  # keep it admissible

    e_coarse = v_cycle(
        torch.zeros_like(r_coarse),
        r_coarse,
        2.0 * h,
        faces_by_n,
        sigma_T_by_n,
        metric_levels,
        n_base,
        omega,
        nu_pre,
        nu_post,
        base_iters,
        diag_by_n,
    )  # coarse-grid correction

    u = u + prolong(impose_electric_potential_uniqueness(e_coarse))  # correct

    return jacobi(
        u, I, h, faces, metric, sig_T, omega, nu_post, diag=diag
    )  # post-smooth
