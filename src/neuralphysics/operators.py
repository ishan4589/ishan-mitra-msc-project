# I used Claude Code to help implement and package the finite-volume operators.

r"""The discretisation: the conservative flux-form finite-volume operator on the
reference cube, and the plain Cartesian 7-point operator it must collapse onto.

The mapped operator uses fixed neighbour-difference `conv3d` kernels for its
diagonal fluxes, multiplies them by spatially varying face coefficients, and adds
the in-plane cross term separately. Conductivity and the diagonal metric factors
are multiplied at nodes and the complete product is harmonically averaged onto
faces. The cross coefficient is instead averaged arithmetically over cell corners.

The insulating (homogeneous Neumann) condition is imposed by omitting the
boundary faces from the flux balance. Reflect padding together with mirrored
end-face coefficients reproduces the corresponding half-control-volume rows.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "NEIGHBOUR_DIFF_KERNELS",
    "face_coefficients",
    "laplacian_metric_3D",
    "laplacian_kernel",
    "laplacian_3D",
]


def _pad(f):
    # Reflect-pad one layer on every axis, matching `laplacian_3D`.
    return F.pad(
        f.view(1, 1, *f.shape), (1, 1, 1, 1, 1, 1), mode="reflect"
    ).view(f.shape[0] + 2, f.shape[1] + 2, f.shape[2] + 2)


# Fixed +/- neighbour-difference kernels in face order
# (+xi, -xi, +eta, -eta, +zeta, -zeta). Each computes the neighbour
# value minus the centre value. PyTorch's conv3d is a
# cross-correlation, not a true convolution, so the kernels are used unflipped.
_FACE_OFFSETS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _build_neighbour_diff_kernels():
    K = torch.zeros(6, 1, 3, 3, 3)
    for f, (dx, dy, dz) in enumerate(_FACE_OFFSETS):
        K[f, 0, 1, 1, 1] = -1.0
        K[f, 0, 1 + dx, 1 + dy, 1 + dz] = 1.0
    return K


NEIGHBOUR_DIFF_KERNELS = _build_neighbour_diff_kernels()  # (6, 1, 3, 3, 3)


def face_coefficients(metric, sigma_T, sigma_L):
    """Face coefficients of the flux form, as (6, n, n, n) in the face order
    (+xi, -xi, +eta, -eta, +zeta, -zeta).

    Each is a single harmonic mean of the whole product:

        a_face = harm(sigma_lo * m_lo, sigma_hi * m_hi)

    and not a product of two separately averaged factors. The flux form needs the number that
    makes a two-point Ohm's law reproduce the true current between two nodes, not the value of
    the coefficient near the face. Since -(a phi')' = 0 has constant flux, that number is
    a* = h / integral(dx / a) over the cell. This integral harmonic mean is the
    exact effective coefficient for a one-dimensional cell, while `harm()` is the
    endpoint trapezoidal approximation to it. Arithmetic averaging instead
    averages conductance, which is not the quantity that accumulates in series.
    The transmission condition at a sharp interface is the epsilon -> 0 corollary of this, not
    its premise; what decides whether the choice of mean matters is epsilon/h, not whether
    sigma is formally continuous.

    Each product is averaged along the axis its metric factor belongs to: sigma_T*mxx onto
    xi-faces, sigma_T*myy onto eta-faces, sigma_L*mL onto zeta-faces. One code path serves all
    six, with no special case for the fibre axis: the extrusion is straight, so sigma_L is
    z-invariant and mL is tiled along z, hence harm(a, a) = a on every zeta-face.

    Inputs must be strictly positive -- harm has a pole at a_lo = -a_hi. TissueModel enforces
    sigma > 0, and mxx, myy, mL > 0 follow from the mesh-validity gate det J2 > 0.

    An axis with n nodes has only n-1 internal faces, so the outermost entry describes no face
    and is filled by mirroring. With the reflect padding of phi this reproduces exactly the row
    obtained by dropping the boundary face and halving the control volume, so the mirrored
    value supplies the mirrored coefficient used to form the half-volume boundary
    row; it is not an independently sampled exterior-face coefficient.

    mxy does not appear here: a cross term has no two-point face flux, and it is sign-indefinite,
    so no harmonic mean exists for it. It is assembled cell-centred in the operator, with sigma_T
    folded into its four-corner arithmetic average. The algebra is in the final report (final-report.pdf).
    """
    mxx, _mxy, myy, mL = metric
    harm = lambda a, b: 2.0 * a * b / (a + b)

    coeffs = []
    for m_field, s_field, axis in (
        (mxx, sigma_T, 0),
        (myy, sigma_T, 1),
        (mL, sigma_L, 2),
    ):
        a_field = s_field * m_field  # the whole face coefficient
        n = a_field.shape[axis]
        a_lo, a_hi = a_field.narrow(axis, 0, n - 1), a_field.narrow(
            axis, 1, n - 1
        )

        face = harm(a_lo, a_hi)
        plus = torch.cat(
            [face, face.narrow(axis, n - 2, 1)], dim=axis
        )  # no +face at i = n-1
        minus = torch.cat(
            [face.narrow(axis, 0, 1), face], dim=axis
        )  # no -face at i = 0
        coeffs += [plus, minus]
    return torch.stack(coeffs, dim=0)


# I forgot to update this stale function name as it no longer describes the operator.
# A more accurate name would be "apply_mapped_conductivity_operator".
def laplacian_metric_3D(phi, h, faces, metric, sigma_T):
    """div_xi(sigma-tilde grad_xi phi) as a conservative finite-volume operator on the reference
    cube, with the homogeneous Neumann boundary condition imposed by omitting the boundary
    faces from the flux balance.

    Diagonal terms are two-point face fluxes: the precomputed (6, n, n, n) face coefficients are
    contracted against the six fixed +-1 neighbour-difference kernels on the reflect-padded phi.
    Reflect padding with the mirrored end-face coefficient is algebraically identical to dropping
    the boundary face and dividing by the halved control volume, so no explicit 1/v is needed here.

    sigma is not a scalar multiplier because a conductivity pulled outside the difference cannot
    encode a jump between tissue interfaces. Instead it lives inside `faces`, harmonically averaged onto
    each face by `face_coefficients()`, so the coefficient remains inside the
    discrete divergence and represents jumps through a face conductance.
    Consequently the old split A(sigma) = sigma_T A_T + sigma_L A_L is no longer correct, as harmonic means are not additive.
    What remains is degree-one homogeneity, A(lambda sigma) = lambda A(sigma).

    Passing `faces` in (rather than building them here) keeps the operator and the smoother
    from ever disagreeing about what sigma is: both read the same tensor.

    The cross term has no two-point face flux, and is assembled cell-centred from the symmetric
    bilinear form  -int sigma_T m_xy (phi_xi psi_eta + phi_eta psi_xi) with Q1 shape functions and
    one-point quadrature at the cell centre. sigma_T is folded into the four-corner average, i.e.
    the average is taken over the product sigma_T*m_xy. There is no transmission condition at a cell
    centre for a harmonic mean to encode, and one-point quadrature of a discontinuous coefficient is
    first-order. The four-corner average used here is arithmetic.
    It carries 1/(v_i v_j) explicitly; the zero slabs in the outer difference are the boundary condition.

    Parameters
    ----------
    faces   : (6, n, n, n) face coefficients from face_coefficients(metric, sigma_T, sigma_L),
              in the order (+xi, -xi, +eta, -eta, +zeta, -zeta).
    metric  : (mxx, mxy, myy, mL). Only mxy is read here; the diagonal entries have already been
              consumed into `faces`.
    sigma_T : (n, n, n) transverse conductivity field, for the cross term only.
    """
    mxx, mxy, myy, mL = metric
    n = phi.shape[0]

    # --- diagonal terms: two-point face fluxes, boundary faces omitted ------------------------
    d = F.conv3d(_pad(phi)[None, None], NEIGHBOUR_DIFF_KERNELS.to(phi))[
        0
    ]  # (6, n, n, n)
    out = (faces * d).sum(0) / h**2

    # --- cross term: cell-centred variational assembly ----------------------------------------
    # Skip work when the identity metric makes the cross coefficient identically zero.
    if torch.count_nonzero(mxy) > 0:
        v = torch.ones(n, dtype=phi.dtype, device=phi.device)
        v[0] = v[-1] = 0.5  # trapezoidal / control-volume factor

        # Q1 derivatives at the centre of each in-plane cell: mean of the two opposite edge diffs
        dxi_c = (
            (phi[1:, :-1] - phi[:-1, :-1]) + (phi[1:, 1:] - phi[:-1, 1:])
        ) / (2 * h)
        deta_c = (
            (phi[:-1, 1:] - phi[:-1, :-1]) + (phi[1:, 1:] - phi[1:, :-1])
        ) / (2 * h)

        smxy = sigma_T * mxy  # average the product, per D3
        m_c = 0.25 * (
            smxy[:-1, :-1] + smxy[1:, :-1] + smxy[:-1, 1:] + smxy[1:, 1:]
        )
        P, Q = m_c * deta_c, m_c * dxi_c  # (n-1, n-1, n)

        # scatter each cell value onto the two nodes bounding it in the transverse axis
        Pbar = F.pad(P, (0, 0, 0, 1)) + F.pad(P, (0, 0, 1, 0))  # (n-1, n,   n)
        Qbar = F.pad(Q, (0, 0, 0, 0, 0, 1)) + F.pad(
            Q, (0, 0, 0, 0, 1, 0)
        )  # (n,   n-1, n)

        # outer difference; the zero slabs at each end are the insulating boundary condition
        Pz = F.pad(Pbar, (0, 0, 0, 0, 1, 1))  # (n+1, n,   n)
        Qz = F.pad(Qbar, (0, 0, 1, 1))  # (n,   n+1, n)
        cross = ((Pz[1:] - Pz[:-1]) + (Qz[:, 1:] - Qz[:, :-1])) / (2 * h)

        out = out + cross / (v.view(-1, 1, 1) * v.view(1, -1, 1))

    return out


def laplacian_kernel(h, sigma_T, sigma_L):
    """
    Stage 0 oracle only - never used to solve.

    7-point face-neighbour stencil for the diagonal, homogeneous anisotropic
    operator div(sigma . grad), fibres along z (sigma_x = sigma_y = sigma_T,
    sigma_z = sigma_L).
    """
    k = torch.zeros(3, 3, 3)
    k[1, 1, 1] = -2.0 * (
        2.0 * sigma_T + sigma_L
    )  # centre: -2(sigma_x+sigma_y+sigma_z)
    k[0, 1, 1] = k[2, 1, 1] = sigma_T  # ±x face-neighbours (transverse)
    k[1, 0, 1] = k[1, 2, 1] = sigma_T  # ±y face-neighbours (transverse)
    k[1, 1, 0] = k[1, 1, 2] = (
        sigma_L  # ±z face-neighbours (longitudinal, along fibre)
    )
    k = k / h**2
    return k.view(1, 1, 3, 3, 3)


def laplacian_3D(phi, h, sigma_T, sigma_L):
    """
    Stage 0 oracle only - never used to solve.

    3D anisotropic operator div(sigma . grad phi) for a diagonal, homogeneous
    conductivity tensor (fibres along z), applied as a fixed-structure 7-point
    conv3d whose weights come from the arguments sigma_T and sigma_L.

    Reflect padding sets the ghost nodes on all six faces
    (phi[-1]=phi[1], phi[N]=phi[N-2] in each axis).

    This is the identity-map limit of laplacian_metric_3D, and exists so that the
    mapped operator can be checked against an implementation that shares none of
    its machinery. Nothing in the solver path calls it.
    """
    n = phi.shape[0]
    assert phi.shape == (n, n, n)
    phi_in = phi.view(1, 1, n, n, n)  # (batch, channel, D, H, W)
    phi_pad = F.pad(
        phi_in, (1, 1, 1, 1, 1, 1), mode="reflect"
    )  # pad all 3 spatial axes - mirror about the boundary node
    return F.conv3d(phi_pad, laplacian_kernel(h, sigma_T, sigma_L)).view(
        n, n, n
    )
