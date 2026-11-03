# I used Claude Code to help implement and package tests for operators.py.

"""Tests for neuralphysics.operators."""

import numpy as np
import pytest
import torch

from neuralphysics.grid import grid_at
from neuralphysics.metric import identity_metric
from neuralphysics.operators import (
    NEIGHBOUR_DIFF_KERNELS,
    face_coefficients,
    laplacian_3D,
    laplacian_metric_3D,
    _pad,
)
from neuralphysics.tissue import LIMB


def problem(n):
    """(h, metric, sigma_T, sigma_L, faces) for the layered limb at resolution n.

    The coefficient field the solver actually sees, at the tissue model's own radii
    and transition width -- so these tests exercise the operator on the medium the
    experiment runs on rather than on a second convention invented here.
    """
    g = grid_at(n)
    sT, sL = LIMB.sigma_fields(g)
    return g.h, g.metric, sT, sL, face_coefficients(g.metric, sT, sL)


# ---- the identity oracle ----------------------------------------------------


@pytest.mark.parametrize("n", [9])
def test_identity_metric_collapses_onto_the_seven_point_stencil(configured, n):
    """With J = I and a homogeneous coefficient the flux form must reproduce the
    plain Cartesian operator exactly. The two implementations share no code, so
    this is the sharpest available check on the face assembly and the padding.
    """
    h = 1.0 / (n - 1)
    sT_s, sL_s = 0.09, 0.40
    metric = identity_metric(n)
    sT = torch.full((n, n, n), sT_s)
    sL = torch.full((n, n, n), sL_s)
    faces = face_coefficients(metric, sT, sL)
    phi = torch.as_tensor(np.random.default_rng(0).standard_normal((n, n, n)))

    got = laplacian_metric_3D(phi, h, faces, metric, sT)
    want = laplacian_3D(phi, h, sT_s, sL_s)
    assert float((got - want).abs().max()) < 1e-14 * float(want.abs().max())


def test_cross_term_is_skipped_only_when_mxy_vanishes(configured):
    """The identity branch is taken on the strength of mxy being identically zero.
    On the mapped grid it is not, and the cross term must therefore contribute.
    """
    n, h = 9, 1.0 / 8
    metric = grid_at(n).metric
    sT = torch.full((n, n, n), 0.09)
    sL = torch.full((n, n, n), 0.40)
    faces = face_coefficients(metric, sT, sL)
    phi = torch.as_tensor(np.random.default_rng(2).standard_normal((n, n, n)))

    mxx, mxy, myy, mL = metric
    with_cross = laplacian_metric_3D(phi, h, faces, metric, sT)
    without = laplacian_metric_3D(
        phi, h, faces, (mxx, torch.zeros_like(mxy), myy, mL), sT
    )
    assert float((with_cross - without).abs().max()) > 1e-6 * float(
        with_cross.abs().max()
    )


def test_seven_point_oracle_is_exact_on_a_quadratic(configured):
    """div(sigma grad) of a quadratic is constant, and the 7-point stencil is
    exact on quadratics. Any error in the kernel weights or the 1/h^2 shows here.
    """
    n = 9
    h = 1.0 / (n - 1)
    t = torch.linspace(0.0, 1.0, n)
    X, Y, Z = torch.meshgrid(t, t, t, indexing="ij")
    phi = X**2 + Y**2 + Z**2
    out = laplacian_3D(phi, h, 0.09, 0.40)
    interior = out[1:-1, 1:-1, 1:-1]
    assert torch.allclose(
        interior, torch.full_like(interior, 2 * (2 * 0.09 + 0.40)), atol=1e-10
    )


# ---- properties of the face coefficients ------------------------------------


def test_face_coefficient_is_the_harmonic_mean_of_the_product(configured):
    """a_face = harm(sigma*m), one average over the whole product. On xi and eta the
    metric varies, so this is strictly below the old harm(sigma)*arith(m); on zeta
    both factors are z-invariant and harm(a, a) = a, which is why the fibre axis
    needs no special case."""
    n = 9
    metric = grid_at(n).metric
    sT = torch.full((n, n, n), 0.09)
    sL = torch.full((n, n, n), 0.40)
    faces = face_coefficients(metric, sT, sL)

    mxx, _, myy, mL = metric
    for k, (m, s, axis) in enumerate(
        ((mxx, 0.09, 0), (myy, 0.09, 1), (mL, 0.40, 2))
    ):
        lo, hi = m.narrow(axis, 0, n - 1), m.narrow(axis, 1, n - 1)
        harm = 2.0 * lo * hi / (lo + hi)
        plus = faces[2 * k].narrow(axis, 0, n - 1)
        assert torch.allclose(plus, s * harm, rtol=1e-12, atol=0.0)

    # zeta: both factors are z-invariant, so the face carries the nodal value itself
    plus_z = faces[4].narrow(2, 0, n - 1)
    assert torch.allclose(
        plus_z, 0.40 * mL.narrow(2, 0, n - 1), rtol=1e-12, atol=0.0
    )


def test_end_face_coefficients_are_mirrored(configured):
    """An axis with n nodes has n-1 internal faces; the outermost entry describes
    no face and is filled by mirroring. Together with the reflect padding this is
    what imposes the insulating condition, so the mirroring is load-bearing."""
    n = 9
    _, _, _, _, faces = problem(n)
    for axis in (0, 1, 2):
        plus, minus = faces[2 * axis], faces[2 * axis + 1]
        assert torch.equal(
            plus.narrow(axis, n - 1, 1), plus.narrow(axis, n - 2, 1)
        )
        assert torch.equal(minus.narrow(axis, 0, 1), minus.narrow(axis, 1, 1))


# ---- properties of the operator ---------------------------------------------


@pytest.mark.parametrize("lam", [2.0])
def test_operator_is_degree_one_homogeneous_in_sigma(configured, lam):
    """A(lambda sigma) = lambda A(sigma). Harmonic means are not additive, so the
    split A = sigma_T A_T + sigma_L A_L is unavailable; degree-one homogeneity
    under a common scaling still holds."""
    n = 9
    h, metric, sT, sL, faces = problem(n)
    phi = torch.as_tensor(np.random.default_rng(3).standard_normal((n, n, n)))

    base = laplacian_metric_3D(phi, h, faces, metric, sT)
    scaled = laplacian_metric_3D(
        phi, h, face_coefficients(metric, lam * sT, lam * sL), metric, lam * sT
    )
    assert torch.allclose(scaled, lam * base, rtol=1e-12, atol=0.0)


# ---- padding and kernels ----------------------------------------------------


def test_pad_mirrors_about_the_boundary_node(configured):
    """phi[-1] = phi[1] on every axis. Together with mirrored end-face
    coefficients, this produces the half-control-volume boundary row."""
    n = 5
    f = torch.as_tensor(np.random.default_rng(6).standard_normal((n, n, n)))
    p = _pad(f)
    assert p.shape == (n + 2, n + 2, n + 2)
    assert torch.equal(p[1:-1, 1:-1, 1:-1], f)
    assert torch.equal(p[0, 1:-1, 1:-1], f[1])
    assert torch.equal(p[-1, 1:-1, 1:-1], f[-2])
    assert torch.equal(p[1:-1, 0, 1:-1], f[:, 1])
    assert torch.equal(p[1:-1, 1:-1, 0], f[:, :, 1])


def test_neighbour_kernels_are_plus_minus_one_differences(configured):
    """Six kernels, each -1 at the centre and +1 at one face neighbour, in the
    order (+xi, -xi, +eta, -eta, +zeta, -zeta)."""
    K = NEIGHBOUR_DIFF_KERNELS
    assert K.shape == (6, 1, 3, 3, 3)
    offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    for f, (a, b, c) in enumerate(offsets):
        assert K[f, 0, 1, 1, 1].item() == -1.0
        assert K[f, 0, 1 + a, 1 + b, 1 + c].item() == 1.0
        assert K[f].abs().sum().item() == 2.0
