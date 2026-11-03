# I used Claude Code to help implement and package the structural-verification tests.

"""Tests for neuralphysics.verification.structural.

The six structural properties hold for any positive coefficient field, so most of
what is below is property-based rather than pinned. The pinned comparisons exist
to catch a change that preserves the properties while moving the operator.
"""

import numpy as np
import pytest
import torch

from neuralphysics.grid import grid_at
from neuralphysics.metric import identity_metric
from neuralphysics.operators import face_coefficients, laplacian_metric_3D
from neuralphysics.projections import trapezoid_weight_field
from neuralphysics.tissue import LIMB
from neuralphysics.verification import (
    assemble_dense,
    matrix_free_null_checks,
    structural_metrics,
)

# The homogeneous case is a control: sigma constant, isotropy ratio 5, so any
# structural defect it shows belongs to the operator or the mesh and not to the
# layering. Matches the manufactured-solution constants in test_mms.
SIGMA_T_TRUE, SIGMA_L_TRUE = 1.0, 5.0
METRICS = ["identity", "cylinder"]
SMALL = [5, 9]


def metric_by_name(name, n):
    return identity_metric(n) if name == "identity" else grid_at(n).metric


def homogeneous_sigma(n):
    return torch.full((n, n, n), SIGMA_T_TRUE), torch.full(
        (n, n, n), SIGMA_L_TRUE
    )


def random_positive_sigma(n, seed):
    """A coefficient field with no structure at all: anisotropic, discontinuous and
    varying along z, which the physical model never is. The properties are proved
    by counting faces and tracking signs, so they must survive this."""
    g = torch.Generator().manual_seed(seed)
    return (
        0.01 + torch.rand(n, n, n, generator=g),
        0.01 + torch.rand(n, n, n, generator=g),
    )


def assert_six_properties(m, tag):
    sc = m["scale"]
    assert m["r_null"] < 1e-9 * sc, f"{tag}: A 1 != 0"
    assert m["l_null"] < 1e-9 * sc, (
        f"{tag}: A^T W != 0 - suspect the end-face mirroring in "
        f"face_coefficients, or the cross-term 1/(v_i v_j)"
    )
    assert (
        m["l_ones"] > 1e-6 * sc
    ), f"{tag}: A^T 1 == 0, so the two null spaces are not distinct"
    assert (
        m["rank_deficiency"] == 1
    ), f"{tag}: rank deficiency is {m['rank_deficiency']}"
    assert (
        m["symmetry"] < 1e-9 * sc
    ), f"{tag}: diag(W) A is not symmetric - suspect the cross term"
    assert (
        m["cosine"] > 1 - 1e-8
    ), f"{tag}: smallest left-singular direction is not W"


# ---- the six properties, under coefficient fields the model never produces ----


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("n", [5, 9])
def test_properties_hold_for_a_random_positive_coefficient_field(
    configured, n, metric_name
):
    """The strongest form of the claim in the module docstring. If this passes and
    a pinned case fails, the fault is in the coefficient construction, not the
    operator."""
    sT, sL = random_positive_sigma(n, seed=7 * n + len(metric_name))
    assert_six_properties(
        structural_metrics(n, metric_by_name(metric_name, n), sT, sL),
        f"n={n} {metric_name} random",
    )


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("n", [5, 9])
def test_properties_hold_for_the_homogeneous_medium(
    configured, n, metric_name
):
    assert_six_properties(
        structural_metrics(
            n, metric_by_name(metric_name, n), *homogeneous_sigma(n)
        ),
        f"n={n} {metric_name} homogeneous",
    )


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("n", SMALL)
def test_properties_hold_for_the_layered_medium(configured, n, metric_name):
    sT, sL = LIMB.sigma_fields(grid_at(n))
    assert_six_properties(
        structural_metrics(n, metric_by_name(metric_name, n), sT, sL),
        f"n={n} {metric_name} layered",
    )


def test_properties_survive_a_thousandfold_conductivity_contrast(configured):
    """Sigma-independence has to be measured rather than argued: the old argument
    for probing once was the linearity that harmonic means destroy.

    The contrast is imposed radially, so the field spans three decades between the
    axis and the rim -- far beyond anything the tissue table produces.
    """
    n, contrast = 9, 1000.0
    g = grid_at(n)
    r = torch.as_tensor(
        g.radius_field(LIMB.shell_norm)
    )  # 0 at the axis, 0.5 at the rim
    scale = (contrast ** (2.0 * r)).unsqueeze(-1).expand(n, n, n)
    sT, sL = LIMB.sigma_fields(g)
    assert_six_properties(
        structural_metrics(n, g.metric, sT * scale, sL * scale),
        "n=9 cylinder contrast=1000",
    )


def test_W_is_the_left_null_vector_of_a_small_assembled_A(configured):
    """Deferred from step 2, where projections.py landed without the operator
    needed to state it. impose_net_zero_CSD removes the W-weighted mean because W
    spans ker(A^T); this is the assembled evidence for that claim."""
    n = 5
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = homogeneous_sigma(n)
    M = assemble_dense(n, h, face_coefficients(metric, sT, sL), metric, sT)
    W = trapezoid_weight_field(n).reshape(-1)
    assert float((M.T @ W).abs().max()) < 1e-9 * float(M.abs().max())
    # and it is W specifically, not any weighting: the plain ones vector is not.
    assert float((M.T @ torch.ones(n**3)).abs().max()) > 1e-6 * float(
        M.abs().max()
    )


# ---- assemble_dense ---------------------------------------------------------


def test_assembled_matrix_reproduces_the_operator(configured):
    """The assembly is the operator applied to basis fields, so M phi must equal
    A phi for an arbitrary phi. Without this the structural numbers could be
    measuring a matrix that is not the operator."""
    n = 5
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = homogeneous_sigma(n)
    faces = face_coefficients(metric, sT, sL)
    M = assemble_dense(n, h, faces, metric, sT)
    phi = torch.as_tensor(np.random.default_rng(11).standard_normal((n, n, n)))
    assert torch.allclose(
        M @ phi.reshape(-1),
        laplacian_metric_3D(phi, h, faces, metric, sT).reshape(-1),
        atol=1e-10,
    )


def test_operator_is_not_symmetric_in_the_plain_inner_product(configured):
    """Negative control for property 5: diag(W) A is symmetric, A alone is not.
    Testing only the weighted form would pass on an operator that happened to be
    plainly symmetric, which would mean the boundary rows were wrong."""
    n = 5
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = homogeneous_sigma(n)
    M = assemble_dense(n, h, face_coefficients(metric, sT, sL), metric, sT)
    assert float((M - M.T).abs().max()) > 1e-6 * float(M.abs().max())


def test_a_perturbed_face_coefficient_breaks_the_left_null_space(configured):
    """The six checks are only evidence if they can fail. Nudging one face
    coefficient leaves A 1 = 0 intact -- the flux form still telescopes -- but
    destroys A^T W = 0 and the W-symmetry, which is exactly the failure mode the
    assertion messages name."""
    n = 5
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = homogeneous_sigma(n)
    faces = face_coefficients(metric, sT, sL).clone()
    faces[0, 2, 2, 2] *= 1.05  # +xi face at one interior node only
    M = assemble_dense(n, h, faces, metric, sT)
    W = trapezoid_weight_field(n).reshape(-1)
    sc = float(M.abs().max())
    assert float((M.T @ W).abs().max()) > 1e-6 * sc
    assert (
        float((torch.diag(W) @ M - (torch.diag(W) @ M).T).abs().max())
        > 1e-6 * sc
    )


# ---- matrix-free ------------------------------------------------------------


def test_matrix_free_agrees_with_the_dense_assembly(configured):
    """The autograd adjoint product and the assembled A^T W must give the same
    number; the matrix-free path is what certifies N = 65, so it needs checking
    where both are available."""
    n = 9
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = homogeneous_sigma(n)
    faces = face_coefficients(metric, sT, sL)
    free = matrix_free_null_checks(n, h, faces, metric, sT)
    dense = structural_metrics(n, metric, sT, sL)
    assert free["r_null"] == pytest.approx(dense["r_null"], abs=1e-10)
    assert free["l_null"] == pytest.approx(dense["l_null"], abs=1e-10)


@pytest.mark.parametrize("n", [17, 33])
def test_matrix_free_null_spaces_hold_at_larger_n(configured, n):
    """Larger production sizes are omitted here for runtime. Random positive
    coefficients exercise the identities without relying on shell structure."""
    h = 1.0 / (n - 1)
    metric = grid_at(n).metric
    sT, sL = random_positive_sigma(n, seed=n)
    out = matrix_free_null_checks(
        n, h, face_coefficients(metric, sT, sL), metric, sT
    )
    assert out["r_null"] < 1e-9 and out["l_null"] < 1e-9
