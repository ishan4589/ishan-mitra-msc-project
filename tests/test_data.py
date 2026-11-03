# I used Claude Code to help implement and package tests for data.py.

"""Tests for neuralphysics.data."""

import numpy as np
import pytest
import torch

from neuralphysics.data import (
    SOURCE_SPECS,
    ElectrodeArray,
    Observation,
    add_measurement_noise,
    dipole_centres,
    gaussian_dipole,
    lateral_surface_mask,
    layer_bounds,
    weighted_sum,
)
from neuralphysics.grid import grid_at
from neuralphysics.projections import trapezoid_weight_field
from neuralphysics.tissue import LIMB

SIZES = [17, 33]


# ---- the source: admissibility ----------------------------------------------


@pytest.mark.parametrize("n", [17])
def test_the_source_is_admissible(configured, n):
    """W . I = 0 to machine precision.

    This is the discrete compatibility condition, and without it A phi = I has no
    solution at all: a limb sealed by an insulating boundary cannot accumulate net
    charge.
    """
    I = gaussian_dipole(grid_at(n), LIMB)
    assert abs(weighted_sum(I)) < 1e-14 * float(I.abs().max())


@pytest.mark.parametrize("n", [17])
def test_the_raw_dipole_is_already_balanced(configured, n):
    """The z nodes are symmetric about the mid-plane, the two centres are placed
    symmetrically about it, and the
    transverse factors are identical, so source and sink are exact mirror images
    and the raw difference integrates to zero to round-off. The projection is
    therefore insurance and not correction -- for this placement. It stays because
    admissibility is a precondition of the solve, and a caller who moves one blob
    breaks the symmetry that makes it free.
    """
    g = grid_at(n)
    c = dipole_centres(g, LIMB)
    X, Y = g.coords
    Xp = torch.as_tensor(X)[:, :, None].expand(n, n, n)
    Yp = torch.as_tensor(Y)[:, :, None].expand(n, n, n)
    Zp = torch.linspace(0.0, g.L, n)[None, None, :].expand(n, n, n)

    def blob(cz):
        return torch.exp(
            -(
                ((Xp - c["c_x"]) ** 2 + (Yp - c["c_y"]) ** 2 + (Zp - cz) ** 2)
                / 0.05**2
            )
        )

    raw = blob(c["cz_source"]) - blob(c["cz_sink"])
    assert abs(weighted_sum(raw)) < 1e-13 * float(raw.abs().max())


@pytest.mark.parametrize("n", [17])
def test_the_source_is_odd_about_the_mid_plane(configured, n):
    """Source and sink are mirror images about z = L/2 and share (x, y), so the
    field is odd under z -> L - z. This is what makes it a dipole rather than two
    unrelated blobs, and it holds only because the separation is along zeta
    alone."""
    I = gaussian_dipole(grid_at(n), LIMB)
    assert float((I + I.flip(2)).abs().max()) < 1e-14 * float(I.abs().max())


# ---- the source: placement --------------------------------------------------


@pytest.mark.parametrize("n", [17, 65])
def test_both_blobs_sit_in_muscle(configured, n):
    """At every level, and not only the finest. The membership is a fraction of the
    normalised radius, which is level-independent, but the nodes are not: a peak that
    holds muscle at n = 65 could fall on a node the coarse grid resolves differently.

    Stated as "muscle holds most of the membership at the peak node", which is the
    weakest form of the claim that still means the source is in muscle."""
    g = grid_at(n)
    I = gaussian_dipole(g, LIMB)
    w = LIMB.layer_weights(g)
    muscle = LIMB.layer_names.index("muscle")

    for tag, flat in (("source", int(I.argmax())), ("sink", int(I.argmin()))):
        p = np.unravel_index(flat, I.shape)
        assert (
            float(w[(muscle, *p)]) > 0.5
        ), f"{tag} peak is not majority muscle"


@pytest.mark.parametrize("n", [17])
def test_the_extrema_straddle_the_mid_plane(configured, n):
    """The positive blob sits below the mid-plane and the negative one above,
    which fixes the sign convention: I > 0 is the source."""
    g = grid_at(n)
    I = gaussian_dipole(g, LIMB)
    k_src = np.unravel_index(int(I.argmax()), I.shape)[2]
    k_snk = np.unravel_index(int(I.argmin()), I.shape)[2]
    assert k_src < (n - 1) / 2 < k_snk


def test_layer_bounds_reach_the_axis_and_the_rim():
    """The innermost shell starts at the axis and the outermost reaches the rim, so
    those two bounds come from the geometry rather than from the interface list.
    """
    assert layer_bounds(LIMB, "bone")[0] == 0.0
    assert layer_bounds(LIMB, "skin")[1] == 0.5
    assert layer_bounds(LIMB, "muscle") == (
        LIMB.interfaces[0],
        LIMB.interfaces[1],
    )


def test_the_offset_is_normalised_not_a_fixed_length(configured):
    """The same normalised offset is a different physical length at every
    resolution, because the disc is a different physical size at each. A fixed
    physical offset would sit at a different fraction of the limb on each grid.
    """
    r = [dipole_centres(grid_at(n), LIMB)["r_blob"] for n in (9, 17, 33, 65)]
    assert len(set(r)) == len(r)
    assert max(r) - min(r) > 1e-3


# ---- the source: the clearance rule -----------------------------------------


def test_a_blob_that_does_not_clear_its_interfaces_is_rejected(configured):
    """A blob overlapping an interface would excite more than the intended tissue
    region. The rule is an experiment-design
    property and not a defensive check, which is why it raises rather than asserts:
    under python -O an assert would turn a design error into a silently wrong
    experiment.

    Increasing the width to 0.08 makes the default placement violate the required
    two-width clearance, exercising the guard rather than only its valid path.
    """
    with pytest.raises(ValueError, match="does not clear"):
        gaussian_dipole(grid_at(17), LIMB, width=0.08)


def test_the_configured_sources_clear_their_interfaces(configured):
    """The other half: the design the experiment actually runs must satisfy its own
    precondition. Nothing else in the suite places a blob at SOURCE_SPECS -- the
    tests above use the default mid-annulus offset -- so this directly covers the
    configured source table at n = 65, where the margin is tightest."""
    g = grid_at(65)
    for r_blob_norm, angle_deg in SOURCE_SPECS:
        gaussian_dipole(g, LIMB, r_blob_norm=r_blob_norm, angle_deg=angle_deg)


# ---- the noise --------------------------------------------------------------


def test_a_different_seed_gives_a_different_draw(configured):
    phi = torch.randn(9, 9, 9)
    a, _ = add_measurement_noise(phi, 5.0, 0)
    b, _ = add_measurement_noise(phi, 5.0, 1)
    assert not torch.equal(a, b)


def test_the_draw_is_independent_of_the_global_rng(configured):
    """The generator is created and seeded locally, so an unrelated draw between
    the two calls cannot move the noise. Without that, the noise realisation would
    depend on how much code ran before it."""
    phi = torch.randn(9, 9, 9)
    a, _ = add_measurement_noise(phi, 5.0, 0)
    torch.randn(1000)
    b, _ = add_measurement_noise(phi, 5.0, 0)
    assert torch.equal(a, b)


def test_zero_noise_returns_the_field_unchanged(configured):
    phi = torch.randn(9, 9, 9)
    obs, stats = add_measurement_noise(phi, 0.0, 0)
    assert torch.equal(obs, phi)
    assert stats["snr"] == float("inf")
    assert stats["noise_std"] == 0.0


def test_the_noise_level_is_relative_to_the_field(configured):
    """Scaling the field scales the noise by the same factor, so the realised SNR
    is invariant. An absolute noise level would make the SNR depend on the
    conductivities through the amplitude of phi."""
    phi = torch.randn(17, 17, 17)
    _, a = add_measurement_noise(phi, 5.0, 0)
    _, b = add_measurement_noise(1000.0 * phi, 5.0, 0)
    assert b["noise_std"] == pytest.approx(1000.0 * a["noise_std"], rel=1e-12)
    assert b["snr"] == pytest.approx(a["snr"], rel=1e-12)


def test_the_realised_snr_is_near_the_target(configured):
    """A specification of 5% implies an SNR near 20, but the realised value is a
    property of one draw and is not exactly 20. Both numbers are reported so the
    write-up can quote the realised one."""
    phi = torch.randn(33, 33, 33)
    _, stats = add_measurement_noise(phi, 5.0, 0)
    assert stats["snr"] == pytest.approx(20.0, rel=0.05)


# ---- the electrodes ---------------------------------------------------------


@pytest.mark.parametrize("n", [17])
def test_dense_coverage_selects_every_lateral_node(configured, n):
    array = ElectrodeArray(1.0)
    assert array.step == 1
    assert array.coverage(n) == 1.0
    assert torch.equal(array.mask(n), lateral_surface_mask(n))


def test_the_stride_uses_ceil_rather_than_round(configured):
    """round(1 / sqrt(0.5)) = round(1.414) = 1 selects every node, so a nominal
    half coverage would silently deliver full coverage. ceil gives 2. This is the
    one behavioural change in this module and it is inert at coverage 1, where
    both give a stride of 1."""
    assert ElectrodeArray(0.5).step == 2
    assert ElectrodeArray(1.0).step == 1
    assert ElectrodeArray(0.25).step == 2
    assert ElectrodeArray(0.2).step == 3


def test_realised_coverage_is_quantised_below_the_target(configured):
    """Realised coverage is 1/step^2 per face up to face-edge effects, so it is
    generally below the target rather than equal to it. Quote the realised
    figure."""
    array = ElectrodeArray(0.5)
    realised = array.coverage(33)
    assert realised < 0.5
    assert realised == pytest.approx(0.25, rel=0.15)


def test_the_full_field_control_reads_every_node(configured):
    array = ElectrodeArray(1.0, use_surface=False)
    assert bool(array.mask(17).all())
    assert array.coverage(17) == 1.0


# ---- the observation --------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_the_observation_grid_is_the_solve_grid(configured, n):
    """The observation stores the same grid that determines its field shape, so
    downstream code can explicitly choose that grid for the inverse solve.
    """
    g = grid_at(n)
    obs = Observation(torch.zeros(n, n, n), ElectrodeArray().mask(n), g)
    assert obs.n == g.n == n
    assert obs.phi.shape == (g.n,) * 3


def test_a_mask_from_another_grid_is_rejected(configured):
    """The three objects are bundled so that the grid a field lives on is answerable
    from the object rather than by tracing which variable came from where. A stale
    n from an earlier notebook cell is the natural way to break that."""
    with pytest.raises(ValueError, match="but the grid is"):
        Observation(
            torch.zeros(17, 17, 17), ElectrodeArray().mask(33), grid_at(17)
        )


def test_an_integer_mask_is_rejected(configured):
    """Stated on an integer mask, not a float one. A float mask is caught by torch
    itself -- phi[mask] raises IndexError -- so a test written that way passes
    without the guard. An integer mask is the case that needs it: phi[mask] is then
    advanced indexing rather than boolean selection, and returns a tensor of shape
    (n, n, n, n, n) with no error raised anywhere."""
    with pytest.raises(TypeError, match="boolean"):
        Observation(
            torch.zeros(17, 17, 17),
            torch.ones(17, 17, 17, dtype=torch.long),
            grid_at(17),
        )


def test_an_empty_mask_is_rejected(configured):
    """With an empty mask, both norms in the relative misfit are zero and the ratio
    is undefined, so the malformed observation is rejected at construction.
    """
    with pytest.raises(ValueError, match="nothing to fit"):
        Observation(
            torch.zeros(17, 17, 17),
            torch.zeros(17, 17, 17, dtype=torch.bool),
            grid_at(17),
        )


def test_the_observation_reads_only_the_masked_nodes(configured):
    n = 17
    phi = torch.randn(n, n, n)
    mask = ElectrodeArray().mask(n)
    obs = Observation(phi, mask, grid_at(n))
    assert obs.values.numel() == int(mask.sum())
    assert torch.equal(obs.values, phi[mask])


def test_the_misfit_ignores_nodes_off_the_electrodes(configured):
    """The functional the optimiser minimises lives on the lateral faces only, so a
    perturbation in the interior must not move it."""
    n = 17
    phi = torch.randn(n, n, n)
    obs = Observation(phi, ElectrodeArray().mask(n), grid_at(n))
    perturbed = phi.clone()
    perturbed[1:-1, 1:-1] += 1.0
    assert obs.misfit(perturbed) == 0.0


def test_the_misfit_refuses_a_prediction_of_the_wrong_shape(configured):
    """misfit carries its own shape check, separate from the constructor's: the
    observation is fixed once built, but a prediction arrives fresh on every
    objective evaluation, and is the thing an inversion on the wrong grid would
    deliver."""
    n = 17
    obs = Observation(
        torch.randn(n, n, n), ElectrodeArray().mask(n), grid_at(n)
    )
    with pytest.raises(ValueError, match="prediction is"):
        obs.misfit(torch.zeros(33, 33, 33))


def test_the_misfit_is_relative_to_the_observed_norm(configured):
    n = 17
    phi = torch.randn(n, n, n)
    obs = Observation(phi, ElectrodeArray().mask(n), grid_at(n))
    assert obs.misfit(2.0 * phi) == pytest.approx(1.0, rel=1e-12)


# ---- end to end -------------------------------------------------------------


@pytest.mark.slow
def test_the_forward_field_has_zero_weighted_mean_not_zero_sum(configured):
    """The gauge removes the volume-weighted mean, because a boundary node owns
    less volume than an interior one. The plain sum is a different functional, so
    this test asserts only the weighted gauge condition.
    """
    from neuralphysics.forward import forward
    from neuralphysics.hierarchy import CoefficientHierarchy

    n = 65
    g = grid_at(n)
    I = gaussian_dipole(g, LIMB)
    phi = forward(I, CoefficientHierarchy.from_vectors(LIMB, n)).detach()

    W = trapezoid_weight_field(n)
    weighted_mean = ((W * phi).sum() / W.sum()).abs().item()
    assert weighted_mean < 1e-16 * float(phi.abs().max())
