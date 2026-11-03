# I used Claude Code to help implement and package the manufactured-solution tests.

"""Tests for the manufactured-solution convergence study.

The finest study is marked slow because it dominates the suite. The fast tests
exercise the map, rate arithmetic and coordinate alignment at smaller N.
"""

import numpy as np
import pytest
import torch

from neuralphysics.verification import (
    SMOOTH_MAP,
    L_phys,
    SmoothAnalyticMap,
    mms_error,
    phi_mms,
    refinement_study,
)

SIGMA_T_TRUE, SIGMA_L_TRUE = 1.0, 5.0


# ---- the map ----------------------------------------------------------------

def test_map_is_the_identity_at_zero_amplitude(configured):
    flat = SmoothAnalyticMap(amplitude=0.0)
    xi, eta = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 7), indexing="ij")
    assert np.allclose(flat.x(xi, eta), xi)
    assert np.allclose(flat.y(xi, eta), eta)
    assert np.allclose(flat.det_J(xi, eta), 1.0)


def test_map_jacobian_stays_positive(configured):
    """det J2 = 1 + A pi sin(pi(xi + eta)) is positive exactly when A < 1/pi. A map
    that folded would make the study meaningless rather than merely inaccurate."""
    xi, eta = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 65), indexing="ij")
    assert SMOOTH_MAP.det_J(xi, eta).min() > 0.0


def test_amplitude_at_or_beyond_one_over_pi_is_rejected(configured):
    SmoothAnalyticMap(amplitude=0.3)                      # below 1/pi = 0.3183...
    with pytest.raises(ValueError):
        SmoothAnalyticMap(amplitude=1.0 / np.pi)
    with pytest.raises(ValueError):
        SmoothAnalyticMap(amplitude=-0.01)


def test_analytic_jacobian_matches_a_finite_difference(configured):
    """det_J is written out by hand rather than differentiated, so it needs
    checking against the map it claims to describe."""
    e = 1e-6
    rng = np.random.default_rng(41)
    xi, eta = rng.uniform(0.1, 0.9, 200), rng.uniform(0.1, 0.9, 200)
    dxdxi = (SMOOTH_MAP.x(xi + e, eta) - SMOOTH_MAP.x(xi - e, eta)) / (2 * e)
    dxdeta = (SMOOTH_MAP.x(xi, eta + e) - SMOOTH_MAP.x(xi, eta - e)) / (2 * e)
    dydxi = (SMOOTH_MAP.y(xi + e, eta) - SMOOTH_MAP.y(xi - e, eta)) / (2 * e)
    dydeta = (SMOOTH_MAP.y(xi, eta + e) - SMOOTH_MAP.y(xi, eta - e)) / (2 * e)
    assert np.allclose(dxdxi * dydeta - dxdeta * dydxi, SMOOTH_MAP.det_J(xi, eta), atol=1e-7)


def test_manufactured_pair_is_consistent(configured):
    """L_phys must actually be div(sigma grad phi_mms). Checked by a central
    difference on a fine physical grid, away from the boundary."""
    e = 1e-4
    rng = np.random.default_rng(42)
    x, y, z = (rng.uniform(0.2, 0.8, 300) for _ in range(3))
    lap = 0.0
    for axis, sig in ((0, SIGMA_T_TRUE), (1, SIGMA_T_TRUE), (2, SIGMA_L_TRUE)):
        p = [x, y, z]
        hi, lo = list(p), list(p)
        hi[axis] = p[axis] + e
        lo[axis] = p[axis] - e
        lap = lap + sig * (phi_mms(*hi) - 2 * phi_mms(x, y, z) + phi_mms(*lo)) / e ** 2
    assert np.allclose(lap, L_phys(x, y, z, SIGMA_T_TRUE, SIGMA_L_TRUE), atol=1e-4)


# ---- the rate arithmetic ----------------------------------------------------

def test_refinement_study_recovers_a_known_rate(configured):
    """A synthetic error_fn decaying exactly like h^2 must give rates of exactly 2,
    so a failure here is in the log2 arithmetic and not in the operator."""
    out = refinement_study(lambda N: (1.0 / (N - 1)) ** 2, [17, 33, 65])
    assert out["Ns"] == [17, 33, 65]
    assert out["rates"] == pytest.approx([2.0, 2.0], abs=1e-12)


def test_refinement_study_rejects_sizes_that_do_not_halve_h(configured):
    """log2 of the error ratio is only a convergence rate if h halves between
    consecutive entries; silently reporting a number for [17, 40] would be worse
    than refusing."""
    with pytest.raises(ValueError):
        refinement_study(lambda N: 1.0, [17, 40])
    with pytest.raises(ValueError):
        refinement_study(lambda N: 1.0, [33])


# ---- the study itself -------------------------------------------------------

def test_second_order_is_visible_at_small_n(configured):
    """The cheap end of the sweep, so the suite has an order-of-accuracy check
    that runs by default. The rate is not yet at 2 here -- the coarse grids are
    pre-asymptotic -- so this asserts only that the error is falling faster than
    first order."""
    out = refinement_study(lambda N: mms_error(N, SIGMA_T_TRUE, SIGMA_L_TRUE), [9, 17, 33])
    assert out["rates"][-1] > 1.5


@pytest.mark.slow
def test_boundary_strip_barely_matters_on_a_fine_grid(configured):
    """By N = 65 the max error has moved off the boundary region: widening the
    excluded strip from three layers to five changes it by under a percent. That
    is what licenses reading the fine-grid errors as the operator's own."""
    a = mms_error(65, SIGMA_T_TRUE, SIGMA_L_TRUE, boundary_strip=3)
    b = mms_error(65, SIGMA_T_TRUE, SIGMA_L_TRUE, boundary_strip=5)
    assert b == pytest.approx(a, rel=0.02)


def test_boundary_strip_dominates_on_a_coarse_grid(configured):
    """At N = 17 it does not: the max error still sits within a few cells of the
    face, and widening the strip halves it. The coarse entries of the sweep are
    therefore boundary-contaminated, which is why the first rate comes out near
    1.6 rather than 2 -- pre-asymptotic, not a defect. Recorded as a test so the
    coarse numbers are not quoted as if they were clean."""
    a = mms_error(17, SIGMA_T_TRUE, SIGMA_L_TRUE, boundary_strip=3)
    b = mms_error(17, SIGMA_T_TRUE, SIGMA_L_TRUE, boundary_strip=6)
    assert b < 0.7 * a


def test_misaligned_sampling_destroys_second_order(configured):
    """The metric at node k is built from the extended reference node (k+1)h, so
    phi is sampled there too. Sampling on the plain grid i*h instead pairs each
    metric value with a neighbouring potential value; the error is then dominated
    by an O(h) offset and the observed rate collapses to about one. This is the
    negative control for the alignment comment in the module docstring."""
    def misaligned(N):
        from neuralphysics.metric import jacobian_num, metric_from_jacobian
        from neuralphysics.operators import face_coefficients, laplacian_metric_3D
        hN = 1.0 / (N - 1)
        r = np.arange(N + 2) * hN
        ETA_g, XI_g = np.meshgrid(r, r, indexing="ij")
        J2 = jacobian_num(torch.as_tensor(SMOOTH_MAP.x(XI_g, ETA_g)),
                          torch.as_tensor(SMOOTH_MAP.y(XI_g, ETA_g)), N, N)
        metric = metric_from_jacobian(J2, N, z_zeta=1.0)
        ax = np.arange(N) * hN                            # <- the mistake: no +1 shift
        Xi, Eta, Z = np.meshgrid(ax, ax, np.arange(N) * hN, indexing="ij")
        X, Y = SMOOTH_MAP.x(Xi, Eta), SMOOTH_MAP.y(Xi, Eta)
        sTf = torch.full((N, N, N), SIGMA_T_TRUE)
        sLf = torch.full((N, N, N), SIGMA_L_TRUE)
        faces = face_coefficients(metric, sTf, sLf)
        num = laplacian_metric_3D(torch.as_tensor(phi_mms(X, Y, Z)), hN, faces, metric, sTf)
        target = torch.as_tensor(SMOOTH_MAP.det_J(Xi, Eta)
                                 * L_phys(X, Y, Z, SIGMA_T_TRUE, SIGMA_L_TRUE))
        return (num[3:-3, 3:-3, 3:-3] - target[3:-3, 3:-3, 3:-3]).abs().max().item()

    out = refinement_study(misaligned, [17, 33, 65])
    assert out["rates"][-1] < 1.4


@pytest.mark.slow
def test_finest_pair_is_second_order(configured):
    """By the finest tested pair, the observed interior rate should be close to two
    for the smooth analytic map and homogeneous coefficients."""
    out = refinement_study(lambda N: mms_error(N, SIGMA_T_TRUE, SIGMA_L_TRUE), [17, 33, 65, 129])
    assert out["rates"][-1] > 1.8
