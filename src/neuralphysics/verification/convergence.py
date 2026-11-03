# I used Claude Code to help implement and package the convergence verification.

r"""Order-of-accuracy verification by the method of manufactured solutions.

The operator computes the reference-space quantity div_xi(sigma-tilde grad_xi phi),
which for a smooth map equals det(J) times the physical div(sigma grad phi). Both
sides of that identity are available in closed form here, so the difference is
pure discretisation error and its decay rate under refinement is the operator's
order of accuracy.

Two implementation details are important when interpreting the measured rate:

  * `jacobian_num` builds the metric at node k from the extended reference node
    (k+1)h, so phi must be sampled at (i+1)h too. Sampling phi on the plain grid
    i*h instead pairs each metric value with a neighbouring potential value and
    introduces an O(h) coordinate offset.
  * A strip of width `m` is dropped at every face. The reflect padding makes the
    boundary rows impose the insulating condition rather than approximate the PDE,
    and the cross-term treatment also has boundary-specific rows. The study drops
    a configurable strip so its reported maximum measures the interior stencil.
"""

import math

import numpy as np
import torch

from ..metric import jacobian_num, metric_from_jacobian
from ..operators import face_coefficients, laplacian_metric_3D
from .manufactured import SMOOTH_MAP, L_phys, phi_mms

__all__ = ["mms_error", "refinement_study"]


def mms_error(N, sigma_T, sigma_L, mapping=SMOOTH_MAP, boundary_strip=3):
    """Max interior error of the mapped operator against the manufactured solution at grid size N.

    `boundary_strip` is the number of layers dropped at each face. The default of
    three excludes the rows most directly affected by the boundary treatment; it
    is exposed so sensitivity to that choice can be checked.
    """
    hN = 1.0 / (N - 1)

    r = np.arange(N + 2) * hN                        # extended reference nodes for jacobian_num
    ETA_g, XI_g = np.meshgrid(r, r, indexing="ij")   # [row = eta, col = xi]
    J2 = jacobian_num(torch.as_tensor(mapping.x(XI_g, ETA_g)),
                      torch.as_tensor(mapping.y(XI_g, ETA_g)), N, N)
    metric = metric_from_jacobian(J2, N, z_zeta=1.0)

    ax = (np.arange(N) + 1) * hN                     # operator node i -> reference (i+1)h  (aligned)
    Xi, Eta, Z = np.meshgrid(ax, ax, np.arange(N) * hN, indexing="ij")
    X, Y = mapping.x(Xi, Eta), mapping.y(Xi, Eta)

    sTf = torch.full((N, N, N), float(sigma_T))      # homogeneous coefficients isolate the smooth
    sLf = torch.full((N, N, N), float(sigma_L))      # mapped stencil from material-interface effects
    faces = face_coefficients(metric, sTf, sLf)

    num = laplacian_metric_3D(torch.as_tensor(phi_mms(X, Y, Z)), hN, faces, metric, sTf)
    target = torch.as_tensor(mapping.det_J(Xi, Eta) * L_phys(X, Y, Z, sigma_T, sigma_L))

    m = boundary_strip
    return (num[m:-m, m:-m, m:-m] - target[m:-m, m:-m, m:-m]).abs().max().item()


def refinement_study(error_fn, Ns):
    """Errors at each N and the log2 rate estimates between consecutive pairs.

    Returns a dict with `Ns`, `errors` and `rates`, where
    rates[k] = log2(errors[k] / errors[k+1]). Successive Ns are assumed to halve
    h, which for the 2^k + 1 sizes used here they do; the rate is meaningless
    otherwise, so the assumption is checked.
    """
    Ns = list(Ns)
    if len(Ns) < 2:
        raise ValueError("a refinement study needs at least two grid sizes")
    for a, b in zip(Ns, Ns[1:]):
        if b - 1 != 2 * (a - 1):
            raise ValueError(f"consecutive sizes must halve h: {a} -> {b} does not")

    errors = [error_fn(N) for N in Ns]
    rates = [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    return {"Ns": Ns, "errors": errors, "rates": rates}
