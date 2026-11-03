# I used Claude Code to help implement and package the manufactured solutions.

r"""The manufactured solution and the smooth analytic map it is posed on.

The map is smooth and analytic, deliberately unlike the one the experiment uses.
The production map is built by a max-norm radial rescaling followed by discrete
Jacobi smoothing and is represented only by its nodes. Measuring order on it would
mix stencil error with the regularity and numerical construction of that map. The
analytic map below instead supplies a smooth target and an exact determinant.

The coefficient is likewise homogeneous, so this study certifies the stencil and
says nothing about the interface treatment -- a smooth sigma has no transmission
condition to encode. The planar-interface residual is what covers that.
"""

import numpy as np

__all__ = ["SmoothAnalyticMap", "SMOOTH_MAP", "phi_mms", "L_phys"]


class SmoothAnalyticMap:
    r"""A sinusoidal perturbation of the identity on the reference square.

        x = xi  + A sin(pi xi) sin(pi eta)
        y = eta + A sin(pi xi) sin(pi eta)

    with det J2 = 1 + A pi sin(pi(xi + eta)), which is positive -- the map is a
    diffeomorphism -- precisely when A < 1/pi. The default A = 0.06 leaves a
    comfortable margin while still distorting the mesh enough that a metric bug
    would show.
    """

    def __init__(self, amplitude=0.06):
        if not 0.0 <= amplitude < 1.0 / np.pi:
            raise ValueError(f"amplitude must lie in [0, 1/pi); got {amplitude}")
        self.amplitude = amplitude

    def _bump(self, xi, eta):
        return self.amplitude * np.sin(np.pi * xi) * np.sin(np.pi * eta)

    def x(self, xi, eta):
        return xi + self._bump(xi, eta)

    def y(self, xi, eta):
        return eta + self._bump(xi, eta)

    def det_J(self, xi, eta):
        """Analytic det(J2), used directly in the manufactured target."""
        return 1.0 + self.amplitude * np.pi * np.sin(np.pi * (xi + eta))


SMOOTH_MAP = SmoothAnalyticMap()


def phi_mms(x, y, z):
    """The manufactured potential, in PHYSICAL coordinates."""
    return np.sin(np.pi * x) * np.cos(np.pi * y) + z ** 2


def L_phys(x, y, z, sigma_T, sigma_L):
    """div(sigma grad phi_mms) in physical coordinates, for homogeneous sigma.

    The transverse part contributes -2 pi^2 sin(pi x) cos(pi y) and the
    longitudinal part contributes 2 from the z^2 term. The operator computes the
    reference-space quantity det(J) * L_phys, so this is multiplied by the map's
    Jacobian determinant before comparison.
    """
    return sigma_T * (-2 * np.pi ** 2 * np.sin(np.pi * x) * np.cos(np.pi * y)) + sigma_L * 2.0
