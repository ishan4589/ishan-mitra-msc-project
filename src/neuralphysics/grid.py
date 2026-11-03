# I used Claude Code to help implement and package the cylindrical grid.

r"""The square-to-disc map, the cylinder grid it generates, and the normalised
radius field the tissue shells are assigned on.

The initial max-norm projection has a closed form, but the subsequent discrete
Laplacian smoothing does not. The smoothed node positions are therefore the
representation of the final geometry, and everything downstream -- the Jacobian,
the metric and the shell labelling -- is measured from them.

The projection-and-smoothing approach is adapted from Sun (2025), the previous
project on this topic.

All spacing is passed explicitly rather than read from module scope, so a grid at
one resolution cannot silently inherit the spacing of another.
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch

from .metric import jacobian_num, metric_from_jacobian

__all__ = [
    "SmoothingPolicy",
    "cartesian_to_circular",
    "smooth_grid",
    "create_cylindrical_grid",
    "CylinderGrid",
    "grid_at",
    "levels",
]


@dataclass(frozen=True)
class SmoothingPolicy:
    """Parameters of the Laplacian mesh smoothing applied after the disc projection.

    Frozen so that a grid's geometry is fully determined by its constructor
    arguments, and hashable so those arguments can key a cache.
    """

    iterations: int = 100
    del_t: float = 0.2


# ---- the map ----------------------------------------------------------------


# Adapted and refactored from Sun (2025).
def cartesian_to_circular(nx, ny, x, y, dx, dy):
    """Project the reference square onto a disc by rescaling each node's offset
    from the centre by the ratio of its L-inf radius to its L2 radius.

    A node at max-norm radius s lands at Euclidean radius s, so the image of each
    L-inf level set (a square ring) is an exact circle. Returns transposed arrays:
    the caller's [row, col] becomes [col, row].
    """
    # Work on copies so the caller's coordinate arrays are unchanged.
    x = x.copy()
    y = y.copy()

    # Centre of the haloed reference lattice.
    o_x = dx * (nx + 1) / 2
    o_y = dy * (ny + 1) / 2

    # Map each square level set to a concentric circle of the same radius.
    for i in range(nx + 2):
        for j in range(ny + 2):
            rx = np.abs(x[i][j] - o_x)
            ry = np.abs(y[i][j] - o_y)
            r = np.sqrt(rx**2 + ry**2)
            r_inner = max(rx, ry)

            # Leave the centre fixed without dividing by zero.
            if r == 0:
                r = 1.0

            x[i][j] = (x[i][j] - o_x) * r_inner / r + o_x
            y[i][j] = (y[i][j] - o_y) * r_inner / r + o_y

    return x.T, y.T


# Adapted and refactored from Sun (2025).
def smooth_grid(x, y, iterations=100, del_t=0.2):
    """Laplacian mesh smoothing of the interior nodes with the outer ring fixed.

    Each iteration applies an explicit five-point Laplacian update with step
    `del_t`. At the default `del_t=0.2`, this is the average of the centre and its
    four axial neighbours.

    This is a Jacobi sweep: it writes into fresh arrays while reading the old
    ones, so the update at a node never sees the current sweep's
    changes at its neighbours.

    The raw radial map changes formula across the four diagonals, where the
    coordinate attaining the max norm switches. Smoothing reduces the resulting
    directional kinks but does not turn the numerically defined map into a smooth
    analytic one.
    """
    nx, ny = x.shape

    for _ in range(iterations):

        x_new = x.copy()
        y_new = y.copy()

        for i in range(1, nx - 1):  # exclude the fixed boundary
            for j in range(1, ny - 1):

                # Explicit five-point Laplacian update.
                x_new[i, j] = x[i, j] + del_t * (
                    x[i + 1, j]
                    + x[i - 1, j]
                    + x[i, j + 1]
                    + x[i, j - 1]
                    - 4 * x[i, j]
                )
                y_new[i, j] = y[i, j] + del_t * (
                    y[i + 1, j]
                    + y[i - 1, j]
                    + y[i, j + 1]
                    + y[i, j - 1]
                    - 4 * y[i, j]
                )

        x, y = x_new, y_new

    return x, y


# Adapted and refactored from Sun (2025).
def create_cylindrical_grid(nx, ny, nz, x, y, dx, dy, dz, smoothing=None):
    """Full (nz+2, ny+2, nx+2) node coordinates of the cylinder: the smoothed disc
    replicated on every z-level, and z linearly spaced.

    The extrusion is straight, so every z-level carries an identical disc.
    """
    smoothing = SmoothingPolicy() if smoothing is None else smoothing

    x_2d, y_2d = cartesian_to_circular(nx, ny, x, y, dx, dy)

    x_2d_sm, y_2d_sm = smooth_grid(
        x_2d, y_2d, iterations=smoothing.iterations, del_t=smoothing.del_t
    )  # smooth the projected interior while retaining the circular boundary

    z_coords = np.linspace(
        0, (nz + 1) * dz, nz + 2
    )  # equally spaced axial levels

    # Stack identical copies of the disc along a new leading z axis.
    x = np.tile(x_2d_sm, (nz + 2, 1, 1))
    y = np.tile(y_2d_sm, (nz + 2, 1, 1))
    z = np.tile(z_coords[:, np.newaxis, np.newaxis], (1, ny + 2, nx + 2))

    return x, y, z


# ---- the grid ---------------------------------------------------------------


class CylinderGrid:
    """The cylinder cross-section at one resolution, and the fields measured on it.

    Interior node coordinates are laid out [xi = axis 0, eta = axis 1], matching
    the operator's convention. The disc centre is not (0.5, 0.5): the nodes
    run from h to n*h, so the centre sits at h*(n+1)/2, and that value drifts with
    n. It is computed here and never hard-coded.

    Parameters
    ----------
    n         : nodes per axis.
    L         : side length represented by the n solved nodes, with spacing
                h = L/(n-1). The transverse coordinate interval is shifted by h
                because those nodes are the interior of a one-node halo. The map is
                positively homogeneous in the coordinates, so scaling L scales
                every physical length -- node positions, the centre, the rim
                radius -- while leaving every dimensionless quantity alone: the
                Jacobian, the four metric fields, det J, and the normalised
                radius field the shell labelling reads. The operator carries
                1/h^2, so A scales as 1/L^2.
    smoothing : SmoothingPolicy for the mesh smoothing after projection.
    z_zeta    : dz/dzeta, the axial stretch of the extrusion. Does not affect the
                cross-section; carried here because the metric is built from it.
    mapped    : False bypasses the square-to-disc map entirely, giving the
                undistorted reference lattice. This is the J = I limit against
                which the mapped machinery is checked.
    """

    def __init__(self, n, L=1.0, smoothing=None, z_zeta=1.0, mapped=True):
        if n < 3:
            raise ValueError(f"n must be at least 3, got {n}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        self.n = int(n)
        self.L = float(L)
        self.smoothing = SmoothingPolicy() if smoothing is None else smoothing
        self.z_zeta = float(z_zeta)
        self.mapped = bool(mapped)

        self.h = self.L / (self.n - 1)
        self.centre = self.h * (self.n + 1) / 2.0

        self._radius_cache = {}

    def __repr__(self):
        return (
            f"CylinderGrid(n={self.n}, L={self.L}, smoothing={self.smoothing}, "
            f"z_zeta={self.z_zeta}, mapped={self.mapped})"
        )

    # -- coordinates --

    @property
    def reference_nodes(self):
        """The n interior reference coordinates along one axis: h, 2h, ..., n*h.

        Node i sits at (i+1)*h, not i*h. The grid carries a one-node halo used by
        the central differences that measure the Jacobian, and the interior of that
        halo is what the operator solves on.
        """
        return np.linspace(0.0, (self.n + 1) * self.h, self.n + 2)[1:-1]

    @functools.cached_property
    def coords_haloed(self):
        """(X, Y), each (n+2, n+2), in the [eta, xi] convention, INCLUDING the
        one-node halo.

        This is the raw output of the map and the layout the central differences
        that measure the Jacobian require: they read a neighbour on both sides of
        every interior node, so the ring must be present.
        """
        if not self.mapped:
            full = np.linspace(0.0, (self.n + 1) * self.h, self.n + 2)
            XI, ETA = np.meshgrid(full, full, indexing="ij")
            return XI.T, ETA.T

        ix = np.linspace(0, (self.n + 1) * self.h, self.n + 2)
        iy = np.linspace(0, (self.n + 1) * self.h, self.n + 2)
        Xc, Yc, _ = create_cylindrical_grid(
            self.n,
            self.n,
            self.n,
            *np.meshgrid(ix, iy, indexing="ij"),
            dx=self.h,
            dy=self.h,
            dz=self.h,
            smoothing=self.smoothing,
        )
        return Xc[0], Yc[0]

    @functools.cached_property
    def coords(self):
        """(X, Y), each (n, n), the physical positions of the interior reference
        nodes on the cross-section, in the [xi, eta] convention.

        One disc serves every z-level.
        """
        X, Y = self.coords_haloed
        return X[1:-1, 1:-1].T, Y[1:-1, 1:-1].T

    # -- geometry --

    @functools.cached_property
    def jacobian(self):
        """(2, 2, n+2, n+2) field of d(x,y)/d(xi,eta), measured by central
        differences against the reference spacing h.

        The spacing is passed explicitly rather than left to default. Both the
        coordinates and h scale with L, so the quotient is independent of it --
        but only if the denominator is this grid's h and not an assumed 1/(n-1).
        """
        X, Y = self.coords_haloed
        return jacobian_num(
            torch.as_tensor(X),
            torch.as_tensor(Y),
            self.n,
            self.n,
            h_xi=self.h,
            h_eta=self.h,
        )

    @functools.cached_property
    def metric(self):
        """(mxx, mxy, myy, mL), each (n, n, n): the coefficient fields the operator
        consumes.

        Raises if the sampled Jacobian determinant is non-positive at any solved
        node. Such a value indicates a locally inverted or degenerate numerical map,
        for which these metric coefficients are invalid.
        """
        m = metric_from_jacobian(self.jacobian, self.n, self.z_zeta)
        det_J = m[3] * (self.z_zeta**2)
        if not bool((det_J > 0).all()):
            raise ValueError(
                f"mesh invalid at n = {self.n}: det(J) <= 0 somewhere (folded cells)"
            )
        return m

    @property
    def det_J(self):
        """Sampled local volume-scale factor det(J2) * z_zeta."""
        return self.metric[3] * (self.z_zeta**2)

    # -- radius --

    def _radius(self, norm):
        """(normalised radius field, raw rim radii), cached per norm."""
        if norm not in ("physical", "reference"):
            raise ValueError(
                f"norm must be 'physical' or 'reference', got {norm!r}"
            )

        if norm not in self._radius_cache:
            o = self.centre

            if norm == "physical":
                Xd, Yd = self.coords
                r = np.hypot(Xd - o, Yd - o)
            else:
                ref = self.reference_nodes
                XI, ETA = np.meshgrid(ref, ref, indexing="ij")
                r = np.maximum(np.abs(XI - o), np.abs(ETA - o))  # max-norm radius

            rim = np.concatenate(
                [r[0, :], r[-1, :], r[:, 0], r[:, -1]]
            )  # four edge traces; corners occur once in each incident edge
            r_norm = 0.5 * r / rim.mean()

            rim_n = np.concatenate(
                [r_norm[0, :], r_norm[-1, :], r_norm[:, 0], r_norm[:, -1]]
            )
            assert (
                abs(rim_n.mean() - 0.5) < 1e-12
            ), "normalisation must place the mean rim at 0.5"
            assert r_norm.min() >= 0.0 and np.isfinite(r_norm).all()
            self._radius_cache[norm] = (r_norm, rim)

        return self._radius_cache[norm]

    def radius_field(self, norm):
        """(n, n) normalised transverse radius, with the edge-trace mean at 0.5.

        norm = 'physical'  : radius of the MAPPED node position. Geometrically
                             faithful; interfaces staircase in index space.
        norm = 'reference' : L-inf radius of the node's REFERENCE position.
                             Face-aligned in the reference cube, and exactly
                             circular under the raw map -- but the mesh smoothing
                             displaces those faces from the intended circle.

        The normalisation is what makes shell radii mean the same fraction of the
        limb at every resolution: the disc is a different physical size at each,
        so absolute radii would make each level a slightly different limb.
        """
        return self._radius(norm)[0]

    def rim_mean(self, norm):
        """Mean raw radius over the four edge traces used for normalisation.

        Concatenating the traces counts each corner twice, once for each incident
        edge; this is the same convention used when constructing the radius field.
        """
        return float(self._radius(norm)[1].mean())

    def rim_stats(self, norm):
        """Raw edge-trace geometry as a dict.

        The normalisation fixes the sampled mean, not the spread, so residual
        non-circularity remains visible. Corners follow the same twice-counted
        convention as `rim_mean`.
        """
        rim = self._radius(norm)[1]
        return {
            "n": self.n,
            "h": self.h,
            "mean": float(rim.mean()),
            "min": float(rim.min()),
            "max": float(rim.max()),
            "spread": float(rim.max() - rim.min()),
            "spread_in_h": float((rim.max() - rim.min()) / self.h),
        }

    def physical_shell_radius(self, R_norm, norm):
        """Normalised shell radius -> physical length on this grid's disc.

        Inverts r_norm = 0.5 * r / rim_mean.
        """
        return 2.0 * R_norm * self.rim_mean(norm)


@functools.lru_cache(maxsize=None)
def grid_at(n, L=1.0, smoothing=None, z_zeta=1.0, mapped=True):
    """Cached CylinderGrid at resolution n.

    Generating a grid runs the smoothing loop and is requested repeatedly at the
    same resolutions. The cache therefore reuses one instance for each complete
    constructor signature. Callers must treat the returned grid as read-only;
    `CylinderGrid` itself does not prevent attribute mutation.
    """
    return CylinderGrid(
        n, L=L, smoothing=smoothing, z_zeta=z_zeta, mapped=mapped
    )


# ---- the level hierarchy ----------------------------------------------------


def _is_power_of_two_plus_one(n):
    """Whether n = 2^k + 1, which is what vertex-centred coarsening requires."""
    return n >= 3 and (n - 1) & (n - 2) == 0


def levels(n, n_base=9):
    """Grid sizes from `n` down to `n_base` by vertex-centred coarsening, fine first.

    [n, (n+1)//2, ...], stopping at the last size still at or above n_base. This is
    the single statement of the coarsening floor: the coefficient hierarchy, the
    solver's base level and any level-consistency check read it from here, so moving
    the floor moves all three together instead of one of them.

    n_base = 9 rather than 3 because muscle is empty at n = 3 and fat has four nodes
    per slice at n = 5, so coarsening further would solve a different medium. The
    coarse operator is a rediscretisation rather than a Galerkin product, so every
    level has to resolve the geometry it claims to.

    Both sizes must be 2^k + 1. Full-weighting restriction reads fine indices
    2J-1, 2J, 2J+1 and trilinear prolongation sends coarse node J to fine node 2J, so
    a size that is not one more than a power of two loses the endpoint alignment both
    transfers depend on. This is a property of the transfers rather than of a grid, so
    it is checked here and not in CylinderGrid, which will build a mesh at any n >= 3.
    """
    if not _is_power_of_two_plus_one(n):
        raise ValueError(f"n must be 2^k + 1 for the V-cycle, got {n}")
    if not _is_power_of_two_plus_one(n_base):
        raise ValueError(f"n_base must be 2^k + 1, got {n_base}")
    if n_base > n:
        raise ValueError(f"n_base = {n_base} exceeds n = {n}")

    sizes = []
    while n >= n_base:
        sizes.append(n)
        n = (n + 1) // 2
    return sizes
