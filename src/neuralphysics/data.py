# I used Claude Code to help implement and package the observation model.

r"""The observation model: the current source, the measurement noise, and positioning of
the surface electrodes.

This module specifies the synthetic experiment rather than the tissue physics.
The dipole is a pair of smooth Gaussian blobs used as an idealised current-source
density. The noise scale is relative to the clean field's nodewise RMS.
The electrodes read the four lateral faces of the reference cube,
which on the mapped grid are exactly the skin surface, so the mask is index
arithmetic and not a geometric search.

One resolution throughout. The observation carries the grid it was produced on,
so "which grid is this on" is answerable from the object rather than by
inspection, and the intended inversion path can use that same grid. Sampling a fine field down
to a coarser inversion grid is not offered: the disc is a different physical size
at every resolution, so a coarser grid is a slightly different limb rather than
the same limb more coarsely resolved.
"""

import math
from dataclasses import dataclass

import torch

from .projections import impose_net_zero_CSD, trapezoid_weight_field

__all__ = [
    "SOURCE_SPECS",
    "layer_bounds",
    "dipole_centres",
    "gaussian_dipole",
    "add_measurement_noise",
    "lateral_surface_mask",
    "ElectrodeArray",
    "Observation",
    "weighted_sum",
]


# ---- the source -------------------------------------------------------------

# (r_blob_norm, angle_deg) per source, in normalised radius. Radius is the
# main design variable. The two values sample different depths within muscle;
# their 180-degree angular separation also keeps the source markers apart in
# figures. At width 0.05, both satisfy the clearance interval used at n = 65.
SOURCE_SPECS = ((0.250, 0.0), (0.312, 180.0))


def layer_bounds(tissue, layer):
    """(inner, outer) normalised radii of one shell.

    The innermost layer starts at the axis and the outermost reaches the rim, so
    those two bounds come from the geometry rather than from `interfaces`.
    """
    k = tissue.layer_names.index(layer)
    inner = 0.0 if k == 0 else tissue.interfaces[k - 1]
    outer = 0.5 if k == tissue.n_layers - 1 else tissue.interfaces[k]
    return inner, outer


def dipole_centres(
    grid, tissue, r_blob_norm=None, dz=0.13, layer="muscle", angle_deg=0.0
):
    """The position of the two blobs in physical coordinates on this grid.

    Returned separately from the field so that a notebook can report the placement
    without recomputing the blob, and so that the placement can be tested without
    catching an exception.

    r_blob_norm defaults to the mid-radius of `layer`, in normalised units. It is
    normalised rather than physical because the disc is a different physical size
    at every resolution: a fixed physical offset would sit at a different fraction
    of the limb on each grid, and the interfaces it has to clear would have moved.

    `angle_deg` is the azimuth of the blob about the limb axis, measured from +x.
    It controls placement rather than tissue properties. In the ideal continuous
    shell model, changing only this angle rotates the source; the discrete mapped
    grid is not continuously rotationally invariant, so the resulting sampled
    fields need not be exact rotated copies.

    Keys: `r_blob` (physical transverse offset), `angle_deg`, `c_x`, `c_y`,
    `cz_source`, `cz_sink`, and the two shell radii bounding `layer`, `r_inner` and
    `r_outer`, also physical.
    """
    r_in, r_out = layer_bounds(tissue, layer)
    if r_blob_norm is None:
        r_blob_norm = 0.5 * (r_in + r_out)

    norm = tissue.shell_norm
    r_blob = grid.physical_shell_radius(r_blob_norm, norm)

    X, Y = grid.coords
    t = math.radians(angle_deg)
    return {
        "r_blob": r_blob,
        "angle_deg": float(angle_deg),
        # Use the mean mapped position as the disc centre, not a literal 0.5.
        # They agree to round-off by symmetry; this is the form the field is built
        # from, so taking it from anywhere else would move the blob in the last bit.
        "c_x": float(X.mean())
        + r_blob * math.cos(t),  # azimuth measured from +x
        "c_y": float(Y.mean()) + r_blob * math.sin(t),
        # Split along z about the mid-plane. Both blobs share (x, y) and separate
        # along zeta only, aligning the synthetic dipole with the assumed fibre
        # direction.
        "cz_source": 0.5 * grid.L - dz,
        "cz_sink": 0.5 * grid.L + dz,
        "r_inner": grid.physical_shell_radius(r_in, norm),
        "r_outer": grid.physical_shell_radius(r_out, norm),
    }


def gaussian_dipole(
    grid,
    tissue,
    width=0.05,
    r_blob_norm=None,
    dz=0.13,
    layer="muscle",
    angle_deg=0.0,
):
    """(n, n, n) source-sink Gaussian dipole, aligned with the fibre axis and placed
    off the limb axis inside `layer`.

    Built on the PHYSICAL disc coordinates, not the reference square: a blob placed
    on the reference grid would be distorted by the map into something that is not a
    Gaussian in physical space.

    `width` is the 1/e blob radius as a physical length; w/h = 3.2 at n = 65 for
    the default unit extent. `dz` is the half-separation along the
    fibre axis, also a physical length. Neither is normalised, so both must be
    rescaled by hand if the domain extent L changes.

    The two preconditions below are properties the experimental design must have,
    not defensive checks: a blob that overlaps an interface is not a source in one
    tissue, and a blob near a z face is truncated by the domain rather than
    resolved by it. They raise rather than assert, so that `python -O` cannot turn
    a design error into a silently wrong experiment. Both are stated in physical
    radius, which under soft membership is the essence of what being "inside a shell"
    means.

    The field is projected onto the zero-weighted-mean subspace before it is
    returned, because under the insulating Neumann condition the compatibility
    condition W . I = 0 is what makes A phi = I solvable at all.

    For this source the projection is insurance rather than correction. The z nodes
    are symmetric about the mid-plane and the two blob centres are placed
    symmetrically about it, with identical transverse factors, so source and sink
    are mirror images on the symmetric axial grid, so the raw difference has zero
    weighted sum up to floating-point round-off. The
    projection stays because admissibility is a precondition of the solve and must
    not depend on a symmetry the caller can break with one argument.
    """
    n = grid.n
    c = dipole_centres(
        grid,
        tissue,
        r_blob_norm=r_blob_norm,
        dz=dz,
        layer=layer,
        angle_deg=angle_deg,
    )

    if not c["r_inner"] + 2 * width < c["r_blob"] < c["r_outer"] - 2 * width:
        raise ValueError(
            f"blob offset {c['r_blob']:.4f} does not clear both {layer} interfaces "
            f"by 2w = {2 * width:.4f}: the shell runs from {c['r_inner']:.4f} to "
            f"{c['r_outer']:.4f}"
        )
    if not (2 * width < c["cz_source"] and c["cz_sink"] < grid.L - 2 * width):
        raise ValueError(
            f"blob too close to a z face: source at {c['cz_source']:.4f}, sink at "
            f"{c['cz_sink']:.4f}, domain [0, {grid.L}], 2w = {2 * width:.4f}"
        )

    X, Y = grid.coords
    X_phys = torch.as_tensor(X)[:, :, None].expand(n, n, n)
    Y_phys = torch.as_tensor(Y)[:, :, None].expand(n, n, n)
    Z_phys = torch.linspace(0.0, grid.L, n)[None, None, :].expand(n, n, n)

    # A smooth, rapidly decaying synthetic source profile.
    def blob(cz):
        return torch.exp(
            -(
                (
                    (X_phys - c["c_x"]) ** 2
                    + (Y_phys - c["c_y"]) ** 2
                    + (Z_phys - cz) ** 2
                )
                / width**2
            )
        )

    return impose_net_zero_CSD(blob(c["cz_source"]) - blob(c["cz_sink"]))


# ---- the noise --------------------------------------------------------------


def add_measurement_noise(phi, percent=5.0, seed=0):
    """Additive Gaussian noise at `percent` of the clean field's nodewise RMS.

    The RMS is the unweighted mean over all array entries. It is not a
    trapezoidal volume RMS and not the electrode-only relative error; the latter
    must be computed after restricting both noise and field to the mask.

    Returns `(phi_obs, stats)`, where stats carries `rms`, `noise_std` and the
    realised `snr` = ||phi|| / ||noise||. Both norms are over the whole array,
    so this SNR is likewise an all-node quantity.

    Relative rather than absolute because the field amplitude changes with the
    conductivities. The
    generator is created and seeded locally, so the draw is independent of any
    earlier use of the global RNG and re-running this call alone reproduces it.
    """
    if percent < 0:
        raise ValueError(
            f"noise percentage must be non-negative, got {percent}"
        )

    rms = phi.pow(2).mean().sqrt()
    noise_std = (percent / 100.0) * rms

    gen = torch.Generator().manual_seed(int(seed))
    noise = noise_std * torch.randn(phi.shape, generator=gen, dtype=phi.dtype)

    snr = (phi.norm() / noise.norm()).item() if percent > 0 else float("inf")
    return phi + noise, {
        "rms": rms.item(),
        "noise_std": noise_std.item(),
        "snr": snr,
    }


# ---- the electrodes ---------------------------------------------------------


def lateral_surface_mask(n):
    """All nodes on the four lateral faces -- the denominator for a coverage figure.

    The two z faces are excluded throughout: they are the cut ends of the limb, not
    skin, and nothing is measured there.
    """
    surf = torch.zeros(n, n, n, dtype=torch.bool)
    surf[0] = surf[-1] = True
    surf[:, 0] = surf[:, -1] = True
    return surf


@dataclass(frozen=True)
class ElectrodeArray:
    """Where the array reads the field.

    On the mapped grid the skin surface IS the four lateral faces of the reference
    cube, so the mask is index arithmetic with no geometric search.

    coverage_target : requested fraction of each lateral face. The stride makes
                      realised coverage approximately 1/step^2, with finite-grid
                      and shared-edge effects; obtain the exact value from
                      `coverage`.
    use_surface     : False gives synthetic full-field data at every node and
                      exists as a control rather than a surface-electrode layout.

    The stride is uniform in the two index directions spanning each reference-cube
    face, not in physical arclength on the mapped rim. It is computed with `ceil`
    rather than
    `round`: round(1/sqrt(0.5)) = round(1.414) = 1 would select every node and
    silently deliver full coverage for a nominal one half. Both are inert at
    coverage 1, where the stride is 1 either way.
    """

    coverage_target: float = 1.0
    use_surface: bool = True

    def __post_init__(self):
        if not 0.0 < self.coverage_target <= 1.0:
            raise ValueError(
                f"coverage_target must lie in (0, 1], "
                f"got {self.coverage_target}"
            )

    @property
    def step(self):
        """Node stride within each lateral face."""
        return max(1, math.ceil(1.0 / math.sqrt(self.coverage_target)))

    def mask(self, n):
        """Boolean (n, n, n) mask selecting the electrode nodes."""
        if not self.use_surface:
            return torch.ones(n, n, n, dtype=torch.bool)

        s = self.step
        face_sel = torch.zeros(n, n, dtype=torch.bool)
        face_sel[::s, ::s] = True

        mask = torch.zeros(n, n, n, dtype=torch.bool)
        mask[0] |= face_sel
        mask[-1] |= face_sel  # x = 0, x = n-1  (pattern on y, z)
        mask[:, 0] |= face_sel
        mask[:, -1] |= face_sel  # y = 0, y = n-1  (pattern on x, z)
        return mask

    def coverage(self, n):
        """Realised fraction of the lateral surface that is read.

        Measured as the selected lateral nodes over all lateral nodes, so the
        full-field control returns 1.0 rather than a ratio above one.
        """
        surf = lateral_surface_mask(n)
        return int((self.mask(n) & surf).sum()) / int(surf.sum())


# ---- the observation --------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """A noisy field, the mask it is read through, and the grid it was produced on.

    The grid is carried rather than implied. The inversion solves on the grid the
    data was generated on, and bundling the three together is what makes that
    checkable at the point of use instead of by tracing which variable came from
    where.
    """

    phi: torch.Tensor
    mask: torch.Tensor
    grid: object

    def __post_init__(self):
        n = self.grid.n
        if tuple(self.phi.shape) != (n, n, n):
            raise ValueError(
                f"field is {tuple(self.phi.shape)} but the grid is n = {n}"
            )
        if tuple(self.mask.shape) != (n, n, n):
            raise ValueError(
                f"mask is {tuple(self.mask.shape)} but the grid is n = {n}"
            )
        if self.mask.dtype is not torch.bool:
            raise TypeError(f"mask must be boolean, got {self.mask.dtype}")
        if not bool(self.mask.any()):
            raise ValueError(
                "the mask selects no nodes, so there is nothing to fit"
            )

    @property
    def n(self):
        return self.grid.n

    @property
    def values(self):
        """The measured values, as a flat vector in mask order."""
        return self.phi[self.mask]

    def misfit(self, phi_pred):
        """Relative misfit of a predicted field at the electrode nodes.

        The denominator is the norm of the observed values, so this is the
        quantity the convergence plots report and is comparable across noise
        levels and resolutions.
        """
        if tuple(phi_pred.shape) != tuple(self.phi.shape):
            raise ValueError(
                f"prediction is {tuple(phi_pred.shape)}, "
                f"observation is {tuple(self.phi.shape)}"
            )
        return float(
            torch.linalg.norm((phi_pred - self.phi)[self.mask])
            / torch.linalg.norm(self.values)
        )


def weighted_sum(field):
    """Trapezoidal-weighted sum of a field, without the common spacing factor.

    Exposed because it is what makes the source's admissibility checkable: a
    dipole is balanced when this vanishes, and the plain sum is not the same
    statement on a vertex-centred grid. Multiplying the result by h^3 gives the
    corresponding quadrature integral on a cube with uniform spacing h.
    """
    W = trapezoid_weight_field(
        field.shape[0], dtype=field.dtype, device=field.device
    )
    return float((W * field).sum())
