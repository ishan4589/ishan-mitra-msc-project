# I used Claude Code to help implement and package the tissue model.

r"""The concentric-shell tissue model and its smoothly blended conductivity fields.

Four tissues as concentric shells about the limb axis, invariant along z (the
fibre axis): bone core, then muscle, fat, skin. The outer boundary of the domain
is the curved rim of the disc, so the skin shell reaches the rim and the
insulating Neumann condition on the boundary is the skin-air interface.
Air conductivity is neglected, as in Mesin (2013).

Interface radii are fractions of the normalised radius, on which the mean of the
four sampled edge traces sits at 0.5. That normalisation makes them
level-independent: the disc is
a different physical size at every multigrid level -- sampled edge mean 0.50144 at
n = 65 against 0.52341 at n = 9, so absolute radii would silently make each
coarse level a different limb.

The (n, n) cross-section is broadcast along zeta to (n, n, n). The extrusion is
straight, so sigma is z-invariant exactly. This is what makes every
zeta-face join two nodes of the same tissue, so the harmonic mean along
zeta is an identity and one code path serves all six faces.

Membership is soft throughout. Each interface is a logistic of width epsilon
instead of a step, so the radii are differentiable and a shape derivative comes
out of the same backward pass that returns the conductivity gradients. The
justification is numerical - we do not model the layer boundaries as discrete
because that requires a step function, which has zero derivative almost everywhere,
so it cannot be inverted through.
"""

from dataclasses import dataclass
from typing import Mapping

import torch

__all__ = [
    "TissueModel",
    "LIMB",
    "interfaces_from_logits",
    "logits_from_interfaces",
]


def interfaces_from_logits(logits, r_outer):
    """Interface radii from unconstrained logits, with the outermost radius held fixed.

    The radii must satisfy 0 < R_1 < ... < R_last, and L-BFGS-B has no way to express
    that as a box. Optimising the layer thicknesses through a softmax removes the
    constraint instead of enforcing it: a softmax is positive and sums to one, so the
    partial sums are automatically increasing and automatically inside (0, r_outer),
    for any real logits the optimiser cares to try.

    One fewer logit than thicknesses, with the last softmax logit pinned at zero:
    a softmax is
    invariant under adding a constant to every logit, so leaving all of them free
    would give the objective a flat direction that is nothing to do with the physics.

    r_outer is not recovered. Skin thickness is taken as known rather than fitted, so
    the outermost interface is a fixed constant of the forward model and the recovered
    thicknesses partition the region inside it.
    """
    u = torch.as_tensor(logits).reshape(-1)
    t = torch.softmax(torch.cat([u, torch.zeros_like(u[:1])]), dim=0)
    inner = r_outer * torch.cumsum(t[:-1], dim=0)
    fixed = torch.as_tensor(r_outer).to(inner.dtype).reshape(1)
    return torch.cat(
        [inner, fixed]
    )  # the last entry is the constant, not a sum to it


def logits_from_interfaces(interfaces):
    """The inverse of `interfaces_from_logits`, for turning an initial guess stated as
    radii into the logits the optimiser actually carries."""
    R = torch.as_tensor(interfaces).reshape(-1)
    t = torch.diff(R, prepend=torch.zeros_like(R[:1])) / R[-1]
    if not bool((t > 0).all()):
        raise ValueError(
            f"interfaces must increase outwards from 0, got {tuple(R)}"
        )
    return torch.log(t[:-1]) - torch.log(t[-1])


@dataclass(frozen=True)
class TissueModel:
    """Shell geometry, layer conductivities, and the width of the interfaces.

    The dataclass is frozen, so its attributes cannot be rebound after
    construction. Its conductivity mapping may itself be mutable, so instances
    should still be treated as configuration objects rather than immutable values.

    Parameters
    ----------
    layer_names   : tissues from the axis outwards. Layer order counts along this.
    sigma_layers  : name -> (sigma_T, sigma_L) in S/m.
    interfaces    : normalised radii separating consecutive layers, so
                    len(interfaces) == len(layer_names) - 1. Ordered, and all
                    strictly below 0.5 so that the outermost layer reaches the rim.
    shell_norm    : which radius field the shells are assigned by. 'physical' is
                    geometrically faithful and staircases the interfaces in index
                    space, with grid-scale placement error. 'reference' is face-aligned
                    in the reference cube and exactly circular under the raw map,
                    but subsequent mesh smoothing moves those mapped faces away
                    from their raw circular locations.
    epsilon       : interface transition width, as a fraction of the extent
                    grid.L. One convention throughout the package; every method
                    that accepts an epsilon accepts it in these units and converts
                    internally. A fraction rather than a length so that the same
                    model describes the same limb at any extent: an absolute width
                    would be a different share of a longer domain. Defaults to h/4
                    at n = 65 and L = 1.
    provisional   : layers whose conductivities are placeholders awaiting a cited
                    value. Carried in the type so it cannot be forgotten.
    """

    layer_names: tuple
    sigma_layers: Mapping
    interfaces: tuple
    shell_norm: str = "physical"
    epsilon: float = 1.0 / 256.0
    provisional: frozenset = frozenset()

    def __post_init__(self):
        if self.shell_norm not in ("physical", "reference"):
            raise ValueError(
                f"shell_norm must be 'physical' or 'reference', "
                f"got {self.shell_norm!r}"
            )
        if tuple(self.sigma_layers) != tuple(self.layer_names):
            raise ValueError(
                "sigma_layers must name exactly layer_names, in the same order; "
                f"got {tuple(self.sigma_layers)} against {tuple(self.layer_names)}"
            )
        if len(self.interfaces) != len(self.layer_names) - 1:
            raise ValueError(
                f"{len(self.layer_names)} layers need "
                f"{len(self.layer_names) - 1} interfaces, "
                f"got {len(self.interfaces)}"
            )
        if not all(
            a < b for a, b in zip(self.interfaces, self.interfaces[1:])
        ):
            raise ValueError(
                f"interfaces must increase outwards, got {self.interfaces}"
            )
        if not 0.0 < self.interfaces[0] or not self.interfaces[-1] < 0.5:
            raise ValueError(
                "interfaces must lie in (0, 0.5): the outermost layer has to reach "
                f"the rim at 0.5, got {self.interfaces}"
            )
        if not self.epsilon > 0:
            raise ValueError(
                "epsilon must be strictly positive: zero width is the hard "
                "labelling, whose derivative with respect to a radius is zero "
                f"almost everywhere, so no radius could be recovered; "
                f"got {self.epsilon}"
            )
        if not all(sT > 0 and sL > 0 for sT, sL in self.sigma_layers.values()):
            raise ValueError(
                "every conductivity must be positive; a non-positive value flips "
                "the sign of the operator"
            )
        if not set(self.provisional) <= set(self.layer_names):
            raise ValueError(
                f"provisional names an unknown layer: "
                f"{set(self.provisional) - set(self.layer_names)}"
            )

    # -- derived -------------------------------------------------------------

    @property
    def n_layers(self):
        return len(self.layer_names)

    @property
    def anisotropic_layers(self):
        """Layers whose two conductivities differ. A property rather than a
        constraint: which tissues are anisotropic is a fact about a particular
        model, not about the class."""
        return frozenset(
            nm for nm, (sT, sL) in self.sigma_layers.items() if sT != sL
        )

    # -- fields --------------------------------------------------------------

    def epsilon_norm(self, grid, epsilon=None, norm=None):
        """`epsilon` converted from a fraction of the extent into this level's
        normalised radius units.

        The sharpness of a tissue boundary is a statement about the model, not about
        the mesh, so epsilon is the same fraction of the limb on every level and only
        its resolution changes. The normalised radius puts the sampled edge-trace
        mean at 0.5, so
        the conversion is 0.5 * epsilon * L / rim_mean. That is not a no-op across
        levels, since the sampled edge mean runs from 0.50144 at n = 65 to 0.52341 at
        n = 9, but it is exact under a change of extent, because rim_mean scales with
        L and the two cancel.
        """
        eps = self.epsilon if epsilon is None else epsilon
        return (
            0.5
            * eps
            * grid.L
            / grid.rim_mean(self.shell_norm if norm is None else norm)
        )

    def layer_weights(self, grid, interfaces=None, epsilon=None, norm=None):
        """Soft layer membership on `grid`, shape (n_layers, n, n, n), summing to 1
        along the leading axis.

        A discrete labelling would be a step function of the interface radii: its
        derivative is zero almost everywhere, so autograd cannot reach a radius
        through it. Replacing each step by a logistic of width "epsilon" gives a
        partition of unity that is smooth in the radii, so a radius can be recovered
        by the same backward pass that already recovers the conductivities.

        Writing s_k = sigmoid((r - R_k) / epsilon), the weights telescope:

            w_0 = 1 - s_1,   w_k = s_k - s_{k+1},   w_last = s_last

        which sums to 1 by construction and is non-negative whenever the interfaces
        increase outwards. "epsilon" is a fraction of the extent grid.L and is
        converted to this level's normalised units by `epsilon_norm`; it is not a
        multiple of h, so the transition stays the same share of the limb as the grid
        is refined and only becomes better resolved.

        Deliberately not cached. The interfaces are optimiser iterates, so a cache
        would either miss every call or serve a stale field.

        Both arguments default to the model's own values, which is what the
        data-generation path wants; the inversion passes its iterates instead.
        """
        norm = self.shell_norm if norm is None else norm
        n = grid.n
        r = torch.as_tensor(grid.radius_field(norm))  # (n, n), rim at 0.5
        R = torch.as_tensor(
            self.interfaces if interfaces is None else interfaces
        )
        s = torch.sigmoid(
            (r.unsqueeze(0) - R.reshape(-1, 1, 1))
            / self.epsilon_norm(grid, epsilon, norm)
        )
        upper = torch.cat([torch.ones_like(s[:1]), s])  # 1, s_1, ..., s_last
        lower = torch.cat([s, torch.zeros_like(s[:1])])  # s_1, ..., s_last, 0
        w = upper - lower
        return w.unsqueeze(-1).expand(-1, n, n, n)  # broadcast along zeta

    def sigma_vectors(self):
        """The two length-n_layers (sigma_T, sigma_L) vectors, ordered as layer_names.

        torch.stack, not torch.tensor, so this path also works when the entries are
        differentiable tensors: the inverse problem passes its iterates through here.
        """
        sT = torch.stack(
            [
                torch.as_tensor(self.sigma_layers[nm][0])
                for nm in self.layer_names
            ]
        )
        sL = torch.stack(
            [
                torch.as_tensor(self.sigma_layers[nm][1])
                for nm in self.layer_names
            ]
        )
        return sT, sL

    def sigma_fields(
        self,
        grid,
        sT_vec=None,
        sL_vec=None,
        interfaces=None,
        epsilon=None,
        norm=None,
    ):
        """(sigma_T, sigma_L) as two (n, n, n) fields on `grid`.

        Every argument defaults to the model's own value, so `sigma_fields(grid)` is
        the ground truth and each argument is an independent lever over it.

        sT_vec, sL_vec : per-layer conductivities ordered like `layer_names`. The
                         inversion passes its current iterates here; the weighted sum
                         is differentiable, so gradients reach them.
        interfaces     : the shell radii. Also differentiable, through the logistic
                         membership, which is what supplies the shape derivative.
        epsilon        : transition width, as a fraction of the extent grid.L.
        """
        if sT_vec is None or sL_vec is None:
            sT_vec, sL_vec = self.sigma_vectors()
        w = self.layer_weights(grid, interfaces, epsilon, norm)
        return (
            (sT_vec.reshape(-1, 1, 1, 1) * w).sum(0),
            (sL_vec.reshape(-1, 1, 1, 1) * w).sum(0),
        )


# The limb this project inverts on. A named constant rather than a default
# argument: these values carry sources and a rationale, so the call site should say
# which limb it means instead of inheriting one silently.
LIMB = TissueModel(
    layer_names=("bone", "muscle", "fat", "skin"),
    sigma_layers={
        # Conductivity values attributed to Farina et al. (2004) in the project data.
        "bone": (0.02, 0.02),
        "muscle": (0.10, 0.50),
        "fat": (0.05, 0.05),
        "skin": (1.00, 1.00),
    },
    # Bone, muscle and fat surfaces at 0.7, 2 and 2.3 cm on a 2.4 cm limb, scaled
    # by 0.5 / 2.4 so the skin surface sits at the normalised rim. The radii are
    # those of the four-layer cylinder Maksymenko et al. (2023) validate against;
    # they are used as a test geometry there.
    interfaces=(0.5 * 0.7 / 2.4, 0.5 * 2.0 / 2.4, 0.5 * 2.3 / 2.4),
    shell_norm="physical",
    epsilon=1.0
    / 256.0,  # h/4 at n = 65 and L = 1; a fraction of L, not a cell count
    provisional=frozenset(),
)
