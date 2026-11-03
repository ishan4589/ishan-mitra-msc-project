# I used Claude Code to help implement and package the coefficient hierarchy.

r"""The per-level coefficients the V-cycle reads: metric, face coefficients, sigma_T.

Every level supplies its own coefficients rather than restricted ones, because
the coarse operator is a rediscretisation of the same shell geometry at the
coarser grid and not a Galerkin product R A_h P. That matters more here than on a
Cartesian mesh: the coarse disc is a different physical size -- sampled rim radius
0.50144 at n = 65 against 0.52341 at n = 9. So a restricted fine-grid label
field would place the interfaces at the wrong physical radii. Normalising the
radius per level is what keeps the interface fractions meaning the same part of
the limb on every grid.

sigma_T is carried separately from the face tensor because the cross term is a
cell-centred coefficient, and a cell-centred coefficient is not a face
coefficient. There is no Cartesian counterpart: that operator has no cross term.

One constructor, `from_vectors`, serves data generation and inversion. Its
conductivity, interface and transition-width arguments default to the tissue
model; grid extent, axial stretch and coarsening floor have their own defaults.
"""

from dataclasses import dataclass, field

from .grid import grid_at, levels
from .operators import face_coefficients

__all__ = ["CoefficientHierarchy"]


@dataclass(frozen=True)
class CoefficientHierarchy:
    """Metric, face coefficients and sigma_T at every level of one V-cycle hierarchy.

    Attributes
    ----------
    levels    : grid sizes from fine to coarse.
    metric    : {level -> (mxx, mxy, myy, mL)}.
    faces     : {level -> (6, n, n, n) face coefficients}.
    sigma_T   : {level -> (n, n, n) transverse conductivity}.
    L         : the domain extent every level was built at. Carried here rather
                than supplied alongside, because the coefficients already encode it
                and a separately passed extent could disagree with them -- which
                would hand the solve the wrong h and raise nothing. `forward` reads
                it from here, as it already reads the coarsening floor from
                `levels[-1]`.
    diagonals : {level -> probed operator diagonal}, empty until something fills it.
    """

    levels: tuple
    metric: dict
    faces: dict
    sigma_T: dict
    L: float = 1.0
    diagonals: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not self.levels:
            raise ValueError("a hierarchy needs at least one level")
        if self.L <= 0:
            raise ValueError(
                f"the domain extent must be positive, got {self.L}"
            )
        want = set(self.levels)
        for name in ("metric", "faces", "sigma_T"):
            got = set(getattr(self, name))
            if got != want:
                raise ValueError(
                    "the metric, face and sigma_T hierarchies must cover exactly "
                    f"the same levels; {name} covers {sorted(got, reverse=True)} "
                    f"against {sorted(want, reverse=True)}"
                )
        for n in self.levels:
            if self.faces[n].shape != (6, n, n, n):
                raise ValueError(
                    f"faces at level {n} have shape "
                    f"{tuple(self.faces[n].shape)}, expected {(6, n, n, n)}"
                )
            if self.sigma_T[n].shape != (n, n, n):
                raise ValueError(
                    f"sigma_T at level {n} has shape "
                    f"{tuple(self.sigma_T[n].shape)}, expected {(n, n, n)}"
                )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_vectors(
        cls,
        tissue,
        n_fine,
        sT_vec=None,
        sL_vec=None,
        interfaces=None,
        epsilon=None,
        n_base=None,
        L=1.0,
        z_zeta=1.0,
    ):
        """Coefficients at every level, from per-layer conductivities and radii.

        This is what the inversion calls each evaluation. Gradients reach the
        conductivity vectors through the weighted sum and the harmonic-mean face
        averaging, and reach the radii through the logistic membership -- the shape
        derivative, obtained with no separate derivation.

        `tissue` and `n_fine` are required. The tissue model carries sourced values
        and the resolution decides what medium the coarse levels resolve, so neither
        is the sort of thing to inherit silently. Every remaining argument defaults:
        the conductivities, radii and width fall back to the tissue model's own, so
        `(tissue, n)` is the ground-truth hierarchy and iterates give the current
        one; `epsilon` is a fraction of the extent L, converted per level by
        `TissueModel.epsilon_norm`; `n_base` defers to `levels`; and `L` and
        `z_zeta` configure the grids built at every level.
        """
        sizes = levels(n_fine) if n_base is None else levels(n_fine, n_base)
        metric, faces, sig_T = {}, {}, {}
        for n in sizes:
            g = grid_at(n, L=L, z_zeta=z_zeta)
            sT, sL = tissue.sigma_fields(
                g,
                sT_vec=sT_vec,
                sL_vec=sL_vec,
                interfaces=interfaces,
                epsilon=epsilon,
            )
            metric[n] = g.metric
            faces[n] = face_coefficients(g.metric, sT, sL)
            sig_T[n] = sT
        return cls(tuple(sizes), metric, faces, sig_T, L=L)

    # -- convenience ---------------------------------------------------------

    @property
    def n_fine(self):
        return self.levels[0]

    @property
    def requires_grad(self):
        """Whether anything here is attached to the autograd graph. True when the
        vectors or radii it was built from carry requires_grad, false at the truth.
        """
        return any(self.faces[n].requires_grad for n in self.levels)
