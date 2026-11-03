# I used Claude Code to help implement and package the forward solver.

r"""The forward solve: given a current source, the potential it induces.

The operator is singular under the insulating boundary condition, so two things
have to happen for phi to be a function of (I, sigma) rather than of the solver's
iteration history. The source is projected onto the admissible subspace, because
the system is solvable only for sources orthogonal to the operator's left null
space -- physically, a limb sealed by an insulating boundary cannot accumulate
net charge. The returned field is then gauge-fixed, because the solution is
determined only up to an additive constant and the iteration would otherwise
leave whichever constant it happened to accumulate.

Both matter for the inverse problem specifically. An unprojected source would
make the misfit depend on an inadmissible component that no conductivity can
explain, and an unfixed gauge would put a null direction into the parameter
sensitivities.

A fixed number of V-cycles is used instead of a residual tolerance. This gives
each objective evaluation the same amount of solver work and the same unrolled
graph structure. The result is an approximate PDE solution whose accuracy is
controlled by `num_cycles`; no claim of convergence is implied by the fixed count.
"""

import torch

from .multigrid import v_cycle
from .projections import (
    impose_electric_potential_uniqueness,
    impose_net_zero_CSD,
)

__all__ = ["forward"]


def forward(
    I,
    hierarchy,
    num_cycles=8,
    omega=2.0 / 3.0,
    nu_pre=2,
    nu_post=2,
    base_iters=50,
):
    """Approximately solve div_xi(sigma-tilde grad_xi phi) = I on the reference cube.

    Parameters
    ----------
    I           : (n, n, n) current source density. Projected to net zero before use;
                  the projection is a no-op for a source that is already admissible,
                  which the dipole is by construction.
    hierarchy   : CoefficientHierarchy whose finest level matches I. Which constructor
                  built it is what decides whether this solve is differentiable and in
                  what -- there is no conductivity argument here, and no branch on one.
    num_cycles  : how many V-cycles to run. A fixed count rather than a convergence
                  criterion, so it is an argument of the solve: calling this in a loop
                  over cycle counts is how the residual history is measured, and the
                  measurement is only meaningful if the argument reaches the loop
                  bound.
    omega, nu_pre, nu_post, base_iters :
                  the smoother's damping and sweep counts, passed through to the
                  cycle unchanged.

    The domain extent and the coarsening floor are both read from the hierarchy rather
    than passed here. Its coefficients were built at one extent and descend to one
    coarsest level, so either given separately could disagree with them -- and the
    extent silently, since it enters only through h.
    """
    n = I.shape[0]
    if n != hierarchy.n_fine:
        raise ValueError(
            f"source is {n}^3 but the hierarchy is built for "
            f"{hierarchy.n_fine}^3"
        )

    cycles = int(num_cycles)
    if cycles < 1:
        raise ValueError(f"a solve needs at least one V-cycle, got {cycles}")

    h_grid = hierarchy.L / (n - 1)
    n_base = hierarchy.levels[-1]

    b = impose_net_zero_CSD(I)
    phi = torch.zeros(n, n, n)
    for _ in range(cycles):
        phi = v_cycle(
            phi,
            b,
            h_grid,
            hierarchy.faces,
            hierarchy.sigma_T,
            hierarchy.metric,
            n_base,
            omega,
            nu_pre,
            nu_post,
            base_iters,
            diag_by_n=hierarchy.diagonals,
        )
    return impose_electric_potential_uniqueness(phi)
