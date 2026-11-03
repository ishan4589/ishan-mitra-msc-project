# I used Claude Code to help implement and package the verification norms.

"""Error norms for the verification studies.

Two are needed. The max norm is what a manufactured-solution study reports,
because it certifies the worst node rather than an average that a few bad rows
can hide in. The volume-weighted L2 supports comparisons across reference-grid
resolutions: an unweighted sum of squares grows like n^(3/2) under refinement, so
its magnitude is not directly comparable between grid sizes.

The weight is the same trapezoidal field the operator's left null space uses, so
the norm and the compatibility projection agree about what the discrete measure
is. That field is unnormalised -- its entries are 1, 1/2, 1/4, 1/8 and it sums to
(n-1)^3 -- so `l2_w` supplies the h^3 itself.

Limitation, deliberate: this is the measure of the reference cube, not of the
mapped physical domain. A physical-domain field norm would carry a
det J factor as well. Nothing here needs that yet; revisit when the refinement
study wants to compare solutions across mapped grids.
"""

import torch

__all__ = ["linf", "l2_w"]


def linf(field, mask=None):
    """Max-norm of `field`, over `mask` if one is given.

    An all-False mask has no defined maximum and returns 0.0 rather than raising,
    so a caller sweeping a family of masks does not have to special-case the
    empty one.
    """
    f = field if mask is None else field[mask]
    if f.numel() == 0:
        return 0.0
    return f.abs().max().item()


def l2_w(field, W, mask=None):
    r"""Volume-weighted L2 norm over the reference cube, sqrt(h^3 sum W f^2).

    h is taken as 1/(n-1) from the shape of `W`, which is the reference spacing on
    [0, 1]^3. With that factor the weights integrate to 1, so a constant field c
    returns |c| at every resolution, and over a mask the result is the L2 norm on
    the masked region rather than an RMS.
    """
    n = W.shape[0]
    h3 = (1.0 / (n - 1)) ** 3
    f, w = (field, W) if mask is None else (field[mask], W[mask])
    if f.numel() == 0:
        return 0.0
    return torch.sqrt(h3 * (w * f ** 2).sum()).item()
