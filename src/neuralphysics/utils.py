# I used Claude Code to help implement and package the numerical utilities.

"""Dependency-free helpers: array conversion and global numerical configuration."""

import numpy as np
import torch

__all__ = ["to_numpy", "configure"]


# Convert tensors for NumPy-based reporting and plotting code.
def to_numpy(tensor):
    """np.gradient needs a numpy array: pass numpy straight through, otherwise
    detach the torch tensor from the graph and convert."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def configure(dtype=torch.float64, seed=42):
    """Set the global numerical configuration: default dtype and RNG seeds.

    Called explicitly rather than on import because changing global dtype and RNG
    state is intrusive in a larger session. This project defaults to float64 to
    reduce round-off in the differentiated solver, and making that choice explicit
    keeps it visible to callers.

    Parameters
    ----------
    dtype : torch dtype used as the default for newly created tensors.
    seed  : seed applied to both the torch and numpy global generators.

    Returns
    -------
    dict recording what was set, for provenance reporting.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(dtype)
    return {"dtype": str(dtype), "seed": seed}
