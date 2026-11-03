# I used Claude Code to help implement and package the verification tools.

"""Verification: the checks that certify the discretisation, kept separate from
the code that runs the experiment.

Each module computes numbers and returns them. Deciding whether a number is
acceptable is the caller's job -- a test asserting a threshold, or a notebook
printing a table -- so that the same routine serves both without one of them
having to catch an AssertionError.
"""
from .convergence import mms_error, refinement_study
from .manufactured import SMOOTH_MAP, L_phys, SmoothAnalyticMap, phi_mms
from .norms import l2_w, linf
from .structural import (
    assemble_dense,
    matrix_free_null_checks,
    planar_interface_residual,
    structural_metrics,
)

__all__ = [
    "assemble_dense",
    "structural_metrics",
    "matrix_free_null_checks",
    "planar_interface_residual",
    "SmoothAnalyticMap",
    "SMOOTH_MAP",
    "phi_mms",
    "L_phys",
    "mms_error",
    "refinement_study",
    "linf",
    "l2_w",
]
