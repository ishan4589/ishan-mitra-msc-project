# I used Claude Code to help implement and package neuralphysics.

r"""Neural Physics for inverse problems in biophysics.

Recovers five tissue conductivity parameters (bone, muscle_T, muscle_L, fat, skin)
and the two inner interface radii that set the bone, muscle and fat thicknesses,
from simulated surface-electrode potentials by backpropagating through a
differentiable multigrid solver. The forward PDE is the electro-quasi-static
equation div(sigma grad phi) = I under the sign convention used here, on a
cylindrical limb domain,
mapped to a reference cube with the geometry carried in the metric, discretised as
a conservative flux-form finite-volume operator whose fixed neighbour-difference
kernels are combined with spatially varying coefficient fields.

"Neural Physics" names the implementation:
the discretisation is expressed as neural-network layers so that an AI library becomes
the PDE solver and autograd supplies the gradients. See the following paper:
  author  = {B. Chen and C. E. Heaney and C. C. Pain},
  title   = {Neural physics: Using {AI} libraries to develop physics-based solvers for incompressible computational fluid dynamics},
  journal = {Computers \& Fluids},
  volume  = {308},
  pages   = {106981},
  year    = {2026},
  doi     = {10.1016/j.compfluid.2026.106981}

This module performs no side effects on import.
Call `configure()` explicitly to select the default dtype and seed the generators;
the project uses float64 by default to reduce round-off in differentiated solves.
"""

__version__ = "0.1.0"  # package version

from .data import (
    SOURCE_SPECS,
    ElectrodeArray,
    Observation,
    add_measurement_noise,
    dipole_centres,
    gaussian_dipole,
    lateral_surface_mask,
    layer_bounds,
    weighted_sum,
)
from .ensemble import (
    EnsembleResult,
    RunResult,
    invert,
    load_geometry,
    parameter_names,
    run_ensemble,
    save_geometry,
    sample_initial_parameters,
    sigma_indices,
    thicknesses,
    true_parameters,
    true_sigma,
    unpack,
)
from .forward import forward
from .grid import CylinderGrid, SmoothingPolicy, grid_at, levels
from .hierarchy import CoefficientHierarchy
from .metric import identity_metric, jacobian_num, metric_from_jacobian
from .multigrid import jacobi, probe_diagonal, prolong, restrict, v_cycle
from .operators import (
    NEIGHBOUR_DIFF_KERNELS,
    face_coefficients,
    laplacian_3D,
    laplacian_kernel,
    laplacian_metric_3D,
)
from .tissue import (
    LIMB,
    TissueModel,
    interfaces_from_logits,
    logits_from_interfaces,
)
from .projections import (
    impose_electric_potential_uniqueness,
    impose_net_zero_CSD,
    trapezoid_weight_field,
)
from .utils import configure, to_numpy

__all__ = [
    "configure",
    "to_numpy",
    "trapezoid_weight_field",
    "impose_net_zero_CSD",
    "impose_electric_potential_uniqueness",
    "CylinderGrid",
    "SmoothingPolicy",
    "grid_at",
    "levels",
    "face_coefficients",
    "laplacian_metric_3D",
    "laplacian_kernel",
    "laplacian_3D",
    "NEIGHBOUR_DIFF_KERNELS",
    "CoefficientHierarchy",
    "forward",
    "SOURCE_SPECS",
    "layer_bounds",
    "dipole_centres",
    "gaussian_dipole",
    "add_measurement_noise",
    "lateral_surface_mask",
    "ElectrodeArray",
    "Observation",
    "weighted_sum",
    "restrict",
    "prolong",
    "probe_diagonal",
    "jacobi",
    "v_cycle",
    "TissueModel",
    "LIMB",
    "interfaces_from_logits",
    "logits_from_interfaces",
    "parameter_names",
    "sigma_indices",
    "true_sigma",
    "true_parameters",
    "unpack",
    "thicknesses",
    "sample_initial_parameters",
    "invert",
    "run_ensemble",
    "RunResult",
    "EnsembleResult",
    "save_geometry",
    "load_geometry",
    "__version__",
]
