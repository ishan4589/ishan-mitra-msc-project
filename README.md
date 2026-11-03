# neuralphysics

A differentiable, conservative flux-form finite-difference solver for recovering tissue properties from synthetic surface-electrode potentials.

The bundled model represents a limb as four concentric cylindrical layers:
bone, anisotropic muscle, fat, and skin. The inverse problem estimates five
conductivity parameters (including transverse and longitudinal muscle
conductivity) and the two inner interface radii. The fat-skin interface is fixed,
so skin thickness is assumed known.

The forward model solves

```text
div(sigma grad(phi)) = I
```

with an insulating outer boundary. A square-to-disc map carries the cylindrical
geometry onto a reference cube, where the equation is discretised by a
conservative flux-form operator. PyTorch autograd differentiates through a fixed
number of multigrid V-cycles, providing gradients with respect to both
conductivities and interface radii. Because the cycle count is fixed, `forward`
returns an approximate discrete solution rather than iterating to a residual
tolerance.

This repository was produced for an Independent Research Project at Imperial
College London. The project was supervised by Dr Claire Heaney and Professor
Christopher Pain.

The project was originally developed in a university-hosted GitHub repository and
moved to this personal account after it finished, so its development commit
history was not carried over.

## Installation

Python 3.10 or later is required.

```bash
pip install -e ".[dev]"
```

The main dependencies are PyTorch, NumPy, SciPy, and Matplotlib.

## Quick start

```python
from neuralphysics import (
    LIMB,
    CoefficientHierarchy,
    ElectrodeArray,
    Observation,
    add_measurement_noise,
    configure,
    forward,
    gaussian_dipole,
    grid_at,
)

configure()  # Select float64 and seed the NumPy and PyTorch generators.

n = 33
grid = grid_at(n)
source = gaussian_dipole(grid, LIMB)
coefficients = CoefficientHierarchy.from_vectors(LIMB, n)

potential = forward(source, coefficients)
observed, noise = add_measurement_noise(potential, percent=5.0, seed=0)
observation = Observation(observed, ElectrodeArray().mask(n), grid)

print(f"{observation.values.numel()} electrode values")
print(f"realised full-field SNR: {noise['snr']:.1f}")
```

Grid sizes used by the multigrid hierarchy have the form `2^k + 1`, such as 33
or 65. See the
[end-to-end experiment](notebook-demos/end-to-end-experiment.ipynb) for data
generation, seven-parameter inversion, and visualisation.

## Tests

```bash
pytest                         # default suite; excludes slow tests
pytest -m slow                 # slow tests only
pytest -m "slow or not slow"  # complete suite
```

## Repository layout

| Path | Contents |
|---|---|
| `src/neuralphysics/` | Solver, tissue model, observation model, inversion, and plotting code |
| `src/neuralphysics/verification/` | Structural and manufactured-solution checks |
| `tests/` | Pytest suite |
| `notebook-demos/` | Notebooks that run the experiments and generate the figures used in the final report |

## Reference

The Neural Physics implementation follows:

B. Chen, C. E. Heaney, and C. C. Pain, “Neural physics: Using AI libraries to
develop physics-based solvers for incompressible computational fluid dynamics,”
*Computers & Fluids*, vol. 308, 106981, 2026.
[doi:10.1016/j.compfluid.2026.106981](https://doi.org/10.1016/j.compfluid.2026.106981)

The square-to-disc mapping and mesh-smoothing approach is adapted from Fanxin
Sun, *Solving Inverse Problems in Bio-Physics Using Neural Physics* (2025).
