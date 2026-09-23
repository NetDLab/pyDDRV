<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pyddrv-mark-dark.svg">
    <img src="docs/assets/pyddrv-mark-light.svg" alt="pyDDRV logo" width="140">
  </picture>
</p>

<h1 align="center">pyDDRV: Data-Driven Recurrence-based Verification</h1>

<p align="center">
  <a href="https://github.com/NetDLab/pyDDRV/actions/workflows/ci.yml"><img src="https://github.com/NetDLab/pyDDRV/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://arxiv.org/abs/2608.26447"><img src="https://img.shields.io/badge/arXiv-2608.26447-b31b1b.svg" alt="arXiv"></a>
</p>

pyDDRV certifies exponential decay toward an equilibrium, down to a small ball
around it, and computes inner approximations of regions of attraction. It uses
simulated trajectories and implements the Recurrent Lyapunov Function (RLF)
method of Siegelmann, Paganini and Mallada
([arXiv:2608.26447](https://arxiv.org/abs/2608.26447)).

A classical Lyapunov certificate needs a function whose sublevel sets are
invariant. An RLF only needs to be recurrent: from every initial condition, the
function must return to a smaller value within a horizon `τ`, although it may
increase in between. pyDDRV uses the plain norm `V(x) = ‖x − x*‖` as the RLF
and checks the recurrence condition on trajectories, so no Lyapunov function has
to be constructed.

The method is data-driven in the sense that the certificate is computed from
trajectories rather than from an analysis of the equations. It is not
model-free. pyDDRV simulates a vector field `f` that you provide, from initial
conditions it chooses, and it needs an upper bound `L` on the one-sided
Lipschitz constant of `f` over the states those trajectories visit. `L` can be
given or estimated.

## Example

```python
import numpy as np
from pyddrv import verify_stability, matrix_measure

A = np.array([[0.0, 2.0], [-1.0, -1.0]])

def f(x):                      # batched vector field: (N, d) -> (N, d)
    return x @ A.T

report = verify_stability(f, R=0.7, d=2, tau=3.0, L=matrix_measure(A))
print(report.summary())
```

```
verify_stability[2-norm, R=0.7, eps=0.007, tau=3]: certified: alpha >= 0.4500 (ceiling 0.4999, gap 10.0%) | L=0.207, 45720 cubes, 4 refinements, backend=numpy
```

For a linear field the one-sided Lipschitz constant is the matrix measure of
`A`, so `L` is exact here. The result certifies that every trajectory starting
in the box `‖x‖∞ ≤ 0.7` satisfies `‖x(t)‖ ≤ C e^{−0.45 t} ‖x(0)‖`, with
`C = e^{(0.45 + L)τ}`, until it enters the ball of radius `eps` around the
equilibrium. The eigenvalues of `A` are `−0.5 ± 1.32i`, so no rate above 0.5
is possible.

Regions of attraction are computed for a given rate. For this function the
field has to be written with `jax.numpy`:

```python
import jax.numpy as jnp
from pyddrv import verify_roa

def pendulum(x):
    return jnp.stack([x[:, 1], -jnp.sin(x[:, 0]) - x[:, 1]], axis=1)

roa = verify_roa(pendulum, R=3.0, d=2, alpha=0.2, tau=5.0, L=0.62, trim=True)
print(roa.summary())
```

The result is a union of cubes (`roa.centers`, `roa.halfs`) covering 84% of the
box, computed in about 3 seconds on a laptop CPU. The bound `L=0.62` holds for
the pendulum everywhere; [GETTING_STARTED.md](GETTING_STARTED.md) shows how it
is obtained.

## What the guarantee depends on

A certified rate is a lower bound under the following conditions.

- `L` bounds the one-sided Lipschitz constant of `f` over the reachable set.
  This holds with certainty when `L` is given as a proven bound, as in both
  examples above, or when an analytic Jacobian that is affine in the state is
  supplied. When `L` is estimated for a general field, the default estimator
  fits an extreme-value distribution to sampled values of the matrix measure,
  and the bound holds with probability `rho` (0.95 by default).
- When `L` is estimated, pyDDRV also checks that every trajectory from the box
  stays inside the region over which `L` was estimated. The result is in
  `report.discretization_ok`.
- Trajectories are computed with a fixed-step RK4 integrator. Integration
  error is not included in the certificate.

`report.certified` is true only when a positive rate was found and the check
above passed. If no rate is certified, nothing follows about the stability of
the equilibrium.

## How it works

For a point `x`, a single simulated trajectory gives a rate that holds for
every point in a ball around `x`: the one-sided Lipschitz constant bounds how
far neighbouring trajectories can separate. pyDDRV covers the box `Q_R` minus a
small ball around the equilibrium with cubes that are coarse far from the
equilibrium and finer near it, certifies each cube from the trajectory through
its center, and splits the cubes that limit the overall rate. The certified
rate is the minimum over cubes. Under the conditions above, each intermediate
value is a valid lower bound, so a run can be stopped at any time. Refinement
usually raises the rate toward `report.alpha_upper`, the rate certified at the
cube centers alone.

Simulation and the certification test run in one compiled JAX kernel. A NumPy
implementation gives the same results more slowly, and a PyTorch
implementation targets Apple and NVIDIA GPUs.

## Installation

Python 3.10 or later.

```bash
pip install "pyddrv[jax] @ git+https://github.com/NetDLab/pyDDRV"
```

The package and import name is `pyddrv`, in lowercase. Optional extras: `jax`
(compiled CPU kernel, recommended), `jax-cuda` (NVIDIA GPUs), `torch` (PyTorch
kernel), `viz` (plotting), `examples` (plotting and JupyterLab, for the
notebooks), `sos` (sum-of-squares baseline, needs a MOSEK license), `dev`
(tests).

For development:

```bash
git clone https://github.com/NetDLab/pyDDRV && cd pyDDRV
conda env create -f environment.yml && conda activate pyddrv
pip install -e ".[dev]"
pytest
```

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md): installation, the bundled
  examples and their expected output, using pyDDRV on your own system,
  choosing `L` and `τ`, plotting, platforms.
- `examples/`: four scripts (a pendulum, the bilinear benchmark from the paper,
  and Kuramoto oscillators in two and three dimensions), and the same four as
  notebooks in `examples/notebooks/`.
- The docstrings of `verify_stability` and `verify_roa` in
  `src/pyddrv/api.py` list every option.

## Repository layout

```
src/pyddrv/
  api.py            verify_stability, verify_roa
  verification/     kernels, cube grid and splitting, the two algorithms,
                    reachable-set and Lipschitz estimation
  lipschitz.py      matrix measures; Lipschitz constants from Jacobians or data
  lipschitz_evt.py  extreme-value estimate of the one-sided Lipschitz constant
  viz.py            plotting
  systems/          example vector fields (NumPy and JAX)
  baselines/        sum-of-squares comparison
examples/           scripts and notebooks
tests/              test suite
```

## Citation

R. Siegelmann, F. Paganini, E. Mallada. Stability Analysis and Data-driven
Verification via Recurrent Lyapunov Functions. arXiv:2608.26447, 2026.

`CITATION.cff` has the metadata in machine-readable form.
