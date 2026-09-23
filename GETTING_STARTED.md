# Getting started

The method is summarized in the [README](README.md) and described in the paper
([arXiv:2608.26447](https://arxiv.org/abs/2608.26447)).

## 1. Installation

pyDDRV needs Python 3.10 or later. The `python3` that ships with macOS is
usually 3.9, so use a Python from Homebrew, python.org or conda.

```bash
python3.12 -m venv pyddrv-env
source pyddrv-env/bin/activate
pip install "pyddrv[jax,dev] @ git+https://github.com/NetDLab/pyDDRV"
```

`jax` installs the compiled kernel and `dev` installs pytest and matplotlib.
Without JAX, `verify_stability` still works, using a slower NumPy kernel that
gives the same results; `verify_roa` then needs the PyTorch backend.

To work on the code itself, clone the repository instead:

```bash
git clone https://github.com/NetDLab/pyDDRV && cd pyDDRV
conda env create -f environment.yml
conda activate pyddrv
pip install -e ".[dev]"
```

## 2. Checking the installation

From a clone of the repository:

```bash
pytest -q
```

All tests should pass. Tests for optional components that are not installed
(JAX, PyTorch, the sum-of-squares baseline) are skipped. A full run takes one
to three minutes, most of it in the sum-of-squares tests.

## 3. The examples

```bash
python examples/pendulum_stability.py
python examples/bilinear2d_stability.py
python examples/kuramoto_roa.py
python examples/kuramoto_roa_3d.py
```

Each script prints its result and writes the raw data and figures to
`examples/output/`. Results should match the table below to within a few
percent. Times were measured on a laptop CPU.

| Script | System | Result | Time |
|---|---|---|---|
| `pendulum_stability.py` | damped pendulum, `R = 0.8` | rate ≥ 0.388 | 5 s |
| `bilinear2d_stability.py` | bilinear benchmark of the paper, `R = 0.7` | rate ≥ 0.474 | 2 s |
| `kuramoto_roa.py` | 3 Kuramoto oscillators, rate 1 | 82% of the box | 10 s |
| `kuramoto_roa_3d.py` | 4 Kuramoto oscillators, rate 1 | 66% of the box (first pass only) | 40 s |

The pendulum example uses the NumPy kernel, the others the JAX kernel. The
same four examples are available as notebooks in `examples/notebooks/`, with
more explanation. To run them, install the `examples` extra, which adds
JupyterLab:

```bash
pip install "pyddrv[jax,examples] @ git+https://github.com/NetDLab/pyDDRV"
jupyter lab
```

## 4. Certifying a decay rate for your own system

You need a vector field that accepts a batch of states of shape `(N, d)` and
returns the derivatives in the same shape. Writing it with `jax.numpy` lets
pyDDRV use the compiled kernel. A field written with NumPy also works for
`verify_stability`, more slowly.

```python
import jax.numpy as jnp
from pyddrv import verify_stability

def pendulum(x):
    theta, omega = x[:, 0], x[:, 1]
    return jnp.stack([omega, -jnp.sin(theta) - omega], axis=1)

report = verify_stability(pendulum, R=0.8, d=2, tau=5.0)
print(report.summary())
```

```
verify_stability[2-norm, R=0.8, eps=0.008, tau=5]: certified: alpha >= 0.4349 (ceiling 0.4832, gap 10.0%) | L=0.0947, 30152 cubes, 4 refinements, backend=jax
```

`pendulum_stability.py` reports 0.388 for the same box because it uses
`tau=6` and stops refining at a 20% gap to `alpha_upper` (`delta=0.2`) instead
of the default 10%. Here `L` is estimated (section 5), so the rate holds with probability 0.95.

The main arguments:

| Argument | Meaning |
|---|---|
| `R` | half-width of the box `Q_R = {‖x − x*‖∞ ≤ R}` to certify |
| `d` | state dimension |
| `tau` | recurrence horizon (section 9) |
| `equilibrium` | the equilibrium `x*`, if it is not the origin |
| `norm` | norm used for `V(x) = ‖x − x*‖`: `"2"` (default), `"inf"` or `"1"` |
| `eps` | radius of the ball around `x*` excluded from the certificate (default `R/100`) |
| `L`, `jac`, `L_method` | how the Lipschitz constant is obtained (section 5) |
| `max_seconds`, `max_refine` | time and refinement budgets |
| `delta` | stop refining when the rate is within this relative gap of `alpha_upper` |

The full list is in the docstring of `verify_stability`.

## 5. The Lipschitz constant

The certificate uses an upper bound `L` on the one-sided Lipschitz constant of
`f`, which for a smooth field is `sup μ(∂f/∂x)`, the largest matrix measure of
the Jacobian, taken over the states that trajectories from `Q_R` visit. If `L`
is too small the certificate is not valid. There are three options, from
strongest to weakest.

1. Pass a bound you have derived, as `L=...`. For the pendulum above the
   Jacobian is `[[0, 1], [−cos θ, −1]]`, whose 2-norm matrix measure never
   exceeds `(√5 − 1)/2 ≈ 0.618`, so `L=0.62` is valid everywhere.
2. Pass an analytic Jacobian as `jac=...`. pyDDRV first estimates a box that
   contains the reachable set. If the Jacobian is affine in the state, as for
   polynomial fields of degree two, its matrix measure is largest at a corner
   of that box, and pyDDRV evaluates it exactly there.
3. Otherwise, the default (`L_method="auto"`) samples the matrix measure
   inside the box that contains the reachable set and fits an extreme-value
   distribution to the sampled maxima, following Knuth et al. The resulting
   bound holds with probability `rho` (0.95 by default) rather than with
   certainty.

`report.lipschitz.method` records which method was used, and
`report.lipschitz.evt` holds the diagnostics of the extreme-value fit.

The extreme-value estimate can fail on large boxes. For the pendulum with
`R = 3`, trajectories from the corners swing over the top, the reachable set is
large, the sampled maxima all take the same value, and the fit breaks down: it
returns an `L` in the millions and nothing is certified. With `L=0.62` the same
call certifies 84% of the box. When a closed-form bound is available, use it.

If you have measured states and derivatives but no model, the one-sided
Lipschitz constant can be estimated from those samples with
`pyddrv.evt_one_sided_lipschitz_from_data`. The estimate only covers the
region where the samples lie, so they must cover the reachable set. This only
provides `L`; the certificate itself still needs a simulator that pyDDRV can
start from initial conditions of its choosing.

## 6. Regions of attraction

`verify_roa` takes a target rate `alpha` and returns the set of cubes in `Q_R`
from which convergence at that rate is certified. It requires the compiled
kernel, so the field must be written with `jax.numpy` (or with PyTorch, using
`backend="torch"`).

```python
from pyddrv import verify_roa

roa = verify_roa(pendulum, R=3.0, d=2, alpha=0.2, tau=5.0, L=0.62, trim=True)
print(roa.summary())
```

```
verify_roa[alpha=0.2, 2-norm, R=3, tau=5]: 29582 cubes (4.8% of 614624 tested), volume 29.35 | L=0.62, Trim=True (fixed point after 15 passes), stop=complete
```

The certified set is `roa.centers` and `roa.halfs` (cube centers and half
widths) and its volume is `roa.volume`; here that is 82% of the box.

With `trim=False`, a cube is accepted when the decay condition holds for the
whole cube, which is checked with the trajectory through its center and the
Lipschitz bound: for some return time `t ≤ tau`, the norm of every trajectory
from the cube has dropped by at least a factor `e^{−alpha t}`. This says
nothing about where the trajectories go afterwards.

With `trim=True`, the returned region is also closed under these returns: at
the return time, the trajectories from each cube lie inside the returned region
or inside the ball of radius `eps` around the equilibrium. Every trajectory
starting in the region therefore reaches that ball, and its norm drops by at
least `e^{−alpha t}` at each return. The ball has to count as a landing zone:
a chain of returns that stayed in the region forever would shrink towards the
equilibrium, but the region excludes a neighborhood of the equilibrium.

`verify_roa` gets there by repeating a Trim pass, which checks each cube
against the region from the previous pass and drops or splits the cubes that
fail, until a pass changes nothing. `roa.trim_passes` is the number of passes,
and `roa.trim_history` the cube count and volume after each. If
`max_trim_passes` (default 30) is reached first, `roa.trim_converged` is false,
a warning is issued, and the region is not certified against itself. The
check uses a raster of the region with `raster_n` cells per axis, and the cells
must be small compared to `eps`, since chains that end in the ball have to
land on cells lying entirely inside it. `max_seconds` limits the first pass
only.

For long runs in higher dimensions, `verify_roa` accepts further options
(`max_seconds`, `priority`, `inner_first`, `max_pending_parents`, `spill_dir`);
they are documented in `pyddrv.verification.find_alpha_roa_fused`. None of them
are needed for the examples here.

## 7. Plotting

The functions in `pyddrv.viz` take the report objects directly and return a
matplotlib `Axes`. They need matplotlib (the `viz`, `examples` or `dev` extra).

```python
import matplotlib.pyplot as plt
from pyddrv.viz import plot_anytime, plot_stability_2d, plot_roa_2d

report = verify_stability(pendulum, R=0.8, d=2, tau=5.0, record_trace=True)
plot_anytime(report)                     # certified rate against wall time
plot_stability_2d(report, f=pendulum)    # the box over the phase portrait
plot_stability_2d(report, show_grid=True, grid_color_by="alpha")
plot_roa_2d(roa, color_by="width")       # the certified cubes
plt.show()
```

`plot_stability_2d` can draw the cubes used by the certificate as outlines over
the flow (`grid_color_by="outline"`), or filled by their width (`"width"`) or
by the rate each one certifies (`"alpha"`). For regions in more than two
dimensions, `plot_roa_slice(roa, dims=(0, 1), at=...)` draws the cubes that
intersect an axis-aligned plane.

## 8. Reading a report

`verify_stability` returns a `StabilityReport` with these fields:

- `alpha`: the certified rate.
- `alpha_upper`: the rate certified at the cube centers alone. Refinement can
  bring `alpha` close to it but not above it. It is not a bound on the rate of
  the system.
- `certified`: true when `alpha > 0` and the discretization check passed.
- `discretization_ok`: whether the boundary trajectories used to bound the
  reachable set cover all trajectories from the box. If it is false, a warning
  is issued and the result is not certified.
- `L`, `lipschitz`: the Lipschitz constant and how it was obtained.
- `trace`: with `record_trace=True`, the list `(seconds, alpha, n_cubes)` after
  each refinement round.

If no positive rate is certified, try a smaller `R`, a longer `tau`, a larger
time budget, or a tighter `L`. Failing to certify does not show that the
equilibrium is unstable.

## 9. Choosing `tau`

`tau` is how long a trajectory may take to return to a smaller value of `V`.
The certificate takes the best return time in `(0, tau]`, so for a fixed `L` a
longer `tau` cannot lower the certified rate. When `L` is estimated, it is
estimated for the chosen horizon and can grow if trajectories travel further,
which can lower the rate. The cost grows in proportion to `tau`, since every
trajectory is integrated over the whole horizon.

If `tau` is too short, trajectories that need longer to return certify nothing,
and the rate is low or negative even for a stable system. A few characteristic
time constants of the system are a reasonable start; the examples use 3 to 6.
`method="ladder"` lets each cube start at `tau/4` and use the full horizon only
when it limits the rate.

## 10. Platforms and GPUs

A GPU is not needed for the examples. The test suite runs on Linux (Python
3.10 and 3.12) and Windows (Python 3.11) on every commit, with and without
JAX.

On NVIDIA GPUs under Linux (or WSL2 on Windows), install the `jax-cuda` extra
instead of `jax`; the same code then runs on the GPU. We have not benchmarked
this configuration. JAX does not support CUDA on native Windows. There, and on
Apple GPUs, the PyTorch kernel can be used: install the `torch` extra, write
the field with PyTorch operations, and pass `backend="torch"`.

## 11. Troubleshooting

- Import errors, or an old version of pyddrv being used: another editable
  install may be shadowing this one. The tests use `src/` directly. For the
  example scripts in a clone, set `PYTHONPATH=$PWD/src`.
- `verify_roa` raises "needs the fused JAX kernel": the field is not written
  with `jax.numpy`, or JAX is not installed.
- The sum-of-squares tests fail or hang: that baseline uses MOSEK, which needs a
  license file (free for academic use) at `~/mosek/mosek.lic`.
- `L` in the thousands or more: see section 5.
