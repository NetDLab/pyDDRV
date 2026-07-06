# pyddrv — Data-Driven Recurrence-based Verification

Certify **exponential stability** and **regions of attraction** of a dynamical
system **from sampled trajectories** — no Lyapunov function search, no model
of the dynamics beyond a one-sided Lipschitz bound (which can itself be
estimated from data).

pyddrv implements verification via **Recurrent Lyapunov Functions (RLFs)**
(Siegelmann, Shen, Paganini, Mallada): instead of constructing a Lyapunov
function whose sublevel sets are invariant, take the plain norm
`V(x) = ‖x − x*‖` and certify from trajectories that it is **τ-recurrent** —
within a horizon `τ` it dips back below `e^{−ατ} V`. Recurrence permits
excursions that invariance forbids, so the certificates are less conservative
than classical (e.g. sum-of-squares) ones, and they come with an **anytime**
guarantee: more compute never hurts, stopping early is sound.

## Quick start

```python
import numpy as np
from pyddrv import verify_stability

A = np.array([[0.0, 2.0], [-1.0, -1.0]])         # stable spiral

def f(x):                                        # batched field (N,d) -> (N,d)
    return x @ A.T

report = verify_stability(f, R=0.7, d=2, tau=3.0)
print(report.summary())
# certified: alpha >= 0.47 ... — a GUARANTEED exponential decay rate on Q_R
```

`verify_stability` runs the full pipeline: it estimates the one-sided
Lipschitz constant `L` from boundary trajectories (with a
discretization-robustness check that lifts the sampled claim to the
continuum), covers `Q_R \ B_ε` with an exponentially-efficient layered grid
of cubes, certifies each cube from **one** trajectory (Theorem 8), and
adaptively refines the cubes that limit the global rate.

Regions of attraction, at a target rate `alpha`:

```python
from pyddrv import verify_roa

roa = verify_roa(f, R=np.pi, d=2, alpha=1.0, tau=1.9, trim=True)
print(roa.summary())     # union of certified cubes: roa.centers, roa.halfs
```

Non-origin equilibria are handled with `equilibrium=x_star`; results are
reported in original coordinates.

## What you provide

- **A batched vector field** `f(X: (N,d)) -> (N,d)`. Written with
  backend-agnostic ops (or `jax.numpy`) it runs on the fused, jitted JAX
  kernel; a plain-NumPy field automatically uses the (identical, slower)
  NumPy kernel. A torch field with `backend="torch"` runs on Apple-GPU/CUDA.
- Optionally an **analytic Jacobian** (`jac=`) — makes the Lipschitz constant
  exact for fields with state-affine Jacobians — or a precomputed **`L=`**
  (closed-form bound, or `one_sided_lipschitz_from_data` from experimental
  `(x, f(x))` samples).

## What you get back

A certificate, not a heuristic: every reported rate is a sound lower bound.

- `verify_stability` → `StabilityReport`: certified rate `alpha`, the
  data-driven ceiling `alpha_upper`, the sub-optimality gap, the `L`
  estimate, and the full anytime trace.
- `verify_roa` → `RoAReport`: a union of certified cubes (an inner
  RoA approximation), its volume, and per-cube refinement depths.

## How it works (the 60-second version)

The ε-ERLF condition (paper eq. 24): for every `x` in `Q_R \ B_ε` there is a
return time `t ∈ (0, τ]` with `e^{αt} ‖φ(t,x)‖ ≤ ‖x‖`. A single trajectory
certifies a whole ball `B_r(x)` via **Theorem 8**, using the one-sided
Lipschitz constant `L = sup μ(∂f/∂x)` (a matrix measure / log-norm) to bound
neighbour drift. A layered grid — coarse far from the equilibrium,
geometrically finer near it (`O(3^d log(R/ε))` cubes instead of a uniform
`O((R/ε)^d)`) — turns ball certificates into set certificates, and adaptive
`3^d` splitting concentrates compute where the rate is limited. Simulation
and certification are fused in a jitted kernel (RK4 inside `lax.scan`,
running-max over return times, tiled batches, `O(N·d)` memory — no stored
trajectories), which is what makes hundreds of millions of cube certificates
per run practical.

## Install

```bash
pip install "pyddrv[jax] @ git+https://github.com/NetDLab/pyddrv"   # recommended
# or, for development:
git clone https://github.com/NetDLab/pyddrv && cd pyddrv
conda env create -f environment.yml && conda activate pyddrv        # numpy+scipy+jax
pip install -e ".[dev]"
pytest
```

Extras: `[jax]` fused CPU kernel (recommended), `[jax-cuda]` NVIDIA GPUs,
`[torch]` Apple-MPS/CUDA kernel for very large RoA sweeps, `[sos]` the
model-based sum-of-squares comparison baseline, `[dev]` tests + plotting.

## Layout

```
src/pyddrv/
  api.py           verify_stability / verify_roa (start here)
  verification/    the certifier: fused kernels (JAX/NumPy/torch), Theorem-8
                   ball bounds, layered grid + 3^d splitting, Algorithms 1 & 2,
                   one-sided Lipschitz estimation with eq-(38) robustness check
  lipschitz.py     matrix measures / log-norms; L from data (Definition 2)
  systems/         example dynamics (NumPy + JAX variants) and an RK4 sampler
  data/            TrajectorySet container
  baselines/       optional box-SoS comparison (SumOfSquares/PICOS)
examples/          runnable demos (stability, anytime frontier, Kuramoto RoA)
tests/             pytest suite
```

## Examples

```bash
python examples/pendulum_stability.py     # NumPy field, full pipeline
python examples/bilinear2d_stability.py   # JAX fast path + anytime trace
python examples/kuramoto_roa.py           # region of attraction with Trim
```

Each script writes its figures and raw results to `examples/output/`.

## Relationship to the paper

This is the general-purpose successor to the (private) research code
developed for the paper below; the certifier core (Theorem-8 kernel, layered
grids, Algorithms 1–2, renewal-terminated discretization check) is identical
to the version validated there, including reproduction of the paper's
benchmark tables and figures.

## Citing

R. Siegelmann, Y. Shen, F. Paganini, E. Mallada. *Stability Analysis and
Data-driven Verification via Recurrent Lyapunov Functions.*
([preprint](https://mallada.ece.jhu.edu/pubs/2025-Preprint-SSPM.pdf))

See `CITATION.cff` for BibTeX-ready metadata.
