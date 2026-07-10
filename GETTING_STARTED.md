# Getting started / testing guide

A hands-on walkthrough for someone evaluating pyddrv for the first time: install
it, confirm it works, reproduce the bundled results, then point it at your own
system. No prior knowledge of the codebase is assumed.

For *what the method does* and the theory, see [`README.md`](README.md). This
document is purely operational.

---

## 0. What you will be able to do

- **Certify a decay rate** for a dynamical system: a guaranteed exponential
  rate `alpha` such that `‖x(t)‖ ≤ C e^{-alpha t} ‖x(0)‖` on a box `Q_R`
  around an equilibrium. (`verify_stability`)
- **Certify a region of attraction**: a union of cubes from which trajectories
  provably converge at a target rate. (`verify_roa`)

Everything is computed **from simulated (or logged) trajectories** — you supply
a vector field (or trajectory data), not a hand-crafted Lyapunov function.

---

## 1. Install

You need Python ≥ 3.10. Two supported paths:

### Option A — quick, from GitHub (recommended for a tester)

```bash
python -m venv pyddrv-env && source pyddrv-env/bin/activate
pip install "pyddrv[jax,dev] @ git+https://github.com/NetDLab/pyddrv"
```

`[jax]` pulls the fast fused kernel (CPU; still fast). `[dev]` adds pytest and
matplotlib so you can run the test suite and the example figures.

### Option B — development clone (to read/modify the code)

```bash
git clone https://github.com/NetDLab/pyddrv && cd pyddrv
conda env create -f environment.yml     # creates the `pyddrv` env: numpy+scipy+jax
conda activate pyddrv
pip install -e ".[dev]"
```

### Minimal / no-JAX install

The certifier runs on a pure-NumPy fallback with **identical certificates**
(just slower, and `verify_roa` needs JAX or torch — see §5). If JAX is a
problem on your platform:

```bash
pip install "pyddrv @ git+https://github.com/NetDLab/pyddrv"   # numpy only
```

### Optional extras

| Extra          | Enables                                                        |
|----------------|---------------------------------------------------------------|
| `[jax-cuda]`   | NVIDIA GPU (same code; ~20–50× on large runs)                 |
| `[torch]`      | Apple-MPS / CUDA kernel for very large RoA sweeps             |
| `[sos]`        | the sum-of-squares comparison baseline (needs a MOSEK license)|

---

## 2. Sanity check: run the test suite

```bash
pytest -q
```

Expected: **`74 passed`** on a full `[jax,sos]` install (about 45 s; most of the
time is the SoS baseline tests). On a **numpy-only** install you should see
roughly **`35 passed, 5 skipped`** — the skips are the JAX/torch/SoS tests,
which is correct and not a failure.

> If pytest reports import errors about `pyddrv`, see Troubleshooting (§8):
> almost always a stale editable install shadowing the source tree.

---

## 3. Reproduce the bundled results

Three runnable demos live in `examples/`. Each prints a one-line certificate
and writes raw results + a figure to `examples/output/`.

```bash
python examples/pendulum_stability.py     # NumPy field, full pipeline, ~5 s
python examples/bilinear2d_stability.py   # JAX fast path + anytime trace, ~2 s
python examples/kuramoto_roa.py           # region of attraction with Trim, ~1–2 min
```

**Expected output (numbers are stable to a few %; wall times are machine-dependent):**

| Example        | Certified result                                  | Backend | Notes |
|----------------|---------------------------------------------------|---------|-------|
| `pendulum`     | `alpha >= 0.388` (ceiling 0.484), eq-(38) `True`  | numpy   | damped pendulum, `R=0.8`, `tau=6`; `L` via EVT (nonlinear Jacobian) |
| `bilinear2d`   | `alpha >= 0.474` (ceiling 0.499), gap 5%          | jax     | paper eq. (39), `eta=0.3`, matches the paper's ~0.47 |
| `kuramoto_roa` | certified 1-RoA ≈ **82% of Q_π**                  | jax     | the sync basin minus the two splay-corner basins |

If you see certificates in these ballparks, the install is sound and the method
is doing what the paper claims. `pendulum` runs on the NumPy fallback on
purpose (its `np.sin` field does not JIT-trace), so it also confirms the no-JAX
path.

**Prefer notebooks?** The same three walkthroughs, step by step with the
reasoning inline, are in [`examples/notebooks/`](examples/notebooks). Install the
`examples` extra (matplotlib + JupyterLab) and open one — see that folder's
README:

```bash
pip install "pyddrv[jax,examples] @ git+https://github.com/NetDLab/pyddrv"
jupyter lab
```

---

## 4. Verify your own system (stability)

The whole API is two functions. Here is the complete recipe:

```python
import numpy as np
from pyddrv import verify_stability

# 1. A BATCHED vector field: takes (N, d) states, returns (N, d) derivatives.
#    (Batched so N trajectories integrate at once — this is required.)
def f(x):
    x1, x2 = x[:, 0], x[:, 1]
    return np.stack([x2, -np.sin(x1) - x2], axis=1)   # e.g. a pendulum

# 2. Certify a rate on the box Q_R (inf-norm radius R) about the origin.
report = verify_stability(f, R=0.8, d=2, tau=5.0)

print(report.summary())
print("certified:", report.certified)      # True / False
print("rate alpha:", report.alpha)         # the guaranteed lower bound
```

**Key arguments:**

| Arg           | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `R`           | inf-norm radius of the verification box `Q_R` (required)                |
| `d`           | state dimension (required)                                              |
| `tau`         | recurrence horizon; larger never hurts the rate, costs compute (def 5)  |
| `equilibrium` | pass `x_star` if the equilibrium is not the origin; results come back in original coordinates |
| `norm`        | `"2"` (default), `"inf"`, or `"1"` — the norm used for `V(x)=‖x‖`       |
| `eps`         | inner radius excluded around the equilibrium (default `R/100`)          |
| `jac`         | optional analytic Jacobian `x->(d,d)`; makes `L` exact for nice fields  |
| `L`           | pass a precomputed one-sided Lipschitz bound to skip estimation         |
| `max_seconds` | wall-clock budget; the result is an *anytime* sound lower bound         |
| `max_refine`  | max refinement rounds (default 12)                                     |
| `delta`       | stop once the rate is within this relative gap of the ceiling (def 0.1) |

**Writing the field so it uses the fast kernel.** If you write `f` with
`jax.numpy` (or ops that trace under `jit`), the fused JAX kernel is selected
automatically. A plain-NumPy field silently uses the NumPy kernel — same
answer, slower. You do not choose the backend; it is probed. Force it with
`backend="jax" | "numpy" | "torch"` if needed.

### Choosing how the Lipschitz constant `L` is estimated

The certificate rests on a one-sided Lipschitz constant `L = sup μ(∂f/∂x)` over
the reachable set. **A too-small `L` makes the certificate unsound**, so how you
get it matters. Three options, best first:

1. **You have a closed-form bound** → pass it as `L=...`. Rigorous, no sampling.
   (Example: Kuramoto's `L ≤ 2k(n−1)/n`.)
2. **General nonlinear field** → `L_method="evt"`. This uses the extreme-value
   (reverse-Weibull) estimator of Knuth et al.: it samples `μ(∂f/∂x)` in the
   *interior* of the box and extrapolates to a **high-probability upper bound**
   (confidence `rho`, default 0.95). Diagnostics land in `report.lipschitz.evt`.

   ```python
   report = verify_stability(f, R=0.8, d=2, tau=5.0, L_method="evt", rho=0.95)
   print(report.L, report.lipschitz.evt.summary())
   ```
3. **State-affine Jacobian only** (e.g. bilinear systems) → `L_method="corners"`
   (the default) is *exact* there. **Do not** trust `corners` for a general
   nonlinear field: the true `sup μ` is often in the box interior, and corners
   can miss it (we've seen it under-shoot by 3–6× on pendulum-like fields),
   yielding an unsound `L`. Use `"evt"` instead.

You can also estimate `L` straight from logged `(x, f(x))` data with no model:
`pyddrv.evt_one_sided_lipschitz_from_data(states, derivatives, rho=0.95)`.

---

## 5. Verify a region of attraction

```python
from pyddrv import verify_roa

roa = verify_roa(f, R=np.pi, d=2, alpha=1.0, tau=1.9, trim=True)
print(roa.summary())
# roa.centers (N,d), roa.halfs (N,) — the certified union of cubes
# roa.volume — total certified volume
```

- `alpha` is the **target** rate you demand (unlike `verify_stability`, which
  *finds* the best rate).
- `trim=True` runs the paper's two-pass protocol (grow, then re-certify that
  trajectories stay inside the grown region). Sound RoA claims with the 2-norm
  generally need it.
- **`verify_roa` requires JAX** (or `backend="torch"` with a torch field). On a
  numpy-only install it raises with a clear message; use the low-level
  `pyddrv.verification.find_alpha_roa` for a slow pure-NumPy fallback.

---

## 6. Visualizing results

Reusable plotting helpers live in `pyddrv.viz` (they take the report objects
directly; matplotlib required — it is in the `[dev]` extra). Each returns a
matplotlib `Axes` you can restyle.

```python
from pyddrv.viz import plot_anytime, plot_stability_2d, plot_roa_2d
import matplotlib.pyplot as plt

# stability: the anytime certified-rate-vs-time curve (needs record_trace=True)
rep = verify_stability(f, R=0.8, d=2, tau=5.0, record_trace=True)
plot_anytime(rep)

# stability: the verified box over a phase portrait of the field (2-D)
plot_stability_2d(rep, f=f)
# ...and the certified covering grid itself (the layered + adaptively-refined
# cubes). "outline" draws the tiling over the flow; "alpha"/"width" fill cubes
# by their per-cube certified rate / size:
plot_stability_2d(rep, f=f, show_grid=True, grid_color_by="outline")
plot_stability_2d(rep, show_grid=True, grid_color_by="alpha")

# region of attraction: the certified cube union, colored by cube size (2-D)
roa = verify_roa(f, R=np.pi, d=2, alpha=1.0, tau=1.9, trim=True)
plot_roa_2d(roa, color_by="width")     # or color_by="depth"

plt.show()   # or ax.figure.savefig("out.png", dpi=150)
```

The three `examples/` scripts each use one of these helpers, so they double as
worked usage. `plot_roa_2d` is 2-D only; for higher-dimensional regions, slice
to two coordinates before plotting.

## 7. Reading a report

`verify_stability` returns a `StabilityReport`:

- **`.certified`** — `True` iff a positive rate was certified **and** the
  discretization-robustness check passed. This is the headline boolean.
- **`.alpha`** — the guaranteed rate (a sound *lower* bound). Report this.
- **`.alpha_upper`** — the data-driven *ceiling*: the best rate the data could
  support here. The gap `.alpha` → `.alpha_upper` is how much refinement might
  still gain.
- **`.discretization_ok`** — whether eq. (38) lifted the finite-sample claim to
  the continuum. If `False`, the numbers are still informative but the formal
  continuum guarantee is not established (a warning is emitted).
- **`.L`, `.lipschitz`** — the one-sided Lipschitz constant used and the full
  estimate (`R_bar`, `R_max`, ...).
- **`.trace`** — with `record_trace=True`, the anytime curve
  `[(seconds, alpha, n_cubes), ...]`.

**Two things testers commonly misread:**

1. **Not certified ≠ unstable.** A negative or `-inf` `alpha` means *this run*
   could not certify a rate — not that the system diverges. Try a smaller `R`,
   a larger `tau`, or a longer budget.
2. The rate is deliberately **conservative**. `alpha` is a floor; the true rate
   is between `alpha` and `alpha_upper`. Tightening `delta` and raising
   `max_refine`/`max_seconds` pushes `alpha` toward the ceiling.

---

## 8. Troubleshooting

- **`ImportError` / wrong version of `pyddrv` in pytest or examples.** A
  previously `pip install -e`'d copy elsewhere can shadow this tree. Run from
  the repo root; pytest is already configured with `pythonpath=["src"]`. For
  the example scripts on a dev clone, `export PYTHONPATH=$PWD/src` before
  running them so they pick up *this* source and not another editable install.
- **`verify_roa` raises "needs the fused JAX kernel".** Install `[jax]`, or pass
  a torch field with `backend="torch"`, or use the low-level NumPy fallback.
- **SoS baseline tests/examples fail or hang.** The `[sos]` baseline uses MOSEK,
  which needs a free academic license at `~/mosek/mosek.lic`. Without it, skip
  the SoS parts — they are only the comparison baseline, not the certifier.
- **GPU.** JAX on Apple Silicon is CPU-only here and is already fast. Real GPU
  speedups come from `[jax-cuda]` on an NVIDIA machine (no code change) or the
  `[torch]` MPS/CUDA kernel for very large RoA sweeps.

---

## Where to go next

- `examples/` — copy the closest demo and swap in your field.
- `src/pyddrv/api.py` — full docstrings for every `verify_stability` /
  `verify_roa` argument.
- `src/pyddrv/verification/` — the low-level building blocks (Theorem-8 ball
  bounds, layered grid, Algorithms 1 & 2) if you want to go under the hood.
</content>
</invoke>
