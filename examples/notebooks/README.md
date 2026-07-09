# Example notebooks

Interactive, explained versions of the scripts in `examples/`. Each notebook
walks through one use case step by step, with the reasoning inline.

| Notebook | Covers |
|----------|--------|
| [`01_pendulum_stability.ipynb`](01_pendulum_stability.ipynb) | Minimal `verify_stability` pipeline; NumPy fallback; **EVT** Lipschitz estimation for a nonlinear field |
| [`02_bilinear2d_stability.ipynb`](02_bilinear2d_stability.ipynb) | JAX fast path; analytic Jacobian; the **anytime** certified-rate frontier |
| [`03_kuramoto_roa.ipynb`](03_kuramoto_roa.ipynb) | `verify_roa`; closed-form `L`; the two-pass **Trim** protocol; RoA cube-map plot |

## Running them

```bash
pip install "pyddrv[jax,dev] @ git+https://github.com/NetDLab/pyddrv"
pip install jupyterlab            # if you don't already have it
jupyter lab                       # then open a notebook and "Run All"
```

On a development checkout, make sure `pyddrv` is importable (either
`pip install -e ".[dev]"`, or launch Jupyter with `PYTHONPATH=$PWD/src`).

The notebooks ship **un-executed** (clean diffs); "Run All" regenerates every
figure. `01` runs on the pure-NumPy path (no JAX needed); `02` and `03` use the
JAX kernel.
