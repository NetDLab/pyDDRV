# Example notebooks

Interactive, explained versions of the scripts in `examples/`. Each notebook
walks through one use case step by step, with the reasoning inline.

| Notebook | Covers |
|----------|--------|
| [`01_pendulum_stability.ipynb`](01_pendulum_stability.ipynb) | Minimal `verify_stability` pipeline; NumPy fallback; **EVT** Lipschitz estimation for a nonlinear field |
| [`02_bilinear2d_stability.ipynb`](02_bilinear2d_stability.ipynb) | JAX fast path; analytic Jacobian; the **anytime** certified-rate frontier |
| [`03_kuramoto_roa.ipynb`](03_kuramoto_roa.ipynb) | `verify_roa`; closed-form `L`; the two-pass **Trim** protocol; RoA cube-map plot |

## Running them

Use a clean Python **3.10+** environment (the macOS system `python3` is often
3.9 — too old):

```bash
python3.12 -m venv pyddrv-env && source pyddrv-env/bin/activate   # any 3.10+
pip install "pyddrv[jax,examples] @ git+https://github.com/NetDLab/pyddrv"
jupyter lab                       # then open a notebook and "Run All"
```

The `examples` extra pulls in matplotlib **and** JupyterLab, so that one line is
everything you need. Launch `jupyter lab` from inside the activated environment
so the notebook uses its kernel.

On a development checkout, install from the repo instead
(`pip install -e ".[jax,examples]"`), or launch Jupyter with
`PYTHONPATH=$PWD/src`.

The notebooks ship **un-executed** (clean diffs); "Run All" regenerates every
figure. `01` runs on the pure-NumPy path (no JAX needed); `02` and `03` use the
JAX kernel.
