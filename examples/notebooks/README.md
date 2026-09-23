# Example notebooks

The four scripts in `examples/`, as notebooks with more explanation.

| Notebook | Content |
|---|---|
| [`01_pendulum_stability.ipynb`](01_pendulum_stability.ipynb) | `verify_stability` on a damped pendulum; NumPy kernel; extreme-value estimate of the Lipschitz constant |
| [`02_bilinear2d_stability.ipynb`](02_bilinear2d_stability.ipynb) | the bilinear benchmark of the paper; JAX kernel; certified rate during refinement |
| [`03_kuramoto_roa.ipynb`](03_kuramoto_roa.ipynb) | `verify_roa` for three Kuramoto oscillators, with a closed-form Lipschitz bound and `trim=True` |
| [`04_kuramoto_roa_3d.ipynb`](04_kuramoto_roa_3d.ipynb) | four oscillators (a three-dimensional region), drawn as cross-sections with `plot_roa_slice` |

## Running them

Use Python 3.10 or later (the `python3` that ships with macOS is usually 3.9):

```bash
python3.12 -m venv pyddrv-env
source pyddrv-env/bin/activate
pip install "pyddrv[jax,examples] @ git+https://github.com/NetDLab/pyDDRV"
jupyter lab
```

The `examples` extra installs matplotlib and JupyterLab. Start `jupyter lab`
from the activated environment so that the notebooks use it. In a clone of the
repository, install with `pip install -e ".[jax,examples]"` instead.

The notebooks are stored without output; run all cells to produce the results
and figures. Notebook 01 runs without JAX; the others need it.
