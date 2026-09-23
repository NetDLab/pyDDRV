# Changelog

Changes to pyDDRV, by version. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Documentation
- Logo in the README, with light and dark variants in `docs/assets/`.
- The project is named pyDDRV and the repository is `NetDLab/pyDDRV`. The
  package and import name remain `pyddrv`.
- README, getting-started guide, notebooks and docstrings rewritten. They now
  state that the method needs a simulator for the vector field, that estimated
  Lipschitz constants give a guarantee that holds with probability `rho`, that
  integration error is not included in the certificate, and that
  `alpha_upper` is the rate certified at the cube centers of the current grid,
  not a bound on the rate of the system. All code examples were checked to run.

## [0.1.0](https://github.com/NetDLab/pyDDRV/releases/tag/v0.1.0) (2026-09-17)

First public release.

### Certification
- `verify_stability`: certified exponential decay rate on a box around an
  equilibrium, computed from simulated trajectories with adaptive refinement of
  a cube grid.
- `verify_roa`: inner approximation of the region of attraction at a given
  rate. With `trim=True`, a second pass also requires certifying trajectories
  to stay inside the region found by the first pass.
- Compiled JAX kernel, NumPy implementation with identical results, and a
  PyTorch backend for Apple and NVIDIA GPUs.
- Trajectory-local contraction bound (`verification/contraction.py`), which can
  replace the global Lipschitz bound in the PyTorch backend.
- Options for long region-of-attraction runs: frontier size limit, free-disk
  check, priority orderings, a stopping rule for stalled progress.

### Lipschitz estimation
- `L_method="auto"` (default): box corners when the analytic Jacobian is affine
  in the state, otherwise the extreme-value estimate of Knuth et al.
- Estimation from measured states and derivatives.

### Other
- Plotting in `pyddrv.viz`.
- Four examples with matching notebooks, and `GETTING_STARTED.md`.
- Tests on Linux (Python 3.10, 3.12) and Windows (3.11), with and without JAX.
