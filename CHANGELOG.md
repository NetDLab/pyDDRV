# Changelog

All notable changes to pyddrv are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-17

First public release.

### Certification
- `verify_stability`: guaranteed exponential decay rate on a box `Q_R` from
  simulated trajectories (Theorem-8 ball certificates, layered
  `O(3^d log(R/ε))` covering grid, adaptive refinement, anytime lower bound).
- `verify_roa`: certified inner approximation of a region of attraction at a
  target rate, with the two-pass Trim protocol.
- Fused JAX kernel (RK4 inside `lax.scan`, tiled, no stored trajectories);
  identical NumPy fallback; torch backend for Apple-MPS / CUDA.
- Trajectory-local contraction bound (`verification/contraction.py`) as a
  tighter alternative to the global `r e^{Lt}` inflation (torch backend).
- Large-run controls for the region-of-attraction grower: bounded frontier
  with sound eviction, disk-headroom guard, inner-first seeding, frontier
  priority modes, stall-proof plateau stop.

### Lipschitz estimation
- `L_method="auto"` (default): exact box-corner estimate when an analytic
  Jacobian is state-affine, otherwise the extreme-value (reverse-Weibull)
  high-probability upper bound of Knuth et al.; model-free variant from
  `(x, f(x))` data.

### Tooling
- `pyddrv.viz`: anytime curves, verified box with the covering grid, RoA cube
  maps, and 2-D slices of higher-dimensional regions.
- Four runnable examples with matching notebooks; `GETTING_STARTED.md`.
- Continuous integration on Linux (Python 3.10, 3.12) and Windows (3.11),
  with and without JAX.

[0.1.0]: https://github.com/NetDLab/pyddrv/releases/tag/v0.1.0
