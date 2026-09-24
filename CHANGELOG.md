# Changelog

Changes to pyDDRV, by version. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed
- `verify_roa(trim=True)` ran one Trim pass against the first-pass region and
  reported a smaller region that was never checked against itself, so a cube
  could be certified by landing in the part that pass removed. For the linear
  spiral `A = [[0, 2], [-1, -1]]` at `alpha = 1`, above the spectral abscissa
  0.5, this kept a quarter of the box. The Trim pass is now repeated until it
  changes nothing: every trajectory from the returned region lands, at its
  return time, in the region or in `B_eps`, so it reaches `B_eps`.
  - `B_eps` counts as a landing zone (`find_alpha_roa_fused(trim_eps_ball=True)`).
    Without it no nonempty region passes its own check.
  - `RoAReport.trim_passes`, `trim_converged` and `trim_history`;
    `max_trim_passes` (default 30) bounds the loop, with a warning if reached.
    `max_seconds` limits the first pass only.
  - A warning when the raster (`raster_n`) is coarse relative to `eps`.
- The region raster used by Trim ignored the edge of `Q_R` (cells there
  overstated their distance to the outside, and a region filling the box got
  no usable distances) and was sensitive to the float32 rounding of stored
  cubes. Both are fixed.
- The documented pendulum example certifies 82% of the box instead of 84%.
  About 0.5% of the old region was not closed under its own returns. The
  Kuramoto region of attraction (notebook 03) grows from 82.154% to 82.170%
  of the box.
- Extreme-value Lipschitz estimate on large boxes. For the damped pendulum
  written with `jax.numpy`, `verify_roa(..., R=3.0)` estimated `L = 1.78e6`
  and certified nothing; it now estimates `L = 0.6181` (exact value 0.618).
  - Finite-difference Jacobians (`numerical_jacobian` and the extreme-value
    sampler) use a step matched to the precision of the field. The fixed step
    `1e-6` gave errors near 0.15 for float32 fields. The estimated error of
    each sample is added to `L` (`EVTEstimate.resolution`).
  - The reverse-Weibull fit tries several starting points, and each bootstrap
    fit starts from the full-sample fit. SciPy's default start could converge
    to a poor local optimum or diverge.
  - When the fit is ill-posed (tied block maxima, KS rejection, failed or
    divergent bootstrap), `L` is the sample maximum plus a margin calibrated
    to hold with probability `rho` under the reverse-Weibull model, and
    `EVTEstimate.status` is `"fallback"` with a `reason`. Well-posed fits
    (`status == "fit"`) report `gamma + Phi^{-1}(rho) * se` as before.

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
