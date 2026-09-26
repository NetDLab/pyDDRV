r"""The two main functions of pyDDRV.

- :func:`verify_stability` certifies an exponential decay rate ``alpha`` for
  ``V(x) = ||x - x*||`` on the box ``Q_R`` around an equilibrium ``x*``.
- :func:`verify_roa` certifies, for a given rate ``alpha``, an inner
  approximation of the region of attraction as a union of cubes.

Both simulate a batched vector field ``f: (N, d) -> (N, d)`` supplied by the
user. A field written with ``jax.numpy`` runs on the compiled JAX kernel; a
field written with NumPy runs on a NumPy kernel with the same results
(``verify_stability`` only). The lower-level functions are in
:mod:`pyddrv.verification`.

The computations assume the equilibrium is at the origin. With
``equilibrium=x_star`` the field is shifted internally, and results are
reported in the original coordinates.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .systems.examples import simulate
from .verification.estimate import (
    LipschitzEstimate,
    estimate_L,
    estimate_L_roa,
)
from .verification.fused import (
    FusedAlphaMinResult,
    FusedRoAResult,
    _has_jax,
    find_alpha_min_fused,
    find_alpha_min_ladder,
    find_alpha_roa_fused,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def shifted_field(f: Callable, equilibrium) -> Callable:
    """Wrap a batched field so the equilibrium moves to the origin.

    Returns ``g(y) = f(y + x*)``, i.e. ``g`` in the shifted coordinates
    ``y = x - x*``. Works with NumPy, JAX, and torch batches (the shift is
    cast to the incoming array type).
    """
    xs = np.asarray(equilibrium, dtype=float)

    def g(y):
        s = xs
        if type(y).__module__.startswith("torch"):
            import torch
            s = torch.as_tensor(xs, dtype=y.dtype, device=y.device)
        return f(y + s)

    return g


def shifted_jacobian(jac: Callable, equilibrium) -> Callable:
    """Shift an analytic Jacobian ``x -> (d, d)`` the same way."""
    xs = np.asarray(equilibrium, dtype=float)
    return lambda y: jac(np.asarray(y, dtype=float) + xs)


def _rk4_rollout(f: Callable) -> Callable:
    """Batched RK4 rollout ``(points (N,d), t_grid (T,)) -> (N, T, d)`` from a
    batched NumPy-compatible field (used for the Lipschitz estimation passes,
    which need materialized trajectories)."""

    def rollout(points, t_grid):
        t = np.asarray(t_grid, dtype=float)
        dt = float(t[1] - t[0])
        return simulate(f, np.asarray(points, dtype=float), dt, len(t))

    return rollout


def _probe_backend(f: Callable, d: int, backend: str, *,
                   jit_fallback: bool = False) -> str:
    """Resolve ``backend="auto"``: use JAX only if ``f`` is a JAX field.

    A field written with ``jax.numpy`` returns a ``jax.Array`` even for a NumPy
    input, so ``f`` is first evaluated on a small NumPy batch and the type of
    the output is checked. No exception is raised for a NumPy field. (An
    earlier version tried ``jax.jit`` and caught the resulting
    ``TracerArrayConversionError``; debuggers set to stop on raised exceptions,
    such as VS Code's, then paused inside the library.) A field that returns a
    JAX array is then checked to trace under jit.

    ``jit_fallback=True`` (used by :func:`verify_roa`, which has no NumPy
    kernel) additionally attempts jit on ndarray-returning fields: operator-only
    NumPy fields such as ``x @ A.T`` trace fine and can still use the fused
    kernel; ``np.sin``-style fields raise inside the attempt (caught) and
    resolve to "numpy"."""
    if backend != "auto":
        return backend
    if not _has_jax():
        return "numpy"
    import jax

    try:
        is_jax = isinstance(f(np.zeros((2, d))), jax.Array)
    except Exception:
        return "numpy"
    if not is_jax and not jit_fallback:
        return "numpy"                          # plain-NumPy field: no jit try
    try:
        import jax.numpy as jnp

        out = jax.jit(f)(jnp.zeros((2, d)))
        if tuple(out.shape) != (2, d):
            raise ValueError(f"field returned shape {out.shape}, expected (2, {d})")
        return "jax"
    except Exception:
        return "numpy"


def _default_eps(R: float) -> float:
    return R / 100.0


def _default_steps(tau: float) -> int:
    return max(120, int(round(50.0 * tau)))


# --------------------------------------------------------------------------- #
# stability
# --------------------------------------------------------------------------- #
@dataclass
class StabilityReport:
    """Result of :func:`verify_stability`.

    ``alpha`` is the certified rate. Every trajectory starting in
    ``Q_R \\ B_eps`` (relative to the equilibrium) satisfies
    ``min_{0<s<=tau} e^{alpha s} V(x(t+s)) <= V(x(t))``, and hence
    ``V(x(t)) <= C e^{-alpha t} V(x(0))`` until it enters ``B_eps``. This holds
    provided ``L`` bounds the one-sided Lipschitz constant; see ``lipschitz``.
    A negative or infinite ``alpha`` means no rate was certified, which does
    not show that the equilibrium is unstable.
    """

    alpha: float                    # certified rate
    alpha_upper: float              # rate certified at the cube centers alone
    suboptimality: float            # relative gap at the binding cube
    converged: bool                 # gap <= delta (False: budget-limited)
    L: float                        # one-sided Lipschitz constant used
    discretization_ok: bool         # boundary samples cover the reachable set
    equilibrium: np.ndarray
    norm: str
    R: float
    eps: float
    tau: float
    n_boxes: int
    refinements: int
    backend: str
    lipschitz: LipschitzEstimate = field(repr=False, default=None)
    result: FusedAlphaMinResult = field(repr=False, default=None)
    trace: list = field(repr=False, default=None)

    @property
    def certified(self) -> bool:
        """True when a positive rate is certified and the discretization
        check validates the continuum claim."""
        return bool(np.isfinite(self.alpha) and self.alpha > 0
                    and self.discretization_ok)

    def summary(self) -> str:
        if self.certified:
            head = (f"certified: alpha >= {self.alpha:.4f} "
                    f"(ceiling {self.alpha_upper:.4f}, "
                    f"gap {self.suboptimality:.1%})")
        else:
            why = ("discretization check failed" if not self.discretization_ok
                   else "no positive rate certified")
            head = f"NOT certified ({why}; best lower bound {self.alpha:.4f})"
        return (f"verify_stability[{self.norm}-norm, R={self.R:g}, "
                f"eps={self.eps:g}, tau={self.tau:g}]: {head} | "
                f"L={self.L:.3g}, {self.n_boxes} cubes, "
                f"{self.refinements} refinements, backend={self.backend}")


def verify_stability(
    f: Callable,
    R: float,
    d: int,
    *,
    jac: Optional[Callable] = None,
    L: Optional[float] = None,
    eps: Optional[float] = None,
    norm: str = "2",
    tau: float = 5.0,
    equilibrium=None,
    method: str = "fused",
    L_method: str = "auto",
    rho: float = 0.95,
    n_steps: Optional[int] = None,
    delta: float = 0.1,
    max_refine: int = 12,
    max_seconds: Optional[float] = None,
    backend: str = "auto",
    record_trace: bool = False,
    **kwargs,
) -> StabilityReport:
    r"""Certify an exponential decay rate on ``Q_R`` by simulating ``f``.

    Parameters
    ----------
    f : callable
        Batched vector field ``(N, d) -> (N, d)``. A field written with
        ``jax.numpy`` runs on the compiled JAX kernel; a field written with
        NumPy runs on the NumPy kernel.
    R : float
        Half-width of the box ``Q_R = {||x - x*||_inf <= R}``.
    d : int
        State dimension.
    jac : callable, optional
        Analytic Jacobian ``x -> (d, d)``. If it is affine in the state, the
        one-sided Lipschitz constant is computed exactly at the box corners.
        Otherwise a numerical Jacobian is used.
    L : float, optional
        Upper bound on the one-sided Lipschitz constant of ``f`` over the
        states visited by trajectories from ``Q_R``, for example a bound
        derived by hand. If omitted, the reachable set is estimated from
        trajectories started on the boundary of ``Q_R``, ``L`` is estimated
        over it (see ``L_method``), and a check that the boundary samples
        cover all trajectories is recorded in ``discretization_ok``.
    eps : float, optional
        Radius of the ball ``B_eps`` around the equilibrium that is excluded
        from the certificate. Default ``R/100``.
    norm : {"2", "inf", "1"}
        Norm used for ``V``.
    tau : float
        Recurrence horizon: the time within which ``V`` must return to a
        smaller value. For a fixed ``L``, a longer horizon cannot lower the
        certified rate; the cost grows in proportion to ``tau``.
    equilibrium : array_like, optional
        Equilibrium ``x*``, if it is not the origin.
    method : {"fused", "ladder"}
        ``"fused"`` uses the horizon ``tau`` for every cube. ``"ladder"``
        starts each cube at ``tau/4`` and extends it to ``tau/2`` and ``tau``
        only for cubes that limit the rate, which is usually cheaper.
    L_method : {"auto", "corners", "evt"}
        How ``L`` is estimated when it is not given. ``"corners"`` evaluates
        the matrix measure of the Jacobian at the corners of the box. This is
        exact when the Jacobian is affine in the state and can underestimate
        ``L`` otherwise. ``"evt"`` samples the matrix measure inside the box
        and fits an extreme-value distribution to the sampled maxima (Knuth et
        al.); the resulting bound holds with probability ``rho``. ``"auto"``
        (default) uses ``"corners"`` when ``jac`` is given and passes a sampled
        test of affinity, and ``"evt"`` otherwise. The choice and the fit
        diagnostics are stored in ``report.lipschitz``.
    rho : float
        Probability with which the ``"evt"`` estimate exceeds the true
        constant (default 0.95). A higher value gives a larger ``L`` and a
        lower certified rate.
    delta : float
        Refinement stops once the certified rate is within this relative gap
        of ``alpha_upper``, the rate certified at the cube centers.
    max_refine, max_seconds :
        Limits on refinement rounds and wall-clock time. The rate reported
        when a limit is reached is valid, but may be lower than a longer run
        would give.
    record_trace : bool
        Record ``[(seconds, alpha, n_cubes), ...]`` after each refinement
        round.
    **kwargs :
        Forwarded to the underlying
        :func:`pyddrv.verification.find_alpha_min_fused` /
        :func:`~pyddrv.verification.find_alpha_min_ladder`.

    Returns
    -------
    StabilityReport
    """
    eps = _default_eps(R) if eps is None else float(eps)
    n_steps = _default_steps(tau) if n_steps is None else int(n_steps)
    if equilibrium is not None:
        f = shifted_field(f, equilibrium)
        jac = shifted_jacobian(jac, equilibrium) if jac is not None else None
        eq = np.asarray(equilibrium, dtype=float)
    else:
        eq = np.zeros(d)

    backend = _probe_backend(f, d, backend)

    if L is None:
        est = estimate_L(_rk4_rollout(f), R, eps, d, norm=norm, tau=tau,
                         jac=jac, f=f, L_method=L_method, rho=rho)
        if not np.isfinite(est.L):
            return StabilityReport(
                alpha=float("-inf"), alpha_upper=float("-inf"),
                suboptimality=float("inf"), converged=False, L=est.L,
                discretization_ok=False, equilibrium=eq, norm=norm, R=R,
                eps=eps, tau=tau, n_boxes=0, refinements=0, backend=backend,
                lipschitz=est)
        if not est.discretization_ok:
            warnings.warn(
                "The boundary trajectories used to estimate the reachable set "
                "could not be shown to cover all trajectories from the box, even "
                "on the finest boundary grid tried. The result is reported but "
                "not certified (report.certified is False).",
                stacklevel=2)
        L_val = est.L
    else:
        est = LipschitzEstimate(L=float(L), R_bar=float("nan"),
                                R_max=float("nan"), discretization_ok=True)
        L_val = float(L)

    if method == "ladder":
        res = find_alpha_min_ladder(
            f, L_val, R, eps, d, norm=norm, tau_max=tau, n_levels=3,
            steps_per_unit=max(1, int(round(n_steps / tau))), delta=delta,
            max_refine=max_refine, backend=backend, max_seconds=max_seconds,
            record_trace=record_trace, **kwargs)
    elif method == "fused":
        res = find_alpha_min_fused(
            f, L_val, R, eps, d, norm=norm, tau=tau, n_steps=n_steps,
            delta=delta, max_refine=max_refine, backend=backend,
            max_seconds=max_seconds, record_trace=record_trace, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; use 'fused' or 'ladder'")

    return StabilityReport(
        alpha=res.alpha_certified,
        alpha_upper=res.alpha_upper,
        suboptimality=res.suboptimality,
        converged=res.converged,
        L=L_val,
        discretization_ok=bool(est.discretization_ok),
        equilibrium=eq,
        norm=norm, R=float(R), eps=eps, tau=float(tau),
        n_boxes=res.n_boxes, refinements=res.refinements,
        backend=res.backend,
        lipschitz=est, result=res, trace=res.trace)


# --------------------------------------------------------------------------- #
# region of attraction
# --------------------------------------------------------------------------- #
@dataclass
class RoAReport:
    """Result of :func:`verify_roa`: an inner approximation of the region of
    attraction as a union of cubes, with centers ``centers`` and half-widths
    ``halfs`` in the original coordinates.

    With ``trim=True`` and ``trim_converged``, every trajectory starting in
    the region reaches ``B_eps``, and its norm drops by at least
    ``e^{-alpha t}`` at each return. ``n_tested`` counts the cubes evaluated
    over all passes."""

    alpha: float
    volume: float                   # total volume of the certified union
    n_certified: int
    n_tested: int
    equilibrium: np.ndarray
    norm: str
    R: float
    eps: float
    tau: float
    L: float
    discretization_ok: bool
    trim: bool
    stop_reason: str
    trim_passes: int = 0            # Trim passes run after the first pass
    trim_converged: bool = False    # last Trim pass changed nothing
    centers: np.ndarray = field(repr=False, default=None)   # (N, d)
    halfs: np.ndarray = field(repr=False, default=None)     # (N,)
    depths: np.ndarray = field(repr=False, default=None)    # split depth
    lipschitz: LipschitzEstimate = field(repr=False, default=None)
    result: FusedRoAResult = field(repr=False, default=None)
    trace: list = field(repr=False, default=None)
    # (n_certified, volume) after the first pass and after each Trim pass
    trim_history: list = field(repr=False, default=None)

    def summary(self) -> str:
        frac = self.n_certified / self.n_tested if self.n_tested else 0.0
        if not self.trim:
            trim = "Trim=False"
        elif self.trim_converged:
            trim = f"Trim=True (fixed point after {self.trim_passes} passes)"
        else:
            trim = (f"Trim=True (NOT a fixed point after {self.trim_passes} "
                    f"passes)")
        return (f"verify_roa[alpha={self.alpha:g}, {self.norm}-norm, "
                f"R={self.R:g}, tau={self.tau:g}]: {self.n_certified} cubes "
                f"({frac:.1%} of {self.n_tested} tested), volume {self.volume:.4g}"
                f" | L={self.L:.3g}, {trim}, stop={self.stop_reason}")


def verify_roa(
    f: Callable,
    R: float,
    d: int,
    alpha: float,
    *,
    jac: Optional[Callable] = None,
    L: Optional[float] = None,
    eps: Optional[float] = None,
    norm: str = "2",
    tau: float = 2.0,
    equilibrium=None,
    n_steps: Optional[int] = None,
    max_refine: int = 5,
    trim: bool = False,
    max_trim_passes: int = 30,
    raster_n: int = 729,
    max_seconds: Optional[float] = None,
    plateau_rel: Optional[float] = None,
    backend: str = "auto",
    spill_dir: Optional[str] = None,
    verbose: bool = False,
    L_method: str = "auto",
    rho: float = 0.95,
    **kwargs,
) -> RoAReport:
    r"""Certify an inner approximation of the region of attraction.

    Returns the cubes in ``Q_R`` for which the recurrence condition at rate
    ``alpha`` is certified, given the bound ``L``. Requires the compiled
    kernel: a field written with ``jax.numpy`` (install the ``jax`` extra), or
    a PyTorch field with ``backend="torch"``.

    The parameters are those of :func:`verify_stability`, plus:

    trim : bool
        If false, a cube is accepted when the decay condition holds for the
        whole cube: for some return time ``t <= tau``, every trajectory from
        the cube has ``e^{alpha t} ||x(t)|| <= ||x(0)||``. If true, the
        returned region is also closed under these returns: at the return
        time, trajectories from each cube lie inside the returned region or
        inside ``B_eps``. Every trajectory from the region then reaches
        ``B_eps``, and its norm drops by at least ``e^{-alpha t}`` at each
        return. The first pass is followed by Trim passes, each checking the
        previous region against itself and dropping or splitting cubes that
        fail, until a pass changes nothing (``trim_converged``). The region
        is rasterized with ``raster_n`` cells per axis for this check; the
        cells must be small compared to ``eps`` (a warning is issued
        otherwise), because trajectories that end in ``B_eps`` have to land
        on cells that lie entirely inside it.
    max_trim_passes : int
        Upper bound on the number of Trim passes. If it is reached before a
        pass changes nothing, a warning is issued, ``trim_converged`` is
        false, and the returned region is not certified against itself.
    max_seconds : float, optional
        Time budget for the first pass. The Trim passes run to completion.
    plateau_rel : float, optional
        Stop when the certified volume has grown by less than this fraction
        over the last doubling of the elapsed time, and the volume expected
        from the cubes still queued is also below this fraction.
    spill_dir : str, optional
        Write certified cubes to files in this directory instead of keeping
        them in memory. Needed for runs that certify more cubes than fit in
        memory.
    **kwargs :
        Options for long runs, passed to
        :func:`pyddrv.verification.find_alpha_roa_fused`: ``priority``
        (``"gain"``, ``"norm"`` or ``"sizedist"``), ``inner_first`` and
        ``inner_first_min_h``, ``max_pending_parents`` (limits memory by
        discarding the least promising cubes, which then remain uncertified),
        ``min_disk_gb`` and ``disk_check_every`` (stop before the disk fills,
        with ``spill_dir``), and ``local_jac`` and ``local_M`` (a bound on
        trajectory separation computed along each trajectory, torch backend
        only). See that function's docstring.
    """
    eps = _default_eps(R) if eps is None else float(eps)
    n_steps = _default_steps(tau) if n_steps is None else int(n_steps)
    if equilibrium is not None:
        f = shifted_field(f, equilibrium)
        jac = shifted_jacobian(jac, equilibrium) if jac is not None else None
        eq = np.asarray(equilibrium, dtype=float)
    else:
        eq = np.zeros(d)

    backend = _probe_backend(f, d, backend, jit_fallback=True)
    if backend == "numpy":
        raise RuntimeError(
            "verify_roa needs the fused JAX kernel (pip install \"pyddrv[jax]\") "
            "with a jit-compatible field, or backend='torch' with a torch field. "
            "For a pure-NumPy fallback use "
            "pyddrv.verification.find_alpha_roa (rollout-based, slower).")

    if L is None:
        est = estimate_L_roa(_rk4_rollout(f), R, d, norm=norm, tau=tau,
                             jac=jac, f=f, L_method=L_method, rho=rho)
        if not np.isfinite(est.L):
            return RoAReport(alpha=float(alpha), volume=0.0, n_certified=0,
                             n_tested=0, equilibrium=eq, norm=norm, R=R,
                             eps=eps, tau=tau, L=est.L,
                             discretization_ok=False, trim=trim,
                             stop_reason="lipschitz-diverged", lipschitz=est)
        L_val = est.L
    else:
        est = LipschitzEstimate(L=float(L), R_bar=float("nan"),
                                R_max=float("nan"), discretization_ok=True)
        L_val = float(L)

    res = find_alpha_roa_fused(
        f, L_val, R, eps, d, alpha, norm=norm, tau=tau, n_steps=n_steps,
        max_refine=max_refine, backend=backend, max_seconds=max_seconds,
        plateau_rel=plateau_rel, spill_dir=spill_dir, verbose=verbose,
        **kwargs)

    def _volume(r):
        h = np.asarray(r.halfs, dtype=float)
        return float(np.sum((2.0 * h) ** d)) if h.size else 0.0

    # Trim: repeat the pass against the previous result until a pass changes
    # nothing, i.e. every cube of the region certifies against the region
    # itself (plus B_eps) without being split. Regions only shrink from pass
    # to pass. max_seconds bounds the first pass only.
    stop_reason = res.stop_reason
    n_tested = res.n_tested
    history = [(res.n_certified, _volume(res))]
    passes, converged = 0, False
    if trim and 2.0 * R / raster_n > eps / 2.0:
        warnings.warn(
            f"verify_roa: raster cells (width {2.0 * R / raster_n:.3g}) are "
            f"coarse relative to eps={eps:.3g}, so few cells fit inside B_eps "
            f"and the Trim passes may shrink the region to nothing. Use "
            f"raster_n >= {int(np.ceil(4.0 * R / eps))}.", stacklevel=2)
    if trim:
        while res.n_certified and passes < max_trim_passes:
            n_in = res.n_certified
            region = (np.asarray(res.centers), np.asarray(res.halfs))
            res = find_alpha_roa_fused(
                f, L_val, R, eps, d, alpha, norm=norm, tau=tau,
                n_steps=n_steps, max_refine=max_refine, backend=backend,
                trim_region=region,
                grid0=region + (np.asarray(res.depths, dtype=int),),
                raster_n=raster_n, verbose=verbose, **kwargs)
            passes += 1
            n_tested += res.n_tested
            history.append((res.n_certified, _volume(res)))
            if res.n_tested == n_in and res.n_certified == n_in:
                converged = True
                break
        converged = converged or res.n_certified == 0
        if not converged:
            warnings.warn(
                f"verify_roa: the Trim passes did not reach a fixed point "
                f"within max_trim_passes={max_trim_passes}; the returned "
                f"region is not certified against itself.", stacklevel=2)

    halfs = np.asarray(res.halfs, dtype=float)
    volume = _volume(res)
    centers = (np.asarray(res.centers, dtype=float) + eq
               if res.n_certified else np.empty((0, d)))
    return RoAReport(
        alpha=float(alpha), volume=volume,
        n_certified=res.n_certified, n_tested=n_tested,
        equilibrium=eq, norm=norm, R=float(R), eps=eps, tau=float(tau),
        L=L_val, discretization_ok=bool(est.discretization_ok),
        trim=bool(trim), stop_reason=stop_reason,
        trim_passes=passes, trim_converged=converged,
        centers=centers, halfs=halfs, depths=np.asarray(res.depths),
        lipschitz=est, result=res, trace=res.trace, trim_history=history)
