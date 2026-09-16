r"""High-level, one-call verification API.

This module is the recommended entry point for library users. It wires the
pieces of the pipeline together with sound defaults:

1. :func:`verify_stability` -- certify a guaranteed exponential decay rate
   ``alpha`` for ``V(x) = ||x - x*||`` over the box ``Q_R`` around the
   equilibrium ``x*`` (Algorithm 1: one-sided Lipschitz estimation, layered
   covering grid, Theorem-8 ball certificates, adaptive refinement).
2. :func:`verify_roa` -- given a target rate ``alpha``, grow a certified
   inner approximation of the region of attraction as a union of cubes
   (Algorithm 2).

Both accept a *batched* vector field ``f : (N, d) -> (N, d)``. If the field is
written with backend-agnostic ops (or ``jax.numpy``), the fused JAX kernel is
used automatically; otherwise the NumPy kernel runs (same certificates, slower).
Everything low-level remains available in :mod:`pyddrv.verification`.

Coordinates: all internal machinery assumes the equilibrium at the origin.
Passing ``equilibrium=x_star`` shifts the field for you, and reported
regions/cubes are shifted back to original coordinates.
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

    A field written with ``jax.numpy`` ops returns a ``jax.Array`` even for a
    NumPy input, so we first evaluate ``f`` on a small NumPy batch and check
    the output type -- **no exception is raised for the common plain-NumPy
    case** (an earlier version probed by attempting ``jax.jit`` and catching
    ``TracerArrayConversionError``, which worked but made debuggers configured
    to break on raised exceptions -- e.g. VS Code -- pause inside the library
    on the very first example). A field that returned a JAX array is then
    confirmed to trace under jit.

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
# stability (Algorithm 1)
# --------------------------------------------------------------------------- #
@dataclass
class StabilityReport:
    """Outcome of :func:`verify_stability`.

    ``alpha`` is the *guaranteed* exponential rate: every trajectory starting
    in ``Q_R \\ B_eps`` (coordinates relative to the equilibrium) satisfies the
    tau-recurrence ``min_{0<s<=tau} e^{alpha s} V(x(t+s)) <= V(x(t))``, which
    yields ``V(x(t)) <= C e^{-alpha t} V(x(0))`` until ``B_eps`` is reached.
    A negative/-inf ``alpha`` means no rate was certified (not a disproof).
    """

    alpha: float                    # certified rate (the guarantee)
    alpha_upper: float              # data-driven ceiling (best achievable here)
    suboptimality: float            # relative gap at the binding cube
    converged: bool                 # gap <= delta (False: budget-limited)
    L: float                        # one-sided Lipschitz constant used
    discretization_ok: bool         # eq-(38) continuum check passed
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
    r"""Certify an exponential decay rate over ``Q_R`` from simulated data.

    Parameters
    ----------
    f : callable
        Batched vector field ``(N, d) -> (N, d)``. Written with
        backend-agnostic ops (or ``jax.numpy``) it runs on the fused JAX
        kernel; plain-NumPy fields automatically use the NumPy kernel.
    R : float
        Inf-norm radius of the verification box ``Q_R`` (about the
        equilibrium).
    d : int
        State dimension.
    jac : callable, optional
        Analytic Jacobian ``x -> (d, d)``. Makes the one-sided Lipschitz
        constant exact at box corners for fields with state-affine Jacobian;
        otherwise a numerical Jacobian is used.
    L : float, optional
        One-sided Lipschitz constant over the reachable set, if you already
        have one (e.g. a closed-form bound, or
        :func:`pyddrv.lipschitz.one_sided_lipschitz_from_data` from
        experimental data). If omitted it is estimated from boundary
        trajectories (paper §IX-B), including the discretization-robustness
        check.
    eps : float, optional
        Radius of the excluded ball ``B_eps`` around the equilibrium (the
        certificate cannot extend to the equilibrium itself). Default
        ``R/100``.
    norm : {"2", "inf", "1"}
        Working norm for ``V``.
    tau : float
        Recurrence horizon: trajectories may wander for up to ``tau`` time
        units before ``V`` must have dipped. Larger tau never hurts the rate
        but costs compute.
    equilibrium : array_like, optional
        Equilibrium ``x*`` if not the origin. The field is shifted
        internally.
    method : {"fused", "ladder"}
        ``"fused"`` = Algorithm 1 at fixed horizon tau. ``"ladder"`` =
        per-cube horizon escalation up to ``tau`` (cheaper at matched
        accuracy; horizons ``tau/4, tau/2, tau``).
    L_method : {"auto", "corners", "evt"}
        How the one-sided Lipschitz constant is estimated when ``L`` is not
        supplied. ``"corners"`` evaluates the matrix measure at the box corners
        -- exact only when the Jacobian is affine in the state (the paper's
        bilinear systems), and an UNDER-estimate (unsound) for general
        nonlinear fields. ``"evt"`` uses the extreme-value (reverse-Weibull)
        estimator of Knuth et al. over interior samples, a high-probability
        upper bound that is the sound choice for nonlinear fields. ``"auto"``
        (default) picks ``"corners"`` only when an analytic ``jac`` is given
        and passes a sampled affinity test, and ``"evt"`` otherwise. The
        resolved choice and diagnostics live in ``report.lipschitz``
        (``.method``, ``.evt`` with fitted ``gamma``, KS p-value, ``validated``).
    rho : float
        Confidence level for ``L_method="evt"``: the estimated ``L``
        over-estimates the true constant with probability ``rho`` (default
        0.95). Higher ``rho`` -> larger, safer ``L`` (more conservative rate).
    delta : float
        Relative sub-optimality target; refinement stops once the certified
        rate is within ``delta`` of the data-driven ceiling.
    max_refine, max_seconds :
        Refinement-round / wall-clock budgets (the result is an *anytime*
        lower bound; stopping early is sound, just conservative).
    record_trace : bool
        Record the anytime curve ``[(seconds, alpha, n_cubes), ...]``.
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
                "estimate_L: discretization-robustness check (eq. 38) did not "
                "pass at the densest boundary grid tried; the continuum "
                "containment claim is unverified (report.certified is False).",
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
# region of attraction (Algorithm 2)
# --------------------------------------------------------------------------- #
@dataclass
class RoAReport:
    """Outcome of :func:`verify_roa`: a certified inner approximation of the
    region of attraction as a union of cubes (centers/halfs in ORIGINAL
    coordinates; subtract ``equilibrium`` to recover the internal shifted
    frame)."""

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
    centers: np.ndarray = field(repr=False, default=None)   # (N, d)
    halfs: np.ndarray = field(repr=False, default=None)     # (N,)
    depths: np.ndarray = field(repr=False, default=None)    # split depth
    lipschitz: LipschitzEstimate = field(repr=False, default=None)
    result: FusedRoAResult = field(repr=False, default=None)
    trace: list = field(repr=False, default=None)

    def summary(self) -> str:
        frac = self.n_certified / self.n_tested if self.n_tested else 0.0
        return (f"verify_roa[alpha={self.alpha:g}, {self.norm}-norm, "
                f"R={self.R:g}, tau={self.tau:g}]: {self.n_certified} cubes "
                f"({frac:.1%} of {self.n_tested} tested), volume {self.volume:.4g}"
                f" | L={self.L:.3g}, Trim={self.trim}, stop={self.stop_reason}")


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
    r"""Grow a certified inner approximation of the region of attraction.

    Every cube in the returned union satisfies the recurrence condition at
    rate ``alpha`` (Theorem 8), so trajectories from the union converge to
    ``B_eps`` around the equilibrium at rate ``alpha``.

    Requires the fused JAX kernel (or ``backend="torch"`` with a torch
    field); install the ``jax`` extra.

    Parameters largely mirror :func:`verify_stability` (including
    ``L_method``/``rho`` for the Lipschitz estimate). Extra knobs:

    trim : bool
        Run the paper's two-pass protocol: pass 1 grows a tentative region
        from the decay condition alone; pass 2 re-certifies it enforcing that
        certifying trajectories remain inside the *grown region itself*
        (checked against the exact union via a rasterized inner distance
        map, ``raster_n`` cells per axis). Sound RoA claims with non-inf
        working norms generally need this.
    plateau_rel : float, optional
        Stall-proof plateau stop (e.g. ``1e-3``): stop when certified volume
        growth over the last time-doubling AND the expected one-split gain
        still queued both fall below this fraction.
    spill_dir : str, optional
        Stream certified cubes to disk (memmap-backed result) -- required
        for very long runs where the union outgrows RAM.
    **kwargs :
        Large-run controls forwarded to
        :func:`pyddrv.verification.find_alpha_roa_fused`: ``priority``
        (``"gain"`` | ``"norm"`` | ``"sizedist"``), ``inner_first`` /
        ``inner_first_min_h``, ``max_pending_parents`` (bounded frontier with
        sound eviction), ``min_disk_gb`` / ``disk_check_every`` (disk guard
        with ``spill_dir``), and ``local_jac`` / ``local_M`` (trajectory-local
        contraction bound, torch backend). See that function's docstring.
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

    if trim and res.n_certified:
        res = find_alpha_roa_fused(
            f, L_val, R, eps, d, alpha, norm=norm, tau=tau, n_steps=n_steps,
            max_refine=max_refine, backend=backend, max_seconds=max_seconds,
            trim_region=(np.asarray(res.centers), np.asarray(res.halfs)),
            grid0=(np.asarray(res.centers), np.asarray(res.halfs),
                   np.asarray(res.depths, dtype=int)),
            raster_n=raster_n, verbose=verbose, **kwargs)

    halfs = np.asarray(res.halfs, dtype=float)
    volume = float(np.sum((2.0 * halfs) ** d)) if halfs.size else 0.0
    centers = (np.asarray(res.centers, dtype=float) + eq
               if res.n_certified else np.empty((0, d)))
    return RoAReport(
        alpha=float(alpha), volume=volume,
        n_certified=res.n_certified, n_tested=res.n_tested,
        equilibrium=eq, norm=norm, R=float(R), eps=eps, tau=float(tau),
        L=L_val, discretization_ok=bool(est.discretization_ok),
        trim=bool(res.trim), stop_reason=res.stop_reason,
        centers=centers, halfs=halfs, depths=np.asarray(res.depths),
        lipschitz=est, result=res, trace=res.trace)
