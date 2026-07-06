r"""Fused, tiled, float32 accelerator for the per-box Theorem-8 alpha-maximization.

This is an *additive* fast path for the layered RLF certifier. It computes exactly
the same quantity as :func:`pyddrv.verification.ball.alpha_max` -- the best
certified rate ``alpha_max(x, r)`` over the return time ``t in (0, tau]`` for a
batch of box centres -- but with three structural wins:

1. **Fusion / no T-axis.** Instead of materializing ``rollout(centers) -> (N,T,d)``
   and then reducing, each centre is integrated with RK4 *inside* a ``lax.scan``
   (JAX) or a python step-loop (numpy) that carries only ``(state, running-max
   alpha_lo, running-max alpha_up)``. Peak memory is ``O(N*d)`` instead of
   ``O(N*T*d)``, and both radii (``r = c*h`` for the cube bound ``alpha_lo`` and
   ``r = 0`` for the centre bound ``alpha_up``) are done in one integration pass.
3. **Tiling.** The kernel is mapped over box-tiles of ``tile_size`` so peak memory
   is ``O(tile_size*d)`` regardless of how large the refined grid grows.
4. **Dtype.** ``dtype="float32"`` by default; ``exp``/``log``/norm run in that
   dtype. ``float64`` is supported via the numpy backend (JAX here has x64 off).
5. **Cheaper t-max.** The max over ``t`` happens step-by-step in the scan -- no
   length-``T`` alpha array, no post-hoc argmax. ``n_steps`` sets the cost.

The vector field ``f`` is taken **batched**: ``f(X:(N,d)) -> (N,d)`` (the repo's
:mod:`pyddrv.systems` convention). A single-state ``jnp`` field ``g(x:(d,))`` is
adapted with ``f = jax.vmap(g)``.

Nothing here changes :mod:`pyddrv.verification.algorithm1`; it is a parallel path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .grid import initial_grid, norm_equiv_c, split_many

_TINY64 = 1e-300


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #
def _norm_key(norm: str) -> str:
    key = str(norm).lower()
    if key in ("2", "l2", "euclidean"):
        return "2"
    if key in ("inf", "linf", "oo", "infinity"):
        return "inf"
    if key in ("1", "l1"):
        return "1"
    raise ValueError(f"unsupported norm {norm!r}")


def _norm_np(x, key: str):
    if key == "2":
        return np.sqrt(np.sum(x * x, axis=-1))
    if key == "inf":
        return np.max(np.abs(x), axis=-1)
    return np.sum(np.abs(x), axis=-1)


# --------------------------------------------------------------------------- #
# numpy backend: fused RK4 + running-max alpha over t, no T-axis
# --------------------------------------------------------------------------- #
def _span_tile_np(f, x_init, Vx, halfs, *, t_lo, dt, n_steps, L, R, c, enforce_S,
                  key, dtype):
    """Integrate the span ``(t_lo, t_lo + n_steps*dt]`` from state ``x_init`` and
    reduce alpha(t) on the fly. ``Vx = ||x(0)||`` is the norm at the ORIGINAL
    centre (t=0), so a span starting from a stored state continues the same
    certificate. Returns ``(a_lo, a_up, x_end)``."""
    dt = dtype(dt)
    x = np.asarray(x_init, dtype=dtype)
    r_lo = dtype(c) * np.asarray(halfs, dtype=dtype)             # (N,)
    Vx = np.asarray(Vx, dtype=dtype)
    neg = np.full(x.shape[0], -np.inf)
    a_lo, a_up = neg.copy(), neg.copy()
    tiny = _TINY64 if dtype is np.float64 else np.finfo(dtype).tiny
    Rd, Ld, t0 = dtype(R), dtype(L), dtype(t_lo)

    for k in range(1, n_steps + 1):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        tk = t0 + dtype(k) * dt                                   # absolute time
        Vphi = _norm_np(x, key)
        phi_inf = np.max(np.abs(x), axis=-1)
        eLt = np.exp(Ld * tk)
        a_lo = np.maximum(a_lo, _alpha_step_np(Vx, Vphi, phi_inf, r_lo, tk, eLt,
                                               Rd, enforce_S, tiny))
        a_up = np.maximum(a_up, _alpha_step_np(Vx, Vphi, phi_inf, 0.0, tk, eLt,
                                               Rd, enforce_S, tiny))
    return (np.asarray(a_lo, dtype=float), np.asarray(a_up, dtype=float),
            np.asarray(x, dtype=float))


def _alpha_step_np(Vx, Vphi, phi_inf, r, tk, eLt, R, enforce_S, tiny):
    """alpha(t) at one return time for radius r (r scalar 0 or (N,) array)."""
    margin = Vx - r
    denom = Vphi + r * eLt
    valid = (margin > 0) & (denom > tiny)
    if enforce_S:
        valid = valid & (phi_inf <= R - r * eLt)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, margin / np.where(denom > 0, denom, 1.0), np.nan)
        a = np.log(ratio) / tk
    return np.where(valid, a, -np.inf)


# --------------------------------------------------------------------------- #
# JAX backend: same kernel inside lax.scan, jitted
# --------------------------------------------------------------------------- #
def _make_jax_kernel(f, key: str, enforce_S: bool):
    import jax
    import jax.numpy as jnp
    from jax import lax

    def jnorm(x):
        if key == "2":
            return jnp.sqrt(jnp.sum(x * x, axis=-1))
        if key == "inf":
            return jnp.max(jnp.abs(x), axis=-1)
        return jnp.sum(jnp.abs(x), axis=-1)

    def alpha_step(Vx, Vphi, phi_inf, r, tk, eLt, R):
        margin = Vx - r
        denom = Vphi + r * eLt
        valid = (margin > 0) & (denom > 0)
        if enforce_S:
            valid = valid & (phi_inf <= R - r * eLt)
        denom_s = jnp.where(denom > 0, denom, 1.0)
        a = jnp.log(jnp.where(valid, margin / denom_s, 1.0)) / tk
        return jnp.where(valid, a, -jnp.inf)

    @jax.jit
    def kernel(x_init, Vx, halfs, tk_arr, dt, L, R, c):
        # Span form: integrate from x_init over the (absolute) times tk_arr, with
        # Vx = ||x(0)|| of the ORIGINAL centre. For a fresh box x_init IS the
        # centre and Vx its norm; for a horizon continuation x_init is the stored
        # state at the previous rung and the certificate simply extends.
        x = x_init
        r_lo = c * halfs
        neg = jnp.full(Vx.shape, -jnp.inf)

        def body(carry, tk):
            x, a_lo, a_up = carry
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            Vphi = jnorm(xn)
            phi_inf = jnp.max(jnp.abs(xn), axis=-1)
            eLt = jnp.exp(L * tk)
            a_lo = jnp.maximum(a_lo, alpha_step(Vx, Vphi, phi_inf, r_lo, tk, eLt, R))
            a_up = jnp.maximum(a_up, alpha_step(Vx, Vphi, phi_inf, 0.0, tk, eLt, R))
            return (xn, a_lo, a_up), None

        (x_end, a_lo, a_up), _ = lax.scan(body, (x, neg, neg), tk_arr)
        return a_lo, a_up, x_end

    # (5) Tile-level early exit (stub). `lax.scan` always runs all n_steps. A
    # `lax.while_loop` could stop once *every* box in the tile has certified AND
    # further integration cannot raise min(a_lo) -- e.g. carry `min(a_lo)` and a
    # monotone upper bound on any remaining gain, break when the bound <= min(a_lo):
    #
    #   def cond(c): step, x, a_lo, a_up, done = c; return (step < n_steps) & ~done
    #   def body_w(c): ... ; done = jnp.all(a_lo > -inf) & (max_possible_gain <= 0)
    #   lax.while_loop(cond, body_w, (0, x, neg, neg, False))
    #
    # Per-box breaking does NOT vectorize (each box certifies at a different t), so
    # the gain is only tile-level and depends on tile homogeneity; left unimplemented.
    return kernel


def _span_tile_jax(kernel, x_init, Vx, halfs, *, t_lo, dt, n_steps, L, R, c, dtype):
    import jax.numpy as jnp
    # Pad the batch to the next power of two (>=256) so the jitted kernel sees only
    # a handful of leading-dim shapes and recompiles are reused across refinement
    # rounds. Padding states are 0 -> Vx=0 -> margin<0 -> -inf, sliced off below.
    N = len(x_init)
    P = 1 << max(8, (N - 1).bit_length()) if N > 0 else 0
    if P > N:
        x_init = np.concatenate([x_init, np.zeros((P - N, x_init.shape[1]))])
        Vx = np.concatenate([Vx, np.zeros(P - N)])
        halfs = np.concatenate([halfs, np.ones(P - N)])
    tk_arr = (t_lo + dt * jnp.arange(1, n_steps + 1)).astype(dtype)
    a_lo, a_up, x_end = kernel(
        jnp.asarray(x_init, dtype), jnp.asarray(Vx, dtype),
        jnp.asarray(halfs, dtype), tk_arr,
        jnp.asarray(dt, dtype), jnp.asarray(L, dtype), jnp.asarray(R, dtype),
        jnp.asarray(c, dtype))
    return (np.asarray(a_lo, dtype=float)[:N], np.asarray(a_up, dtype=float)[:N],
            np.asarray(x_end, dtype=float)[:N])


# --------------------------------------------------------------------------- #
# backend resolution + public bounds API
# --------------------------------------------------------------------------- #
def _has_jax() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def _resolve(backend: str, dtype):
    """Return ('jax'|'numpy'|'torch', numpy-dtype). float64 always routes to
    numpy (JAX here runs with x64 disabled; torch kernels are float32)."""
    npd = np.float32 if str(dtype) in ("float32", "f4") else np.float64
    if backend == "torch":
        return "torch", np.float32
    if backend == "numpy" or npd is np.float64 or not _has_jax():
        return "numpy", npd
    if backend in ("auto", "jax"):
        return "jax", np.float32
    raise ValueError(f"unknown backend {backend!r}")


def fused_alpha_span(f, x_init, Vx, halfs, *, t_lo, t_hi, n_steps, L, R, norm="2",
                     c=None, enforce_S=True, dtype="float32", tile_size=100_000,
                     backend="auto", _kernel=None):
    """Fused ``(alpha_lo, alpha_up, x_end)`` over the time span ``(t_lo, t_hi]``,
    integrating from ``x_init`` with ``Vx = ||x(0)||`` of the original centres.
    With ``t_lo=0, x_init=centers`` this is a fresh evaluation; with ``x_init``
    the stored states at ``t_lo`` it CONTINUES the certificate over the new span
    only (a horizon escalation re-integrates nothing). Tiled; ``x_end`` enables
    the next continuation."""
    x_init = np.asarray(x_init, dtype=float)
    Vx = np.asarray(Vx, dtype=float)
    halfs = np.asarray(halfs, dtype=float)
    key = _norm_key(norm)
    d = x_init.shape[1]
    if c is None:
        c = norm_equiv_c(norm, d)
    which, npd = _resolve(backend, dtype)
    if which == "jax":
        kernel = _kernel if _kernel is not None else _make_jax_kernel(f, key, enforce_S)
    elif which == "torch":
        if _kernel is not None:
            kernel = _kernel
        else:
            from .fused_torch import make_torch_span_kernel
            kernel = make_torch_span_kernel(f, key, enforce_S)
    else:
        kernel = None
    dt = (t_hi - t_lo) / n_steps
    los, ups, xes = [], [], []
    for s in range(0, len(x_init), tile_size):
        xs, vs, hs = x_init[s:s + tile_size], Vx[s:s + tile_size], halfs[s:s + tile_size]
        if which == "jax":
            lo, up, xe = _span_tile_jax(kernel, xs, vs, hs, t_lo=t_lo, dt=dt,
                                        n_steps=n_steps, L=L, R=R, c=c, dtype=npd)
        elif which == "torch":
            lo, up, xe = kernel(xs, vs, hs, float(t_lo), float(dt),
                                int(n_steps), float(L), float(R), float(c))
        else:
            lo, up, xe = _span_tile_np(f, xs, vs, hs, t_lo=t_lo, dt=dt,
                                       n_steps=n_steps, L=L, R=R, c=c,
                                       enforce_S=enforce_S, key=key, dtype=npd)
        los.append(lo)
        ups.append(up)
        xes.append(xe)
    if not los:
        return np.empty(0), np.empty(0), np.empty((0, d))
    return np.concatenate(los), np.concatenate(ups), np.concatenate(xes)


def fused_alpha_bounds(f, centers, halfs, *, tau, n_steps, L, R, norm="2", c=None,
                       enforce_S=True, dtype="float32", tile_size=100_000,
                       backend="auto", _kernel=None):
    """Fused ``(alpha_lo, alpha_up)`` for a batch of boxes -- equals
    ``ball.alpha_max`` at radii ``c*h`` and ``0`` respectively, computed without a
    T-axis and tiled to bound memory. ``f`` is a batched field ``(N,d)->(N,d)``.

    ``_kernel`` (internal) lets a caller pass a pre-built jitted kernel so it is
    compiled once and reused across many calls (e.g. the refinement loop)."""
    centers = np.asarray(centers, dtype=float)
    Vx = _norm_np(centers, _norm_key(norm))
    lo, up, _ = fused_alpha_span(f, centers, Vx, halfs, t_lo=0.0, t_hi=tau,
                                 n_steps=n_steps, L=L, R=R, norm=norm, c=c,
                                 enforce_S=enforce_S, dtype=dtype,
                                 tile_size=tile_size, backend=backend,
                                 _kernel=_kernel)
    return lo, up


def alpha_bounds_materialized_np(ys, t_grid, r, L, R, norm="2", enforce_S=True):
    """Materialized numpy reference: alpha_max for radius ``r`` from a full
    ``(N,T,d)`` trajectory array. An independent re-derivation of the (32a)/(32b)
    formula (does not call :mod:`ball`), for cross-checking the fused path."""
    ys = np.asarray(ys, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    key = _norm_key(norm)
    N = ys.shape[0]
    tk = t[1:]                                             # (T-1,)
    Vx = _norm_np(ys[:, 0, :], key)[:, None]              # (N,1)
    Vphi = _norm_np(ys[:, 1:, :], key)                    # (N,T-1)
    phi_inf = np.max(np.abs(ys[:, 1:, :]), axis=2)        # (N,T-1)
    r = np.broadcast_to(np.asarray(r, dtype=float), (N,))[:, None]
    eLt = np.exp(L * tk)[None, :]
    margin = Vx - r
    denom = Vphi + r * eLt
    valid = (margin > 0) & (denom > _TINY64)
    if enforce_S:
        valid = valid & (phi_inf <= R - r * eLt)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, margin / np.where(denom > 0, denom, 1.0), np.nan)
        a = np.log(ratio) / tk[None, :]
    return np.where(valid, a, -np.inf).max(axis=1)


# --------------------------------------------------------------------------- #
# find_alpha_min_fused: mirrors algorithm1's incremental refinement loop
# --------------------------------------------------------------------------- #
@dataclass
class FusedAlphaMinResult:
    alpha_certified: float
    alpha_upper: float
    suboptimality: float
    converged: bool
    refinements: int
    n_boxes: int
    backend: str
    trace: list = field(repr=False, default=None)
    centers: np.ndarray = field(repr=False, default=None)
    halfs: np.ndarray = field(repr=False, default=None)
    alpha_lo: np.ndarray = field(repr=False, default=None)
    alpha_up: np.ndarray = field(repr=False, default=None)

    def __repr__(self):
        tag = "converged" if self.converged else f"budget ({self.refinements} ref)"
        return (f"FusedAlphaMinResult(alpha={self.alpha_certified:.4f} "
                f"(<= {self.alpha_upper:.4f}, sub-opt {self.suboptimality:.1%}, {tag}), "
                f"{self.n_boxes} boxes, {self.backend})")


def find_alpha_min_fused(f, L, R, eps, d, *, norm="2", tau=2.0, n_steps=200,
                         delta=0.2, max_refine=6, enforce_S=True, dtype="float32",
                         tile_size=100_000, max_split_per_round=20_000,
                         backend="auto", max_seconds=None, plateau_eps=None,
                         plateau_window=3, record_trace=False):
    """Algorithm 1 with the fused per-box kernel. Mirrors
    :func:`pyddrv.verification.algorithm1.find_alpha_min` with ``gap_ref="global"``
    (global-gap stop, incremental refinement -- only new sub-boxes evaluated each
    round), plus the same optional wall-clock budget / plateau stop / anytime
    trace. ``f`` is a batched field ``(N,d)->(N,d)``."""
    c = norm_equiv_c(norm, d)
    which, _ = _resolve(backend, dtype)
    key = _norm_key(norm)
    # Build the jitted kernel ONCE and reuse across all rounds (else every
    # evaluate() would recompile).
    if which == "jax":
        kernel = _make_jax_kernel(f, key, enforce_S)
    elif which == "torch":
        from .fused_torch import make_torch_span_kernel
        kernel = make_torch_span_kernel(f, key, enforce_S)
    else:
        kernel = None
    centers, halfs = initial_grid(R, eps, norm, d)

    def evaluate(cs, hs):
        return fused_alpha_bounds(f, cs, hs, tau=tau, n_steps=n_steps, L=L, R=R,
                                  norm=norm, c=c, enforce_S=enforce_S, dtype=dtype,
                                  tile_size=tile_size, backend=backend, _kernel=kernel)

    t0 = time.time()
    trace = []
    a_lo, a_up = evaluate(centers, halfs)
    if record_trace:
        trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))
    refinements = 0
    cert_hist = []
    gap = np.inf
    converged = False
    while True:
        i = int(np.argmin(a_lo))
        worst_lo = a_lo[i]
        ref_up = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else np.inf
        gap = np.inf
        if np.isfinite(worst_lo) and worst_lo > 0 and np.isfinite(ref_up):
            gap = (ref_up - worst_lo) / ref_up
        converged = gap <= delta
        timed_out = max_seconds is not None and (time.time() - t0) >= max_seconds
        plateaued = False
        if plateau_eps is not None and np.isfinite(worst_lo) and worst_lo > 0:
            cert_hist.append(float(worst_lo))
            if (len(cert_hist) > plateau_window
                    and cert_hist[-1] - cert_hist[-1 - plateau_window] < plateau_eps):
                plateaued = True
        if refinements >= max_refine or converged or timed_out or plateaued:
            break
        ceiling = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else 0.0
        if ceiling > 0:
            target = ceiling * (1.0 - delta)
        else:
            target = np.max(a_lo[np.isfinite(a_lo)]) if np.isfinite(a_lo).any() else 0.0
        blocking = ~(a_lo >= target)
        order = np.argsort(a_lo)
        cap = max(1, min(int(blocking.sum()), max_split_per_round))
        worst = order[:cap]
        keep = np.ones(len(centers), dtype=bool)
        keep[worst] = False
        sub_c, sub_h = split_many(centers[worst], halfs[worst])
        sub_lo, sub_up = evaluate(sub_c, sub_h)          # only the new sub-boxes
        centers = np.concatenate([centers[keep], sub_c], axis=0)
        halfs = np.concatenate([halfs[keep], sub_h], axis=0)
        a_lo = np.concatenate([a_lo[keep], sub_lo], axis=0)
        a_up = np.concatenate([a_up[keep], sub_up], axis=0)
        refinements += 1
        if record_trace:
            trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))

    return FusedAlphaMinResult(
        alpha_certified=float(np.min(a_lo)),
        alpha_upper=float(ref_up),
        suboptimality=float(gap),
        converged=bool(converged),
        refinements=refinements,
        n_boxes=len(centers),
        backend=which,
        trace=trace if record_trace else None,
        centers=centers, halfs=halfs, alpha_lo=a_lo, alpha_up=a_up,
    )


# --------------------------------------------------------------------------- #
# find_alpha_min_ladder: per-box horizon escalation (tau-ladder)
# --------------------------------------------------------------------------- #
def find_alpha_min_ladder(f, L, R, eps, d, *, norm="2", tau_max=20.0, n_levels=4,
                          steps_per_unit=50, delta=0.2, max_refine=25,
                          enforce_S=True, dtype="float32", tile_size=100_000,
                          max_split_per_round=20_000, backend="auto",
                          max_seconds=None, plateau_eps=None, plateau_window=3,
                          record_trace=False):
    r"""Algorithm 1 with **per-box horizon escalation** (the tau-ladder).

    Rationale: with ``L`` estimated over the *saturated* reachable set (it is
    tau-independent once trajectories stop excursing), the Theorem-8 drift
    ``r e^{Lt}`` is paid at the certifying return time ``t`` -- NOT at ``tau``.
    Enlarging tau therefore only widens the search window for ``t``: per box,
    ``alpha_max`` is monotone non-decreasing in the horizon, and the only cost is
    integration steps. So instead of one global tau, each box climbs a ladder
    ``tau_max/2^(n_levels-1), ..., tau_max/2, tau_max`` **on demand**:

    * every box is first evaluated at the shortest horizon (the role of the
      paper's minimal-return tau); most certify near the ceiling and stop there;
    * a box still *blocking* the certified rate is escalated one rung (linear
      cost) before it is ever split (3^d cost); only boxes already at ``tau_max``
      are split, children inheriting the parent's rung.

    This dominates any fixed tau at matched compute: easy boxes never pay for a
    long horizon, hard boxes get the full window. ``alpha_lo/alpha_up`` are
    running maxima across rungs (sound: same ``L``, nested t-windows, same RK4
    step ``1/steps_per_unit`` at every rung).

    Caveat: soundness of the max across rungs needs ``L`` valid over the
    reachable set for ``[0, tau_max]`` -- pass an L estimated at ``tau_max``.
    """
    c = norm_equiv_c(norm, d)
    which, _ = _resolve(backend, dtype)
    key = _norm_key(norm)
    if which == "jax":
        kernel = _make_jax_kernel(f, key, enforce_S)
    elif which == "torch":
        from .fused_torch import make_torch_span_kernel
        kernel = make_torch_span_kernel(f, key, enforce_S)
    else:
        kernel = None
    taus = [tau_max / (2.0 ** (n_levels - 1 - k)) for k in range(n_levels)]
    # spans between rungs (rung 0 spans (0, tau_0]); same dt everywhere
    span_lo = [0.0] + taus[:-1]
    span_steps = [max(1, int(round((hi - lo) * steps_per_unit)))
                  for lo, hi in zip(span_lo, taus)]

    def span(x_init, Vx, hs, lvl):
        """Certify the span (taus[lvl-1], taus[lvl]] from the stored states.
        Escalations CONTINUE the integration -- nothing is re-integrated."""
        return fused_alpha_span(f, x_init, Vx, hs, t_lo=span_lo[lvl],
                                t_hi=taus[lvl], n_steps=span_steps[lvl],
                                L=L, R=R, norm=norm, c=c, enforce_S=enforce_S,
                                dtype=dtype, tile_size=tile_size,
                                backend=backend, _kernel=kernel)

    def full(cs, hs, lvl):
        """Fresh box (no stored state): certify (0, taus[lvl]] from the centre."""
        Vx = _norm_np(np.asarray(cs, dtype=float), key)
        return fused_alpha_span(f, cs, Vx, hs, t_lo=0.0, t_hi=taus[lvl],
                                n_steps=max(1, int(round(taus[lvl] * steps_per_unit))),
                                L=L, R=R, norm=norm, c=c, enforce_S=enforce_S,
                                dtype=dtype, tile_size=tile_size,
                                backend=backend, _kernel=kernel)

    centers, halfs = initial_grid(R, eps, norm, d)

    # Warm-up: trigger the JIT compiles OUTSIDE the timed region, at the padded
    # batch shapes the run will actually hit -- the initial grid's shape for the
    # rung-0 span, and the escalation-cap shape for the higher rungs. Dummy zero
    # states are harmless (Vx=0 -> infeasible -> -inf) and the compute is trivial
    # next to the compile itself.
    if which == "jax":
        d_ = centers.shape[1]
        for wl in range(n_levels):
            for n_warm in {len(centers), min(max_split_per_round, 4096)}:
                span(np.zeros((n_warm, d_)), np.zeros(n_warm), np.ones(n_warm), wl)

    t0 = time.time()
    trace = []
    levels = np.zeros(len(centers), dtype=int)
    a_lo, a_up, xs_end = full(centers, halfs, 0)
    if record_trace:
        trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))
    refinements = 0
    cert_hist = []
    gap = np.inf
    converged = False
    ref_up = np.inf
    while True:
        i = int(np.argmin(a_lo))
        worst_lo = a_lo[i]
        ref_up = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else np.inf
        gap = np.inf
        if np.isfinite(worst_lo) and worst_lo > 0 and np.isfinite(ref_up):
            gap = (ref_up - worst_lo) / ref_up
        converged = gap <= delta
        timed_out = max_seconds is not None and (time.time() - t0) >= max_seconds
        plateaued = False
        if plateau_eps is not None and np.isfinite(worst_lo) and worst_lo > 0:
            cert_hist.append(float(worst_lo))
            if (len(cert_hist) > plateau_window
                    and cert_hist[-1] - cert_hist[-1 - plateau_window] < plateau_eps):
                plateaued = True
        if refinements >= max_refine or converged or timed_out or plateaued:
            break

        ceiling = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else 0.0
        if ceiling > 0:
            target = ceiling * (1.0 - delta)
        else:
            target = np.max(a_lo[np.isfinite(a_lo)]) if np.isfinite(a_lo).any() else 0.0
        blocking = ~(a_lo >= target)
        if not blocking.any():
            break
        order = np.argsort(a_lo)
        order = order[blocking[order]]                 # blocking boxes, worst first
        act = order[:max(1, min(len(order), max_split_per_round))]

        esc = act[levels[act] < n_levels - 1]          # escalate horizon (linear)
        spl = act[levels[act] >= n_levels - 1]         # already at tau_max: split
        if esc.size:
            # group by NEXT rung so each level is one batched evaluation; the
            # integration CONTINUES from the stored state at the previous rung,
            # so only the new span (taus[lvl-1], taus[lvl]] is computed
            nxt = levels[esc] + 1
            for lvl in np.unique(nxt):
                idx = esc[nxt == lvl]
                Vx = _norm_np(centers[idx], key)       # ||x(0)|| of the centres
                lo, up, xe = span(xs_end[idx], Vx, halfs[idx], int(lvl))
                a_lo[idx] = np.maximum(a_lo[idx], lo)  # running max across rungs
                a_up[idx] = np.maximum(a_up[idx], up)
                xs_end[idx] = xe
                levels[idx] = lvl
        if spl.size:
            keep = np.ones(len(centers), dtype=bool)
            keep[spl] = False
            sub_c, sub_h = split_many(centers[spl], halfs[spl])
            # children inherit the parent's rung: needing a long horizon is a
            # property of the region, not of the box size. Fresh boxes have no
            # stored state, so they integrate (0, taus[max]] once.
            sub_lo, sub_up, sub_xe = full(sub_c, sub_h, n_levels - 1)
            sub_lv = np.full(len(sub_c), n_levels - 1, dtype=int)
            centers = np.concatenate([centers[keep], sub_c], axis=0)
            halfs = np.concatenate([halfs[keep], sub_h], axis=0)
            levels = np.concatenate([levels[keep], sub_lv], axis=0)
            xs_end = np.concatenate([xs_end[keep], sub_xe], axis=0)
            a_lo = np.concatenate([a_lo[keep], sub_lo], axis=0)
            a_up = np.concatenate([a_up[keep], sub_up], axis=0)
        refinements += 1
        if record_trace:
            trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))

    return FusedAlphaMinResult(
        alpha_certified=float(np.min(a_lo)),
        alpha_upper=float(ref_up),
        suboptimality=float(gap),
        converged=bool(converged),
        refinements=refinements,
        n_boxes=len(centers),
        backend=which,
        trace=trace if record_trace else None,
        centers=centers, halfs=halfs, alpha_lo=a_lo, alpha_up=a_up,
    )


# --------------------------------------------------------------------------- #
# Algorithm 2 (Find-alpha-RoA), fused
# --------------------------------------------------------------------------- #
def region_distance_map(centers, halfs, R_dom, n_cells):
    r"""Rasterize a union of grid-aligned cubes onto an ``n_cells^d`` grid over
    ``[-R_dom, R_dom]^d`` and return the inf-norm INNER distance map: for each
    cell, a sound lower bound on the distance from any point of the cell to the
    complement of the union. Enables the Trim (32b) check ``sd(phi, S) <=
    -r e^{Lt}`` as a single lookup: ``dist(phi) >= r e^{Lt}``.

    Cells only partially covered by the union count as OUTSIDE (conservative
    under-approximation), and one cell width is subtracted from the transform
    (point-in-cell quantization), so the map never overstates the distance."""
    from scipy.ndimage import distance_transform_cdt
    d = centers.shape[1]
    cellw = 2.0 * R_dom / n_cells
    lo = -R_dom
    if d == 2:
        # EXACT area accumulation: a cell is inside iff the summed overlap
        # area of (possibly sub-cell) cubes fills it. Per-axis coverage
        # weights are separable, so each cube contributes the outer product
        # of two per-axis weight vectors over its touched cell range.
        area = np.zeros((n_cells, n_cells), dtype=np.float64)
        for (cvec, h) in zip(centers, halfs):
            wxs, wys, ix, iy = [], [], [], []
            for ax, (c_a, out_w, out_i) in enumerate(
                    [(cvec[0], wxs, ix), (cvec[1], wys, iy)]):
                a0 = (c_a - h - lo) / cellw
                a1 = (c_a + h - lo) / cellw
                j0, j1 = int(np.floor(a0)), int(np.ceil(a1))
                for j in range(max(j0, 0), min(j1, n_cells)):
                    w = min(a1, j + 1) - max(a0, j)
                    if w > 0:
                        out_w.append(w)
                        out_i.append(j)
            if ix and iy:
                area[np.ix_(ix, iy)] += np.outer(wxs, wys)
        grid = area >= 1.0 - 1e-6
    else:
        grid = np.zeros((n_cells,) * d, dtype=bool)
        for (cvec, h) in zip(centers, halfs):
            i0 = np.round((cvec - h - lo) / cellw, 9)
            i1 = np.round((cvec + h - lo) / cellw, 9)
            i0 = np.ceil(i0 - 1e-9).astype(int)
            i1 = np.floor(i1 + 1e-9).astype(int)
            if np.any(i1 <= i0):
                continue                # cube thinner than a raster cell: drop
            sl = tuple(slice(max(a, 0), min(b, n_cells))
                       for a, b in zip(i0, i1))
            grid[sl] = True
    # chessboard = inf-norm distance in cell units to the nearest outside cell
    dist_cells = distance_transform_cdt(grid, metric="chessboard")
    dmap = (dist_cells.astype(np.float64) - 1.0) * cellw
    return np.maximum(dmap, 0.0), lo, cellw


def _make_jax_roa_kernel(f, key: str, use_region: bool):
    """Kernel returning alpha_max at radius r = c*h for a batch of cubes. With
    ``use_region``, (32b) is checked against a rasterized region via its inner
    distance map (same-t AND, as in ball.alpha_max); else (32a) only."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    def jnorm(x):
        if key == "2":
            return jnp.sqrt(jnp.sum(x * x, axis=-1))
        if key == "inf":
            return jnp.max(jnp.abs(x), axis=-1)
        return jnp.sum(jnp.abs(x), axis=-1)

    @jax.jit
    def kernel(centers, halfs, tk_arr, dt, L, R, c, dmap, lo, cellw):
        x = centers
        Vx = jnorm(x)
        r = c * halfs
        neg = jnp.full(Vx.shape, -jnp.inf)
        n_cells = dmap.shape[0]
        dflat = dmap.reshape(-1)
        d_dim = centers.shape[1]

        def region_dist(phi):
            idx = jnp.floor((phi - lo) / cellw).astype(jnp.int32)
            inb = jnp.all((idx >= 0) & (idx < n_cells), axis=-1)
            idx = jnp.clip(idx, 0, n_cells - 1)
            flat = idx[..., 0]
            for j in range(1, d_dim):
                flat = flat * n_cells + idx[..., j]
            return jnp.where(inb, dflat[flat], 0.0)

        def alpha_at(radius, Vphi, xn, eLt, tk):
            margin = Vx - radius
            denom = Vphi + radius * eLt
            valid = (margin > 0) & (denom > 0)
            if use_region:
                valid = valid & (region_dist(xn) >= radius * eLt)   # (32b) vs S
            denom_s = jnp.where(denom > 0, denom, 1.0)
            return jnp.where(valid,
                             jnp.log(jnp.where(valid, margin / denom_s, 1.0))
                             / tk, -jnp.inf)

        def body(carry, tk):
            x, a, a3 = carry
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            Vphi = jnorm(xn)
            eLt = jnp.exp(L * tk)
            a = jnp.maximum(a, alpha_at(r, Vphi, xn, eLt, tk))
            # one-split LOOKAHEAD: alpha at radius r/3 = the exact certificate
            # the cube's centre child will get after one split (free: same
            # rollout). Drives the expected-gain refinement priority.
            a3 = jnp.maximum(a3, alpha_at(r / 3.0, Vphi, xn, eLt, tk))
            return (xn, a, a3), None

        (_, a, a3), _ = lax.scan(body, (x, neg, neg), tk_arr)
        return a, a3

    return kernel


@dataclass
class FusedRoAResult:
    alpha: float
    centers: np.ndarray = field(repr=False, default=None)
    halfs: np.ndarray = field(repr=False, default=None)
    depths: np.ndarray = field(repr=False, default=None)   # split depth per cube
    n_certified: int = 0
    n_tested: int = 0
    refinements: int = 0
    trim: bool = False
    trace: list = field(default_factory=list, repr=False)
    # per chunk: (elapsed_s, half_width_evaluated, n_tested_cum,
    #            n_certified_cum, volume_cum)
    budget_hit: bool = False
    stop_reason: str = "complete"     # complete | budget | plateau

    def summary(self) -> str:
        frac = (self.n_certified / self.n_tested) if self.n_tested else 0.0
        return (f"Find-alpha-RoA-fused(alpha={self.alpha:g}): {self.n_certified} "
                f"cubes of {self.n_tested} tested ({frac:.1%}), "
                f"depth<={self.refinements}, Trim={self.trim}")


def find_alpha_roa_fused(f, L, R, eps, d, alpha, *, norm="2", tau=2.0,
                         n_steps=200, max_refine=5, trim_region=None,
                         raster_n=2187, grid0=None, dtype="float32",
                         tile_size=100_000, max_seconds=None,
                         plateau_rel=None, verbose=False, backend="jax",
                         spill_dir=None, odd_field_sym=False):
    r"""Algorithm 2 with the fused kernel. Mirrors
    :func:`pyddrv.verification.algorithm2.find_alpha_roa`: grow ``Positives`` at
    the target rate ``alpha`` from the layered grid (or ``grid0``), splitting
    failing cubes up to depth ``max_refine``.

    ``trim_region=(centers, halfs)`` enables the paper's ``Trim=True`` pass:
    (32b) is checked against the EXACT union of those cubes (rasterized inner
    distance map at ``raster_n^d``), not a bounding box. Paper usage: pass 1
    with ``trim_region=None`` grows ``S0``; pass 2 with ``trim_region=S0`` and
    ``grid0=S0`` prunes it.

    ``max_seconds`` makes the loop ANYTIME: the budget is checked per TILE, so
    a round larger than the remaining budget stops mid-round -- cubes evaluated
    so far keep their verdicts, unevaluated cubes of that round are dropped
    (sound: nothing is certified without its check; same semantics as hitting
    ``max_refine``, but the truncation is in grid order, i.e. spatially biased).
    The per-round ``trace`` records (elapsed, depth, tested, certified, volume).

    ``plateau_rel`` (e.g. 1e-3) enables the STALL-PROOF plateau stop: terminate
    once BOTH (a) the certified volume grew by less than ``plateau_rel``
    (relative) over the last time-DOUBLING, and (b) the expected one-split
    certifiable volume still pending in the queue (sum of chunk volume times
    its measured lookahead pass-fraction) is below ``plateau_rel`` times the
    current volume. Condition (b) keeps the rule from misfiring during the
    staircase stalls where a large jump is one split away.

    ``spill_dir``: stream certified cubes to raw binary files in this
    directory (centers.f32 / halfs.f32 / depths.u8) instead of accumulating
    them in RAM -- resident memory stays O(tile) regardless of budget; the
    returned centers/halfs/depths are read-only memmaps of those files.
    REQUIRED for long runs: a 10^4 s GPU run certifies ~10^9 cubes (~25 GB),
    which OOM-kills the process if held in RAM (observed 2026-07-05)."""
    c = norm_equiv_c(norm, d)
    key = _norm_key(norm)
    use_region = trim_region is not None
    if backend == "torch":
        # f must be a BATCHED TORCH field; region (Trim) supported via dmap
        from .fused_torch import make_torch_roa_kernel
        t_kernel = make_torch_roa_kernel(f, key)
        jnp = None
    else:
        import jax.numpy as jnp
        kernel = _make_jax_roa_kernel(f, key, use_region)
    if use_region:
        dmap, lo, cellw = region_distance_map(np.asarray(trim_region[0]),
                                              np.asarray(trim_region[1]),
                                              R, raster_n)
    else:                       # dummy 1-cell map, unused inside the kernel
        dmap, lo, cellw = np.zeros((1,) * d), -R, 2.0 * R
    dmap_j = jnp.asarray(dmap, jnp.float32) if jnp is not None else None

    if grid0 is not None:
        centers, halfs = np.asarray(grid0[0], float), np.asarray(grid0[1], float)
        # carry split depths through (e.g. from a Trim=False pass) so the total
        # split budget stays max_refine and Fig-3-style colors survive pass 2
        depths = (np.asarray(grid0[2], int) if len(grid0) > 2
                  else np.zeros(len(centers), dtype=int))
    else:
        centers, halfs = initial_grid(R, eps, norm, d)
        depths = np.zeros(len(centers), dtype=int)
    npd = np.float32 if str(dtype) in ("float32", "f4") else np.float64
    dt = tau / n_steps
    tk = (dt * np.arange(1, n_steps + 1))

    def evaluate(cs, hs, deadline=None):
        # returns (alpha, alpha_lookahead) for a PREFIX of cs (all of it
        # unless the deadline cuts the round short mid-tile-loop).
        # odd_field_sym: for odd fields f(-x) = -f(x) the certificate of the
        # mirrored cube transfers exactly, so a cube certifies if IT or its
        # mirror does -- keeps the certified tiling symmetric WITHOUT overlap
        if odd_field_sym:
            a, a3 = evaluate_one(cs, hs, deadline)
            am, am3 = evaluate_one(-cs[:len(a)], hs[:len(a)], None)
            return np.maximum(a, am), np.maximum(a3, am3)
        return evaluate_one(cs, hs, deadline)

    def evaluate_one(cs, hs, deadline=None):
        out, out3 = [], []
        for s in range(0, len(cs), tile_size):
            if deadline is not None and s > 0 and time.perf_counter() > deadline:
                break
            c_t, h_t = cs[s:s + tile_size], hs[s:s + tile_size]
            N = len(c_t)
            if backend == "torch":      # no pow2 padding: no XLA recompiles
                a, a3 = t_kernel(c_t, h_t, float(tau), n_steps, float(L),
                                 float(c),
                                 dmap=dmap if use_region else None,
                                 lo=float(lo), cellw=float(cellw))
                out.append(a)
                out3.append(a3)
                continue
            P = 1 << max(8, (N - 1).bit_length())
            if P > N:           # pad (zero centres are infeasible -> -inf)
                c_t = np.concatenate([c_t, np.zeros((P - N, d))])
                h_t = np.concatenate([h_t, np.ones(P - N)])
            a, a3 = kernel(jnp.asarray(c_t, npd), jnp.asarray(h_t, npd),
                           jnp.asarray(tk, npd), jnp.asarray(dt, npd),
                           jnp.asarray(L, npd), jnp.asarray(R, npd),
                           jnp.asarray(c, npd), dmap_j,
                           jnp.asarray(lo, npd), jnp.asarray(cellw, npd))
            out.append(np.asarray(a, float)[:N])
            out3.append(np.asarray(a3, float)[:N])
        if not out:
            return np.empty(0), np.empty(0)
        return np.concatenate(out), np.concatenate(out3)

    pos_c, pos_h, pos_d = [], [], []
    spill = None
    if spill_dir is not None:
        import os as _os
        _os.makedirs(spill_dir, exist_ok=True)
        spill = {name: open(_os.path.join(spill_dir, name), "wb")
                 for name in ("centers.f32", "halfs.f32", "depths.u8")}
    n_tested = 0
    max_depth_seen = 0
    n_cert, vol = 0, 0.0
    trace, budget_hit = [], False
    trace_t, trace_v = [], []
    t_start = time.perf_counter()
    deadline = None if max_seconds is None else t_start + max_seconds
    # EXPECTED-GAIN priority queue of UNSPLIT parent chunks. Priority of a
    # chunk = (child volume) * (measured fraction of parents whose one-split
    # LOOKAHEAD alpha(x, r/3) already meets the target) + a small floor so
    # large unexplored cubes are never starved (floor-only order degrades to
    # largest-first). Failing parents are sorted by lookahead before chunking
    # so high-yield chunks surface first. Children are materialized only one
    # chunk (~4 tiles) at a time, so peak memory is O(tile_size) in any
    # dimension, and the deadline is honored between chunks.
    import heapq
    P_FLOOR = 0.05
    par_batch = max(1, (4 * tile_size) // 3 ** d)
    heap, tie = [], 0
    pending_gain = 0.0      # sum over queue of chunk volume x lookahead p_hat

    def push(cs, hs, ds, needs_split, p_hat):
        nonlocal tie, pending_gain
        if not len(cs):
            return
        h_eval = float(hs[0]) / 3.0 if needs_split else float(hs[0])
        gain = (2.0 * h_eval) ** d * (p_hat + P_FLOOR)
        # expected one-split certified volume of this chunk: splitting
        # preserves volume, so it is p_hat x the chunk's own volume
        exp_gain = p_hat * float(np.sum((2.0 * hs) ** d))
        heapq.heappush(heap, (-gain, tie, (cs, hs, ds, needs_split, exp_gain)))
        pending_gain += exp_gain
        tie += 1

    hr = np.round(halfs, 12)
    for h in np.unique(hr):                 # uniform-size initial chunks
        m_ = hr == h
        push(centers[m_], halfs[m_], depths[m_], False, 0.0)
    t_last_log, v_last_log = -1e30, 0.0
    stop_reason = "complete"
    while heap:
        _, _, (cs, hs, ds, needs_split, exp_gain) = heapq.heappop(heap)
        pending_gain -= exp_gain
        if needs_split:
            cs, hs = split_many(cs, hs)
            ds = np.repeat(ds + 1, 3 ** d)
        a, a3 = evaluate(cs, hs, deadline)
        if len(a) < len(cs):            # deadline cut this batch short
            budget_hit = True
            cs, hs, ds = cs[:len(a)], hs[:len(a)], ds[:len(a)]
        if not len(cs):
            break
        n_tested += len(cs)
        max_depth_seen = max(max_depth_seen, int(ds.max()))
        passed = a >= alpha
        # float32 accumulation (exact for 3-adic sizes); with spill_dir the
        # chunks go straight to disk and RAM stays O(tile)
        if spill is not None:
            cs[passed].astype(np.float32).tofile(spill["centers.f32"])
            hs[passed].astype(np.float32).tofile(spill["halfs.f32"])
            ds[passed].astype(np.uint8).tofile(spill["depths.u8"])
        else:
            pos_c.append(cs[passed].astype(np.float32))
            pos_h.append(hs[passed].astype(np.float32))
            pos_d.append(ds[passed].astype(np.uint8))
        n_cert += int(passed.sum())
        vol += float(np.sum((2.0 * hs[passed]) ** d))
        elapsed = time.perf_counter() - t_start
        trace.append((elapsed, float(hs[0]), n_tested, n_cert, vol))
        trace_t.append(elapsed)
        trace_v.append(vol)
        if verbose and (elapsed - t_last_log > 10.0 or not heap
                        or vol > 1.02 * max(v_last_log, 1e-12)):
            t_last_log, v_last_log = elapsed, vol
            print(f"  [roa] {elapsed:7.1f}s h={float(hs[0]):.4g} "
                  f"tested={n_tested} certified={n_cert} vol={vol:.2f}",
                  flush=True)
        if budget_hit or (deadline is not None
                          and time.perf_counter() >= deadline):
            budget_hit = True
            stop_reason = "budget"
            break
        if plateau_rel is not None and vol > 0:
            import bisect
            j = bisect.bisect_left(trace_t, elapsed / 2.0)
            if (j < len(trace) - 1 and trace_v[j] > 0
                    and vol < (1 + plateau_rel) * trace_v[j]
                    and pending_gain < plateau_rel * vol):
                stop_reason = "plateau"
                break
        # split only failing cubes still below the total split budget; failing
        # cubes already AT max_refine are discarded. Parents queue as float32
        # (children inherit the dtype): the pending frontier of a long run is
        # hundreds of millions of cubes, f64 doubles its footprint for nothing
        splittable = (~passed) & (ds < max_refine)
        order = np.argsort(-a3[splittable])          # best lookahead first
        pc = cs[splittable][order].astype(np.float32, copy=False)
        ph = hs[splittable][order].astype(np.float32, copy=False)
        pd, pa3 = ds[splittable][order], a3[splittable][order]
        for s in range(0, len(pc), par_batch):
            p_hat = float(np.mean(pa3[s:s + par_batch] >= alpha))
            push(pc[s:s + par_batch], ph[s:s + par_batch],
                 pd[s:s + par_batch], True, p_hat)

    if spill is not None:
        import os as _os
        for fh in spill.values():
            fh.close()
        C = np.memmap(_os.path.join(spill_dir, "centers.f32"), np.float32,
                      "r").reshape(-1, d) if n_cert else np.empty((0, d))
        H = (np.memmap(_os.path.join(spill_dir, "halfs.f32"), np.float32, "r")
             if n_cert else np.empty((0,)))
        D = (np.memmap(_os.path.join(spill_dir, "depths.u8"), np.uint8, "r")
             if n_cert else np.empty((0,), dtype=int))
    else:
        C = np.concatenate(pos_c) if pos_c else np.empty((0, d))
        H = np.concatenate(pos_h) if pos_h else np.empty((0,))
        D = np.concatenate(pos_d) if pos_d else np.empty((0,), dtype=int)
    return FusedRoAResult(alpha=float(alpha), centers=C, halfs=H, depths=D,
                          n_certified=n_cert, n_tested=n_tested,
                          refinements=max_depth_seen, trim=use_region,
                          trace=trace, budget_hit=budget_hit,
                          stop_reason=stop_reason)
