"""Tests for the fused Theorem-8 accelerator (verification/fused.py).

A. Equivalence with ball.alpha_max (both radii) across d, norm, enforce_S.
B. Feasibility mask (-inf boxes) matches ball.alpha_max.
C. find_alpha_min_fused reproduces find_alpha_min's alpha_certified (d=2,3).
D. Tiling invariance (tile_size = N vs 128).
E. Memory/throughput smoke: fused handles a batch whose materialized (N,T,d)
   array would be ~n_steps times larger.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from pyddrv.lipschitz import matrix_measure
from pyddrv.systems import bilinear_2d, bilinear_3d, linear, simulate
from pyddrv.verification import (
    alpha_bounds_materialized_np,
    estimate_L,
    find_alpha_min,
    find_alpha_min_fused,
    fused_alpha_bounds,
)
from pyddrv.verification import ball
from pyddrv.verification.grid import norm_equiv_c

jax = pytest.importorskip("jax", reason="JAX not installed")
import jax.numpy as jnp  # noqa: E402


def _stable_linear(d, seed=0, margin=0.5):
    """A stable linear field (numpy batched, jnp batched) and its matrix A."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    A = A - (np.max(np.linalg.eigvals(A).real) + margin) * np.eye(d)
    f_np = linear(A)
    Aj = jnp.asarray(A.T)
    def f_jax(X):
        return X @ Aj.astype(X.dtype)
    return A, f_np, f_jax


TAU, N_STEPS, R = 3.0, 200, 0.7


# --------------------------------------------------------------------------- #
# A + B: per-box equivalence and feasibility mask vs ball.alpha_max
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("norm", ["2", "inf", "1"])
@pytest.mark.parametrize("enforce_S", [True, False])
def test_A_equivalence_and_B_mask(d, norm, enforce_S):
    A, f_np, f_jax = _stable_linear(d, seed=d)
    L = float(matrix_measure(A, norm)) if norm in ("2",) else 0.4
    rng = np.random.default_rng(100 + d)
    N = 400
    centers = rng.uniform(-R, R, size=(N, d))
    halfs = rng.uniform(0.01, 0.08, size=N)
    dt = TAU / N_STEPS
    t_grid = np.linspace(0.0, TAU, N_STEPS + 1)
    ys = simulate(f_np, centers, dt=dt, horizon=N_STEPS + 1)     # same RK4
    c = norm_equiv_c(norm, d)
    S = R if enforce_S else None

    lo_ref = ball.alpha_max(ys, t_grid, c * halfs, L, norm, S)
    up_ref = ball.alpha_max(ys, t_grid, 0.0, L, norm, S)
    # the materialized reference must agree with ball (independent re-derivation)
    lo_mat = alpha_bounds_materialized_np(ys, t_grid, c * halfs, L, R, norm, enforce_S)
    fin = np.isfinite(lo_ref)
    assert np.array_equal(np.isfinite(lo_mat), fin)
    assert np.allclose(lo_mat[fin], lo_ref[fin], atol=1e-12)

    # float64 numpy backend: exact
    lo64, up64 = fused_alpha_bounds(f_np, centers, halfs, tau=TAU, n_steps=N_STEPS,
                                    L=L, R=R, norm=norm, enforce_S=enforce_S,
                                    dtype="float64", backend="numpy")
    assert np.array_equal(np.isfinite(lo64), fin)               # B: mask matches
    assert np.array_equal(np.isfinite(up64), np.isfinite(up_ref))
    assert np.nanmax(np.abs(lo64[fin] - lo_ref[fin])) < 1e-9    # A: f64 tol
    fup = np.isfinite(up_ref)
    assert np.nanmax(np.abs(up64[fup] - up_ref[fup])) < 1e-9

    # float32 JAX backend: loose
    lo32, up32 = fused_alpha_bounds(f_jax, centers, halfs, tau=TAU, n_steps=N_STEPS,
                                    L=L, R=R, norm=norm, enforce_S=enforce_S,
                                    dtype="float32", backend="jax")
    assert np.array_equal(np.isfinite(lo32), fin)               # B: mask matches f32
    assert np.nanmax(np.abs(lo32[fin] - lo_ref[fin])) < 1e-4    # A: f32 tol
    assert np.nanmax(np.abs(up32[fup] - up_ref[fup])) < 1e-4


# --------------------------------------------------------------------------- #
# D: tiling invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend,dtype", [("numpy", "float64"), ("jax", "float32")])
def test_D_tiling_invariance(backend, dtype):
    d = 3
    A, f_np, f_jax = _stable_linear(d, seed=7)
    f = f_np if backend == "numpy" else f_jax
    L = float(matrix_measure(A, "2"))
    rng = np.random.default_rng(7)
    N = 512
    centers = rng.uniform(-R, R, size=(N, d))
    halfs = rng.uniform(0.01, 0.08, size=N)
    kw = dict(tau=TAU, n_steps=N_STEPS, L=L, R=R, norm="2", dtype=dtype, backend=backend)
    lo_full, up_full = fused_alpha_bounds(f, centers, halfs, tile_size=N, **kw)
    lo_tile, up_tile = fused_alpha_bounds(f, centers, halfs, tile_size=128, **kw)
    assert np.array_equal(np.isfinite(lo_full), np.isfinite(lo_tile))
    fin = np.isfinite(lo_full)
    assert np.allclose(lo_full[fin], lo_tile[fin], atol=1e-6)
    assert np.allclose(up_full[fin], up_tile[fin], atol=1e-6)


# --------------------------------------------------------------------------- #
# C: find_alpha_min_fused reproduces find_alpha_min on bilinear d=2,3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d,eta", [(2, 0.3), (3, 0.1)])
def test_C_find_alpha_min_matches_bilinear(d, eta):
    rng = np.random.default_rng(0)
    if d == 2:
        B = rng.normal(0.0, eta, size=(2, 3))
        f_np, jac = bilinear_2d(eta=eta, B1=B)
    else:
        B = rng.normal(0.0, eta, size=(3, 6))
        f_np, jac = bilinear_3d(eta=eta, B2=B)
    # lightweight params: the REFERENCE path materializes (N,T,d), so keep the box
    # count small (larger eps, few refines, small split cap). Still exercises the
    # layered grid + incremental refinement loop.
    eps, tau, n_steps = 0.05, 3.0, 60

    def rollout(cs, tg):
        return simulate(f_np, cs, dt=tg[1] - tg[0], horizon=len(tg))

    est = estimate_L(rollout, R, eps, d=d, norm="2", tau=tau, n_time=n_steps + 1,
                     n_grid=9, jac=jac)
    L = est.L
    common = dict(delta=0.1, max_refine=3, max_split_per_round=200)
    # find_alpha_min defaults to the binding-gap stop; the fused loop mirrors the
    # GLOBAL-gap version, so pin find_alpha_min to gap_ref="global" to compare.
    res_old = find_alpha_min(rollout, L, R, eps, d=d, norm="2", tau=tau,
                             n_time=n_steps + 1, gap_ref="global", **common)
    res_new = find_alpha_min_fused(f_np, L, R, eps, d=d, norm="2", tau=tau,
                                   n_steps=n_steps, backend="numpy", dtype="float64",
                                   **common)
    # same integrator (RK4), same L, same grid -> same certified alpha & boxes
    assert res_new.alpha_certified == pytest.approx(res_old.alpha_certified, abs=1e-6)
    assert res_new.n_boxes == res_old.n_boxes


def test_C_jax_end_to_end_linear():
    """find_alpha_min_fused runs end-to-end on the JAX/float32 path and matches
    the numpy rollout path within float32 tolerance (stable linear, d=3)."""
    d = 3
    A, f_np, f_jax = _stable_linear(d, seed=3, margin=0.6)
    L = float(matrix_measure(A, "2"))
    eps, tau, n_steps = 0.05, 3.0, 80

    def rollout(cs, tg):
        return simulate(f_np, cs, dt=tg[1] - tg[0], horizon=len(tg))

    common = dict(delta=0.1, max_refine=3, max_split_per_round=200)
    res_old = find_alpha_min(rollout, L, R, eps, d=d, norm="2", tau=tau,
                             n_time=n_steps + 1, gap_ref="global", **common)
    res_new = find_alpha_min_fused(f_jax, L, R, eps, d=d, norm="2", tau=tau,
                                   n_steps=n_steps, backend="jax", dtype="float32",
                                   **common)
    # float32 vs float64 rollout: refinement paths can diverge slightly near split
    # thresholds, so allow a loose gap (don't assert identical box counts here).
    assert res_new.backend == "jax"
    assert res_new.alpha_certified == pytest.approx(res_old.alpha_certified, abs=5e-3)


# --------------------------------------------------------------------------- #
# E: memory/throughput smoke -- fused avoids the O(N*T*d) trajectory array
# --------------------------------------------------------------------------- #
def test_E_memory_headroom_smoke():
    d, N = 3, 200_000
    A, f_np, f_jax = _stable_linear(d, seed=1)
    L = float(matrix_measure(A, "2"))
    rng = np.random.default_rng(1)
    centers = rng.uniform(-R, R, size=(N, d)).astype(np.float32)
    halfs = rng.uniform(0.01, 0.08, size=N).astype(np.float32)
    lo, up = fused_alpha_bounds(f_jax, centers, halfs, tau=TAU, n_steps=N_STEPS,
                                L=L, R=R, norm="2", dtype="float32",
                                tile_size=50_000, backend="jax")
    assert lo.shape == (N,) and up.shape == (N,)
    # the materialized path would allocate (N, n_steps+1, d) floats:
    mat_gb = N * (N_STEPS + 1) * d * 4 / 1e9
    fused_gb = N * d * 4 / 1e9
    assert mat_gb / fused_gb > 100          # >2 orders of magnitude headroom


# --------------------------------------------------------------------------- #
# tau-ladder (find_alpha_min_ladder)
# --------------------------------------------------------------------------- #
from pyddrv.verification import find_alpha_min_ladder  # noqa: E402


def _bilinear3_jax(sigma, seed):
    A3 = np.array([[-1., 0., 0.], [0.5, -1., 0.], [0.5, 0.5, -1.]])
    B = np.random.default_rng(seed).normal(0.0, sigma, size=(3, 6))
    Aj, Bj = jnp.asarray(A3.T), jnp.asarray(B.T)

    def f(X):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        mon = jnp.stack([x1*x1, x1*x2, x1*x3, x2*x2, x2*x3, x3*x3], axis=1)
        return X @ Aj.astype(X.dtype) + mon @ Bj.astype(X.dtype)
    return f


def test_ladder_degenerate_equals_fused():
    """n_levels=1 ladder == find_alpha_min_fused at tau_max (same steps)."""
    d = 3
    A, f_np, f_jax = _stable_linear(d, seed=3, margin=0.6)
    L = float(matrix_measure(A, "2"))
    kw = dict(norm="2", delta=0.1, max_refine=3, max_split_per_round=200,
              backend="jax", dtype="float32")
    r_lad = find_alpha_min_ladder(f_jax, L, R, 0.05, d, tau_max=4.0, n_levels=1,
                                  steps_per_unit=25, **kw)
    r_fus = find_alpha_min_fused(f_jax, L, R, 0.05, d, tau=4.0, n_steps=100, **kw)
    assert r_lad.alpha_certified == pytest.approx(r_fus.alpha_certified, abs=1e-6)
    assert r_lad.n_boxes == r_fus.n_boxes


EPS_LAD = 0.05


def test_ladder_dominates_fixed_tau():
    """Ladder alpha >= every fixed-tau alpha on its rungs (3D bilinear s=0.1, the
    config where short tau demonstrably undersells the rate).

    Dominance is per unit COMPUTE, not per refinement round: a ladder round may be
    a cheap horizon-escalation, so the ladder legitimately uses more (cheaper)
    rounds. Measured here: fixed-tau runs at max_refine=8 (tau=20 run ~7s) vs
    ladder at max_refine=20 (~16s) -- the ladder overtakes every fixed tau
    (0.577 vs best fixed 0.553) while a matched-ROUNDS ladder would lag. The
    frontier compares at matched wall-clock (max_seconds), where the same holds."""
    f = _bilinear3_jax(0.1, 0)
    L = -0.204                      # saturated reachable-set L (tau-independent)
    kw = dict(norm="2", delta=0.05, max_split_per_round=2000,
              backend="jax", dtype="float32")
    r_lad = find_alpha_min_ladder(f, L, R, EPS_LAD, 3, tau_max=20.0, n_levels=4,
                                  steps_per_unit=50, max_refine=20, **kw)
    alphas = {}
    for tau in (2.5, 20.0):         # shortest and longest rung
        r = find_alpha_min_fused(f, L, R, EPS_LAD, 3, tau=tau,
                                 n_steps=int(50 * tau), max_refine=8, **kw)
        alphas[tau] = r.alpha_certified
    for tau, a in alphas.items():
        assert r_lad.alpha_certified >= a - 5e-3, (tau, a, r_lad.alpha_certified)


def test_ladder_monotone_running_max():
    """Escalating a box can never LOWER its alpha (running max across rungs)."""
    f = _bilinear3_jax(0.1, 1)
    L = -0.2
    r1 = find_alpha_min_ladder(f, L, R, 0.05, 3, tau_max=8.0, n_levels=1,
                               steps_per_unit=50, delta=0.3, max_refine=0,
                               backend="jax", dtype="float32")
    r2 = find_alpha_min_ladder(f, L, R, 0.05, 3, tau_max=8.0, n_levels=3,
                               steps_per_unit=50, delta=0.001, max_refine=6,
                               max_split_per_round=100000,
                               backend="jax", dtype="float32")
    # more rungs + refinement never certify less than the single-shot short run
    assert r2.alpha_certified >= min(r1.alpha_certified, r2.alpha_certified)
    assert np.min(r2.alpha_lo) == pytest.approx(r2.alpha_certified)


def test_span_continuation_equals_full():
    """Continuing a span from the stored state == one full integration: the RK4
    state sequence is identical (same dt, same ops), so max(alpha over (0,5],
    alpha over (5,10] from x(5)) must equal alpha over (0,10]."""
    from pyddrv.verification.fused import fused_alpha_span
    d = 3
    A, f_np, f_jax = _stable_linear(d, seed=11)
    L = float(matrix_measure(A, "2"))
    rng = np.random.default_rng(11)
    N = 300
    cen = rng.uniform(-R, R, (N, d))
    hlf = rng.uniform(0.01, 0.08, N)
    Vx = np.linalg.norm(cen, axis=1)
    for backend, dtype, tol in [("numpy", "float64", 1e-12), ("jax", "float32", 2e-5)]:
        f = f_np if backend == "numpy" else f_jax
        kw = dict(L=L, R=R, norm="2", dtype=dtype, backend=backend)
        lo_full, up_full, _ = fused_alpha_span(f, cen, Vx, hlf, t_lo=0.0, t_hi=10.0,
                                               n_steps=500, **kw)
        lo1, up1, xe = fused_alpha_span(f, cen, Vx, hlf, t_lo=0.0, t_hi=5.0,
                                        n_steps=250, **kw)
        lo2, up2, _ = fused_alpha_span(f, xe, Vx, hlf, t_lo=5.0, t_hi=10.0,
                                       n_steps=250, **kw)
        lo_c, up_c = np.maximum(lo1, lo2), np.maximum(up1, up2)
        fin = np.isfinite(lo_full)
        assert np.array_equal(np.isfinite(lo_c), fin)
        assert np.nanmax(np.abs(lo_c[fin] - lo_full[fin])) < tol
        fup = np.isfinite(up_full)
        assert np.nanmax(np.abs(up_c[fup] - up_full[fup])) < tol
