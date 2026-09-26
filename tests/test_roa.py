"""Tests for the fused Algorithm 2 (find_alpha_roa_fused) and the RoA
L-estimation (estimate_L_roa, paper sec IX-C)."""
import numpy as np
import pytest

from pyddrv.systems import kuramoto_reduced, linear, simulate
from pyddrv.verification import (
    estimate_L_roa,
    find_alpha_roa,
    find_alpha_roa_fused,
    region_distance_map,
)
from pyddrv.verification import ball
from pyddrv.verification.fused import _make_jax_roa_kernel
from pyddrv.verification.grid import norm_equiv_c

jax = pytest.importorskip("jax", reason="JAX not installed")
import jax.numpy as jnp  # noqa: E402

R = 0.7


def _jax_linear(A):
    Aj = jnp.asarray(np.asarray(A).T)
    return lambda X: X @ Aj.astype(X.dtype)


def test_roa_kernel_matches_ball_alpha_max():
    """No-region RoA kernel == ball.alpha_max at radius c*h, S_radius=None."""
    d, tau, n_steps = 3, 3.0, 200
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d))
    A = A - (np.max(np.linalg.eigvals(A).real) + 0.5) * np.eye(d)
    f_np, f_jax = linear(A), _jax_linear(A)
    L = 0.5
    cen = rng.uniform(-R, R, (300, d))
    hlf = rng.uniform(0.01, 0.08, 300)
    c = norm_equiv_c("2", d)
    t_grid = np.linspace(0, tau, n_steps + 1)
    ys = simulate(f_np, cen, dt=tau / n_steps, horizon=n_steps + 1)
    ref = ball.alpha_max(ys, t_grid, c * hlf, L, "2", None)     # (32a) only

    kernel = _make_jax_roa_kernel(f_jax, "2", use_region=False)
    dt = tau / n_steps
    tk = dt * np.arange(1, n_steps + 1)
    a, _a3 = kernel(jnp.asarray(cen, jnp.float32), jnp.asarray(hlf, jnp.float32),
                    jnp.asarray(tk, jnp.float32), jnp.float32(dt),
                    jnp.float32(L), jnp.float32(R), jnp.float32(c),
                    jnp.zeros((1, 1), jnp.float32), jnp.float32(-R),
                    jnp.float32(2 * R))
    a = np.asarray(a, float)
    fin = np.isfinite(ref)
    assert np.array_equal(np.isfinite(a), fin)
    assert np.nanmax(np.abs(a[fin] - ref[fin])) < 1e-4


def test_region_distance_map_sound():
    """The inner distance map never overstates the true inf-norm distance to the
    complement of the union."""
    centers = np.array([[0.0, 0.0], [0.4, 0.0]])
    halfs = np.array([0.2, 0.2])          # union: [-0.2,0.6] x [-0.2,0.2]
    dmap, lo, cellw = region_distance_map(centers, halfs, R_dom=1.0, n_cells=400)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (4000, 2))
    idx = np.clip(((pts - lo) / cellw).astype(int), 0, 399)
    d_claim = dmap[idx[:, 0], idx[:, 1]]
    # true inner inf-distance to the complement of the union of the two boxes
    inside = (np.abs(pts[:, 1]) <= 0.2) & (pts[:, 0] >= -0.2) & (pts[:, 0] <= 0.6)
    d_true = np.where(
        inside,
        np.minimum.reduce([np.abs(pts[:, 1] + 0.2), np.abs(0.2 - pts[:, 1]),
                           np.abs(pts[:, 0] + 0.2), np.abs(0.6 - pts[:, 0])]),
        0.0)
    assert np.all(d_claim <= d_true + 1e-9)          # sound (never overstates)
    # and not vacuous: deep-interior points get a positive distance
    deep = inside & (d_true > 0.1)
    assert np.all(d_claim[deep] > 0.05)


def test_estimate_L_roa_partition_saddle():
    """Saddle A=diag(-1, +1): part of dQ_R returns, part diverges (Gamma_nr
    nonempty); the one-sided checks verify after grid refinement."""
    A = np.diag([-1.0, 1.0])
    f_np = linear(A)

    def rollout(c, t):
        t = np.asarray(t, float)
        return simulate(f_np, np.asarray(c, float), dt=float(t[1] - t[0]),
                        horizon=len(t))

    est = estimate_L_roa(rollout, R, d=2, norm="inf", tau=2.0, n_time=200,
                         n_grid=41, jac=lambda x: A, refine_grid=6,
                         n_grid_max=2049)
    assert est.discretization_ok
    assert np.isfinite(est.L)
    assert est.R_bar >= R                     # covers Q_R itself (regime i)


def test_kuramoto_two_pass_roa():
    """Kuramoto (3 oscillators, k=10): pass 1 (Trim=False) grows a nonempty
    region containing the sync equilibrium; pass 2 (Trim=True vs pass-1 union)
    prunes to a subset."""
    f_np, L = kuramoto_reduced(k=10.0, n=3)
    f_jax_ = _jax_kuramoto(10.0, 3)
    Rk, eps, tau, alpha = np.pi, np.pi / 81, 1.9, 1.0
    r1 = find_alpha_roa_fused(f_jax_, L, Rk, eps, 2, alpha, norm="inf", tau=tau,
                              n_steps=200, max_refine=3)
    assert r1.n_certified > 0
    # contains cubes close to the origin (sync basin)
    dmin = np.min(np.max(np.abs(r1.centers), axis=1))
    assert dmin < 0.2
    r2 = find_alpha_roa_fused(f_jax_, L, Rk, eps, 2, alpha, norm="inf", tau=tau,
                              n_steps=200, max_refine=3,
                              grid0=(r1.centers, r1.halfs),
                              trim_region=(r1.centers, r1.halfs), raster_n=2187)
    # pass 2 may SPLIT pass-1 cubes (count can grow) but every certified cube is
    # contained in the pass-1 region, so certified AREA can only shrink
    area = lambda r: float(np.sum((2 * r.halfs) ** 2))
    assert 0 < r2.n_certified
    assert area(r2) <= area(r1) + 1e-9


def _jax_kuramoto(k, n):
    def f(X):
        phi = jnp.concatenate([X, jnp.zeros((X.shape[0], 1), X.dtype)], axis=1)
        diff = phi[:, None, :] - phi[:, :, None]
        dtheta = (k / n) * jnp.sin(diff).sum(axis=2)
        return dtheta[:, :n - 1] - dtheta[:, n - 1:n]
    return f


def test_roa_fused_matches_materialized_linear():
    """Trim=False fused Algorithm 2 certifies the same cubes as the numpy
    find_alpha_roa on a radially-stable linear system."""
    beta = 0.8
    A = -beta * np.eye(2)
    f_np, f_jax = linear(A), _jax_linear(A)

    def rollout(c, t):
        t = np.asarray(t, float)
        return simulate(f_np, np.asarray(c, float), dt=float(t[1] - t[0]),
                        horizon=len(t))

    alpha, tau, m = 0.4, 3.0, 2
    r_old = find_alpha_roa(rollout, -beta, R, 0.05, 2, alpha, norm="2", tau=tau,
                           n_time=201, max_refine=m)
    r_new = find_alpha_roa_fused(f_jax, -beta, R, 0.05, 2, alpha, norm="2",
                                 tau=tau, n_steps=200, max_refine=m)
    assert r_new.n_certified == r_old.n_certified
    assert r_new.n_tested == r_old.n_tested


def test_region_distance_map_edge_of_box_is_outside():
    """Points beyond [-R, R]^d are outside the region: a region covering the
    whole box has distance 0 on the edge cells and grows inward, instead of
    the array edge being ignored (which overstated distances near the edge
    and gave no finite distances at all for a full box)."""
    n = 9
    dmap, lo, cellw = region_distance_map(np.zeros((1, 2)), np.array([1.0]),
                                          1.0, n)
    assert np.all(dmap[0, :] == 0.0) and np.all(dmap[:, -1] == 0.0)
    assert dmap[n // 2, n // 2] == pytest.approx((n // 2) * cellw)
    # every reported distance fits inside the box
    idx = np.arange(n)
    cell_far = np.maximum(np.abs(lo + idx * cellw), np.abs(lo + (idx + 1) * cellw))
    far = np.maximum(cell_far[:, None], cell_far[None, :])
    assert np.all(far + dmap <= 1.0 + 1e-12)


@pytest.mark.parametrize("norm", ["2", "inf", "1"])
def test_region_distance_map_eps_ball_is_sound(norm):
    """With ball=(eps, norm) the target ball joins the region; every cell's
    reported inf-norm clearance stays inside the ball."""
    n, rad = 81, 0.5
    dmap, lo, cellw = region_distance_map(np.empty((0, 2)), np.empty(0), 1.0,
                                          n, ball=(rad, norm))
    assert dmap.max() > 0.2
    idx = np.arange(n)
    cell_far = np.maximum(np.abs(lo + idx * cellw), np.abs(lo + (idx + 1) * cellw))
    # farthest point of the cell grown by its clearance, per axis
    gx = cell_far[:, None] + dmap
    gy = cell_far[None, :] + dmap
    val = {"2": np.hypot(gx, gy), "inf": np.maximum(gx, gy), "1": gx + gy}[norm]
    assert np.all(val[dmap > 0] <= rad + 1e-12)
