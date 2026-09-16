r"""Tests for the trajectory-local contraction bound (contraction.py).

T1 linear sanity, T2 property test (separation <= bound), T3 Riccati guard,
T4 regression (forcing the global-L factor reproduces the old formula).
All CPU / NumPy, small samples -- safe to run alongside a GPU experiment.
"""
import numpy as np
import pytest

from pyddrv.systems import bilinear_2d, bilinear_3d, kuramoto_reduced
from pyddrv.verification.contraction import (
    accumulate_AI,
    alpha_max_local,
    bilinear_grad_mon,
    bilinear_jac_batched,
    finite_difference_jac,
    jac_lipschitz_affine,
    jac_lipschitz_numeric,
    kuramoto_jac_batched,
    local_bound,
    matrix_measure_2,
    rollout_with_measure,
)

_A2 = np.array([[0.0, 2.0], [-1.0, -1.0]])
_A3 = np.array([[-1.0, 0.0, 0.0], [0.5, -1.0, 0.0], [0.5, 0.5, -1.0]])


def _rk4(f, x, dt, n):
    ys = [x.copy()]
    for _ in range(n):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(x.copy())
    return np.stack(ys, axis=1)                    # (N, T, d)


# --------------------------------------------------------------------------- #
# T1 -- linear sanity: M = 0, rho(t) = r * exp(mu_2(A) t)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("A", [
    np.array([[-1.0, 0.3], [-0.3, -1.0]]),        # normal, mu_2 < 0
    np.array([[-1.0, 10.0], [0.0, -1.0]]),        # non-normal: eig -1, mu_2 = 4
])
def test_T1_linear(A):
    d = 2
    f = lambda X: X @ A.T
    jac = lambda X: np.broadcast_to(A, (len(X), d, d))
    mu = matrix_measure_2(A)
    dt, n = 0.02, 100
    tk = dt * np.arange(n + 1)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, (50, d))
    _, Ai, Ii = rollout_with_measure(f, jac, x, dt, n)
    # A(t) == mu * t exactly (constant a); bound with M=0 == r*exp(mu t)
    assert np.allclose(Ai, mu * tk[None, :], atol=1e-9)
    r = 0.05
    bnd = local_bound(r, Ai, Ii, 0.0, 100.0, np.broadcast_to(tk, Ai.shape))
    assert np.allclose(bnd, r * np.exp(mu * tk)[None, :], rtol=1e-9)
    # empirical: ||phi(t,y)-phi(t,x)|| <= bound(t)
    for _ in range(5):
        u = rng.standard_normal(x.shape); u *= r / np.linalg.norm(u, axis=1, keepdims=True)
        yx = _rk4(f, x + u, dt, n)
        xx = _rk4(f, x, dt, n)
        sep = np.linalg.norm(yx - xx, axis=2)
        assert (sep <= bnd * (1 + 1e-6)).all()


# --------------------------------------------------------------------------- #
# T2 -- property test: separation <= bound over many (x, y, t) triples
# --------------------------------------------------------------------------- #
def _system(name):
    # NOTE: the repo's analytic `jac` from bilinear_2d/3d is SINGLE-POINT
    # (it reshapes the batch to one vector); use the batched finite-difference
    # Jacobian for per-sample matrix measures.
    if name == "bilin2":
        f, _ = bilinear_2d(eta=0.3, seed=1)
        jac = finite_difference_jac(f)
        return f, jac, 2, jac_lipschitz_affine(jac, 2), 0.7
    if name == "bilin3":
        f, _ = bilinear_3d(eta=0.3, seed=1)
        jac = finite_difference_jac(f)
        return f, jac, 3, jac_lipschitz_affine(jac, 3), 0.7
    if name == "kur2":
        f, _ = kuramoto_reduced(k=10.0, n=3)
        jac = finite_difference_jac(f)
        return f, jac, 2, jac_lipschitz_numeric(jac, np.pi, 2), np.pi
    if name == "kur3":
        f, _ = kuramoto_reduced(k=10.0, n=4)
        jac = finite_difference_jac(f)
        return f, jac, 3, jac_lipschitz_numeric(jac, np.pi, 3), np.pi
    raise ValueError(name)


@pytest.mark.parametrize("name", ["bilin2", "bilin3", "kur2", "kur3"])
def test_T2_property(name):
    f, jac, d, M, R = _system(name)
    L = 20.0                                       # loose global L for the min()
    dt, n = 1.9 / 120, 120
    tk = dt * np.arange(n + 1)
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.6 * R, 0.6 * R, (60, d))
    _, Ai, Ii = rollout_with_measure(f, jac, x, dt, n)
    ratios = []
    for r in [0.01, 0.03, 0.08]:
        bnd = local_bound(r, Ai, Ii, M, L, np.broadcast_to(tk, Ai.shape))
        xx = _rk4(f, x, dt, n)
        for _ in range(4):
            u = rng.standard_normal(x.shape)
            u *= r / np.linalg.norm(u, axis=1, keepdims=True)
            yy = _rk4(f, x + u, dt, n)
            sep = np.linalg.norm(yy - xx, axis=2)
            assert (sep <= bnd * (1 + 1e-3)).all(), \
                f"{name}: max sep/bound = {np.nanmax(sep / bnd):.4f}"
            fin = np.isfinite(bnd) & (bnd > 0)
            ratios.append(np.median(sep[fin] / bnd[fin]))
    print(f"\n[T2 {name}] M={M:.3f}  median sep/bound = {np.median(ratios):.3f} "
          f"(1 = tight, <<1 = conservative)")


# --------------------------------------------------------------------------- #
# T3 -- Riccati guard: large r trips r*M*I >= 1 -> finite fallback to global L
# --------------------------------------------------------------------------- #
def test_T3_guard():
    f, _ = bilinear_3d(eta=0.6, seed=2)
    jac = finite_difference_jac(f)
    M = jac_lipschitz_affine(jac, 3)
    dt, n = 1.9 / 120, 120
    tk = dt * np.arange(n + 1)
    x = np.array([[0.5, -0.4, 0.3]])
    _, Ai, Ii = rollout_with_measure(f, jac, x, dt, n)
    r = 12.0                                       # large -> r*(M/2)*I crosses 1
    L = 15.0
    bnd = local_bound(r, Ai, Ii, M, L, np.broadcast_to(tk, Ai.shape))
    assert np.isfinite(bnd).all()                  # no NaN/Inf leaks
    tripped = (r * 0.5 * M * Ii) >= 1.0            # guard uses M/2
    assert tripped.any()                           # guard actually trips
    old = r * np.exp(L * tk)[None, :]
    assert np.allclose(bnd[tripped], old[tripped])  # fell back to global L


# --------------------------------------------------------------------------- #
# T4 -- regression: A = L*t, M = 0 reproduces the global-L certificate exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2, 3])
def test_T5_bilinear_jac_analytic_vs_fd(d):
    rng = np.random.default_rng(0)
    B = rng.normal(0, 0.3, (d, 3 if d == 2 else 6))
    A = _A2 if d == 2 else _A3
    f = (bilinear_2d(eta=0.3, B1=B)[0] if d == 2
         else bilinear_3d(eta=0.3, B2=B)[0])
    ja = bilinear_jac_batched(A, B, bilinear_grad_mon(d))
    jf = finite_difference_jac(f)
    X = rng.uniform(-0.8, 0.8, (200, d))
    assert np.abs(ja(X) - jf(X)).max() < 1e-7


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_T5_kuramoto_jac_analytic_vs_fd(n):
    rng = np.random.default_rng(0)
    f, _ = kuramoto_reduced(k=10.0, n=n)
    ja = kuramoto_jac_batched(10.0, n)
    jf = finite_difference_jac(f)
    X = rng.uniform(-2.5, 2.5, (200, n - 1))
    assert np.abs(ja(X) - jf(X)).max() < 1e-6


def test_T4_regression_reduces_to_global_L():
    f, jac = bilinear_2d(eta=0.3, seed=1)
    dt, n = 1.9 / 120, 120
    tk = dt * np.arange(n + 1)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, (40, 2))
    ys = _rk4(f, x, dt, n)
    L, r = 6.0, 0.05
    N, T = len(x), n + 1
    A_globL = L * np.broadcast_to(tk, (N, T))
    a_loc = alpha_max_local(ys, A_globL, np.zeros((N, T)), tk, r, 0.0, L)
    # direct old-formula alpha_max with r*exp(L t)
    Vx = np.linalg.norm(x, axis=1)[:, None]
    Vphi = np.linalg.norm(ys, axis=2)
    bnd = r * np.exp(L * tk)[None, :]
    margin, denom = Vx - r, Vphi + bnd
    valid = (margin > 0) & (denom > 0) & (tk[None, :] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(valid, np.log(np.where(valid, margin / denom, 1.0))
                        / np.where(tk > 0, tk, 1.0)[None, :], -np.inf)
    a_old = np.where(valid, rate, -np.inf).max(axis=1)
    assert np.allclose(a_loc, a_old, atol=1e-9, equal_nan=True)
