"""Tests for the EVT (reverse-Weibull) Lipschitz estimator.

Covers:
A. Batched matrix measure agrees with the per-matrix lipschitz.matrix_measure.
B. Batched numerical Jacobian agrees with lipschitz.numerical_jacobian.
C. Affine-Jacobian field: EVT >= the exact corner value (soundness anchor).
D. Interior-peak field: EVT captures the interior sup that corners MISS.
E. Data-driven path recovers a known one-sided Lipschitz constant.
F. estimate_L(L_method="evt") plumbs through and returns EVT diagnostics.
G. Finite differences of a float32 field use a step matched to its precision.
H. Ill-posed fits (saturated or quantized block maxima) fall back to the sample
   max plus a calibrated margin, flagged in EVTEstimate.status; well-posed fits
   keep the reverse-Weibull bound.
I. The damped pendulum over the large reachable box of R = 3 gives L close to
   the exact (sqrt(5) - 1)/2 and not below it.
"""
import numpy as np
import pytest

from pyddrv.lipschitz import (
    matrix_measure,
    numerical_jacobian,
    one_sided_lipschitz,
)
from pyddrv.lipschitz_evt import (
    _batched_numerical_jacobian,
    _fallback_factor,
    _matrix_measure_batch,
    evt_one_sided_lipschitz,
    evt_one_sided_lipschitz_from_data,
    evt_sup_estimate,
)
from pyddrv.systems import bilinear_2d

pytest.importorskip("scipy")


def _interior_peak_field():
    """f = [x2, -sin(3 x1) - x2]: mu(Jac) peaks in the interior (cos(3x1)=-1),
    NOT at a box corner, so corner sampling under-estimates the true sup."""
    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return np.stack([x2, -np.sin(3 * x1) - x2], axis=1)

    def jac(x):
        x1 = float(x[0])
        return np.array([[0.0, 1.0], [-3 * np.cos(3 * x1), -1.0]])

    return f, jac


def test_matrix_measure_batch_matches_scalar():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((7, 3, 3))
    for norm in ("2", "inf", "1"):
        batch = _matrix_measure_batch(A, norm=norm)
        scalar = np.array([matrix_measure(a, norm=norm) for a in A])
        assert np.allclose(batch, scalar, atol=1e-10)
    P = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]])
    batch = _matrix_measure_batch(A, P=P)
    scalar = np.array([matrix_measure(a, P=P) for a in A])
    assert np.allclose(batch, scalar, atol=1e-8)


def test_batched_numerical_jacobian_matches_single():
    f, _ = bilinear_2d(eta=0.4, seed=1)
    X = np.array([[0.3, -0.2], [0.5, 0.5], [-0.1, 0.4]])
    Jb = _batched_numerical_jacobian(f, X)
    for k, x in enumerate(X):
        assert np.allclose(Jb[k], numerical_jacobian(f, x), atol=1e-5)


def test_evt_affine_jacobian_is_sound_anchor():
    # affine Jacobian -> corners is EXACT; EVT must not report below it.
    f, jac = bilinear_2d(eta=0.3, seed=0)
    corners = one_sided_lipschitz(f, [0, 0], [1.0, 1.0], norm="2",
                                  method="corners", jac=jac)
    est = evt_one_sided_lipschitz(f, [0, 0], [1.0, 1.0], norm="2", jac=jac,
                                  rho=0.95, n_blocks=60, n_per_block=1500)
    assert est.L >= corners - 1e-9          # never below an attained value
    assert est.L == pytest.approx(corners, rel=0.05)


def test_evt_captures_interior_peak_that_corners_miss():
    f, jac = _interior_peak_field()
    box = ([0.0, 0.0], [1.5, 1.5])
    corners = one_sided_lipschitz(f, *box, norm="2", method="corners", jac=jac)
    truth = one_sided_lipschitz(f, *box, norm="2", method="random",
                                n_samples=100_000, jac=jac)
    est = evt_one_sided_lipschitz(f, *box, norm="2", jac=jac, rho=0.95,
                                  n_blocks=80, n_per_block=3000)
    assert corners < 0.6 * truth            # corners genuinely under-estimates
    assert est.L >= 0.95 * truth            # EVT recovers (covers) the true sup


def test_evt_from_data_recovers_linear_constant():
    # x_dot = A x: one-sided constant = mu_2(A) = lambda_max((A+A^T)/2).
    A = np.array([[-0.5, 2.0], [-1.0, -1.5]])
    true_L = matrix_measure(A, norm="2")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(4000, 2))
    F = X @ A.T
    est = evt_one_sided_lipschitz_from_data(X, F, rho=0.9, n_blocks=60,
                                            n_per_block=1500)
    # linear field: the ratio equals mu exactly along the maximizing direction;
    # EVT should land at true_L (within sampling + confidence margin), never far below.
    assert est.L >= true_L - 0.05
    assert est.L == pytest.approx(true_L, abs=0.15)


def test_estimate_L_evt_plumbs_through():
    from pyddrv.systems import simulate
    from pyddrv.verification.estimate import estimate_L

    f, jac = bilinear_2d(eta=0.3, seed=0)

    def rollout(pts, t):
        return simulate(f, np.asarray(pts, float), float(t[1] - t[0]), len(t))

    est = estimate_L(rollout, R=0.7, eps=0.01, d=2, norm="2", tau=3.0,
                     jac=jac, f=f, L_method="evt", rho=0.95,
                     evt_blocks=40, evt_per_block=1000)
    assert est.method == "evt"
    assert est.evt is not None
    assert np.isfinite(est.L) and est.L > 0
    # EVT L over the reachable box should be >= the corner L over the same box
    corner = estimate_L(rollout, R=0.7, eps=0.01, d=2, norm="2", tau=3.0,
                        jac=jac, f=f, L_method="corners")
    assert est.L >= corner.L - 1e-6


def test_jacobian_affinity_test():
    from pyddrv.verification.estimate import _jacobian_is_affine
    _, jac_affine = bilinear_2d(eta=0.3, seed=0)      # J affine in x
    _, jac_nonaff = _interior_peak_field()              # J has cos(3 x1)
    assert _jacobian_is_affine(jac_affine, 1.0, 2)
    assert not _jacobian_is_affine(jac_nonaff, 1.0, 2)


def test_estimate_L_auto_resolves_by_affinity():
    from pyddrv.systems import simulate
    from pyddrv.verification.estimate import estimate_L

    def rollout_of(f):
        return lambda pts, t: simulate(f, np.asarray(pts, float),
                                       float(t[1] - t[0]), len(t))

    # affine analytic Jacobian -> corners (exact), identical to explicit corners
    f, jac = bilinear_2d(eta=0.3, seed=0)
    auto = estimate_L(rollout_of(f), R=0.7, eps=0.01, d=2, tau=3.0, jac=jac, f=f)
    corn = estimate_L(rollout_of(f), R=0.7, eps=0.01, d=2, tau=3.0, jac=jac, f=f,
                      L_method="corners")
    assert auto.method == "corners" and auto.evt is None
    assert auto.L == pytest.approx(corn.L)

    # nonlinear analytic Jacobian -> evt; no Jacobian at all -> evt
    g, jac_g = _interior_peak_field()
    with_jac = estimate_L(rollout_of(g), R=0.5, eps=0.01, d=2, tau=2.0, jac=jac_g,
                          f=g, evt_blocks=30, evt_per_block=500)
    no_jac = estimate_L(rollout_of(g), R=0.5, eps=0.01, d=2, tau=2.0, f=g,
                        evt_blocks=30, evt_per_block=500)
    assert with_jac.method == "evt" and with_jac.evt is not None
    assert no_jac.method == "evt" and no_jac.evt is not None


# --------------------------------------------------------------------------- #
# precision of the finite-difference Jacobian, ill-posed fits, pendulum
# --------------------------------------------------------------------------- #
PENDULUM_L = (np.sqrt(5.0) - 1.0) / 2.0   # sup_theta mu_2([[0,1],[-cos,-1]])


def _pendulum(dtype):
    def f(x):
        x = np.asarray(x).astype(dtype)
        return np.stack([x[:, 1], -np.sin(x[:, 0]) - x[:, 1]], axis=1)
    return f


def _pendulum_mu(Z):
    J = np.zeros((len(Z), 2, 2))
    J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = 1.0, -np.cos(Z[:, 0]), -1.0
    return _matrix_measure_batch(J)


def test_numerical_jacobian_step_follows_field_precision():
    # float32 field: a fixed 1e-6 step is dominated by rounding (error ~0.1);
    # the precision-matched step is accurate to ~1e-4.
    f32 = _pendulum(np.float32)
    Z = np.random.default_rng(0).uniform(-7.0, 7.0, size=(2000, 2))
    exact = _pendulum_mu(Z)
    auto = _matrix_measure_batch(_batched_numerical_jacobian(f32, Z))
    fixed = _matrix_measure_batch(_batched_numerical_jacobian(f32, Z, eps=1e-6))
    assert np.max(np.abs(auto - exact)) < 1e-4
    assert np.max(np.abs(fixed - exact)) > 1e-2
    # float64 field: unchanged accuracy
    f64 = _pendulum(np.float64)
    auto64 = _matrix_measure_batch(_batched_numerical_jacobian(f64, Z))
    assert np.max(np.abs(auto64 - exact)) < 1e-8


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_evt_pendulum_saturated_box(dtype):
    # Reachable box of the pendulum from Q_3: mu(Jac) reaches its maximum on
    # the lines cos(x1) = -1, which the box crosses, so nearly every block
    # attains it. The estimate must be close to the true value and not below.
    est = evt_one_sided_lipschitz(_pendulum(dtype), [0.0, 0.0], [7.08, 7.08],
                                  norm="2")
    assert PENDULUM_L <= est.L <= PENDULUM_L + 2e-3
    assert est.L >= est.sample_max
    if dtype is np.float32:
        assert est.resolution > 0
        assert est.status in ("fallback", "degenerate")
        assert est.reason or est.status == "degenerate"


def test_evt_pendulum_roa_estimate_jax():
    # The path verify_roa(pendulum, R=3, d=2, tau=5) takes with the default
    # L_method="auto": previously L = 1.78e6.
    jnp = pytest.importorskip("jax.numpy")
    from pyddrv.api import _rk4_rollout
    from pyddrv.verification.estimate import estimate_L_roa

    def pendulum(x):
        return jnp.stack([x[..., 1], -jnp.sin(x[..., 0]) - x[..., 1]], axis=-1)

    est = estimate_L_roa(_rk4_rollout(pendulum), 3.0, 2, tau=5.0, f=pendulum)
    assert est.method == "evt"
    assert PENDULUM_L <= est.L <= PENDULUM_L + 2e-3


def test_evt_quantized_samples_fall_back():
    # Samples rounded to a grid of 0.01: most block maxima tie, the continuous
    # reverse-Weibull model does not apply, and the old bootstrap SE exploded.
    rng = np.random.default_rng(3)

    def sampler(n):
        q = np.cos(rng.uniform(-2.0, 2.0, size=n)) + 0.5 * rng.uniform(size=n)
        return np.round(q, 2)                     # true sup = 1.5

    est = evt_sup_estimate(sampler, n_blocks=100, n_per_block=1000,
                           resolution=0.005, max_shape=2.0)
    assert est.status == "fallback" and not est.well_posed
    assert "distinct" in est.reason
    assert 1.5 <= est.L <= 1.6
    assert "FALLBACK" in est.summary()


def test_evt_constant_quantity_is_degenerate():
    est = evt_sup_estimate(lambda n: np.full(n, 0.3), n_blocks=20,
                           n_per_block=100)
    assert est.status == "degenerate" and est.L == pytest.approx(0.3)


@pytest.mark.parametrize("c", [2.0, 4.0])
def test_evt_well_posed_fit_keeps_reverse_weibull_bound(c):
    # q = 1 - U^(1/c): block maxima are reverse Weibull with shape c, sup = 1.
    from scipy import stats
    rng = np.random.default_rng(5)
    est = evt_sup_estimate(lambda n: 1.0 - rng.uniform(size=n) ** (1.0 / c),
                           n_blocks=100, n_per_block=1000)
    assert est.status == "fit" and est.validated
    assert est.shape == pytest.approx(c, rel=0.3)
    z = stats.norm.ppf(est.rho)
    assert est.L == pytest.approx(max(est.gamma + z * est.se, est.sample_max))
    assert 1.0 <= est.L < 1.3                     # covers, and se is not runaway
    assert est.se <= est.spread


def test_fallback_factor_covers_endpoint():
    # Monte Carlo check of the pivot: for reverse-Weibull block maxima with any
    # shape c <= max_shape, max + k * range >= endpoint with probability >= rho.
    rho, m, cmax = 0.95, 100, 3.0
    k = _fallback_factor(rho, m, cmax)
    rng = np.random.default_rng(7)
    for c in (0.5, 1.0, 2.0, cmax):
        s = 2.0 - 0.3 * rng.exponential(size=(4000, m)) ** (1.0 / c)
        top, rng_ = s.max(axis=1), s.max(axis=1) - s.min(axis=1)
        cover = np.mean(top + k * rng_ >= 2.0)
        assert cover >= rho - 0.015
