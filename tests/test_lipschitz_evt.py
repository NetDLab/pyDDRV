"""Tests for the EVT (reverse-Weibull) Lipschitz estimator.

Covers:
A. Batched matrix measure agrees with the per-matrix lipschitz.matrix_measure.
B. Batched numerical Jacobian agrees with lipschitz.numerical_jacobian.
C. Affine-Jacobian field: EVT >= the exact corner value (soundness anchor).
D. Interior-peak field: EVT captures the interior sup that corners MISS.
E. Data-driven path recovers a known one-sided Lipschitz constant.
F. estimate_L(L_method="evt") plumbs through and returns EVT diagnostics.
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
    _matrix_measure_batch,
    evt_one_sided_lipschitz,
    evt_one_sided_lipschitz_from_data,
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
