"""Tests for one-sided Lipschitz (oL) estimation.

Anchored on cases with known closed-form answers: for a linear field
``f(x)=Ax`` the one-sided Lipschitz constant equals the matrix measure
``mu(A)`` exactly, for every norm, region, and method.
"""
import numpy as np
import pytest

from pyddrv.lipschitz import (
    derivatives_from_trajectories,
    matrix_measure,
    numerical_jacobian,
    one_sided_lipschitz,
    one_sided_lipschitz_from_data,
)
from pyddrv.systems import bilinear_2d, linear, sample_linear

A_SPIRAL = np.array([[0.0, 1.0], [-1.0, -0.5]])


def test_matrix_measure_2norm_diagonal():
    assert matrix_measure(np.diag([-1.0, -2.0]), "2") == pytest.approx(-1.0)


def test_matrix_measure_inf_and_1():
    A = np.array([[-1.0, 1.0], [0.0, -2.0]])
    # mu_inf: rows -> max(-1+|1|, -2+|0|) = 0
    assert matrix_measure(A, "inf") == pytest.approx(0.0)
    # mu_1: cols -> max(-1+|0|, -2+|1|) = -1
    assert matrix_measure(A, "1") == pytest.approx(-1.0)


def test_matrix_measure_2_is_lambda_max_symmetric():
    A = A_SPIRAL
    expected = np.max(np.linalg.eigvalsh(0.5 * (A + A.T)))
    assert matrix_measure(A, "2") == pytest.approx(expected)


def test_numerical_jacobian_of_linear_field():
    f = linear(A_SPIRAL)
    J = numerical_jacobian(f, np.array([0.3, -0.7]))
    np.testing.assert_allclose(J, A_SPIRAL, atol=1e-5)


@pytest.mark.parametrize("norm", ["2", "inf", "1"])
@pytest.mark.parametrize("method", ["corners", "random", "grid"])
def test_linear_oL_equals_matrix_measure(norm, method):
    # For f(x)=Ax the Jacobian is constant, so L_S = mu(A) for any box/method.
    f = linear(A_SPIRAL)
    L = one_sided_lipschitz(f, center=[0, 0], half=[1.0, 1.0],
                            norm=norm, method=method, n_samples=64)
    assert L == pytest.approx(matrix_measure(A_SPIRAL, norm), abs=1e-5)


def test_analytic_jacobian_matches_numerical_bilinear():
    f, jac = bilinear_2d(eta=0.3, seed=1)
    x = np.array([0.4, -0.2])
    np.testing.assert_allclose(jac(x), numerical_jacobian(f, x), atol=1e-5)


def test_bilinear_corner_is_upper_bound_vs_interior():
    # Affine Jacobian => corner max >= any interior sample.
    f, jac = bilinear_2d(eta=0.5, seed=2)
    L_corner = one_sided_lipschitz(f, [0, 0], [0.7, 0.7], norm="2",
                                   method="corners", jac=jac)
    L_random = one_sided_lipschitz(f, [0, 0], [0.7, 0.7], norm="2",
                                   method="random", jac=jac, n_samples=500)
    assert L_corner >= L_random - 1e-9


def test_data_driven_recovers_linear_measure():
    # Data-driven sup over pairs recovers mu_2(A) for a linear field.
    A = A_SPIRAL
    f = linear(A)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 2))
    F = f(X)
    L = one_sided_lipschitz_from_data(X, F)
    assert L == pytest.approx(matrix_measure(A, "2"), abs=1e-6)


def test_derivatives_from_trajectories_shape_and_value():
    data = sample_linear(A_SPIRAL, n_traj=20, horizon=50, dt=0.01, seed=0)
    X, F = derivatives_from_trajectories(data.states, data.dt)
    assert X.shape == F.shape == (20 * 49, 2)
    # finite-difference derivative should approximate A x along the data
    err = np.linalg.norm(F - X @ A_SPIRAL.T, axis=1)
    assert np.median(err) < 0.05
