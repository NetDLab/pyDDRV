"""Tests for the rigorous epsilon-ERLF verification (grid, ball, Algorithm 1/2).

Uses a NumPy RK4 rollout so the suite runs without torch/GPU. A radial linear
contraction ``xdot = -beta x`` has ``||phi(t,x)|| = e^{-beta t}||x||`` exactly, so
the certifiable rate approaches ``beta`` -- a clean ground truth.
"""
import numpy as np
import pytest

from pyddrv.lipschitz import matrix_measure
from pyddrv.systems import linear, simulate
from pyddrv.verification import estimate_L, find_alpha_min, find_alpha_roa
from pyddrv.verification.ball import alpha_max
from pyddrv.verification.grid import initial_grid, norm_equiv_c, split


def make_rollout(A):
    f = linear(np.asarray(A, dtype=float))

    def rollout(centers, t_grid):
        dt = float(t_grid[1] - t_grid[0])
        return simulate(f, centers, dt, len(t_grid))

    return rollout


# ---------------- grid ----------------
def test_initial_grid_covers_QR_without_overshoot():
    centers, halfs = initial_grid(R=0.7, eps=0.01, norm="2", d=2)
    inf_extent = np.max(np.abs(centers) + halfs[:, None])
    assert inf_extent == pytest.approx(0.7, abs=1e-9)         # exactly Q_R
    assert np.all(np.max(np.abs(centers), axis=1) > 0)        # origin excluded


def test_split_makes_3d_subboxes():
    sub_c, sub_h = split([0.0, 0.0], 0.9)
    assert sub_c.shape == (9, 2)            # 3^2
    assert np.allclose(sub_h, 0.3)          # h/3


def test_norm_equiv_constants():
    assert norm_equiv_c("inf", 5) == 1.0
    assert norm_equiv_c("2", 4) == pytest.approx(2.0)         # sqrt(4)
    assert norm_equiv_c("1", 3) == 3.0


# ---------------- ball / Procedure 1 ----------------
def test_alpha_max_matches_closed_form_pure_contraction():
    # synthetic: ||phi(t)|| = e^{-0.5 t} ||x||, x at norm 1, r=0 -> alpha = 0.5
    t = np.linspace(0, 4, 400)
    x = np.array([1.0, 0.0])
    ys = (np.exp(-0.5 * t)[None, :, None] * x[None, None, :])   # (1,T,2)
    a = alpha_max(ys, t, r=0.0, L=-0.5, norm="2")[0]
    assert a == pytest.approx(0.5, abs=1e-3)


def test_32b_gating_and_same_time_coupling():
    # decaying trajectory; ||phi(t)||_inf = e^{-t} * 0.5
    t = np.linspace(0, 2, 200)
    x = np.array([0.5, 0.0])
    ys = (np.exp(-t)[None, :, None] * x[None, None, :])     # (1,T,2)

    # Large box: (32b) non-binding -> positive rate.
    a_big = alpha_max(ys, t, r=0.05, L=0.0, norm="2", S_radius=10.0)[0]
    assert a_big > 0

    # Tiny box: ||phi||_inf <= R - r = 0.01 - 0.05 < 0 at every t -> all times
    # fail (32b), so no certifying time exists.
    a_tiny = alpha_max(ys, t, r=0.05, L=0.0, norm="2", S_radius=0.01)[0]
    assert a_tiny == -np.inf

    # Enforcing (32b) can only lower the achievable rate (added constraint).
    a_free = alpha_max(ys, t, r=0.05, L=0.0, norm="2", S_radius=None)[0]
    a_mid = alpha_max(ys, t, r=0.05, L=0.0, norm="2", S_radius=0.45)[0]
    assert a_mid <= a_free + 1e-12


def test_alpha_max_decreases_with_radius():
    t = np.linspace(0, 4, 400)
    x = np.array([1.0, 0.0])
    ys = (np.exp(-0.5 * t)[None, :, None] * x[None, None, :])
    a0 = alpha_max(ys, t, r=0.0, L=-0.5, norm="2")[0]
    a1 = alpha_max(ys, t, r=0.1, L=-0.5, norm="2")[0]
    assert a1 < a0                          # larger ball -> harder -> smaller rate


# ---------------- Algorithm 1 ----------------
def test_find_alpha_min_radial_contraction():
    beta = 0.5
    A = -beta * np.eye(2)
    L = -beta                                # mu_2(-beta I) = -beta
    res = find_alpha_min(make_rollout(A), L, R=0.5, eps=0.02, d=2, norm="2",
                         tau=5.0, n_time=300, delta=0.05, max_refine=10)
    # certified rate is a lower bound on beta and converges up toward it
    assert 0.30 < res.alpha_certified <= beta + 1e-6
    assert res.alpha_up.min() == pytest.approx(beta, abs=0.05)


def test_unstable_system_not_certified_positive():
    A = 0.3 * np.eye(2)                      # expanding
    L = 0.3
    res = find_alpha_min(make_rollout(A), L, R=0.5, eps=0.02, d=2, norm="2",
                         tau=3.0, n_time=200, delta=0.1, max_refine=2)
    assert res.alpha_certified < 0           # cannot certify decay


# ---------------- Algorithm 2 ----------------
def test_estimate_L_over_reachable_set():
    # Pure contraction xdot = -0.5 x: trajectories never leave Q_R, L = mu_2(A).
    beta = 0.5
    A = -beta * np.eye(2)
    est = estimate_L(make_rollout(A), R=0.7, eps=0.01, d=2, norm="2", tau=4.0,
                     n_grid=15, slack=0.02, f=linear(A))
    assert est.L == pytest.approx(matrix_measure(A, "2"), abs=1e-3)   # -0.5
    assert est.R_max == pytest.approx(0.7, abs=0.05)                  # no excursion
    assert est.discretization_ok


def test_estimate_L_detects_excursion():
    # A spiral excurses beyond Q_R, so R_bar must exceed R.
    A = np.array([[0.0, 2.0], [-1.0, -0.3]])
    est = estimate_L(make_rollout(A), R=0.7, eps=0.01, d=2, norm="2", tau=4.0,
                     n_grid=15, slack=0.02, f=linear(A))
    assert est.R_max > 0.7              # trajectories leave Q_R
    assert est.R_bar > est.R_max
    assert est.L >= matrix_measure(A, "2") - 1e-6   # L over larger box >= over Q_R


def test_find_alpha_roa_certifies_inner_region():
    beta = 0.5
    A = -beta * np.eye(2)
    res = find_alpha_roa(make_rollout(A), L=-beta, R=0.5, eps=0.02, d=2,
                         norm="2", alpha=0.3, tau=5.0, n_time=300, max_refine=4)
    assert res.n_certified > 0
    assert res.n_certified <= res.n_tested
