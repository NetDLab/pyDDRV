"""Tests for the SOS exponential-rate baseline (skipped without the [sos] deps).

Anchored on a linear system with a known answer: for xdot = -beta x and
V = x^T x, V_dot = -2 beta V, so the SOS-certified rate is exactly beta.
"""
import numpy as np
import pytest

pytest.importorskip("SumOfSquares")
import sympy as sp  # noqa: E402

from pyddrv.baselines import (  # noqa: E402
    build_polynomial_field,
    certify_rate_sos,
    certify_rate_sos_box,
)


def test_sos_linear_rate_matches_beta():
    x1, x2 = sp.symbols("x1 x2", real=True)
    xs = [x1, x2]
    beta = 0.5
    f = build_polynomial_field(-beta * np.eye(2), np.zeros((2, 1)), [x1**2], xs)
    alpha = certify_rate_sos(f, xs, R=1.0)
    assert alpha == pytest.approx(beta, abs=0.05)


def test_sos_bilinear_positive_rate():
    x1, x2 = sp.symbols("x1 x2", real=True)
    xs = [x1, x2]
    A = np.array([[0.0, 2.0], [-1.0, -1.0]])
    rng = np.random.default_rng(0)
    B = rng.normal(0.0, 0.3, size=(2, 3))
    f = build_polynomial_field(A, B, [x1**2, x1 * x2, x2**2], xs)
    alpha = certify_rate_sos(f, xs, R=0.7)
    assert np.isfinite(alpha) and alpha > 0.1


def test_sos_box_linear_healthy_rate():
    # 3D linear (non-normal, eigenvalues -1): box V-s certifies Q_R inside an
    # invariant ellipse at a healthy rate (machinery sanity check).
    x = sp.symbols("x1 x2 x3", real=True)
    A = np.array([[-1., 0., 0.], [0.5, -1., 0.], [0.5, 0.5, -1.]])
    mons = [x[0]**2, x[0]*x[1], x[0]*x[2], x[1]**2, x[1]*x[2], x[2]**2]
    f = build_polynomial_field(A, np.zeros((3, 6)), mons, list(x))
    alpha = certify_rate_sos_box(f, list(x), R=0.7)
    assert np.isfinite(alpha) and alpha > 0.3


def test_sos_box_fails_on_divergent_draw():
    # A draw whose trajectories leave Q_R has no invariant ellipse containing the
    # box -> box V-s returns nan (matching RLF), even though the inner-ellipse
    # certify_rate_sos returns a (misleading) finite rate.
    x = sp.symbols("x1 x2 x3", real=True)
    A = np.array([[-1., 0., 0.], [0.5, -1., 0.], [0.5, 0.5, -1.]])
    mons = [x[0]**2, x[0]*x[1], x[0]*x[2], x[1]**2, x[1]*x[2], x[2]**2]
    B = np.random.default_rng(4).normal(0.0, 0.6, size=(3, 6))
    f = build_polynomial_field(A, B, mons, list(x))
    assert np.isnan(certify_rate_sos_box(f, list(x), R=0.7))
    assert np.isfinite(certify_rate_sos(f, list(x), R=0.7))   # inner ellipse "works"
