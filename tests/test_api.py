"""Tests for the high-level API (pyddrv.api).

Anchored on systems with known answers: the stable linear spiral
A = [[0, 2], [-1, -1]] has spectral abscissa 0.5, so any certified rate must
land in (0, ~0.5]; the damped pendulum exercises the plain-NumPy fallback
(np.stack/np.sin do not trace under jit); equilibrium shifts must reproduce
the origin-centered answer.
"""
import numpy as np
import pytest

from pyddrv import (
    StabilityReport,
    shifted_field,
    verify_roa,
    verify_stability,
)
from pyddrv.systems import damped_pendulum, linear

A_SPIRAL = np.array([[0.0, 2.0], [-1.0, -1.0]])   # eigenvalues -0.5 +/- 1.32j


def _spiral_report(**kw):
    kw.setdefault("tau", 3.0)
    kw.setdefault("eps", 0.07)
    kw.setdefault("delta", 0.3)
    kw.setdefault("max_refine", 4)
    return verify_stability(linear(A_SPIRAL), R=0.7, d=2, **kw)


def test_verify_stability_linear_certifies():
    rep = _spiral_report()
    assert isinstance(rep, StabilityReport)
    assert rep.certified
    # rate must be positive and cannot exceed the spectral abscissa (0.5)
    # by more than numerical slack
    assert 0.0 < rep.alpha <= 0.55
    assert rep.alpha <= rep.alpha_upper + 1e-9
    assert rep.discretization_ok
    assert "certified" in rep.summary()


def test_verify_stability_given_L_matches_estimated():
    est = _spiral_report()
    given = _spiral_report(L=est.L)
    assert given.alpha == pytest.approx(est.alpha, abs=1e-9)
    assert given.lipschitz.discretization_ok


def test_verify_stability_numpy_fallback_for_nontraceable_field():
    # np.stack/np.sin inside the pendulum field cannot trace under jit, so
    # backend="auto" must fall back to the NumPy kernel and still certify.
    f = damped_pendulum(damping=1.0)
    rep = verify_stability(f, R=0.5, d=2, tau=6.0, eps=0.05, delta=0.5,
                           max_refine=2)
    assert rep.backend == "numpy"
    assert rep.certified
    assert rep.alpha > 0.0


def test_verify_stability_equilibrium_shift():
    x_star = np.array([1.3, -0.4])

    def g(x):
        return (x - x_star) @ A_SPIRAL.T          # equilibrium at x_star

    origin = _spiral_report()
    shifted = verify_stability(g, R=0.7, d=2, equilibrium=x_star, tau=3.0,
                               eps=0.07, delta=0.3, max_refine=4)
    assert shifted.certified
    assert shifted.alpha == pytest.approx(origin.alpha, rel=1e-6)
    np.testing.assert_allclose(shifted.equilibrium, x_star)


def test_verify_stability_ladder_method():
    rep = _spiral_report(method="ladder")
    assert rep.certified
    assert 0.0 < rep.alpha <= 0.55


def test_shifted_field_numpy():
    x_star = np.array([2.0, 1.0])
    f = linear(A_SPIRAL)
    g = shifted_field(f, x_star)
    y = np.array([[0.1, -0.2], [0.0, 0.0]])
    np.testing.assert_allclose(g(y), f(y + x_star))


def test_verify_stability_unstable_is_not_certified():
    f = linear(np.array([[0.3, 0.0], [0.0, 0.3]]))   # expanding: no rate
    rep = verify_stability(f, R=0.5, d=2, tau=2.0, eps=0.05, max_refine=1)
    assert not rep.certified


@pytest.mark.parametrize("trim", [False, True])
def test_verify_roa_linear(trim):
    pytest.importorskip("jax", reason="verify_roa needs the JAX kernel")
    rep = verify_roa(linear(A_SPIRAL), R=1.0, d=2, alpha=0.2, tau=3.0,
                     eps=1.0 / 27.0, max_refine=2, trim=trim, raster_n=243)
    assert rep.n_certified > 0
    assert rep.volume > 0.0
    # certified cubes must lie inside Q_R (float32 storage -> 1e-6 slack)
    assert np.all(np.abs(rep.centers) + rep.halfs[:, None] <= 1.0 + 1e-6)
    assert rep.trim == trim
    assert "verify_roa" in rep.summary()
