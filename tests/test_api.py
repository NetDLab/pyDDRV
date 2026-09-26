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
from pyddrv.verification import find_alpha_roa_fused

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


def test_probe_backend_numpy_field_never_touches_jit(monkeypatch):
    # Regression (beta feedback, 2026-08): the old probe attempted jax.jit on
    # every field and CAUGHT the TracerArrayConversionError for NumPy fields --
    # correct, but debuggers configured to break on raised exceptions (VS Code)
    # paused inside the library. A NumPy field must now be classified by its
    # output type alone, with no jit attempt and no exception raised.
    jax = pytest.importorskip("jax", reason="probe needs JAX installed")
    from pyddrv.api import _probe_backend

    def boom(*a, **k):
        raise AssertionError("jax.jit must not be called for a NumPy field")

    monkeypatch.setattr(jax, "jit", boom)
    assert _probe_backend(damped_pendulum(), 2, "auto") == "numpy"
    assert _probe_backend(linear(A_SPIRAL), 2, "auto") == "numpy"


def test_probe_backend_jax_field_selected():
    pytest.importorskip("jax", reason="probe needs JAX installed")
    from pyddrv.api import _probe_backend
    from pyddrv.systems.fields_jax import bilinear_2d_jax

    f, _ = bilinear_2d_jax(eta=0.3, seed=0)
    assert _probe_backend(f, 2, "auto") == "jax"


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


# --------------------------------------------------------------------------- #
# Trim: the reported region is a fixed point of the Trim pass
# --------------------------------------------------------------------------- #
L_SPIRAL = float(np.max(np.linalg.eigvalsh((A_SPIRAL + A_SPIRAL.T) / 2)))


def _spiral_roa(alpha, tau, **kw):
    pytest.importorskip("jax", reason="verify_roa needs the JAX kernel")
    kw.setdefault("max_refine", 3)
    return verify_roa(linear(A_SPIRAL), R=np.pi, d=2, alpha=alpha, tau=tau,
                      L=L_SPIRAL, trim=True, raster_n=729, **kw)


def _closure_failures(rep, n_points=400, seed=0):
    """Sample points of the region by volume and count those with no return
    time t <= tau at which e^{alpha t} ||phi|| <= ||x|| and phi lies in the
    region or in B_eps (2-norm, RK4 with the verifier's step)."""
    rng = np.random.default_rng(seed)
    C, H = np.asarray(rep.centers, float), np.asarray(rep.halfs, float)
    w = H ** 2 / np.sum(H ** 2)
    i = rng.choice(len(C), n_points, p=w)
    X = C[i] + H[i, None] * rng.uniform(-1.0, 1.0, (n_points, 2))

    def member(P):
        inside = np.linalg.norm(P, axis=1) <= rep.eps
        for c, h in zip(C, H):
            inside |= np.all(np.abs(P - c) <= h, axis=1)
        return inside

    n_steps = max(120, int(round(50.0 * rep.tau)))
    dt = rep.tau / n_steps
    x, V0 = X.copy(), np.linalg.norm(X, axis=1)
    ok = np.zeros(n_points, bool)
    for k in range(1, n_steps + 1):
        k1 = x @ A_SPIRAL.T
        k2 = (x + 0.5 * dt * k1) @ A_SPIRAL.T
        k3 = (x + 0.5 * dt * k2) @ A_SPIRAL.T
        k4 = (x + dt * k3) @ A_SPIRAL.T
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        cand = ~ok & (np.exp(rep.alpha * k * dt) * np.linalg.norm(x, axis=1)
                      <= V0)
        if cand.any():
            ok[np.flatnonzero(cand)[member(x[cand])]] = True
    return int(np.sum(~ok))


def test_verify_roa_trim_fixed_point_keeps_achievable_rate():
    """alpha = 0.3 is below the spectral abscissa 0.5: the fixed point keeps
    almost all of the box and every cube certifies against the region itself."""
    rep = _spiral_roa(0.3, 3.0)
    assert rep.trim_converged
    assert rep.trim_passes >= 1
    assert rep.volume >= 0.99 * (2 * np.pi) ** 2
    assert rep.trim_history[-1] == (rep.n_certified, pytest.approx(rep.volume))
    # one more pass against the reported region changes nothing
    again = find_alpha_roa_fused(
        linear(A_SPIRAL), L_SPIRAL, np.pi, rep.eps, 2, 0.3, norm="2", tau=3.0,
        n_steps=150, max_refine=3,
        trim_region=(rep.centers, rep.halfs),
        grid0=(rep.centers, rep.halfs, rep.depths), raster_n=729)
    assert again.n_tested == again.n_certified == rep.n_certified
    assert _closure_failures(rep) == 0
    assert "fixed point" in rep.summary()


def test_verify_roa_trim_unachievable_rate_certifies_only_near_eps():
    """alpha = 1 exceeds the spectral abscissa 0.5, so no chain of returns
    can keep that rate for long: only cubes next to B_eps, whose chains end
    there after a return or two, may be certified. A single Trim pass against
    the first-pass region used to keep about a quarter of the box."""
    rep = _spiral_roa(1.0, 1.9, max_refine=4)
    assert rep.trim_converged
    assert rep.trim_history[0][1] > 0.2 * (2 * np.pi) ** 2   # first pass
    assert rep.volume < 1e-4 * (2 * np.pi) ** 2
    if rep.n_certified:
        outer = np.linalg.norm(rep.centers, axis=1) + np.sqrt(2) * rep.halfs
        assert outer.max() <= 3.0 * rep.eps
        assert _closure_failures(rep) == 0


def test_verify_roa_trim_without_eps_ball_is_empty():
    """If chains may not end in B_eps, the norm of a chain decays
    geometrically while every cube stays outside the excluded hole, so no
    nonempty region is a fixed point."""
    rep = _spiral_roa(1.0, 1.9, trim_eps_ball=False)
    assert rep.trim_converged
    assert rep.n_certified == 0


def test_verify_roa_trim_pass_limit_is_reported():
    with pytest.warns(UserWarning, match="fixed point"):
        rep = _spiral_roa(1.0, 1.9, max_refine=4, max_trim_passes=1)
    assert rep.trim_passes == 1
    assert not rep.trim_converged
    assert "NOT a fixed point" in rep.summary()


def test_verify_roa_warns_on_coarse_raster():
    pytest.importorskip("jax", reason="verify_roa needs the JAX kernel")
    with pytest.warns(UserWarning, match="raster_n"):
        verify_roa(linear(A_SPIRAL), R=np.pi, d=2, alpha=0.3, tau=3.0,
                   L=L_SPIRAL, max_refine=1, trim=True, raster_n=81)
