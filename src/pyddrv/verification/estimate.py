r"""Data-driven parameter estimation for verification (paper §IX-B).

Trajectories started in :math:`Q_R` generally **leave** it before returning, so
the one-sided Lipschitz constant used in Theorem 8 must be valid over the
*reachable set*, not over :math:`Q_R`. We estimate the worst-case excursion

.. math::

    R_{\max} := \max_{x\in\Gamma,\; t\in[0,\tau]} \lVert \varphi(t,x)\rVert_\infty,

pick :math:`\bar R > R_{\max}` (small slack for discretization), and compute

.. math::

    L := \sup_{z \in Q_{\bar R}} \mu\!\left(\tfrac{\partial f}{\partial z}\right)

over the **enlarged** box :math:`Q_{\bar R}`. The discretization-robustness check
(eq. 38) confirms that an :math:`h`-neighborhood of every sampled state stays
inside :math:`Q_{\bar R}` -- with **renewal termination**: the drift factor
:math:`h e^{tL}` only runs until the whole tube has returned inside :math:`Q_R`
(after which every covered trajectory restarts inside :math:`Q_R` and the claim
recurses), so the guarantee holds for all :math:`t`, not just :math:`[0,\tau]`,
and the check stays feasible for large :math:`L\tau`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lipschitz import one_sided_lipschitz


@dataclass
class LipschitzEstimate:
    L: float
    R_bar: float        # enlarged box radius over which L is valid
    R_max: float        # worst-case excursion radius (inf-norm)
    discretization_ok: bool


def boundary_grid(R, n_grid, d):
    r"""Uniform grid on the boundary ``∂Q_R`` with spacing ``2R/(n_grid-1)``.

    Returns ``(points (N,d), h_check)`` where ``h_check`` is the cell
    half-spacing (∞-norm coverage radius of the grid on each face). Only
    boundary points are generated (face-wise, never the full ``n^d`` mesh, so
    dense grids stay affordable): since ``||.||`` is convex and the flow is a
    homeomorphism, the worst-case excursion of the reachable set ``R(Q_R)`` is
    attained on ``φ(t, ∂Q_R)``, so the boundary grid suffices.
    """
    ax = np.linspace(-R, R, n_grid)
    faces = []
    for j in range(d):
        if d > 1:
            mesh = np.meshgrid(*([ax] * (d - 1)), indexing="ij")
            flat = [m.reshape(-1) for m in mesh]
            n_face = flat[0].size
        else:
            flat, n_face = [], 1
        cols = [c for c in range(d) if c != j]
        for s in (-R, R):
            pts = np.empty((n_face, d))
            for c, v in zip(cols, flat):
                pts[:, c] = v
            pts[:, j] = s
            faces.append(pts)
    pts = np.unique(np.round(np.concatenate(faces, axis=0), 12), axis=0)
    return pts, R / (n_grid - 1)


def excursion_radius(rollout, points, t_grid) -> float:
    """``R_max``: largest inf-norm reached by any trajectory over the grid."""
    ys = rollout(np.asarray(points, dtype=float), np.asarray(t_grid, dtype=float))
    return float(np.max(np.abs(ys)))


def estimate_L(rollout, R, eps, d, norm="2", tau=4.0, n_time=400, n_grid=21,
               slack=0.05, jac=None, f=None, refine_grid=3, n_grid_max=81
               ) -> LipschitzEstimate:
    r"""Estimate ``L`` over the reachable box ``Q_{R_bar}`` (paper §IX-B).

    Trajectories are launched from a uniform grid on the boundary ``∂Q_R``. We
    measure the worst-case excursion ``R_max = max ||phi(t,x)||_inf``, set the
    (fixed) enlarged radius ``R_bar = R_max (1 + slack)``, and compute
    ``L = sup_{Q_{R_bar}} mu(df/dx)``. The discretization-robustness check (eq 38)

    .. math::  \max_{x,t}\big(\lVert\varphi(t,x)\rVert_\infty + h\,e^{tL}\big)
               \;\le\; R_{bar}

    validates that the boundary samples represent the continuum; if it fails the
    boundary grid ``h`` is *refined* (denser), as in the paper -- ``R_bar`` is
    NOT inflated (that would diverge for fields whose Jacobian grows with the
    box).

    Parameters
    ----------
    rollout : callable
        ``rollout(points (N,d), t_grid (T,)) -> (N, T, d)``.
    n_grid : int
        Initial points per axis on ``∂Q_R`` (controls ``h``).
    slack : float
        Margin: ``R_bar = R_max (1 + slack)``.
    refine_grid : int
        Max grid densifications attempted to satisfy eq (38).
    jac, f : callable, optional
        Analytic Jacobian (exact corner max when affine) or field for a
        numerical Jacobian.

    Returns
    -------
    LipschitzEstimate
        ``discretization_ok`` is True when eq (38) holds at the returned ``L``.
    """
    t_grid = np.linspace(0.0, tau, n_time)
    ng = n_grid
    chunk = 200_000
    L = R_bar = R_max = None
    discretization_ok = False
    for _ in range(refine_grid + 1):
        pts, h = boundary_grid(R, ng, d)

        # pass 1: worst-case excursion (chunked; nothing large is retained)
        R_max = 0.0
        for s in range(0, len(pts), chunk):
            R_max = max(R_max, float(np.max(np.abs(rollout(pts[s:s + chunk],
                                                           t_grid)))))
        R_bar = R_max * (1.0 + slack)
        if not np.isfinite(R_bar) or R_bar > 50.0 * R:
            # trajectories diverged: Q_R is well outside the RoA. Computing L over
            # such a huge box overflows; report it as uncertifiable instead.
            return LipschitzEstimate(L=float("inf"), R_bar=float(R_bar),
                                     R_max=float(R_max), discretization_ok=False)
        L = one_sided_lipschitz(f, [0.0] * d, [R_bar] * d,
                                norm=norm, method="corners", jac=jac)

        # pass 2: RENEWAL-TERMINATED discretization check. The h-tube around a
        # sample, ``||phi(t,x_i)||_inf + h e^{tL}``, covers every unsampled
        # boundary point within h of x_i. Once the WHOLE tube is back inside
        # Q_R, every covered trajectory sits at a point of Q_R and its future is
        # covered by the same claim being proven (any later excursion exits
        # through dQ_R within h of some sample, starting a fresh tube). So the
        # drift factor only needs to run to each sample's tube-RETURN time --
        # not to tau -- and the resulting Q_R' containment holds for ALL t, not
        # just [0, tau]. A sample also passes if its full-horizon tube stays in
        # Q_R' (the original eq-38 condition), whichever is weaker fails last.
        drift = h * np.exp(t_grid * L)                          # (T,)
        T = len(t_grid)
        idx = np.arange(T)[None, :]
        discretization_ok = True
        for s in range(0, len(pts), chunk):
            phi = np.max(np.abs(rollout(pts[s:s + chunk], t_grid)), axis=2)
            tube = phi + drift[None, :]                         # (Nc, T)
            ret = tube <= R                                     # tube inside Q_R
            has = ret.any(axis=1)
            first = np.where(has, np.argmax(ret, axis=1), T - 1)
            pre_max = np.where(idx <= first[:, None], tube, -np.inf).max(axis=1)
            ok = (has & (pre_max <= R_bar)) | (tube.max(axis=1) <= R_bar)
            if not bool(ok.all()):
                discretization_ok = False
                break
        if discretization_ok or ng >= n_grid_max:
            break
        ng = min(n_grid_max, 2 * ng - 1)                # densify (halve h)

    return LipschitzEstimate(L=float(L), R_bar=float(R_bar), R_max=float(R_max),
                             discretization_ok=bool(discretization_ok))


def estimate_L_roa(rollout, R, d, norm="2", tau=2.0, n_time=200, n_grid=21,
                   slack=0.05, jac=None, f=None, refine_grid=6, n_grid_max=1025
                   ) -> LipschitzEstimate:
    r"""Estimate ``L`` for Algorithm 2 (paper §IX-C): candidate sets ``S ⊆ Q_R``.

    A trajectory from ``S`` either (i) stays inside ``Q_R`` or (ii) leaves and
    returns within ``τ``. Boundary samples ``Γ ⊂ ∂Q_R`` are partitioned into

    * ``Γ_ret`` -- return to ``Q_R`` at some ``t ∈ (0, τ]``: these play the role
      of ``Γ`` in §IX-B: ``R_max`` over their excursions, ``R' > R_max``, ``L``
      over ``Q_{R'}``, and the (renewal-terminated) check (38) over ``Γ_ret``.
    * ``Γ_nr`` -- never return: a one-sided check ensures trajectories from an
      ``h``-neighborhood of a non-returning sample also fail to return,

      .. math:: \min_{t\in(0,\tau]} \lVert\varphi(t,x)\rVert_\infty - h\,e^{tL}
                \;>\; R \qquad \forall x \in \Gamma_{nr},

      so no unsampled boundary point is mis-classified. Regions draining through
      ``Γ_nr`` can never satisfy (32b), hence are never certified, and ``L``
      needs no validity along their far excursions.

    If either check fails, the boundary grid is refined and the partition
    recomputed. Returns a :class:`LipschitzEstimate` (``R_max`` over ``Γ_ret``
    only; ``R_bar >= R`` always).
    """
    t_grid = np.linspace(0.0, tau, n_time)
    ng = n_grid
    chunk = 200_000
    L = R_bar = R_max = None
    discretization_ok = False
    for _ in range(refine_grid + 1):
        pts, h = boundary_grid(R, ng, d)

        # pass 1: partition + excursion radius of the RETURNING samples
        R_max = R
        any_ret = False
        for s in range(0, len(pts), chunk):
            phi = np.max(np.abs(rollout(pts[s:s + chunk], t_grid)), axis=2)
            ret = (phi[:, 1:] <= R).any(axis=1)                 # returns at t>0
            if ret.any():
                any_ret = True
                R_max = max(R_max, float(np.max(phi[ret])))
        R_bar = R_max * (1.0 + slack)
        if not np.isfinite(R_bar) or R_bar > 50.0 * R:
            return LipschitzEstimate(L=float("inf"), R_bar=float(R_bar),
                                     R_max=float(R_max), discretization_ok=False)
        L = one_sided_lipschitz(f, [0.0] * d, [R_bar] * d,
                                norm=norm, method="corners", jac=jac)

        # pass 2: renewal-terminated (38) on Gamma_ret; one-sided non-return
        # check on Gamma_nr
        drift = h * np.exp(t_grid * L)                          # (T,)
        T = len(t_grid)
        idx = np.arange(T)[None, :]
        discretization_ok = True
        for s in range(0, len(pts), chunk):
            phi = np.max(np.abs(rollout(pts[s:s + chunk], t_grid)), axis=2)
            is_ret = (phi[:, 1:] <= R).any(axis=1)
            tube = phi + drift[None, :]
            # Gamma_ret: renewal-terminated containment in Q_R'
            retn = tube <= R
            has = retn.any(axis=1)
            first = np.where(has, np.argmax(retn, axis=1), T - 1)
            pre_max = np.where(idx <= first[:, None], tube, -np.inf).max(axis=1)
            ok_ret = (has & (pre_max <= R_bar)) | (tube.max(axis=1) <= R_bar)
            # Gamma_nr: the whole h-neighborhood also fails to return (t > 0)
            low = phi[:, 1:] - drift[None, 1:]
            ok_nr = (low > R).all(axis=1)
            ok = np.where(is_ret, ok_ret, ok_nr)
            if not bool(ok.all()):
                discretization_ok = False
                break
        if (discretization_ok and any_ret) or ng >= n_grid_max:
            break
        ng = min(n_grid_max, 2 * ng - 1)

    return LipschitzEstimate(L=float(L), R_bar=float(R_bar), R_max=float(R_max),
                             discretization_ok=bool(discretization_ok))
