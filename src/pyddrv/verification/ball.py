r"""Theorem 8 ball certification and Procedure 1 (``alpha_max``).

Theorem 8 certifies the :math:`\epsilon`-ERLF recurrence condition on a whole ball
:math:`B_r(x)` from a *single* trajectory started at the center :math:`x`, using
the one-sided Lipschitz constant :math:`L` to bound how far neighbouring
trajectories can drift. With equilibrium at the origin and :math:`V(y)=\lVert
y\rVert`, condition (32a) reads: there exists :math:`t\in(0,\tau]` with

.. math::

    e^{\alpha t}\,\big(\lVert \varphi(t,x)\rVert + r\,e^{Lt}\big)
        \;\le\; \lVert x\rVert - r.                                    (32a)

The Lipschitz drift term :math:`r e^{Lt}` sits *inside* the :math:`e^{\alpha t}`
factor: it bounds :math:`\lVert\varphi(t,y)\rVert` for neighbours
:math:`y\in B_r(x)`. Solving (32a) for the rate gives, at each return time
:math:`t`,

.. math::

    \alpha(t) = \frac{1}{t}\,
        \ln\!\frac{\lVert x\rVert - r}{\lVert \varphi(t,x)\rVert + r\,e^{Lt}},

valid whenever the numerator is positive (and, when the reference set ``S`` is
enforced, :math:`\varphi(t,x)\in S`). **Procedure 1** returns
:math:`\alpha_{\max}(x,r;S)=\max_{t\in(0,\tau]}\alpha(t)` -- a 1-D maximization
over the return time. We evaluate it in closed form over the simulated time grid,
fully vectorized across a batch of balls.
"""
from __future__ import annotations

import numpy as np


def _lp_norm(x, norm: str, axis=-1):
    key = str(norm).lower()
    if key in ("2", "l2", "euclidean"):
        return np.linalg.norm(x, ord=2, axis=axis)
    if key in ("inf", "linf", "oo", "infinity"):
        return np.max(np.abs(x), axis=axis)
    if key in ("1", "l1"):
        return np.sum(np.abs(x), axis=axis)
    raise ValueError(f"unsupported norm {norm!r}")


def alpha_max(trajectories, t_grid, r, L, norm="2", S_radius=None):
    r"""Procedure 1, vectorized: best certified rate for a batch of balls.

    Parameters
    ----------
    trajectories : (N, T, d) ndarray
        Trajectory from each ball center over the shared time grid.
    t_grid : (T,) ndarray
        Increasing times with ``t_grid[0] == 0`` (the center itself).
    r : float or (N,) array
        Ball radius for each center (``c*h`` to certify the cube, ``0`` to
        certify only the center).
    L : float
        One-sided Lipschitz constant over the reference set.
    norm : {"2", "inf", "1"}
        Working norm for ``V``.
    S_radius : float, optional
        Inf-norm radius ``R`` of the reference box ``Q_R``. If given, condition
        (32b) ``||phi(t,x)||_inf <= R - r e^{L t}`` is enforced at the *same*
        certifying time ``t`` as (32a). Required when the working norm is not the
        inf-norm (e.g. 2-norm experiments), where ``Q_R`` is not a sub-level set
        of ``V`` so (32a) does not imply (32b).

    Returns
    -------
    alpha : (N,) ndarray
        ``alpha_max`` per center; ``-inf`` where no feasible time exists.
    """
    ys = np.asarray(trajectories, dtype=float)
    N, T, _ = ys.shape
    t = np.asarray(t_grid, dtype=float)
    Vall = _lp_norm(ys, norm, axis=2)            # (N, T)
    Vx = Vall[:, 0]                              # (N,)
    tk = t[1:]                                   # (T-1,), > 0
    Vphi = Vall[:, 1:]                           # (N, T-1)

    r = np.broadcast_to(np.asarray(r, dtype=float), (N,))[:, None]
    # (32a): e^{a t} ( ||phi(t,x)|| + r e^{L t} ) <= ||x|| - r
    #   => a(t) = (1/t) ln( (||x|| - r) / (||phi(t,x)|| + r e^{L t}) )
    margin = Vx[:, None] - r                                  # (N, 1), need > 0
    denom = Vphi + r * np.exp(L * tk)[None, :]                # (N, T-1), > 0

    valid = (margin > 0) & (denom > 1e-300)
    if S_radius is not None:
        # (32b): sd(phi(t,x), Q_R) + r e^{L t} <= 0. For the inf-norm box Q_R with
        # distance measured in the working norm, sd = ||phi||_inf - R, so this is
        #   ||phi(t,x)||_inf <= R - r e^{L t}.
        # Enforced at the SAME time index as (32a) (both masks AND-ed before the
        # max over t). Required whenever V's norm != inf, since then Q_R is not a
        # sub-level set of V and (32a) does NOT imply (32b) (Remark 3).
        phi_inf = np.max(np.abs(ys[:, 1:, :]), axis=2)              # (N, T-1)
        inside = phi_inf <= S_radius - r * np.exp(L * tk)[None, :]  # same t
        valid &= inside

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, margin / np.where(denom > 0, denom, 1.0), np.nan)
        alpha = np.log(ratio) / tk[None, :]
    alpha = np.where(valid, alpha, -np.inf)
    return alpha.max(axis=1)


def certify_ball(trajectories, t_grid, r, L, alpha, norm="2", S_radius=None):
    """Boolean: does each ball satisfy (32) at the *given* rate ``alpha``?"""
    return alpha_max(trajectories, t_grid, r, L, norm, S_radius) >= alpha
