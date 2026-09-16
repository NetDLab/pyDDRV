r"""Trajectory-local contraction bound ("Coppel with remainder").

Replaces the global-Lipschitz ball-propagation factor ``r*exp(L*t)`` in the
Theorem-8 certificate with a bound computed *along* each center trajectory,
from the matrix measure (log-norm) of the Jacobian plus a second-order
Jacobian-Lipschitz (Hessian) remainder.

Math (2-norm working norm; ``mu = mu_2``, ``M`` a spectral Jacobian-Lipschitz
constant on the region trajectories visit).  Along the center trajectory
``x(s) = phi(s, x)``::

    a(s)  = mu_2( J(x(s)) )            # = lambda_max( sym(J) )
    A(t)  = int_0^t a(s) ds
    I(t)  = int_0^t exp(A(s)) ds

The separation ``e = phi(t,y)-phi(t,x)`` obeys ``de/dt = J(x(t)) e + g`` with
remainder ``||g|| = ||f(y)-f(x)-J(x)e|| <= (M/2) ||e||^2`` (from
``int_0^1 s M ds``), hence ``D^+||e|| <= a(t)||e|| + (M/2)||e||^2``.  The
Riccati comparison ``rho' = a rho + (M/2) rho^2`` , ``rho(0) = r`` has closed
form::

    rho(t) = r*exp(A(t)) / (1 - r*(M/2)*I(t))    valid while  r*(M/2)*I(t) < 1 .

(NB: the M/2 -- not M -- coefficient; the handoff spec's ``M`` is a factor of 2
conservative.  Verified sound by the T2 property test.)

Both the Gronwall bound ``r*exp(L*t)`` and ``rho(t)`` (where its guard holds)
are valid upper bounds, so we always use ``min`` of the two -- the new factor
is pointwise never worse than the current one::

    bound(t) = min( r*exp(L*t),  rho(t) if r*M*I(t) < 1 else +inf ) .

Only the inflation term in (30a)/(30b) changes; global ``L`` is still used for
the ``min`` fallback and (elsewhere) for tau / R' selection.

This module is the NumPy reference + the standalone math; wiring the ``(A, I)``
accumulation into the fused GPU kernel is a separate, additive step.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# matrix measure and quadrature
# --------------------------------------------------------------------------- #
def matrix_measure_2(J):
    r"""``mu_2(J) = lambda_max( (J + J^T)/2 )``, batched over leading axes.

    ``J`` has shape ``(..., d, d)``; returns shape ``(...)``.
    """
    J = np.asarray(J, dtype=np.float64)
    sym = 0.5 * (J + np.swapaxes(J, -1, -2))
    return np.linalg.eigvalsh(sym)[..., -1]


def accumulate_AI(a, dt):
    r"""Cumulative ``A(t) = int a`` and ``I(t) = int exp(A)`` by a CONSERVATIVE
    upper-sum quadrature on the sampled ``a`` (uses the larger endpoint on each
    sub-interval, and evaluates ``exp(A)`` at the larger accumulated value).

    ``a`` has shape ``(..., T)`` (samples along each trajectory); returns
    ``(A, I)`` of the same shape, accumulated in float64 (``exp(A)`` can be
    large even when states are float32).
    """
    a = np.asarray(a, dtype=np.float64)
    T = a.shape[-1]
    A = np.zeros_like(a)
    I = np.zeros_like(a)
    for k in range(1, T):
        A[..., k] = A[..., k - 1] + np.maximum(a[..., k - 1], a[..., k]) * dt
        I[..., k] = I[..., k - 1] + np.exp(A[..., k]) * dt
    return A, I


def local_bound(r, A_t, I_t, M, L, t):
    r"""``bound(t) = min( r*exp(L*t), rho(t) )`` with the Riccati guard.

    ``r`` scalar or ``(...,1)``; ``A_t, I_t, t`` broadcast over the time axis
    (shape ``(..., T)``). Where ``r*M*I_t >= 1`` the new bound is ``+inf`` so
    the ``min`` falls back to the Gronwall factor.
    """
    r = np.asarray(r, dtype=np.float64)
    A_t = np.asarray(A_t, dtype=np.float64)
    I_t = np.asarray(I_t, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    old = r * np.exp(L * t)
    b = 0.5 * M                                    # remainder coefficient M/2
    guard = (r * b * I_t) < 1.0
    denom = 1.0 - r * b * I_t
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(guard, r * np.exp(A_t) / denom, np.inf)
    return np.minimum(old, rho)


# --------------------------------------------------------------------------- #
# center rollout that also returns A, I
# --------------------------------------------------------------------------- #
def rollout_with_measure(f, jac, centers, dt, n_steps, measure=matrix_measure_2):
    r"""RK4-integrate ``centers`` and accumulate ``(A, I)`` from ``mu(J)`` at each
    grid point.  Jacobians are evaluated and reduced on the fly (never stored).

    ``f(X:(N,d)) -> (N,d)`` and ``jac(X:(N,d)) -> (N,d,d)`` (batched).  Returns
    ``ys (N, T, d)``, ``A (N, T)``, ``I (N, T)`` with ``T = n_steps + 1``.
    """
    x = np.asarray(centers, dtype=np.float64)
    N, d = x.shape
    T = n_steps + 1
    ys = np.empty((N, T, d))
    a = np.empty((N, T))
    ys[:, 0] = x
    a[:, 0] = measure(jac(x))
    for k in range(1, T):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[:, k] = x
        a[:, k] = measure(jac(x))
    A, I = accumulate_AI(a, dt)
    return ys, A, I


def alpha_max_local(ys, A, I, t_grid, r, M, L, x_star=None, norm2=True):
    r"""Per-box certified rate using the trajectory-local ``bound(t)``.

    Same shape as the current ``alpha_max`` -- only the inflation term differs.
    ``ys (N,T,d)``, ``A,I (N,T)``, ``t_grid (T,)``, ``r`` scalar or ``(N,)``.
    Returns ``(N,)`` (``-inf`` where no valid return time exists).
    """
    ys = np.asarray(ys, dtype=np.float64)
    N, T, d = ys.shape
    x0 = ys[:, 0, :]
    if x_star is None:
        x_star = np.zeros(d)
    nrm = (lambda z: np.linalg.norm(z, axis=-1)) if norm2 else \
        (lambda z: np.max(np.abs(z), axis=-1))
    Vx = nrm(x0 - x_star)[:, None]                          # (N,1)
    Vphi = nrm(ys - x_star)                                 # (N,T)
    r = np.broadcast_to(np.asarray(r, float), (N,))[:, None]
    tk = np.asarray(t_grid, float)[None, :]                 # (1,T)
    bnd = local_bound(r, A, I, M, L, np.broadcast_to(tk, (N, T)))
    margin = Vx - r
    denom = Vphi + bnd
    valid = (margin > 0) & (denom > 0) & (tk > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(valid, np.log(np.where(valid, margin / denom, 1.0))
                        / np.where(tk > 0, tk, 1.0), -np.inf)
    return np.where(valid, rate, -np.inf).max(axis=1)


# --------------------------------------------------------------------------- #
# Jacobian-Lipschitz constant M
# --------------------------------------------------------------------------- #
def jac_lipschitz_affine(jac, d):
    r"""EXACT ``M`` for a field whose Jacobian is AFFINE in the state (bilinear
    family): ``J(x) = J(0) + sum_j x_j D_j`` with constant ``D_j``, so
    ``M = sup_{||u||=1} || sum_j u_j D_j ||_2``.  Computed by multi-start
    maximization of the spectral norm over the unit sphere (exact for the
    smooth objective at ``d = 2, 3``).
    """
    from scipy.optimize import minimize
    e = np.eye(d)
    J0 = np.asarray(jac(np.zeros((1, d))))[0]
    D = np.stack([np.asarray(jac(e[j][None]))[0] - J0 for j in range(d)])  # (d,d,d)

    def negspec(u):
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            return 0.0
        Mu = np.tensordot(u / nu, D, axes=([0], [0]))       # (d,d)
        return -np.linalg.norm(Mu, 2)

    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(40):
        res = minimize(negspec, rng.standard_normal(d), method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
        best = max(best, -res.fun)
    return float(best)


def jac_lipschitz_numeric(jac, R, d, n_pairs=40000, safety=2.0, seed=0):
    r"""Numerical ``M_hat`` (EXPERIMENT MODE, flag it): max spectral-norm
    difference quotient of ``J`` over random pairs in ``[-R, R]^d``, times a
    ``safety`` factor.  For exploration only -- not for reported results.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-R, R, (n_pairs, d))
    zp = rng.uniform(-R, R, (n_pairs, d))
    dJ = np.asarray(jac(z)) - np.asarray(jac(zp))           # (n,d,d)
    spec = np.linalg.svd(dJ, compute_uv=False)[:, 0]        # largest sing. value
    den = np.linalg.norm(z - zp, axis=1)
    return float((spec / np.maximum(den, 1e-12)).max()) * safety


def bilinear_jac_batched(A, B, monomial_grad):
    r"""Batched analytic Jacobian of ``f(x) = A x + B @ m(x)`` (bilinear
    family), correct per-sample (unlike the single-point ``jac`` shipped with
    ``systems.bilinear_*``).  ``A`` is ``(d,d)``, ``B`` is ``(d, n_mon)``,
    ``monomial_grad(X:(N,d)) -> (N, n_mon, d)`` gives ``d m_a / d x_l``.
    Returns ``jac(X:(N,d)) -> (N,d,d)``, affine in ``X``.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    def jac(X):
        X = np.asarray(X, float)
        dm = monomial_grad(X)                       # (N, n_mon, d)
        return A[None] + np.einsum("am,Nml->Nal", B, dm)

    return jac


def _grad_mon_2d(X):
    x1, x2 = X[:, 0], X[:, 1]
    z = np.zeros_like(x1)
    # rows: d[x1^2, x1 x2, x2^2] ; cols: d/dx1, d/dx2
    return np.stack([np.stack([2 * x1, z], 1),
                     np.stack([x2, x1], 1),
                     np.stack([z, 2 * x2], 1)], axis=1)     # (N,3,2)


def _grad_mon_3d(X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    z = np.zeros_like(x1)
    # monomials [x1^2, x1x2, x1x3, x2^2, x2x3, x3^2]
    return np.stack([
        np.stack([2 * x1, z, z], 1),
        np.stack([x2, x1, z], 1),
        np.stack([x3, z, x1], 1),
        np.stack([z, 2 * x2, z], 1),
        np.stack([z, x3, x2], 1),
        np.stack([z, z, 2 * x3], 1)], axis=1)               # (N,6,3)


def bilinear_grad_mon(d):
    """Monomial-gradient callback for the 2D / 3D bilinear families."""
    return _grad_mon_2d if d == 2 else _grad_mon_3d


def kuramoto_jac_batched(k, n):
    r"""Batched analytic Jacobian of the reduced Kuramoto field (eq. 41).

    Full-coordinate Jacobian ``D[m,l] = d(theta_m dot)/d theta_l``::

        D[m,l] = (k/n) cos(theta_l - theta_m)                  (l != m)
        D[m,m] = -(k/n) sum_{j != m} cos(theta_j - theta_m)

    then reduce ``J = D[:n-1,:n-1] - D[n-1,:n-1]`` (reference index ``n``).
    Returns ``jac(X:(N,n-1)) -> (N,n-1,n-1)``.
    """
    k, n = float(k), int(n)

    def jac(X):
        X = np.asarray(X, float)
        N = len(X)
        th = np.concatenate([X, np.zeros((N, 1))], axis=1)     # (N,n)
        diff = th[:, None, :] - th[:, :, None]                 # diff[.,m,l]=th_l-th_m?
        # want cos(theta_l - theta_m): index [b, m, l]
        C = np.cos(th[:, None, :] - th[:, :, None])            # (N,n,n): [b,m,l]=cos(th_l-th_m)
        D = (k / n) * C
        # diagonal m=m: -(k/n) sum_{j != m} cos(theta_j - theta_m)
        offdiag_sum = (k / n) * (C.sum(axis=2) - 1.0)          # sum_j cos - cos(0)
        idx = np.arange(n)
        D[:, idx, idx] = -offdiag_sum
        Jfull = D                                              # (N,n,n)
        return Jfull[:, :n - 1, :n - 1] - Jfull[:, n - 1:n, :n - 1]

    return jac


def finite_difference_jac(f, eps=1e-6):
    r"""Central-difference batched Jacobian of ``f:(N,d)->(N,d)`` (validation
    / fallback for fields without an analytic Jacobian, e.g. Kuramoto)."""
    def jac(X):
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        J = np.empty((N, d, d))
        for j in range(d):
            dx = np.zeros_like(X)
            dx[:, j] = eps
            J[:, :, j] = (f(X + dx) - f(X - dx)) / (2 * eps)
        return J
    return jac
