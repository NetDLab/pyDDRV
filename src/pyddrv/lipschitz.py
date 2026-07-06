r"""One-sided Lipschitz (oL) constant estimation.

The paper's verification machinery (Theorem 8) certifies a whole ball
:math:`B_r(x)` from a single trajectory by bounding how fast nearby trajectories
can diverge. That bound is the **one-sided Lipschitz constant** of the vector
field over a set :math:`S` (Definition 2):

.. math::

    [f(y) - f(x);\, y - x] \;\le\; L_S\, \lVert y - x\rVert^2,
    \qquad \forall x, y \in S,

where :math:`[\cdot;\cdot]` is the weak pairing of the working norm. For a
:math:`C^1` field on a convex set this equals the supremum of the **logarithmic
matrix measure** (a.k.a. matrix log-norm) of the Jacobian (eq. 3):

.. math::

    L_S = \sup_{x \in S} \mu\!\left(\tfrac{\partial f}{\partial x}(x)\right).

Matrix measures implemented (vector norm -> measure of :math:`A`):

- ``"2"``   : :math:`\mu_2(A) = \lambda_{\max}\!\big(\tfrac{A+A^\top}{2}\big)`
- ``"inf"`` : :math:`\mu_\infty(A) = \max_i\big(a_{ii} + \sum_{j\ne i}|a_{ij}|\big)`
- ``"1"``   : :math:`\mu_1(A) = \max_j\big(a_{jj} + \sum_{i\ne j}|a_{ij}|\big)`
- weighted Euclidean ``P`` (``P = Q^\top Q``): :math:`\mu_{2,P}(A)=\mu_2(Q A Q^{-1})`

Two estimation routes are provided:

1. **Jacobian-based** (`one_sided_lipschitz`) -- the exact eq. (3) supremum over
   a box, evaluated at the corners (exact when the Jacobian is affine in the
   state, as in the paper's bilinear examples) or over a sample/grid.
2. **Data-driven** (`one_sided_lipschitz_from_data`) -- Definition 2 directly,
   from sampled ``(x, f(x))`` pairs, requiring no model. Useful when only
   trajectory data is available; pairs of states are differenced to estimate the
   weak-pairing ratio.
"""
from __future__ import annotations

import itertools
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Matrix measures (logarithmic norms)
# --------------------------------------------------------------------------- #
def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def matrix_measure(A, norm: str = "2", P=None) -> float:
    r"""Logarithmic matrix measure :math:`\mu(A)` for the chosen vector norm.

    Parameters
    ----------
    A : (d, d) array_like
        Matrix (typically a Jacobian).
    norm : {"2", "inf", "1"}
        Underlying vector norm. Ignored when ``P`` is given (weighted-2).
    P : (d, d) array_like, optional
        SPD weight for a weighted Euclidean norm ``||x||_P = sqrt(x^T P x)``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}")

    if P is not None:
        P = np.asarray(P, dtype=float)
        # ||x||_P with P = R^T R: mu_{2,P}(A) = mu_2(R A R^{-1}).
        R = np.linalg.cholesky(P).T
        B = R @ A @ np.linalg.inv(R)
        return float(np.max(np.linalg.eigvalsh(_sym(B))))

    key = str(norm).lower()
    if key in ("2", "l2", "euclidean"):
        return float(np.max(np.linalg.eigvalsh(_sym(A))))
    if key in ("inf", "linf", "oo", "infinity"):
        offdiag = np.abs(A).sum(axis=1) - np.abs(np.diag(A))
        return float(np.max(np.diag(A) + offdiag))
    if key in ("1", "l1"):
        offdiag = np.abs(A).sum(axis=0) - np.abs(np.diag(A))
        return float(np.max(np.diag(A) + offdiag))
    raise ValueError(f"unsupported norm {norm!r}; use '2', 'inf', or '1'")


# --------------------------------------------------------------------------- #
# Jacobians
# --------------------------------------------------------------------------- #
def numerical_jacobian(f: Callable, x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a batched field ``f: (n,d) -> (n,d)``.

    ``f`` is the same batched signature used by :mod:`pyddrv.systems` (it maps a
    stack of states to a stack of derivatives), so we evaluate all ``2d``
    perturbations in a single call.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.size
    E = eps * np.eye(d)
    plus = f(x[None, :] + E)   # (d, d)
    minus = f(x[None, :] - E)  # (d, d)
    # column j is df/dx_j -> (f(x+eps e_j) - f(x-eps e_j)) / (2 eps)
    return ((plus - minus) / (2.0 * eps)).T


# --------------------------------------------------------------------------- #
# Box sampling
# --------------------------------------------------------------------------- #
def box_corners(center, half) -> np.ndarray:
    """The :math:`2^d` corners of the box ``center +/- half`` (per-axis)."""
    center = np.asarray(center, dtype=float)
    half = np.broadcast_to(np.asarray(half, dtype=float), center.shape)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=center.size)))
    return center[None, :] + signs * half[None, :]


def _box_samples(center, half, method, n_samples, seed) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    half = np.broadcast_to(np.asarray(half, dtype=float), center.shape)
    if method == "corners":
        return box_corners(center, half)
    if method == "random":
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0, size=(n_samples, center.size))
        return center[None, :] + u * half[None, :]
    if method == "grid":
        per = max(2, int(round(n_samples ** (1.0 / center.size))))
        axes = [np.linspace(c - h, c + h, per) for c, h in zip(center, half)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], axis=1)
    raise ValueError(f"unknown method {method!r}; use corners|random|grid")


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def one_sided_lipschitz(
    f: Callable,
    center,
    half,
    norm: str = "2",
    method: str = "corners",
    jac: Optional[Callable] = None,
    P=None,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    r"""Estimate :math:`L_S = \sup_{x\in S}\mu(\partial f/\partial x)` over a box.

    ``S`` is the axis-aligned box ``center +/- half``. With ``method="corners"``
    the supremum is evaluated at the box corners -- exact when the Jacobian is
    affine in ``x`` (the paper's bilinear systems), since the matrix measure is
    convex and is maximized at an extreme point. Use ``"random"``/``"grid"`` for
    general nonlinear fields.

    Parameters
    ----------
    f : callable
        Batched vector field ``(n,d) -> (n,d)``. Used only if ``jac`` is None.
    jac : callable, optional
        Analytic Jacobian ``x -> (d,d)``. If given, ``f`` is ignored.
    """
    pts = _box_samples(center, half, method, n_samples, seed)
    jac_fn = jac if jac is not None else (lambda x: numerical_jacobian(f, x))
    return max(matrix_measure(jac_fn(p), norm=norm, P=P) for p in pts)


def one_sided_lipschitz_from_data(
    states,
    derivatives,
    P=None,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> float:
    r"""Data-driven :math:`L` from sampled ``(x, f(x))`` pairs (Definition 2).

    Estimates ``L = sup_{i != j} [f(x_i)-f(x_j); x_i-x_j] / ||x_i-x_j||^2`` using
    the Euclidean (optionally ``P``-weighted) weak pairing, i.e. the standard
    inner product. No model is required -- ``derivatives`` may come from
    finite-differencing trajectory data (see :func:`derivatives_from_trajectories`).

    For more than ``max_pairs`` candidate pairs a random subset is used; the
    result is then a lower bound on the true sup, so prefer dense local sampling.
    """
    X = np.asarray(states, dtype=float)
    F = np.asarray(derivatives, dtype=float)
    if X.shape != F.shape or X.ndim != 2:
        raise ValueError("states and derivatives must both be (N, d)")
    N = X.shape[0]

    def ip(a, b):  # weak pairing = (weighted) inner product, rows
        if P is None:
            return np.sum(a * b, axis=1)
        return np.einsum("ni,ij,nj->n", a, np.asarray(P, float), b)

    total_pairs = N * (N - 1) // 2
    if total_pairs <= max_pairs:
        iu, ju = np.triu_indices(N, k=1)
    else:
        rng = np.random.default_rng(seed)
        iu = rng.integers(0, N, size=max_pairs)
        ju = rng.integers(0, N, size=max_pairs)
        keep = iu != ju
        iu, ju = iu[keep], ju[keep]

    dX = X[iu] - X[ju]
    dF = F[iu] - F[ju]
    denom = ip(dX, dX)
    good = denom > 0
    ratios = ip(dF[good], dX[good]) / denom[good]
    return float(np.max(ratios)) if ratios.size else float("-inf")


def derivatives_from_trajectories(states, dt) -> tuple:
    """Finite-difference ``f(x) ~ (x_{k+1}-x_k)/dt`` from a TrajectorySet array.

    Parameters
    ----------
    states : (n_traj, horizon, d) array
    dt : float

    Returns
    -------
    (X, F) : two (n_traj*(horizon-1), d) arrays of states and estimated f(x).
    """
    s = np.asarray(states, dtype=float)
    if s.ndim != 3:
        raise ValueError("states must be (n_traj, horizon, d)")
    X = s[:, :-1, :].reshape(-1, s.shape[-1])
    F = (np.diff(s, axis=1) / dt).reshape(-1, s.shape[-1])
    return X, F
