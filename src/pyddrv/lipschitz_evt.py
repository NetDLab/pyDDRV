r"""Extreme-value-theory (EVT) estimation of Lipschitz / one-sided-Lipschitz
constants with a high-probability upper bound.

The default box-corner estimate in :mod:`pyddrv.lipschitz` is *exact* only when
the Jacobian is affine in the state (then the matrix measure is convex and its
supremum over a box is attained at a corner). For a general nonlinear field the
corner max can **under**-estimate ``sup_x mu(df/dx)`` -- and an under-estimate of
``L`` makes the Theorem-8 drift term ``r e^{Lt}`` too small, i.e. the certificate
unsound. When no analytic Jacobian is available at all, the raw sample maximum of
Definition-2 ratios is likewise only a *lower* bound on the true supremum.

This module implements the extreme-value approach of Knuth et al., "Planning with
Learned Dynamics: Probabilistic Guarantees on Safety and Reachability via
Lipschitz Constants" (arXiv:2010.08993, Alg. 1), which itself builds on Wood &
Zhang, "Estimation of the Lipschitz constant of a function" (J. Global Optim.,
1996). The supremum of a bounded quantity is estimated as the **upper endpoint**
``gamma`` of a three-parameter reverse Weibull distribution fit to block maxima,
returned with a confidence margin so that the estimate over-estimates the true
constant with a user-chosen probability ``rho``.

Method (Knuth Alg. 1)
---------------------
To bound ``L = sup_{z in Z} q(z)`` for a scalar quantity ``q``:

1. For ``j = 1..n_blocks``: draw ``n_per_block`` i.i.d. samples of ``q`` and take
   the block maximum ``s_j = max_i q(z_{i,j})``.
2. Fit a reverse Weibull (``scipy.stats.weibull_max``) to ``{s_j}`` by MLE,
   obtaining the location parameter ``gamma_hat`` (the distribution's upper
   support limit -- the Lipschitz estimate) and its standard error ``xi``.
3. Validate the fit with a Kolmogorov-Smirnov goodness-of-fit test at
   significance ``0.05``.
4. Report ``L_hat = gamma_hat + Phi^{-1}(rho) * xi`` -- an over-estimate of ``L``
   with probability ``rho`` (asymptotically in ``n_per_block``, via
   Fisher-Tippett-Gnedenko).

The reverse Weibull is the *only* extreme-value class with support bounded above,
so a good fit is itself evidence that the underlying quantity is bounded (finite
Lipschitz constant); a failed KS test is reported, not silently ignored.

The confidence is **probabilistic** and asymptotic -- unlike the exact corner
bound for affine Jacobians, this is a high-probability guarantee, which is the
appropriate tool when no closed-form / affine structure is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .lipschitz import numerical_jacobian


@dataclass
class EVTEstimate:
    """Result of an EVT Lipschitz estimate.

    Attributes
    ----------
    L : float
        The reported high-probability upper bound ``gamma_hat + Phi^{-1}(rho) xi``.
    gamma : float
        Fitted reverse-Weibull location parameter (the raw sup estimate).
    se : float
        Standard error ``xi`` of ``gamma`` (bootstrap over block maxima).
    ks_pvalue : float
        Kolmogorov-Smirnov goodness-of-fit p-value for the reverse-Weibull fit.
    validated : bool
        ``ks_pvalue >= significance`` -- the reverse-Weibull model was not
        rejected (evidence of a finite, well-estimated constant).
    rho : float
        Confidence level requested.
    sample_max : float
        Raw maximum over all samples (a lower bound; for reference/diagnostics).
    n_blocks, n_per_block : int
        Sampling budget used.
    """

    L: float
    gamma: float
    se: float
    ks_pvalue: float
    validated: bool
    rho: float
    sample_max: float
    n_blocks: int
    n_per_block: int

    def summary(self) -> str:
        tag = "validated" if self.validated else "NOT validated (KS rejected)"
        return (f"EVT L_hat={self.L:.4g} (gamma={self.gamma:.4g} + "
                f"z_{self.rho:.2f}*{self.se:.2g}); sample-max={self.sample_max:.4g}; "
                f"KS p={self.ks_pvalue:.3f} [{tag}]")


# --------------------------------------------------------------------------- #
# batched matrix measure (log-norm) for a stack of Jacobians
# --------------------------------------------------------------------------- #
def _matrix_measure_batch(A, norm="2", P=None) -> np.ndarray:
    r"""Vectorized logarithmic matrix measure over a batch ``A: (n, d, d)``.

    Mirrors :func:`pyddrv.lipschitz.matrix_measure` but for a whole stack at
    once (the EVT sampler evaluates thousands of Jacobians)."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        A = A[None]
    if P is not None:
        P = np.asarray(P, dtype=float)
        R = np.linalg.cholesky(P).T
        Rinv = np.linalg.inv(R)
        B = R @ A @ Rinv                     # broadcast (d,d) over batch
        sym = 0.5 * (B + np.swapaxes(B, -1, -2))
        return np.max(np.linalg.eigvalsh(sym), axis=-1)
    key = str(norm).lower()
    if key in ("2", "l2", "euclidean"):
        sym = 0.5 * (A + np.swapaxes(A, -1, -2))
        return np.max(np.linalg.eigvalsh(sym), axis=-1)
    diag = np.diagonal(A, axis1=-2, axis2=-1)               # (n, d)
    absA = np.abs(A)
    if key in ("inf", "linf", "oo", "infinity"):
        offdiag = absA.sum(axis=-1) - np.abs(diag)          # row sums
        return np.max(diag + offdiag, axis=-1)
    if key in ("1", "l1"):
        offdiag = absA.sum(axis=-2) - np.abs(diag)          # col sums
        return np.max(diag + offdiag, axis=-1)
    raise ValueError(f"unsupported norm {norm!r}")


def _batched_numerical_jacobian(f, X, eps=1e-6) -> np.ndarray:
    r"""Central-difference Jacobians for a batch of points ``X: (n, d)`` in a
    single field evaluation. Returns ``(n, d, d)`` with ``J[k, i, j] =
    df_i/dx_j`` at ``X[k]``. ``f`` is the batched field ``(m, d) -> (m, d)``."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    E = eps * np.eye(d)                                     # (d, d)
    # perturbations: for each point, +/- eps along each axis -> (n, 2d, d)
    plus = X[:, None, :] + E[None, :, :]                   # (n, d, d)
    minus = X[:, None, :] - E[None, :, :]
    pts = np.concatenate([plus, minus], axis=1).reshape(n * 2 * d, d)
    vals = np.asarray(f(pts), dtype=float).reshape(n, 2 * d, d)
    fp = vals[:, :d, :]                                     # (n, d_axis, d_out)
    fm = vals[:, d:, :]
    # column j of J is df/dx_j = (fp_j - fm_j)/(2 eps); assemble (n, d_out, d_axis)
    J = np.swapaxes((fp - fm) / (2.0 * eps), 1, 2)
    return J


# --------------------------------------------------------------------------- #
# EVT core: reverse-Weibull sup estimate from a block-maxima sampler
# --------------------------------------------------------------------------- #
def evt_sup_estimate(sampler: Callable[[int], np.ndarray], rho=0.95,
                     n_blocks=100, n_per_block=1000, significance=0.05,
                     n_bootstrap=200, seed=0) -> EVTEstimate:
    r"""Estimate ``sup q`` from a ``sampler(n) -> (n,)`` of i.i.d. draws of ``q``.

    Implements Knuth Alg. 1: block maxima -> reverse-Weibull MLE -> KS test ->
    confidence-inflated location parameter. ``rho`` is the probability with which
    the returned ``L`` over-estimates the true supremum.

    Parameters
    ----------
    sampler : callable
        ``sampler(n)`` returns ``n`` i.i.d. samples of the scalar quantity whose
        supremum is sought.
    rho : float
        Confidence level (default 0.95): higher -> larger safety margin.
    n_blocks : int
        Number of block maxima to fit the reverse Weibull to.
    n_per_block : int
        Samples per block; the EVT limit is in this parameter.
    significance : float
        KS-test significance level (default 0.05, as in Knuth/Wood-Zhang).
    n_bootstrap : int
        Bootstrap resamples for the standard error of the location parameter.
        (The location parameter is a distribution endpoint, so the usual MLE
        asymptotic-normality regularity conditions do not hold; a bootstrap SE
        is more trustworthy than an observed-information one here.)
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    block_max = np.empty(n_blocks)
    sample_max = -np.inf
    for j in range(n_blocks):
        s = np.asarray(sampler(n_per_block), dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            block_max[j] = -np.inf
            continue
        block_max[j] = s.max()
        sample_max = max(sample_max, s.max())
    block_max = block_max[np.isfinite(block_max)]
    if block_max.size < 5:
        raise ValueError("too few finite block maxima to fit an EVT model")

    # Degenerate (near-constant) block maxima: the quantity is effectively
    # deterministic over Z (e.g. a constant Jacobian). Report the constant with
    # zero spread -- no extrapolation needed.
    spread = float(block_max.max() - block_max.min())
    if spread <= 1e-12 * max(1.0, abs(float(block_max.max()))):
        g = float(block_max.max())
        return EVTEstimate(L=g, gamma=g, se=0.0, ks_pvalue=1.0, validated=True,
                           rho=rho, sample_max=float(sample_max),
                           n_blocks=n_blocks, n_per_block=n_per_block)

    # weibull_max is the reverse Weibull: support (-inf, loc], CDF
    # exp(-((loc - x)/scale)^c) for x <= loc. loc == gamma == sup estimate.
    def fit_loc(data):
        c, loc, scale = stats.weibull_max.fit(data)
        return c, loc, scale

    try:
        c, loc, scale = fit_loc(block_max)
    except Exception as exc:                     # pragma: no cover - solver-dependent
        raise ValueError(f"reverse-Weibull fit failed: {exc}") from exc

    # KS goodness-of-fit against the fitted distribution.
    ks = stats.kstest(block_max, "weibull_max", args=(c, loc, scale))
    ks_p = float(ks.pvalue)

    # Bootstrap standard error of the location parameter.
    locs = []
    for _ in range(n_bootstrap):
        bs = rng.choice(block_max, size=block_max.size, replace=True)
        try:
            _, l_b, _ = fit_loc(bs)
            if np.isfinite(l_b):
                locs.append(l_b)
        except Exception:                        # pragma: no cover
            continue
    se = float(np.std(locs)) if len(locs) > 1 else 0.0

    z = float(stats.norm.ppf(rho))
    gamma = float(loc)
    L = gamma + z * se
    # never report below the observed sample max (that is a hard lower bound)
    L = max(L, float(sample_max))
    return EVTEstimate(L=float(L), gamma=gamma, se=se, ks_pvalue=ks_p,
                       validated=bool(ks_p >= significance), rho=float(rho),
                       sample_max=float(sample_max), n_blocks=n_blocks,
                       n_per_block=n_per_block)


# --------------------------------------------------------------------------- #
# public estimators
# --------------------------------------------------------------------------- #
def evt_one_sided_lipschitz(f: Callable, center, half, norm="2", rho=0.95,
                            jac: Optional[Callable] = None, P=None,
                            n_blocks=100, n_per_block=1000, significance=0.05,
                            n_bootstrap=200, seed=0) -> EVTEstimate:
    r"""High-probability upper bound on ``L = sup_{x in box} mu(df/dx)`` over the
    axis-aligned box ``center +/- half``, via EVT instead of corner sampling.

    Uses the analytic Jacobian ``jac`` if given, else a batched central-difference
    Jacobian of ``f``. The matrix measure ``mu`` is the one for ``norm`` (or the
    weighted-``P`` Euclidean measure). This is the recommended replacement for the
    corner estimate when the Jacobian is **not** affine in the state.
    """
    center = np.asarray(center, dtype=float).reshape(-1)
    half = np.broadcast_to(np.asarray(half, dtype=float), center.shape)
    lo, hi = center - half, center + half
    d = center.size
    rng = np.random.default_rng(seed)

    def jac_batch(Z):
        if jac is not None:
            return np.stack([np.asarray(jac(z), dtype=float) for z in Z])
        return _batched_numerical_jacobian(f, Z)

    def sampler(n):
        Z = rng.uniform(lo, hi, size=(n, d))
        return _matrix_measure_batch(jac_batch(Z), norm=norm, P=P)

    est = evt_sup_estimate(sampler, rho=rho, n_blocks=n_blocks,
                           n_per_block=n_per_block, significance=significance,
                           n_bootstrap=n_bootstrap, seed=seed)

    # Soundness anchor: every evaluated mu(Jac(z)) is a value the measure ATTAINS,
    # hence a valid lower bound on the true supremum. Fold in the box corners
    # (the exact max for affine Jacobians) so the report can never fall below an
    # attained value even if the EVT extrapolation is slightly low or KS rejects.
    import itertools
    corners = np.array([[c + s * h for c, s, h in zip(center,
                        signs, half)] for signs in
                        itertools.product([-1.0, 1.0], repeat=d)])
    corner_max = float(np.max(_matrix_measure_batch(jac_batch(corners),
                                                    norm=norm, P=P)))
    if corner_max > est.L:
        est = EVTEstimate(L=corner_max, gamma=est.gamma, se=est.se,
                          ks_pvalue=est.ks_pvalue, validated=est.validated,
                          rho=est.rho, sample_max=max(est.sample_max, corner_max),
                          n_blocks=est.n_blocks, n_per_block=est.n_per_block)
    return est


def evt_one_sided_lipschitz_from_data(states, derivatives, rho=0.95, P=None,
                                      n_blocks=100, n_per_block=1000,
                                      significance=0.05, n_bootstrap=200,
                                      seed=0) -> EVTEstimate:
    r"""High-probability upper bound on the one-sided Lipschitz constant from
    sampled ``(x, f(x))`` pairs -- the fully model-free path (Definition 2), with
    Knuth's EVT extrapolation instead of the raw sample maximum.

    Estimates ``sup_{x != y} <f(x)-f(y), x-y> / ||x-y||^2`` (the 2-norm / weighted
    weak pairing). ``derivatives`` may be finite-differenced from trajectories
    (see :func:`pyddrv.lipschitz.derivatives_from_trajectories`).
    """
    X = np.asarray(states, dtype=float)
    F = np.asarray(derivatives, dtype=float)
    if X.shape != F.shape or X.ndim != 2:
        raise ValueError("states and derivatives must both be (N, d)")
    N = X.shape[0]
    Pm = None if P is None else np.asarray(P, dtype=float)
    rng = np.random.default_rng(seed)

    def ip(a, b):
        if Pm is None:
            return np.sum(a * b, axis=1)
        return np.einsum("ni,ij,nj->n", a, Pm, b)

    def sampler(n):
        i = rng.integers(0, N, size=n)
        j = rng.integers(0, N, size=n)
        keep = i != j
        i, j = i[keep], j[keep]
        dX = X[i] - X[j]
        dF = F[i] - F[j]
        denom = ip(dX, dX)
        good = denom > 0
        return ip(dF[good], dX[good]) / denom[good]

    return evt_sup_estimate(sampler, rho=rho, n_blocks=n_blocks,
                            n_per_block=n_per_block, significance=significance,
                            n_bootstrap=n_bootstrap, seed=seed)
