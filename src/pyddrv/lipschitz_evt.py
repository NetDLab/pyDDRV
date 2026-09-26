r"""Extreme-value-theory (EVT) estimation of Lipschitz / one-sided-Lipschitz
constants with a high-probability upper bound.

The box-corner estimate in :mod:`pyddrv.lipschitz` is exact only when the
Jacobian is affine in the state: the matrix measure is then convex in the state
and its supremum over a box is attained at a corner. For a general nonlinear
field the corner maximum can underestimate ``sup_x mu(df/dx)``, and an
underestimate of ``L`` makes the certificate invalid. The largest value in a
sample is likewise only a lower bound on the supremum.

This module implements the extreme-value approach of Knuth et al., "Planning with
Learned Dynamics: Probabilistic Guarantees on Safety and Reachability via
Lipschitz Constants" (arXiv:2010.08993), which builds on Wood &
Zhang, "Estimation of the Lipschitz constant of a function" (J. Global Optim.,
1996). The supremum of a bounded quantity is estimated as the upper endpoint
``gamma`` of a three-parameter reverse Weibull distribution fit to block maxima,
returned with a confidence margin so that the estimate over-estimates the true
constant with a user-chosen probability ``rho``.

Method
------
To bound ``L = sup_{z in Z} q(z)`` for a scalar quantity ``q``:

1. For ``j = 1..n_blocks``: draw ``n_per_block`` i.i.d. samples of ``q`` and take
   the block maximum ``s_j = max_i q(z_{i,j})``.
2. Fit a reverse Weibull (``scipy.stats.weibull_max``) to ``{s_j}`` by MLE,
   obtaining the location parameter ``gamma_hat`` (the distribution's upper
   support limit, which is the Lipschitz estimate) and its standard error ``xi``.
3. Validate the fit with a Kolmogorov-Smirnov goodness-of-fit test at
   significance ``0.05``.
4. Report ``L_hat = gamma_hat + Phi^{-1}(rho) * xi``, an overestimate of ``L``
   with probability ``rho`` (asymptotically in ``n_per_block``, via
   Fisher-Tippett-Gnedenko).

The reverse Weibull is the *only* extreme-value class with support bounded above,
so a good fit is itself evidence that the underlying quantity is bounded (finite
Lipschitz constant); a failed KS test is reported, not silently ignored.

Ill-posed fits
--------------
The fit in step 2 is only meaningful when the block maxima look like draws from
a continuous distribution. It breaks down when ``q`` saturates at its supremum
(most blocks attain nearly the same value) or when the sampled values are
quantized by rounding, for instance a finite-difference Jacobian of a float32
field. The likelihood then has no well-defined maximum, the bootstrap locations
scatter over orders of magnitude, and ``gamma_hat + Phi^{-1}(rho) xi`` is
useless. The estimate is therefore classified as one of three kinds
(:attr:`EVTEstimate.status`):

* ``"fit"``: the steps above, used when the KS test does not reject, at least
  half of the block maxima are distinct at the sampling resolution, the
  bootstrap fits converge, and ``xi`` is no larger than the range of the block
  maxima. The reported value is exactly the one above.
* ``"degenerate"``: the block maxima agree to within the sampling resolution;
  their common value is reported.
* ``"fallback"``: any other case. The report is
  ``max_j s_j + k * (max_j s_j - min_j s_j)``. Under a reverse-Weibull law for
  the block maxima, ``(gamma - max_j s_j) / (max_j s_j - min_j s_j)`` does not
  depend on the location or scale, only on the shape ``c`` and on
  ``n_blocks``. ``k`` is its ``rho``-quantile at ``c = max_shape``, computed
  by simulation, and the quantile grows with ``c`` (checked numerically).
  The fallback therefore covers the supremum with probability at least
  ``rho`` for every shape ``c <= max_shape`` without estimating any
  parameter. When ``q`` is Lipschitz and sampled uniformly on a box of
  dimension ``m``, the mass of ``{q > sup q - t}`` is at least of order
  ``t^m``, so ``c <= m``; the estimators below pass ``max_shape``
  accordingly.

In every case the sampling ``resolution`` (the error with which each sample of
``q`` is computed, for instance by finite differences) is added to the report,
so that the bound covers the true values rather than the computed ones.

The guarantee is probabilistic and asymptotic, unlike the exact corner bound
for affine Jacobians. It is meant for fields where no closed-form bound or
affine structure is available.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from .lipschitz import batched_numerical_jacobian as _batched_numerical_jacobian


@dataclass
class EVTEstimate:
    """Result of an EVT Lipschitz estimate.

    Attributes
    ----------
    L : float
        The reported high-probability upper bound. For ``status == "fit"`` it is
        ``max(gamma_hat + Phi^{-1}(rho) xi, sample_max) + resolution``; see the
        module docstring for the other cases.
    gamma : float
        Fitted reverse-Weibull location parameter (the raw sup estimate); ``nan``
        when no fit was attempted.
    se : float
        Standard error ``xi`` of ``gamma`` (bootstrap over block maxima).
    ks_pvalue : float
        Kolmogorov-Smirnov goodness-of-fit p-value for the reverse-Weibull fit
        (``nan`` when no fit was attempted).
    validated : bool
        ``ks_pvalue >= significance`` -- the reverse-Weibull model was not
        rejected (evidence of a finite, well-estimated constant).
    rho : float
        Confidence level requested.
    sample_max : float
        Raw maximum over all samples (a lower bound; for reference/diagnostics).
    n_blocks, n_per_block : int
        Sampling budget used.
    status : str
        ``"fit"`` (reverse-Weibull fit used), ``"degenerate"`` (block maxima
        agree to within ``resolution``) or ``"fallback"`` (fit ill-posed;
        ``L`` is the sample maximum plus a margin from the range of the block
        maxima). See the module docstring.
    reason : str
        Why the fit was not used, empty when ``status == "fit"``.
    shape : float
        Fitted reverse-Weibull shape ``c`` (``nan`` when not fitted).
    resolution : float
        Error with which each sample was computed, added to ``L``.
    spread : float
        Range ``max - min`` of the block maxima.
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
    status: str = "fit"
    reason: str = ""
    shape: float = float("nan")
    resolution: float = 0.0
    spread: float = float("nan")

    @property
    def well_posed(self) -> bool:
        """True when ``L`` comes from the reverse-Weibull fit or from a
        degenerate (constant) sample, False for the fallback."""
        return self.status != "fallback"

    def summary(self) -> str:
        if self.status == "fit":
            tag = "validated" if self.validated else "NOT validated (KS rejected)"
            head = (f"EVT L_hat={self.L:.4g} (gamma={self.gamma:.4g} + "
                    f"z_{self.rho:.2f}*{self.se:.2g})")
        elif self.status == "degenerate":
            tag = "degenerate: block maxima constant"
            head = f"EVT L_hat={self.L:.4g}"
        else:
            tag = f"FALLBACK, fit ill-posed: {self.reason}"
            head = (f"EVT L_hat={self.L:.4g} (sample-max + margin from "
                    f"block-max range {self.spread:.2g})")
        ks = "n/a" if not np.isfinite(self.ks_pvalue) else f"{self.ks_pvalue:.3f}"
        res = f"; resolution={self.resolution:.2g}" if self.resolution > 0 else ""
        return (f"{head}; sample-max={self.sample_max:.4g}; KS p={ks}{res} "
                f"[{tag}]")


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


# --------------------------------------------------------------------------- #
# EVT core: reverse-Weibull sup estimate from a block-maxima sampler
# --------------------------------------------------------------------------- #
def _best_weibull_max_fit(x, starts):
    """Best of ``weibull_max`` MLE fits from ``starts`` (``None`` is SciPy's
    default start): the valid fit with the largest finite log-likelihood, or
    ``None``."""
    from scipy import stats

    best, best_ll = None, -np.inf
    for st in starts:
        try:
            p = (stats.weibull_max.fit(x) if st is None else
                 stats.weibull_max.fit(x, st[0], loc=st[1], scale=st[2]))
        except Exception:                        # pragma: no cover - solver-dependent
            continue
        if not (np.all(np.isfinite(p)) and p[0] > 0 and p[2] > 0):
            continue
        ll = float(np.sum(stats.weibull_max.logpdf(x, *p)))
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = tuple(float(v) for v in p), ll
    return best


def _fit_reverse_weibull(x, start=None):
    """Maximum-likelihood ``(c, loc, scale)`` of ``weibull_max`` for ``x``.

    SciPy's default starting point sometimes converges to a poor local optimum
    (for block maxima with shape ``c`` near 2 it can return a log-likelihood
    150 below the optimum), so several starts are tried and the one with the
    largest finite log-likelihood is kept. With ``start`` (the bootstrap passes
    the full-sample fit) only that warm start is tried, and the other starts
    only if it fails. Returns ``None`` when no start gives a valid fit."""
    top = float(np.max(x))
    if start is not None:
        c0, l0, s0 = start
        warm = _best_weibull_max_fit(
            x, [(c0, max(l0, top + 1e-12 * max(1.0, abs(top))), s0)])
        if warm is not None:
            return warm
    sd = float(np.std(x))
    starts = [None]
    if sd > 0:
        starts += [(c0, top + sd, 2.0 * sd) for c0 in (1.0, 2.0, 4.0)]
    return _best_weibull_max_fit(x, starts)


@lru_cache(maxsize=64)
def _fallback_factor(rho: float, n_blocks: int, max_shape: float,
                     n_sim: int = 20000) -> float:
    r"""``rho``-quantile of ``(gamma - M) / (M - m)`` for ``n_blocks`` i.i.d.
    reverse-Weibull draws with shape ``max_shape``, where ``M`` and ``m`` are
    their maximum and minimum and ``gamma`` the upper endpoint.

    The ratio does not depend on location or scale: with ``E_j`` i.i.d.
    standard exponential, ``gamma - s_j = scale * E_j^{1/c}``. Its quantiles
    increase with ``c`` (checked numerically), so ``M + k (M - m)`` with this
    ``k`` covers ``gamma`` with probability at least ``rho`` for every shape
    ``c <= max_shape``."""
    rng = np.random.default_rng(12345)
    k = np.empty(n_sim)
    step = 2000
    for s in range(0, n_sim, step):
        g = rng.exponential(size=(min(step, n_sim - s), n_blocks)) ** (1.0 / max_shape)
        lo, hi = g.min(axis=1), g.max(axis=1)
        k[s:s + g.shape[0]] = lo / (hi - lo)
    return float(np.quantile(k, rho))


def evt_sup_estimate(sampler: Callable[[int], np.ndarray], rho=0.95,
                     n_blocks=100, n_per_block=1000, significance=0.05,
                     n_bootstrap=200, seed=0, resolution=0.0,
                     max_shape=6.0) -> EVTEstimate:
    r"""Estimate ``sup q`` from a ``sampler(n) -> (n,)`` of i.i.d. draws of ``q``.

    Follows Knuth et al.: block maxima, maximum-likelihood fit of a reverse
    Weibull distribution, Kolmogorov-Smirnov test, and the fitted endpoint
    increased by a confidence margin. ``rho`` is the probability with which
    the returned ``L`` over-estimates the true supremum. When the fit is
    ill-posed the estimate falls back to the sample maximum plus a margin; see
    the module docstring and :attr:`EVTEstimate.status`.

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
    resolution : float
        Error with which each sample of ``q`` is computed (0 when exact up to
        float64 rounding). Block maxima closer than this are treated as equal,
        and it is added to the reported ``L``.
    max_shape : float
        Largest reverse-Weibull shape the fallback margin must cover. For a
        Lipschitz ``q`` sampled uniformly on an ``m``-dimensional box the shape
        is at most ``m``.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    resolution = float(max(resolution, 0.0))
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
    m = block_max.size
    top = float(sample_max)
    spread = float(block_max.max() - block_max.min())
    # values closer than this cannot be told apart
    floor = max(1e-12 * max(1.0, abs(top)), resolution)
    common = dict(rho=float(rho), sample_max=top, n_blocks=n_blocks,
                  n_per_block=n_per_block, resolution=resolution, spread=spread)

    # Degenerate (near-constant) block maxima: the quantity is effectively
    # deterministic over Z (e.g. a constant Jacobian), or saturates at its
    # supremum in every block. Report the common value with zero spread.
    if spread <= floor:
        return EVTEstimate(L=top + resolution, gamma=top, se=0.0, ks_pvalue=1.0,
                           validated=True, status="degenerate", **common)

    def fallback(reason, gamma=float("nan"), se=float("nan"),
                 ks_p=float("nan"), c=float("nan")):
        k = _fallback_factor(float(rho), int(m), float(max_shape))
        return EVTEstimate(L=top + k * spread + resolution, gamma=gamma, se=se,
                           ks_pvalue=ks_p, validated=bool(ks_p >= significance),
                           status="fallback", reason=reason, shape=c, **common)

    # Quantized maxima (many ties at the sampling resolution): a continuous
    # model does not describe them, and the MLE/KS test are not meaningful.
    n_distinct = 1 + int(np.sum(np.diff(np.sort(block_max)) > floor))
    if n_distinct < 0.5 * m:
        return fallback(f"only {n_distinct} of {m} block maxima distinct "
                        f"at resolution {floor:.2g}")

    # weibull_max is the reverse Weibull: support (-inf, loc], CDF
    # exp(-((loc - x)/scale)^c) for x <= loc. loc == gamma == sup estimate.
    fit = _fit_reverse_weibull(block_max)
    if fit is None:
        return fallback("reverse-Weibull fit failed")
    c, loc, scale = fit

    # KS goodness-of-fit against the fitted distribution.
    ks_p = float(stats.kstest(block_max, "weibull_max", args=fit).pvalue)

    # Bootstrap standard error of the location parameter, each resample warm
    # started from the full-sample fit.
    locs = []
    for _ in range(n_bootstrap):
        bs = rng.choice(block_max, size=m, replace=True)
        fb = _fit_reverse_weibull(bs, start=fit)
        if fb is not None:
            locs.append(fb[1])
    se = float(np.std(locs)) if len(locs) > 1 else 0.0
    gamma = float(loc)

    reasons = []
    if ks_p < significance:
        reasons.append(f"KS rejected (p={ks_p:.2g})")
    if n_bootstrap > 1 and len(locs) < 0.9 * n_bootstrap:
        reasons.append(f"{n_bootstrap - len(locs)} of {n_bootstrap} bootstrap "
                       f"fits failed")
    if se > spread:
        reasons.append(f"bootstrap se {se:.2g} exceeds block-max range "
                       f"{spread:.2g}")
    if reasons:
        return fallback("; ".join(reasons), gamma=gamma, se=se, ks_p=ks_p, c=c)

    z = float(stats.norm.ppf(rho))
    # never report below the observed sample max (that is a hard lower bound)
    L = max(gamma + z * se, top) + resolution
    return EVTEstimate(L=float(L), gamma=gamma, se=se, ks_pvalue=ks_p,
                       validated=True, status="fit", shape=float(c), **common)


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
    Jacobian of ``f`` with a step matched to the precision of ``f``. The matrix
    measure ``mu`` is the one for ``norm`` (or the weighted-``P`` Euclidean
    measure). This is the recommended replacement for the corner estimate when
    the Jacobian is not affine in the state.

    For a finite-difference Jacobian the error of each sampled ``mu`` is
    estimated on one block of points as the largest change of ``mu`` when the
    step is doubled, and passed to :func:`evt_sup_estimate` as its
    ``resolution``. The fallback margin is calibrated for shapes up to the box
    dimension (see the module docstring).
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

    def mu(Z):
        return _matrix_measure_batch(jac_batch(Z), norm=norm, P=P)

    resolution = 0.0
    if jac is None:
        Zp = np.random.default_rng(seed + 1).uniform(lo, hi, size=(n_per_block, d))
        J2 = _batched_numerical_jacobian(f, Zp, step_factor=2.0)
        resolution = float(np.max(np.abs(
            mu(Zp) - _matrix_measure_batch(J2, norm=norm, P=P))))

    def sampler(n):
        return mu(rng.uniform(lo, hi, size=(n, d)))

    est = evt_sup_estimate(sampler, rho=rho, n_blocks=n_blocks,
                           n_per_block=n_per_block, significance=significance,
                           n_bootstrap=n_bootstrap, seed=seed,
                           resolution=resolution, max_shape=float(d))

    # Soundness anchor: every evaluated mu(Jac(z)) is a value the measure ATTAINS,
    # hence a valid lower bound on the true supremum. Fold in the box corners
    # (the exact max for affine Jacobians) so the report can never fall below an
    # attained value even if the EVT extrapolation is slightly low or KS rejects.
    import itertools
    corners = np.array([[c + s * h for c, s, h in zip(center,
                        signs, half)] for signs in
                        itertools.product([-1.0, 1.0], repeat=d)])
    corner_max = float(np.max(mu(corners)))
    if corner_max + resolution > est.L:
        est = replace(est, L=corner_max + resolution,
                      sample_max=max(est.sample_max, corner_max))
    return est


def evt_one_sided_lipschitz_from_data(states, derivatives, rho=0.95, P=None,
                                      n_blocks=100, n_per_block=1000,
                                      significance=0.05, n_bootstrap=200,
                                      seed=0) -> EVTEstimate:
    r"""High-probability upper bound on the one-sided Lipschitz constant from
    sampled ``(x, f(x))`` pairs. Applies the extreme-value estimate to the
    ratios used in the definition of the one-sided Lipschitz constant, instead
    of taking the largest sampled ratio.

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

    # the ratio is sampled over pairs (x, y), a domain of dimension 2d
    return evt_sup_estimate(sampler, rho=rho, n_blocks=n_blocks,
                            n_per_block=n_per_block, significance=significance,
                            n_bootstrap=n_bootstrap, seed=seed,
                            max_shape=2.0 * X.shape[1])
