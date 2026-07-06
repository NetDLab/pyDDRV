r"""Algorithm 1 -- Find-:math:`\alpha_{\min}` (paper §IX-B).

Tool (T1): given an outer radius :math:`R`, find the largest decay rate
:math:`\alpha` for which :math:`V(x)=\lVert x\rVert` satisfies the
:math:`\alpha`-ERLF condition on :math:`Q_R\setminus B_\epsilon`. Each grid cube
:math:`Q_{h_i}(x_i)` yields two numbers via Procedure 1:

- ``alpha_lo_i = alpha_max(x_i, c*h_i)`` -- certifies the whole cube,
- ``alpha_up_i = alpha_max(x_i, 0)``     -- certifies only the center.

The certified rate over the grid is ``min_i alpha_lo_i``; the relative gap at the
worst cube measures how much refinement could still gain. While that gap exceeds
``delta``, the ``k`` worst cubes are split (Procedure 2) and the bounds
recomputed. Trajectory simulation for all cube centers is batched through the
supplied ``rollout`` callable (the GPU path).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ball import alpha_max
from .grid import initial_grid, norm_equiv_c, split_many


@dataclass
class AlphaMinResult:
    alpha_certified: float          # min over boxes of alpha_lo (the guarantee)
    refinements: int
    n_boxes: int
    norm: str
    L: float
    tau: float
    # Sub-optimality ratio at the worst cube i* = argmin_i alpha_lo_i:
    #   suboptimality = (alpha_up[i*] - alpha_lo[i*]) / alpha_lo[i*]
    # the paper's stopping quantity. Bounds (optimal - certified)/certified at the
    # binding cube. `converged` is True when it fell to <= delta (tight); False
    # means the refinement budget (max_refine) was hit first (conservative).
    suboptimality: float = float("inf")
    converged: bool = False
    delta: float = 0.0
    alpha_upper: float = float("inf")     # alpha_up[i*]: upper bound at the worst cube
    trace: list = field(repr=False, default=None)   # anytime: [(time, alpha, n_boxes), ...]
    centers: np.ndarray = field(repr=False, default=None)
    halfs: np.ndarray = field(repr=False, default=None)
    alpha_lo: np.ndarray = field(repr=False, default=None)
    alpha_up: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        tag = "converged" if self.converged else f"budget-limited ({self.refinements} refines)"
        return (f"Find-alpha_min: certified alpha >= {self.alpha_certified:.4f} "
                f"(<= {self.alpha_upper:.4f}, sub-opt {self.suboptimality:.1%}, {tag})  |  "
                f"{self.n_boxes} boxes, norm={self.norm}, L={self.L:.3g}")


def find_alpha_min(rollout, L, R, eps, d, norm="2", tau=2.0, n_time=200,
                   delta=0.2, max_refine=6, k_split=None,
                   enforce_S=True, max_split_per_round=20000, record_trace=False,
                   gap_ref="binding", max_seconds=None,
                   plateau_eps=None, plateau_window=3):
    r"""Run Algorithm 1.

    Parameters
    ----------
    rollout : callable
        ``rollout(centers (N,d), t_grid (T,)) -> (N, T, d)`` batched simulation.
    L : float
        One-sided Lipschitz constant over ``Q_R`` (see :mod:`pyddrv.lipschitz`).
    R, eps : float
        Outer / inner radii of ``Q_R \ B_eps``.
    d : int
        State dimension.
    norm : {"2", "inf", "1"}
        Working norm.
    tau : float
        Recurrence horizon.
    n_time : int
        Number of time samples in ``[0, tau]`` for the Procedure-1 maximization.
    delta : float
        Relative-gap stopping threshold at the worst box.
    max_refine : int
        Maximum refinement rounds.
    k_split : int, optional
        Number of worst boxes to split per round (default: 10% of the grid).
    enforce_S : bool, default True
        Enforce the (32b) membership check ``||phi(t,x)||_inf <= R - r e^{L t}``
        against ``Q_R``, at the same time as (32a). Necessary for non-inf working
        norms (e.g. the 2-norm bilinear experiments); harmless for the inf-norm.
    """
    c = norm_equiv_c(norm, d)
    centers, halfs = initial_grid(R, eps, norm, d)
    t_grid = np.linspace(0.0, tau, n_time)
    S_radius = R if enforce_S else None

    def evaluate(cs, hs):
        """Simulate a set of box centers once and return (alpha_lo, alpha_up)."""
        ys = rollout(cs, t_grid)                           # (N, T, d) batched
        lo = alpha_max(ys, t_grid, c * hs, L, norm, S_radius)
        up = alpha_max(ys, t_grid, 0.0, L, norm, S_radius)
        return lo, up

    # Incremental simulation: every box is simulated exactly once -- the initial
    # grid up front, then only the freshly split sub-boxes each round (kept boxes
    # reuse their cached alpha). Total work is O(boxes), not O(rounds x boxes).
    t0 = time.time()
    trace = []        # anytime curve: (cumulative_time, certified alpha, n_boxes)
    a_lo, a_up = evaluate(centers, halfs)
    if record_trace:
        trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))
    refinements = 0
    cert_hist = []
    while True:
        i = int(np.argmin(a_lo))
        worst_lo = a_lo[i]
        # Sub-optimality reference + normalization:
        #  "binding" -> (a_up[i*] - a_lo[i*]) / a_lo[i*]   (paper delta_i*, by lower)
        #  "global"  -> (min a_up - min a_lo) / min a_up   (distance to ceiling, as a
        #               fraction OF the ceiling -- "within delta of optimal").
        if gap_ref == "global":
            ref_up = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else np.inf
        else:
            ref_up = a_up[i]
        gap = np.inf
        if np.isfinite(worst_lo) and worst_lo > 0 and np.isfinite(ref_up):
            denom = ref_up if gap_ref == "global" else worst_lo
            gap = (ref_up - worst_lo) / denom

        converged = gap <= delta
        timed_out = max_seconds is not None and (time.time() - t0) >= max_seconds
        # Plateau stop: the certified rate has stopped improving. With a fixed
        # time budget and large tau (so the ceiling is not a meaningful target),
        # this is the honest "refinement no longer helps" criterion -- it stops
        # easy configs early instead of burning the budget / exploding the grid,
        # while a genuinely-still-improving hard config runs on to max_seconds.
        plateaued = False
        if plateau_eps is not None and np.isfinite(worst_lo) and worst_lo > 0:
            cert_hist.append(float(worst_lo))
            if (len(cert_hist) > plateau_window
                    and cert_hist[-1] - cert_hist[-1 - plateau_window] < plateau_eps):
                plateaued = True
        if refinements >= max_refine or converged or timed_out or plateaued:
            break

        # Refine every box that blocks the certified rate: a_lo below the
        # achievable ceiling (min a_up) by more than the tolerance. This drives
        # min(a_lo) -> min(a_up) far faster than splitting a fixed 10%, while a
        # cap bounds per-round growth. (k_split, if set, overrides with a fixed
        # count of the worst boxes.)
        # Split every box whose a_lo leaves the certified rate more than `delta`
        # below the achievable ceiling. Threshold ceiling*(1-delta) matches the
        # GLOBAL stop normalized by the ceiling: a_lo >= ceiling*(1-delta) <=>
        # (ceiling - a_lo)/ceiling <= delta.
        ceiling = np.min(a_up[np.isfinite(a_up)]) if np.isfinite(a_up).any() else 0.0
        target = ceiling * (1.0 - delta) if ceiling > 0 else np.max(a_lo[np.isfinite(a_lo)] if np.isfinite(a_lo).any() else [0.0])
        blocking = ~(a_lo >= target)
        order = np.argsort(a_lo)              # worst first
        if k_split is not None:
            worst = order[:k_split]
        else:
            cap = max(1, min(int(blocking.sum()), max_split_per_round))
            worst = order[:cap]
        keep = np.ones(len(centers), dtype=bool)
        keep[worst] = False
        sub_c, sub_h = split_many(centers[worst], halfs[worst])
        sub_lo, sub_up = evaluate(sub_c, sub_h)            # only the new boxes
        centers = np.concatenate([centers[keep], sub_c], axis=0)
        halfs = np.concatenate([halfs[keep], sub_h], axis=0)
        a_lo = np.concatenate([a_lo[keep], sub_lo], axis=0)
        a_up = np.concatenate([a_up[keep], sub_up], axis=0)
        refinements += 1
        if record_trace:
            trace.append((time.time() - t0, float(np.min(a_lo)), len(centers)))

    return AlphaMinResult(
        alpha_certified=float(np.min(a_lo)),
        refinements=refinements,
        n_boxes=len(centers),
        norm=norm, L=float(L), tau=float(tau),
        suboptimality=float(gap),
        converged=bool(converged),
        delta=float(delta),
        alpha_upper=float(ref_up),
        trace=trace if record_trace else None,
        centers=centers, halfs=halfs, alpha_lo=a_lo, alpha_up=a_up,
    )
