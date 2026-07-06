r"""Algorithm 2 -- Find-:math:`\alpha`-RoA (paper §IX-C).

Tool (T2): given a target rate :math:`\alpha`, grow a certified region
:math:`S \subseteq Q_R` over which :math:`V(x)=\lVert x\rVert` is an
:math:`\alpha`-ERLF. Starting from the layered grid, every cube is tested with
Procedure 1 at radius :math:`c h_i`; cubes meeting the rate join ``Positives``,
and failing cubes are recursively split (Procedure 2) up to ``max_refine``.

The ``Trim`` flag controls the feasibility condition (32b). With ``Trim=False``
only the decay condition (32a) is enforced and a tentative region is grown from
local trajectory information. With ``Trim=True`` a certifying trajectory is also
required to remain inside the current region; here that membership is enforced
approximately via the inf-norm extent of the grown region (a sound box
under-approximation of the union of balls). The paper's recommended use is a
``Trim=False`` pass followed by a ``Trim=True`` prune.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ball import alpha_max
from .grid import initial_grid, norm_equiv_c, split_many


@dataclass
class RoAResult:
    alpha: float
    centers: np.ndarray = field(repr=False, default=None)   # certified cube centers
    halfs: np.ndarray = field(repr=False, default=None)
    n_certified: int = 0
    n_tested: int = 0
    refinements: int = 0
    norm: str = "2"
    trim: bool = False

    def summary(self) -> str:
        frac = (self.n_certified / self.n_tested) if self.n_tested else 0.0
        return (f"Find-alpha-RoA(alpha={self.alpha:g}): {self.n_certified} cubes "
                f"certified of {self.n_tested} tested ({frac:.1%}), "
                f"{self.refinements} refinements, Trim={self.trim}")


def find_alpha_roa(rollout, L, R, eps, d, alpha, norm="2", tau=2.0, n_time=200,
                   max_refine=5, trim=False):
    r"""Run Algorithm 2; returns the certified set of cubes (an inner RoA)."""
    c = norm_equiv_c(norm, d)
    centers, halfs = initial_grid(R, eps, norm, d)
    t_grid = np.linspace(0.0, tau, n_time)

    pos_c, pos_h = [], []
    n_tested = 0
    refinements = 0
    while True:
        if len(centers) == 0:
            break
        ys = rollout(centers, t_grid)
        n_tested += len(centers)

        S_radius = None
        if trim and pos_c:
            allc = np.concatenate(pos_c)
            allh = np.concatenate(pos_h)
            S_radius = float(np.max(np.max(np.abs(allc), axis=1) + c * allh))

        a = alpha_max(ys, t_grid, c * halfs, L, norm, S_radius)
        passed = a >= alpha
        pos_c.append(centers[passed])
        pos_h.append(halfs[passed])

        fail_c, fail_h = centers[~passed], halfs[~passed]
        if refinements >= max_refine or len(fail_c) == 0:
            break
        centers, halfs = split_many(fail_c, fail_h)
        refinements += 1

    C = np.concatenate(pos_c) if pos_c else np.empty((0, d))
    H = np.concatenate(pos_h) if pos_h else np.empty((0,))
    return RoAResult(alpha=float(alpha), centers=C, halfs=H,
                     n_certified=len(C), n_tested=n_tested,
                     refinements=refinements, norm=norm, trim=trim)
