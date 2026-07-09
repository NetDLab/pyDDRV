"""pyddrv -- Data-Driven Recurrence-based Verification.

Certify stability properties of a dynamical system from sampled trajectories
via Recurrent Lyapunov Functions (RLF), using the plain norm
``V(x) = ||x - x*||``. No Lyapunov function search, no model of the dynamics
beyond a one-sided Lipschitz bound (which can itself be estimated from data).

High-level entry points
-----------------------
>>> import numpy as np
>>> from pyddrv import verify_stability
>>> def f(x):                          # batched field (N, d) -> (N, d)
...     return x @ np.array([[0., 2.], [-1., -1.]]).T
>>> report = verify_stability(f, R=0.7, d=2, tau=3.0)
>>> report.certified, report.alpha    # guaranteed exponential rate over Q_R
(True, ...)

:func:`verify_stability` certifies a decay rate over a box (Algorithm 1);
:func:`verify_roa` grows a certified inner region-of-attraction estimate at a
target rate (Algorithm 2). The building blocks (Theorem-8 ball certificates,
layered grids, fused JAX/torch kernels, Lipschitz estimation) live in
:mod:`pyddrv.verification` and :mod:`pyddrv.lipschitz`.

Reference: R. Siegelmann, Y. Shen, F. Paganini, E. Mallada, "Stability Analysis
and Data-driven Verification via Recurrent Lyapunov Functions."
"""
from __future__ import annotations

from .api import (
    RoAReport,
    StabilityReport,
    shifted_field,
    shifted_jacobian,
    verify_roa,
    verify_stability,
)
from .data import TrajectorySet
from .lipschitz import (
    derivatives_from_trajectories,
    matrix_measure,
    one_sided_lipschitz,
    one_sided_lipschitz_from_data,
)
from .lipschitz_evt import (
    EVTEstimate,
    evt_one_sided_lipschitz,
    evt_one_sided_lipschitz_from_data,
)
from .verification import (
    estimate_L,
    estimate_L_roa,
    find_alpha_min,
    find_alpha_min_fused,
    find_alpha_min_ladder,
    find_alpha_roa,
    find_alpha_roa_fused,
)

__version__ = "0.1.0"

__all__ = [
    # high-level API
    "verify_stability",
    "verify_roa",
    "StabilityReport",
    "RoAReport",
    "shifted_field",
    "shifted_jacobian",
    # certifiers
    "find_alpha_min",
    "find_alpha_min_fused",
    "find_alpha_min_ladder",
    "find_alpha_roa",
    "find_alpha_roa_fused",
    # Lipschitz machinery
    "estimate_L",
    "estimate_L_roa",
    "matrix_measure",
    "one_sided_lipschitz",
    "one_sided_lipschitz_from_data",
    "derivatives_from_trajectories",
    "evt_one_sided_lipschitz",
    "evt_one_sided_lipschitz_from_data",
    "EVTEstimate",
    # data
    "TrajectorySet",
    "__version__",
]
