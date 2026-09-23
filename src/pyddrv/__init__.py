"""pyDDRV: Data-Driven Recurrence-based Verification.

Certifies exponential stability and regions of attraction of an equilibrium
with recurrent Lyapunov functions, using the norm ``V(x) = ||x - x*||``. The
certificate is computed from trajectories that pyDDRV simulates from a vector
field supplied by the user, together with an upper bound on the one-sided
Lipschitz constant of the field. No Lyapunov function has to be constructed.

High-level entry points
-----------------------
>>> import numpy as np
>>> from pyddrv import verify_stability
>>> def f(x):                          # batched field (N, d) -> (N, d)
...     return x @ np.array([[0., 2.], [-1., -1.]]).T
>>> report = verify_stability(f, R=0.7, d=2, tau=3.0)
>>> report.certified, report.alpha    # certified exponential rate on Q_R
(True, ...)

:func:`verify_stability` certifies a decay rate on a box, and
:func:`verify_roa` an inner approximation of the region of attraction at a
given rate. The lower-level functions are in :mod:`pyddrv.verification` and
:mod:`pyddrv.lipschitz`.

Reference: R. Siegelmann, F. Paganini, E. Mallada, "Stability Analysis and
Data-driven Verification via Recurrent Lyapunov Functions," arXiv:2608.26447.
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
