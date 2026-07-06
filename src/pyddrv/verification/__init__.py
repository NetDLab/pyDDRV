"""Rigorous ε-ERLF verification (paper §VIII–IX).

Theorem 8 ball certification + Procedure 1 (`ball`), the layered covering grid
and 3^d split (`grid`), and Algorithm 1 / Algorithm 2 (`algorithm1`,
`algorithm2`).
"""
from __future__ import annotations

from .algorithm1 import AlphaMinResult, find_alpha_min
from .algorithm2 import RoAResult, find_alpha_roa
from .ball import alpha_max, certify_ball
from .fused import (
    FusedAlphaMinResult,
    FusedRoAResult,
    alpha_bounds_materialized_np,
    find_alpha_min_fused,
    find_alpha_min_ladder,
    find_alpha_roa_fused,
    fused_alpha_bounds,
    region_distance_map,
)
from .estimate import (
    LipschitzEstimate,
    boundary_grid,
    estimate_L,
    estimate_L_roa,
    excursion_radius,
)
from .grid import initial_grid, norm_equiv_c, split, split_many

__all__ = [
    "alpha_max",
    "certify_ball",
    "initial_grid",
    "split",
    "split_many",
    "norm_equiv_c",
    "estimate_L",
    "excursion_radius",
    "boundary_grid",
    "LipschitzEstimate",
    "find_alpha_min",
    "AlphaMinResult",
    "find_alpha_roa",
    "RoAResult",
    "fused_alpha_bounds",
    "find_alpha_min_fused",
    "find_alpha_min_ladder",
    "find_alpha_roa_fused",
    "FusedRoAResult",
    "region_distance_map",
    "estimate_L_roa",
    "alpha_bounds_materialized_np",
    "FusedAlphaMinResult",
]
