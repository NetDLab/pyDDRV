"""Model-based comparison baselines (require the polynomial vector field).

These are NOT data-driven -- they are the classical certificates pyddrv's RLF
method is compared against (e.g. the paper's SOSTOOLS column). Optional deps:
``pip install -e ".[sos]"``.
"""
from .sos import (
    available_solver,
    build_polynomial_field,
    certify_rate_sos,
    certify_rate_sos_box,
)

__all__ = ["build_polynomial_field", "certify_rate_sos", "certify_rate_sos_box",
           "available_solver"]
