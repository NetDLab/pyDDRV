r"""Layered covering grid and box splitting (paper §IX-A).

The domain :math:`Q_R \setminus B_\epsilon` is tiled by an exponentially-efficient
*layered* grid of cubes rather than a uniform one. Layer :math:`l` uses cubes of
half-spacing :math:`h_l = 3^{l-1}\epsilon/c` and contributes :math:`3^d-1` cells
(the shell :math:`Q_{3h_l}\setminus Q_{h_l}`, i.e. a :math:`3^{\times d}` block of
cubes minus the central one). Layers nest exactly, so :math:`m=\lceil\log_3(Rc/
\epsilon)\rceil` layers cover :math:`Q_R` while excluding :math:`Q_{\epsilon/c}
\approx B_\epsilon`. Each cube :math:`Q_h(x)` is certified through its containing
working-norm ball :math:`B_{ch}(x)` (Theorem 8).
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def norm_equiv_c(norm: str, d: int) -> float:
    r"""Constant ``c`` with ``||x|| <= c ||x||_inf`` so ``Q_h(x) ⊆ B_{ch}(x)``.

    ``c = 1`` for the max norm, ``sqrt(d)`` for Euclidean, ``d`` for the 1-norm.
    """
    key = str(norm).lower()
    if key in ("inf", "linf", "oo", "infinity"):
        return 1.0
    if key in ("2", "l2", "euclidean"):
        return math.sqrt(d)
    if key in ("1", "l1"):
        return float(d)
    raise ValueError(f"unsupported norm {norm!r}")


def n_layers(R: float, eps: float, c: float) -> int:
    r"""Number of layers ``m`` so the inner hole ``Q_{R/3^m}`` is CONTAINED in
    the target ball ``B_eps`` (paper: exclude ``Q_{eps/c}``): the hole is the
    largest R-anchored 3-adic level with ``R/3^m <= eps/c``. Ceiling semantics
    -- rounding the layer count down would exclude a neighborhood LARGER than
    ``B_eps`` and silently weaken the reported epsilon."""
    return max(1, int(math.ceil(math.log(R * c / eps, 3.0) - 1e-9)))


def initial_grid(R: float, eps: float, norm: str, d: int):
    r"""Build the layered grid covering ``Q_R \ Q_{R/3^m}`` (with ``R/3^m ≈ eps/c``).

    Constructed top-down so the outermost layer (``l=m``) has half-spacing
    ``h_m = R/3`` and its cubes span ``[R/3, R]`` in the inf-norm -- i.e. the grid
    covers exactly ``Q_R`` with no overshoot. Inner layers shrink by ``/3`` so the
    excluded inner cube is ``Q_{R/3^m} ≈ B_eps``.

    Returns
    -------
    centers : (N, d) ndarray
    halfs : (N,) ndarray
        Half-spacing ``h_l`` of each cube.
    """
    c = norm_equiv_c(norm, d)
    m = n_layers(R, eps, c)
    centers, halfs = [], []
    for l in range(1, m + 1):
        h = R / (3.0 ** (m - l + 1))   # l=m -> R/3 (outer), l=1 -> R/3^m (inner)
        offsets = (-2.0 * h, 0.0, 2.0 * h)
        for combo in itertools.product(offsets, repeat=d):
            if all(o == 0.0 for o in combo):
                continue  # exclude the central cube (the inner hole / next layer)
            centers.append(combo)
            halfs.append(h)
    return np.asarray(centers, dtype=float), np.asarray(halfs, dtype=float)


def split(center, half) -> tuple:
    r"""Procedure 2: split a cube ``Q_h(x)`` into ``3^d`` sub-cubes of ``h/3``.

    Sub-centers are offset by ``{-2h/3, 0, +2h/3}`` per axis.

    Returns
    -------
    (centers, halfs) : (3^d, d) and (3^d,) ndarrays.
    """
    center = np.asarray(center, dtype=float)
    h = float(half)
    d = center.size
    offs = (-2.0 * h / 3.0, 0.0, 2.0 * h / 3.0)
    subs = np.array([center + np.array(combo)
                     for combo in itertools.product(offs, repeat=d)])
    return subs, np.full(subs.shape[0], h / 3.0)


def split_many(centers, halfs) -> tuple:
    """Split each box in a batch; returns stacked sub-boxes."""
    all_c, all_h = [], []
    for ctr, h in zip(np.asarray(centers), np.asarray(halfs)):
        sc, sh = split(ctr, h)
        all_c.append(sc)
        all_h.append(sh)
    return np.concatenate(all_c, axis=0), np.concatenate(all_h, axis=0)
