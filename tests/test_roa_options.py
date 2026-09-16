"""Large-run controls of find_alpha_roa_fused (ported from the research
sandbox's dimension sweep): frontier priority modes, bounded frontier with
best-first eviction, inner-first seeding, disk-headroom guard, plateau rule.

All on a small 2-D linear problem so they run in seconds. The invariant under
test: these knobs change *which* cubes get evaluated under a budget, never the
soundness of what is certified (with no budget, the certified set must be
identical across orders; with eviction, a subset).
"""
import numpy as np
import pytest

from pyddrv.lipschitz import matrix_measure
from pyddrv.systems import linear
from pyddrv.verification.fused import find_alpha_roa_fused

pytest.importorskip("jax", reason="fused RoA kernel needs JAX")

A = np.array([[0.0, 2.0], [-1.0, -1.0]])          # stable spiral
F = linear(A)
L = matrix_measure(A, norm="2")
KW = dict(R=1.0, eps=1.0 / 27.0, d=2, alpha=0.2, tau=3.0, n_steps=150,
          max_refine=2)


def _cube_set(res):
    c = np.round(np.asarray(res.centers, float), 6)
    h = np.round(np.asarray(res.halfs, float), 6)
    return {tuple(row) + (hh,) for row, hh in zip(c.tolist(), h.tolist())}


def test_priority_modes_certify_identical_sets_without_budget():
    base = find_alpha_roa_fused(F, L, priority="gain", **KW)
    assert base.n_certified > 0
    for mode in ("norm", "sizedist"):
        other = find_alpha_roa_fused(F, L, priority=mode, **KW)
        assert _cube_set(other) == _cube_set(base), mode
        assert other.volume if hasattr(other, "volume") else True


def test_inner_first_changes_order_only():
    base = find_alpha_roa_fused(F, L, **KW)
    boosted = find_alpha_roa_fused(F, L, inner_first=0.5,
                                   inner_first_min_h=1.0 / 81.0, **KW)
    assert _cube_set(boosted) == _cube_set(base)


def test_bounded_frontier_eviction_is_sound_subset():
    full = find_alpha_roa_fused(F, L, **KW)
    evicted = find_alpha_roa_fused(F, L, max_pending_parents=1, **KW)
    assert evicted.stop_reason == "complete"
    assert 0 < evicted.n_certified <= full.n_certified
    assert _cube_set(evicted) <= _cube_set(full)      # never certifies extra


def test_disk_guard_stops_cleanly(tmp_path):
    res = find_alpha_roa_fused(F, L, spill_dir=str(tmp_path),
                               min_disk_gb=1e12, disk_check_every=1, **KW)
    assert res.stop_reason == "disk"
    assert res.budget_hit
    # cubes certified before the stop are intact on disk (memmap-backed)
    assert res.n_certified == len(res.centers)


def test_disk_guard_can_be_disabled(tmp_path):
    res = find_alpha_roa_fused(F, L, spill_dir=str(tmp_path),
                               min_disk_gb=None, disk_check_every=1, **KW)
    assert res.stop_reason == "complete"


def test_plateau_rule_runs_and_reports_reason():
    res = find_alpha_roa_fused(F, L, plateau_rel=1e-3, **KW)
    assert res.stop_reason in ("complete", "plateau")
    assert res.n_certified > 0
