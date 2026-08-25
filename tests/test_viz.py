"""Smoke tests for pyddrv.viz (headless Agg backend).

Verify each helper builds Axes from a report without error, and that the
argument-validation paths raise as documented. Figures are not written.
"""
import types

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyddrv.api import RoAReport, StabilityReport  # noqa: E402
from pyddrv.viz import (  # noqa: E402
    plot_anytime,
    plot_roa_2d,
    plot_roa_slice,
    plot_stability_2d,
)


def _stab_report(trace=True, grid=False):
    res = None
    if grid:
        rng = np.random.default_rng(1)
        res = types.SimpleNamespace(
            centers=rng.uniform(-0.7, 0.7, size=(20, 2)),
            halfs=np.full(20, 0.1),
            alpha_lo=rng.uniform(0.3, 0.6, size=20))
    return StabilityReport(
        alpha=0.4, alpha_upper=0.5, suboptimality=0.2, converged=False,
        L=0.3, discretization_ok=True, equilibrium=np.zeros(2), norm="2",
        R=0.8, eps=0.008, tau=5.0, n_boxes=100, refinements=3, backend="numpy",
        trace=[(0.1, -0.1, 40), (0.3, 0.2, 120), (0.6, 0.4, 360)] if trace
        else None, result=res)


def _roa_report():
    rng = np.random.default_rng(0)
    centers = rng.uniform(-0.6, 0.6, size=(30, 2))
    halfs = np.full(30, 0.1)
    return RoAReport(
        alpha=1.0, volume=float(np.sum((2 * halfs) ** 2)), n_certified=30,
        n_tested=50, equilibrium=np.zeros(2), norm="2", R=1.0, eps=0.01,
        tau=1.9, L=5.0, discretization_ok=True, trim=True, stop_reason="complete",
        centers=centers, halfs=halfs, depths=np.zeros(30, dtype=int))


def test_plot_anytime():
    ax = plot_anytime(_stab_report())
    assert ax.has_data()
    plt.close(ax.figure)


def test_plot_anytime_requires_trace():
    with pytest.raises(ValueError):
        plot_anytime(_stab_report(trace=False))


def test_plot_stability_2d_with_field():
    def f(x):
        return np.stack([x[:, 1], -np.sin(x[:, 0]) - x[:, 1]], axis=1)
    ax = plot_stability_2d(_stab_report(), f=f)
    assert ax.has_data()
    plt.close(ax.figure)


def test_plot_stability_2d_without_field():
    ax = plot_stability_2d(_stab_report(), f=None)   # box only
    assert len(ax.patches) == 1
    plt.close(ax.figure)


def test_plot_stability_2d_show_grid_modes():
    for mode in ("outline", "width", "alpha"):
        ax = plot_stability_2d(_stab_report(grid=True), show_grid=True,
                               grid_color_by=mode)
        assert ax.has_data()
        plt.close(ax.figure)


def test_plot_stability_2d_show_grid_without_result_raises():
    with pytest.raises(ValueError):
        plot_stability_2d(_stab_report(grid=False), show_grid=True)


def test_plot_roa_2d_width_and_depth():
    for color_by in ("width", "depth"):
        ax = plot_roa_2d(_roa_report(), color_by=color_by)
        assert len(ax.collections) >= 1
        plt.close(ax.figure)


def test_plot_roa_2d_rejects_bad_color_by():
    with pytest.raises(ValueError):
        plot_roa_2d(_roa_report(), color_by="nope")


def _roa_report_3d():
    # 3-D region: a 3x3x3 block of unit-ish cubes centered on the origin plane
    grid = np.linspace(-0.4, 0.4, 3)
    centers = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                       -1).reshape(-1, 3)
    halfs = np.full(len(centers), 0.2)
    return RoAReport(
        alpha=1.0, volume=float(np.sum((2 * halfs) ** 3)),
        n_certified=len(centers), n_tested=len(centers),
        equilibrium=np.zeros(3), norm="2", R=1.0, eps=0.01, tau=2.0, L=5.0,
        discretization_ok=True, trim=False, stop_reason="complete",
        centers=centers, halfs=halfs, depths=np.zeros(len(centers), dtype=int))


def test_plot_roa_slice_through_equilibrium():
    rep = _roa_report_3d()
    ax = plot_roa_slice(rep, dims=(0, 1))        # slice at x3 = 0
    # 9 of the 27 cubes contain the x3=0 plane (the x3=0 layer)
    assert len(ax.collections[0].get_paths()) == 9
    plt.close(ax.figure)


def test_plot_roa_slice_off_plane_and_empty():
    rep = _roa_report_3d()
    ax = plot_roa_slice(rep, dims=(0, 2), at=[0.0, 0.35, 0.0])  # x2=0.35 layer
    assert len(ax.collections[0].get_paths()) == 9
    plt.close(ax.figure)
    with pytest.raises(ValueError):              # far outside the region
        plot_roa_slice(rep, dims=(0, 1), at=[0.0, 0.0, 5.0])


def test_plot_roa_slice_validates_dims():
    rep = _roa_report_3d()
    with pytest.raises(ValueError):
        plot_roa_slice(rep, dims=(0, 0))
    with pytest.raises(ValueError):
        plot_roa_slice(rep, dims=(0, 5))
