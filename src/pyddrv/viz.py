r"""Plotting helpers for pyddrv certificates.

Reusable matplotlib visualizations that take the report objects returned by
:func:`pyddrv.verify_stability` / :func:`pyddrv.verify_roa` directly, so you do
not have to reach into their internals. All functions accept an existing
``ax=`` (creating one if omitted) and **return the Axes**, so you can compose
and restyle freely.

matplotlib is an optional dependency (the ``[dev]`` extra); it is imported
lazily inside each function with a clear message if missing.

- :func:`plot_anytime` -- the certified-rate-vs-walltime curve of a
  ``StabilityReport`` (needs ``record_trace=True``).
- :func:`plot_stability_2d` -- the verified box ``Q_R`` over a 2-D phase
  portrait of the field.
- :func:`plot_roa_2d` -- the certified region-of-attraction cube union of a
  ``RoAReport`` (2-D), colored by cube size or split depth.
"""
from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "pyddrv.viz needs matplotlib; install it with "
            "`pip install matplotlib` (or the `[dev]` extra)."
        ) from exc


def _draw_cubes(ax, centers, halfs, values, *, cmap, clabel, edge=True,
                colorbar=True):
    """Draw a 2-D set of axis-aligned cubes as a filled PatchCollection colored
    by ``values`` (big cubes first, so small ones sit on top). Non-finite values
    render transparent. Returns the collection."""
    plt = _plt()
    from matplotlib.collections import PatchCollection

    centers = np.asarray(centers, dtype=float)
    halfs = np.asarray(halfs, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(-halfs)
    cs, hs, vv = centers[order], halfs[order], values[order]
    vv = np.where(np.isfinite(vv), vv, np.nan)
    patches = [plt.Rectangle((c[0] - h, c[1] - h), 2 * h, 2 * h)
               for c, h in zip(cs, hs)]
    kw = dict(cmap=cmap, lw=(0.15 if edge else 0.0))
    if edge:
        kw["edgecolor"] = (0, 0, 0, 0.35)
    col = PatchCollection(patches, **kw)
    col.set_array(vv)
    ax.add_collection(col)
    if colorbar:
        ax.figure.colorbar(col, ax=ax, label=clabel)
    return col


def _grid_arrays(report):
    """(centers-in-original-coords, halfs, alpha_lo) for a StabilityReport's
    certified grid, or None if the run produced no grid."""
    res = getattr(report, "result", None)
    if res is None or getattr(res, "centers", None) is None or len(res.centers) == 0:
        return None
    eq = np.asarray(report.equilibrium, dtype=float).reshape(-1)
    return (np.asarray(res.centers, dtype=float) + eq,
            np.asarray(res.halfs, dtype=float),
            np.asarray(res.alpha_lo, dtype=float))


# --------------------------------------------------------------------------- #
# stability
# --------------------------------------------------------------------------- #
def plot_anytime(report, ax=None, *, show_ceiling=True, label="certified rate"):
    r"""Plot the anytime certified rate ``alpha`` versus wall time.

    ``report`` is a :class:`~pyddrv.StabilityReport` produced with
    ``record_trace=True``. The curve is a sound lower bound at every point;
    the dashed line is the data-driven ceiling (best rate the data supports).
    """
    plt = _plt()
    if report.trace is None:
        raise ValueError("report has no trace; call verify_stability(..., "
                         "record_trace=True)")
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    tr = np.asarray(report.trace, dtype=float)          # (rounds, 3): t, alpha, n
    ax.plot(tr[:, 0], tr[:, 1], "o-", label=label)
    if show_ceiling and np.isfinite(report.alpha_upper):
        ax.axhline(report.alpha_upper, ls="--", color="gray",
                   label=f"ceiling {report.alpha_upper:.3f}")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel(r"certified $\alpha$")
    ax.legend()
    return ax


def plot_stability_2d(report, f=None, ax=None, *, density=1.2,
                      box_color="crimson", show_grid=False,
                      grid_color_by="outline", cmap="viridis"):
    r"""Draw the verified box ``Q_R`` (2-D) over a phase portrait of ``f``, and
    optionally the certified covering grid.

    ``report`` is a 2-D :class:`~pyddrv.StabilityReport`. If a batched field
    ``f: (N,2)->(N,2)`` is given, a streamplot of the flow is drawn underneath
    (NumPy or JAX fields both work). The box is centered at the report's
    equilibrium.

    ``show_grid=True`` overlays the actual cubes the certifier tiled ``Q_R``
    with (from ``report.result``) -- you can see the layered grid (coarse far
    from the equilibrium, finer near it) and where adaptive refinement
    concentrated. ``grid_color_by`` selects how:

    - ``"outline"`` -- cube edges only, drawn over the phase portrait (the flow
      stays visible; best for *seeing the tiling*);
    - ``"width"``  -- cubes filled by ``log10`` cube width (+ colorbar);
    - ``"alpha"``  -- cubes filled by their per-cube certified rate
      ``alpha_lo`` (+ colorbar); reveals which cubes limit the global rate.

    A filled mode (``"width"``/``"alpha"``) suppresses the streamplot, which it
    would otherwise hide.
    """
    plt = _plt()
    eq = np.asarray(report.equilibrium, dtype=float).reshape(-1)
    if eq.size != 2:
        raise ValueError("plot_stability_2d requires a 2-D system")
    R = float(report.R)
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    filled = show_grid and grid_color_by in ("width", "alpha")

    if f is not None and not filled:
        pad = 1.25 * R
        gx = np.linspace(eq[0] - pad, eq[0] + pad, 30)
        gy = np.linspace(eq[1] - pad, eq[1] + pad, 30)
        GX, GY = np.meshgrid(gx, gy)
        pts = np.stack([GX.reshape(-1), GY.reshape(-1)], axis=1)
        vel = np.asarray(f(pts), dtype=float)
        U = vel[:, 0].reshape(GX.shape)
        V = vel[:, 1].reshape(GX.shape)
        ax.streamplot(GX, GY, U, V, density=density, color="0.6",
                      linewidth=0.6, arrowsize=0.7)

    if show_grid:
        grid = _grid_arrays(report)
        if grid is None:
            raise ValueError("report has no certified grid to show "
                             "(report.result is empty)")
        centers, halfs, alpha_lo = grid
        if grid_color_by == "outline":
            from matplotlib.collections import PatchCollection
            patches = [plt.Rectangle((c[0] - h, c[1] - h), 2 * h, 2 * h)
                       for c, h in zip(centers, halfs)]
            ax.add_collection(PatchCollection(
                patches, facecolor="none", edgecolor=(0, 0, 0, 0.4), lw=0.2))
        elif grid_color_by == "width":
            _draw_cubes(ax, centers, halfs, np.log10(2 * halfs), cmap=cmap,
                        clabel=r"$\log_{10}$ cube width")
        elif grid_color_by == "alpha":
            _draw_cubes(ax, centers, halfs, alpha_lo, cmap=cmap,
                        clabel=r"per-cube certified $\alpha$")
        else:
            raise ValueError("grid_color_by must be 'outline', 'width', or "
                             f"'alpha', got {grid_color_by!r}")

    lo = eq - R
    ax.add_patch(plt.Rectangle((lo[0], lo[1]), 2 * R, 2 * R, fill=False,
                               color=box_color, lw=1.5,
                               label=rf"$Q_R$: certified $\alpha \geq "
                                     rf"{report.alpha:.3f}$"))
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return ax


# --------------------------------------------------------------------------- #
# region of attraction
# --------------------------------------------------------------------------- #
def plot_roa_2d(report, ax=None, *, color_by="width", cmap="viridis",
                show_box=True):
    r"""Plot a certified region-of-attraction cube union (2-D).

    ``report`` is a :class:`~pyddrv.RoAReport` from a 2-D ``verify_roa`` run.
    Each certified cube is drawn as a filled square; ``color_by`` selects the
    color scale:

    - ``"width"`` -- ``log10`` of the cube width (the layered grid reads
      directly: coarse cubes far out, geometrically finer near the boundary);
    - ``"depth"`` -- the split depth of each cube (integer refinement level).

    The excluded ball ``B_eps`` around the equilibrium shows as an empty hole.
    """
    plt = _plt()

    centers = np.asarray(report.centers, dtype=float)
    halfs = np.asarray(report.halfs, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("plot_roa_2d requires a 2-D region (centers shape "
                         f"(N,2)); got {centers.shape}. For d>2, slice first.")
    if len(centers) == 0:
        raise ValueError("report has no certified cubes to plot")

    if color_by == "width":
        vals = np.log10(2 * halfs)
        clabel = r"$\log_{10}$ cube width"
    elif color_by == "depth":
        vals = np.asarray(report.depths, dtype=float)
        clabel = "split depth"
    else:
        raise ValueError(f"color_by must be 'width' or 'depth', got {color_by!r}")

    ax = ax or plt.subplots(figsize=(6, 6))[1]
    _draw_cubes(ax, centers, halfs, vals, cmap=cmap, clabel=clabel, edge=False)

    eq = np.asarray(report.equilibrium, dtype=float).reshape(-1)
    R = float(report.R)
    if show_box:
        ax.set_xlim(eq[0] - R, eq[0] + R)
        ax.set_ylim(eq[1] - R, eq[1] + R)
    ax.set_aspect("equal")
    ax.set_title(rf"certified $\alpha={report.alpha:g}$ region "
                 rf"({report.volume:.3g} vol)")
    return ax
