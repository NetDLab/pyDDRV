"""Region of attraction of the synchronized state of 3 Kuramoto oscillators
(reduced coordinates, d = 2), at target rate alpha = 1 — the paper's Fig. 3
setting, with the two-pass Trim protocol.

Writes examples/output/kuramoto_roa.npz (certified cubes) and a figure of the
certified union colored by cube size. Requires the [jax] extra.
"""
import os
import time

import numpy as np

from pyddrv import verify_roa
from pyddrv.systems.fields_jax import kuramoto_reduced_jax

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    k, n = 10.0, 3
    f, L_bound = kuramoto_reduced_jax(k=k, n=n)   # closed-form max-norm L
    R, d = np.pi, n - 1

    t0 = time.time()
    roa = verify_roa(f, R=R, d=d, alpha=1.0, L=L_bound, norm="inf",
                     tau=1.9, eps=np.pi / 81, max_refine=6,
                     trim=True, raster_n=729, max_seconds=120)
    wall = time.time() - t0
    print(roa.summary())
    print(f"  certified fraction of Q_R: {roa.volume / (2 * R) ** d:.1%}, "
          f"wall time {wall:.1f}s")

    np.savez(os.path.join(OUT, "kuramoto_roa.npz"),
             centers=np.asarray(roa.centers), halfs=np.asarray(roa.halfs),
             depths=np.asarray(roa.depths), alpha=roa.alpha, L=roa.L,
             R=R, volume=roa.volume)
    print(f"  raw results -> {OUT}/kuramoto_roa.npz")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(-np.asarray(roa.halfs))    # draw big cubes first
    cs, hs = np.asarray(roa.centers)[order], np.asarray(roa.halfs)[order]
    patches = [Rectangle((c[0] - h, c[1] - h), 2 * h, 2 * h)
               for c, h in zip(cs, hs)]
    col = PatchCollection(patches, cmap="viridis", lw=0)
    col.set_array(np.log10(2 * hs))
    ax.add_collection(col)
    fig.colorbar(col, ax=ax, label=r"$\log_{10}$ cube width")
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi_1$"); ax.set_ylabel(r"$\phi_2$")
    ax.set_title(f"Kuramoto (k={k:g}, n={n}): certified 1-RoA of sync")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kuramoto_roa.png"), dpi=150)
    print(f"  figure -> {OUT}/kuramoto_roa.png")


if __name__ == "__main__":
    main()
