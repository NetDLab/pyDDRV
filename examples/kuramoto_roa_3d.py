"""Region of attraction of synchrony for four Kuramoto oscillators, drawn as
cross-sections.

The settings follow kuramoto_roa.py. With four oscillators the reduced state
has dimension 3, so the region found by the first pass is drawn with
``pyddrv.viz.plot_roa_slice``, which shows the cubes that intersect an
axis-aligned plane. Three planes at increasing ``phi_3`` are drawn.

Only the first pass is run (``trim=False``) to keep the run short; use
``trim=True`` for a result meant to be reported. The area of a cross-section is
not the volume of the region, which is ``roa.volume``.

Writes examples/output/kuramoto3d_slices.{npz,png}. Requires the jax extra.
"""
import os
import time

import numpy as np

from pyddrv import verify_roa
from pyddrv.systems.fields_jax import kuramoto_reduced_jax

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    k, n = 10.0, 4
    f, L_bound = kuramoto_reduced_jax(k=k, n=n)   # closed-form max-norm L
    R, d = np.pi, n - 1                           # d = 3

    t0 = time.time()
    roa = verify_roa(f, R=R, d=d, alpha=1.0, L=L_bound, norm="inf",
                     tau=1.9, eps=np.pi / 27, max_refine=4, max_seconds=90)
    print(roa.summary())
    print(f"  certified fraction of Q_R: {roa.volume / (2 * R) ** d:.1%}, "
          f"wall time {time.time() - t0:.0f}s")

    np.savez(os.path.join(OUT, "kuramoto3d_slices.npz"),
             centers=np.asarray(roa.centers, dtype=np.float32),
             halfs=np.asarray(roa.halfs, dtype=np.float32),
             depths=np.asarray(roa.depths), alpha=roa.alpha, L=roa.L,
             R=R, volume=roa.volume)
    print(f"  raw results -> {OUT}/kuramoto3d_slices.npz")

    try:
        import matplotlib.pyplot as plt
        from pyddrv.viz import plot_roa_slice
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, phi3 in zip(axes, [0.0, 1.2, 2.4]):
        plot_roa_slice(roa, dims=(0, 1), at=[0.0, 0.0, phi3], ax=ax)
        ax.set_xlabel(r"$\phi_1$"); ax.set_ylabel(r"$\phi_2$")
        ax.set_title(rf"slice at $\phi_3 = {phi3:g}$")
    fig.suptitle(f"Kuramoto n={n} (d=3): 2-D slices of the certified 1-RoA",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kuramoto3d_slices.png"), dpi=130,
                bbox_inches="tight")
    print(f"  figure -> {OUT}/kuramoto3d_slices.png")


if __name__ == "__main__":
    main()
