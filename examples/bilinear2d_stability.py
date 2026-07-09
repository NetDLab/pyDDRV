"""The paper's 2D bilinear benchmark on the JAX fast path, with the anytime
trace: the certified rate is a sound lower bound at EVERY point of the curve;
more compute only tightens it toward the data-driven ceiling.

Writes examples/output/bilinear2d_stability.npz and the anytime-curve figure.
Requires the [jax] extra.
"""
import os
import time

import numpy as np

from pyddrv import verify_stability
from pyddrv.systems.fields_jax import bilinear_2d_jax

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    f, jac = bilinear_2d_jax(eta=0.3, seed=0)     # paper eq. (39)
    R, d, tau = 0.7, 2, 5.0

    t0 = time.time()
    report = verify_stability(f, R=R, d=d, jac=jac, tau=tau, eps=0.01,
                              delta=0.05, max_refine=16, max_seconds=60,
                              record_trace=True)
    wall = time.time() - t0
    print(report.summary())
    print(f"  wall time {wall:.1f}s (backend={report.backend})")

    trace = np.array(report.trace, dtype=float)   # (rounds, 3): t, alpha, boxes
    np.savez(os.path.join(OUT, "bilinear2d_stability.npz"),
             alpha=report.alpha, alpha_upper=report.alpha_upper,
             L=report.L, R=R, tau=tau, trace=trace)
    print(f"  raw results -> {OUT}/bilinear2d_stability.npz")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from pyddrv.viz import plot_anytime
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    ax = plot_anytime(report, label="certified rate (anytime)")
    ax.set_title("Bilinear 2D: anytime certified decay rate")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUT, "bilinear2d_stability.png"), dpi=150)
    print(f"  figure -> {OUT}/bilinear2d_stability.png")


if __name__ == "__main__":
    main()
