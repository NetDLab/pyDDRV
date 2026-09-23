"""Certified decay rate for a damped pendulum.

The field is written with NumPy (np.sin), so the NumPy kernel is used and JAX is
not needed.

The Jacobian of the pendulum, [[0, 1], [-cos(theta), -1]], is not affine in the
state, so evaluating its matrix measure at the corners of the box does not
bound L in general: for boxes that contain theta = pi the maximum is inside
the box. L is therefore estimated with the extreme-value method
(L_method="evt"), which samples the inside of the box and gives a bound that
holds with probability rho.

Writes examples/output/pendulum_stability.npz and, if matplotlib is installed,
figures of the certified box and the cubes used by the certificate.
"""
import os

import numpy as np

from pyddrv import verify_stability
from pyddrv.systems import damped_pendulum

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    f = damped_pendulum(damping=1.0)
    R, d, tau = 0.8, 2, 6.0

    report = verify_stability(f, R=R, d=d, tau=tau, delta=0.2, max_refine=6,
                              L_method="evt", rho=0.95, record_trace=True)
    print(report.summary())
    print(f"  L={report.L:.3f} over Q_{report.lipschitz.R_bar:.3f} "
          f"(worst excursion {report.lipschitz.R_max:.3f}), "
          f"discretization check: {report.discretization_ok}")
    print(f"  L via EVT: {report.lipschitz.evt.summary()}")

    np.savez(os.path.join(OUT, "pendulum_stability.npz"),
             alpha=report.alpha, alpha_upper=report.alpha_upper,
             L=report.L, R=R, tau=tau, eps=report.eps,
             trace=np.array(report.trace, dtype=float),
             centers=report.result.centers, halfs=report.result.halfs,
             alpha_lo=report.result.alpha_lo)
    print(f"  raw results -> {OUT}/pendulum_stability.npz")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from pyddrv.viz import plot_stability_2d
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    # (a) phase portrait + verified box, with the certified covering grid
    #     overlaid as cube outlines so the layered/refined tiling is visible.
    ax = plot_stability_2d(report, f=f, show_grid=True, grid_color_by="outline")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\omega$")
    ax.set_title("Damped pendulum: certified decay on $Q_R$ (with grid)")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUT, "pendulum_stability.png"), dpi=150)
    print(f"  figure -> {OUT}/pendulum_stability.png")

    # (b) the same grid, cubes filled by the rate each one certifies; shows
    #     which cubes (near the boundary) limit the global guarantee.
    ax2 = plot_stability_2d(report, show_grid=True, grid_color_by="alpha")
    ax2.set_xlabel(r"$\theta$"); ax2.set_ylabel(r"$\omega$")
    ax2.set_title("Per-cube certified rate")
    ax2.figure.tight_layout()
    ax2.figure.savefig(os.path.join(OUT, "pendulum_grid.png"), dpi=150)
    print(f"  figure -> {OUT}/pendulum_grid.png")

    # (c) cubes filled by their width (log scale): the layered grid
    #     (coarse outside, finer near the origin) and adaptive refinement,
    #     sized directly.
    ax3 = plot_stability_2d(report, show_grid=True, grid_color_by="width")
    ax3.set_xlabel(r"$\theta$"); ax3.set_ylabel(r"$\omega$")
    ax3.set_title("Cube widths (log scale)")
    ax3.figure.tight_layout()
    ax3.figure.savefig(os.path.join(OUT, "pendulum_width.png"), dpi=150)
    print(f"  figure -> {OUT}/pendulum_width.png")


if __name__ == "__main__":
    main()
