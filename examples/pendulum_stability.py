"""Certify a decay rate for the damped pendulum — the minimal pipeline.

The field is plain NumPy (np.sin does not trace under jit), so this exercises
the automatic NumPy-kernel fallback: same certificate, no JAX required.

The pendulum's Jacobian [[0, 1], [-cos(theta), -1]] is NOT affine in the state,
so the box-corner estimate of the one-sided Lipschitz constant L is not
justified here (it happens to be correct only while the box stays inside
|theta| < pi, where mu(dJ) is monotonic; enlarge R past pi and corners would
miss the interior peak at theta=pi and under-estimate L -> an unsound
certificate). We therefore estimate L with the extreme-value (reverse-Weibull)
method L_method="evt", which samples the interior and returns a
high-probability upper bound -- the principled choice for any nonlinear field.

Writes examples/output/pendulum_stability.npz (raw results) and, if
matplotlib is available, a phase portrait with the verified box.
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
          f"eq-(38) check: {report.discretization_ok}")
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

    # (b) the same grid, cubes filled by their per-cube certified rate -- shows
    #     which cubes (near the boundary) limit the global guarantee.
    ax2 = plot_stability_2d(report, show_grid=True, grid_color_by="alpha")
    ax2.set_xlabel(r"$\theta$"); ax2.set_ylabel(r"$\omega$")
    ax2.set_title("Per-cube certified rate")
    ax2.figure.tight_layout()
    ax2.figure.savefig(os.path.join(OUT, "pendulum_grid.png"), dpi=150)
    print(f"  figure -> {OUT}/pendulum_grid.png")

    # (c) cubes filled by their actual WIDTH (log scale) -- the layered grid
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
