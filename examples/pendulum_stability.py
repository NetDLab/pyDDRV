"""Certify a decay rate for the damped pendulum — the minimal pipeline.

The field is plain NumPy (np.sin does not trace under jit), so this exercises
the automatic NumPy-kernel fallback: same certificate, no JAX required.

Writes examples/output/pendulum_stability.npz (raw results) and, if
matplotlib is available, a phase portrait with the verified box.
"""
import os

import numpy as np

from pyddrv import verify_stability
from pyddrv.systems import damped_pendulum, simulate

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    f = damped_pendulum(damping=1.0)
    R, d, tau = 0.8, 2, 6.0

    report = verify_stability(f, R=R, d=d, tau=tau, delta=0.2, max_refine=6,
                              record_trace=True)
    print(report.summary())
    print(f"  L={report.L:.3f} over Q_{report.lipschitz.R_bar:.3f} "
          f"(worst excursion {report.lipschitz.R_max:.3f}), "
          f"eq-(38) check: {report.discretization_ok}")

    np.savez(os.path.join(OUT, "pendulum_stability.npz"),
             alpha=report.alpha, alpha_upper=report.alpha_upper,
             L=report.L, R=R, tau=tau, eps=report.eps,
             trace=np.array(report.trace, dtype=float),
             centers=report.result.centers, halfs=report.result.halfs,
             alpha_lo=report.result.alpha_lo)
    print(f"  raw results -> {OUT}/pendulum_stability.npz")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    x0 = np.stack(np.meshgrid(np.linspace(-R, R, 7),
                              np.linspace(-R, R, 7)), -1).reshape(-1, 2)
    ys = simulate(f, x0, dt=0.02, horizon=400)
    for tr in ys:
        ax.plot(tr[:, 0], tr[:, 1], lw=0.5, color="steelblue", alpha=0.6)
    ax.add_patch(plt.Rectangle((-R, -R), 2 * R, 2 * R, fill=False,
                               color="crimson", lw=1.5,
                               label=f"$Q_R$: certified $\\alpha \\geq "
                                     f"{report.alpha:.3f}$"))
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\omega$")
    ax.legend(loc="upper right"); ax.set_aspect("equal")
    ax.set_title("Damped pendulum: certified exponential decay on $Q_R$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pendulum_stability.png"), dpi=150)
    print(f"  figure -> {OUT}/pendulum_stability.png")


if __name__ == "__main__":
    main()
