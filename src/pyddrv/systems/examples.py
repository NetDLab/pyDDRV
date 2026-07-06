"""Toy dynamical systems and a simple RK4 sampler.

These exist to exercise and demo the certificates. pyddrv itself never needs a
model -- it only consumes :class:`TrajectorySet` data -- but having known-stable
and known-unstable systems makes tests and examples concrete. Real use replaces
``simulate`` with simulator output or logged experimental trajectories.
"""
from __future__ import annotations

import numpy as np

from ..data.trajectory import TrajectorySet


def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(f, x0_batch, dt, horizon) -> np.ndarray:
    """Roll out ``f`` from each initial condition. Returns (n_traj, horizon, dim)."""
    x = np.asarray(x0_batch, dtype=float)
    states = np.empty((x.shape[0], horizon, x.shape[1]))
    states[:, 0] = x
    for k in range(1, horizon):
        x = rk4_step(f, x, dt)
        states[:, k] = x
    return states


def linear(A):
    """Vector field x_dot = A x for a batch of states (n, dim)."""
    A = np.asarray(A, dtype=float)

    def f(x):
        return x @ A.T

    return f


def damped_pendulum(g_over_l=1.0, damping=0.5):
    """Pendulum: theta_dot = omega, omega_dot = -(g/l) sin(theta) - b omega."""

    def f(x):
        theta, omega = x[:, 0], x[:, 1]
        dtheta = omega
        domega = -g_over_l * np.sin(theta) - damping * omega
        return np.stack([dtheta, domega], axis=1)

    return f


def bilinear_2d(eta=0.3, B1=None, seed=0):
    """Paper eq. (39): 2D bilinear family with nonlinearity strength ``eta``.

    ``xdot = A x + B1 [x1^2, x1 x2, x2^2]^T`` with ``A = [[0, 2], [-1, -1]]``
    (stable spiral) and ``B1 in R^{2x3}`` drawn i.i.d. ``N(0, eta)`` where ``eta``
    is the standard deviation (so ``scale=eta``). Returns
    ``(f, jac)`` where ``f`` is the batched field and ``jac(x)`` is the analytic
    Jacobian -- affine in ``x``, so its matrix measure is maximized at box
    corners (used to compute the one-sided Lipschitz constant exactly).
    """
    A = np.array([[0.0, 2.0], [-1.0, -1.0]])
    if B1 is None:
        rng = np.random.default_rng(seed)
        B1 = rng.normal(0.0, eta, size=(2, 3))
    B1 = np.asarray(B1, dtype=float)

    def phi(x):  # monomials [x1^2, x1 x2, x2^2], batched
        x1, x2 = x[:, 0], x[:, 1]
        return np.stack([x1 * x1, x1 * x2, x2 * x2], axis=1)

    def f(x):
        return x @ A.T + phi(x) @ B1.T

    def jac(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x1, x2 = x[0], x[1]
        dphi = np.array([[2 * x1, 0.0], [x2, x1], [0.0, 2 * x2]])  # (3,2)
        return A + B1 @ dphi

    return f, jac


def bilinear_3d(eta=0.3, B2=None, seed=0):
    """Paper eq. (40): 3D bilinear family, returns ``(f, jac)`` (NumPy).

    ``A = [[-1,0,0],[0.5,-1,0],[0.5,0.5,-1]]`` plus ``B2 @ m(x)`` over the 6
    degree-2 monomials ``m = [x1^2, x1 x2, x1 x3, x2^2, x2 x3, x3^2]``. ``B2`` is
    ``3x6`` drawn i.i.d. ``N(0, eta)`` (``eta`` = standard deviation). The
    Jacobian is affine in ``x`` (exact corner max for the one-sided Lipschitz).
    """
    A = np.array([[-1.0, 0.0, 0.0],
                  [0.5, -1.0, 0.0],
                  [0.5, 0.5, -1.0]])
    if B2 is None:
        rng = np.random.default_rng(seed)
        B2 = rng.normal(0.0, eta, size=(3, 6))
    B2 = np.asarray(B2, dtype=float)

    def monomials(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return np.stack([x1 * x1, x1 * x2, x1 * x3,
                         x2 * x2, x2 * x3, x3 * x3], axis=1)

    def f(x):
        return x @ A.T + monomials(x) @ B2.T

    def jac(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x1, x2, x3 = x
        dm = np.array([[2 * x1, 0.0, 0.0],
                       [x2, x1, 0.0],
                       [x3, 0.0, x1],
                       [0.0, 2 * x2, 0.0],
                       [0.0, x3, x2],
                       [0.0, 0.0, 2 * x3]])     # (6,3)
        return A + B2 @ dm

    return f, jac


def random_initial_conditions(n, dim, radius=1.0, seed=0) -> np.ndarray:
    """Uniformly sample ``n`` initial states in a ball of given ``radius``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    r = radius * rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / dim)
    return x * r


def sample_pendulum(n_traj=200, horizon=400, dt=0.02, radius=2.0, damping=0.5,
                    seed=0) -> TrajectorySet:
    """A ready-made stable damped-pendulum dataset about the origin."""
    f = damped_pendulum(damping=damping)
    x0 = random_initial_conditions(n_traj, 2, radius=radius, seed=seed)
    states = simulate(f, x0, dt, horizon)
    return TrajectorySet.from_arrays(states, dt=dt)


def sample_linear(A, n_traj=200, horizon=300, dt=0.02, radius=1.0,
                  seed=0) -> TrajectorySet:
    """A dataset from the linear system x_dot = A x about the origin."""
    f = linear(A)
    dim = np.asarray(A).shape[0]
    x0 = random_initial_conditions(n_traj, dim, radius=radius, seed=seed)
    states = simulate(f, x0, dt, horizon)
    return TrajectorySet.from_arrays(states, dt=dt)


def kuramoto_reduced(k=1.0, n=3):
    r"""Kuramoto oscillators with uniform coupling (paper eq. 41), in reduced
    coordinates ``phi_i = theta_i - theta_n`` (``i = 1..n-1``, ``d = n-1``),
    which removes the rotational symmetry; the synchronized state is ``phi=0``.

    ``theta_dot_i = (k/n) sum_j sin(theta_j - theta_i)``. Closed-form one-sided
    Lipschitz bound (max norm): ``L <= 2 k (n-1) / n`` (each Jacobian row:
    ``|J_ii| + sum |J_ij| <= (k/n)(n-1) + (k/n)(n-1)``), valid globally -- no
    numerical estimation required.

    Returns ``(f, L_bound)`` with ``f`` batched ``(N, n-1) -> (N, n-1)``.
    """
    k, n = float(k), int(n)

    def f(x):
        x = np.asarray(x, dtype=float)
        phi = np.concatenate([x, np.zeros((len(x), 1))], axis=1)      # (N, n)
        diff = phi[:, None, :] - phi[:, :, None]        # diff[b,i,j]=phi_j-phi_i
        dtheta = (k / n) * np.sin(diff).sum(axis=2)                   # (N, n)
        return dtheta[:, :n - 1] - dtheta[:, n - 1:n]

    return f, 2.0 * k * (n - 1) / n
