"""Trajectory data container.

pyddrv is data-driven: certification consumes *sampled trajectories*, whether
from simulation or from experiment. A :class:`TrajectorySet` is a dense batch of
equal-length, uniformly time-sampled trajectories -- the shape the vectorized
recurrence kernels expect. Ragged/variable-length data should be windowed or
padded into this form by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySet:
    """A batch of trajectories sampled on a common uniform time grid.

    Attributes
    ----------
    states : ndarray, shape (n_traj, horizon, dim)
        Sampled state along each trajectory.
    dt : float
        Sampling interval (seconds) between consecutive time indices.
    equilibrium : ndarray, shape (dim,)
        The point about which stability is assessed (default: origin).
    """

    states: np.ndarray
    dt: float
    equilibrium: np.ndarray

    def __post_init__(self):
        s = np.asarray(self.states, dtype=float)
        if s.ndim != 3:
            raise ValueError(
                f"states must be (n_traj, horizon, dim); got shape {s.shape}"
            )
        eq = np.asarray(self.equilibrium, dtype=float)
        if eq.shape != (s.shape[-1],):
            raise ValueError(
                f"equilibrium must have shape ({s.shape[-1]},); got {eq.shape}"
            )
        object.__setattr__(self, "states", s)
        object.__setattr__(self, "equilibrium", eq)
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def n_traj(self) -> int:
        return self.states.shape[0]

    @property
    def horizon(self) -> int:
        return self.states.shape[1]

    @property
    def dim(self) -> int:
        return self.states.shape[2]

    def centered(self) -> np.ndarray:
        """States shifted so the equilibrium is at the origin."""
        return self.states - self.equilibrium

    def window_steps(self, tau: float) -> int:
        """Number of time indices in a recurrence horizon of length ``tau``."""
        w = int(round(tau / self.dt))
        if w < 1:
            raise ValueError(
                f"tau={tau} is shorter than dt={self.dt}; increase tau."
            )
        if w >= self.horizon:
            raise ValueError(
                f"tau={tau} (={w} steps) must be shorter than the trajectory "
                f"horizon ({self.horizon} steps)."
            )
        return w

    @classmethod
    def from_arrays(cls, states, dt, equilibrium=None) -> "TrajectorySet":
        states = np.asarray(states, dtype=float)
        if equilibrium is None:
            equilibrium = np.zeros(states.shape[-1])
        return cls(states=states, dt=float(dt), equilibrium=equilibrium)
