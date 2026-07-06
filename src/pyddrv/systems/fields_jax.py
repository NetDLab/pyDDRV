"""JAX (jit-compatible) variants of the example fields.

The fused certifier runs fastest when the batched field traces under
``jax.jit``. The plain-NumPy fields in :mod:`pyddrv.systems.examples` work on
the NumPy kernel; these mirrors unlock the JAX fast path. Requires the
``jax`` extra.
"""
from __future__ import annotations

import numpy as np


def linear_jax(A):
    """``x_dot = A x`` as a jit-compatible batched field."""
    import jax.numpy as jnp
    Aj = jnp.asarray(np.asarray(A, dtype=float).T)

    def f(x):
        return x @ Aj

    return f


def bilinear_2d_jax(eta=0.3, B1=None, seed=0):
    """JAX mirror of :func:`pyddrv.systems.bilinear_2d` (same B1 draw for the
    same seed). Returns ``(f_jax, jac)`` with ``jac`` the NumPy analytic
    Jacobian (affine in x; exact corner max for the Lipschitz constant)."""
    import jax.numpy as jnp

    from .examples import bilinear_2d

    _, jac = bilinear_2d(eta=eta, B1=B1, seed=seed)
    A = np.array([[0.0, 2.0], [-1.0, -1.0]])
    if B1 is None:
        rng = np.random.default_rng(seed)
        B1 = rng.normal(0.0, eta, size=(2, 3))
    Aj = jnp.asarray(A.T)
    Bj = jnp.asarray(np.asarray(B1, dtype=float).T)

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        phi = jnp.stack([x1 * x1, x1 * x2, x2 * x2], axis=1)
        return x @ Aj + phi @ Bj

    return f, jac


def kuramoto_reduced_jax(k=1.0, n=3):
    """JAX mirror of :func:`pyddrv.systems.kuramoto_reduced` in the O(n) form
    ``sum_j sin(th_j - th_i) = cos(th_i) S - sin(th_i) C``. Returns
    ``(f_jax, L_bound)`` with the global closed-form max-norm one-sided
    Lipschitz bound ``2 k (n-1) / n``."""
    import jax.numpy as jnp
    kk, n = float(k), int(n)

    def f(x):
        phi = jnp.concatenate([x, jnp.zeros((x.shape[0], 1))], axis=1)
        s, co = jnp.sin(phi), jnp.cos(phi)
        S = s.sum(axis=1, keepdims=True)
        C = co.sum(axis=1, keepdims=True)
        dtheta = (kk / n) * (co * S - s * C)
        return dtheta[:, : n - 1] - dtheta[:, n - 1 : n]

    return f, 2.0 * kk * (n - 1) / n
