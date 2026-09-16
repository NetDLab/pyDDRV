r"""Torch (MPS/CUDA) building blocks for the trajectory-local contraction
bound.  MPS supports neither float64 nor ``eigvalsh``, so the matrix measure
``mu_2 = lambda_max(sym J)`` uses closed-form 2x2 / 3x3 symmetric-eigenvalue
formulas (exact, elementwise), and ``A(t)`` accumulates in float32 clipped at
``L*t`` (beyond which the global Gronwall bound wins, so the clip is free).

Mirrors :mod:`pyddrv.verification.contraction` (the NumPy reference) so the two
can be cross-checked. Used by :func:`pyddrv.verification.find_alpha_roa_fused`
via ``local_jac`` / ``local_M`` with ``backend="torch"``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def mu2_sym_closed(S):
    r"""``lambda_max`` of a batch of symmetric matrices ``S (...,d,d)`` for
    ``d in {2,3}`` by closed form (no eigvalsh; MPS-safe).  Returns ``(...)``."""
    import torch
    d = S.shape[-1]
    if d == 1:
        return S[..., 0, 0]
    if d == 2:
        a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
        m = 0.5 * (a + c)
        return m + torch.sqrt(torch.clamp((0.5 * (a - c)) ** 2 + b * b, min=0.0))
    if d == 3:
        # Smith's trigonometric method for the largest eigenvalue of a 3x3 sym.
        s = S
        q = (s[..., 0, 0] + s[..., 1, 1] + s[..., 2, 2]) / 3.0
        p1 = s[..., 0, 1] ** 2 + s[..., 0, 2] ** 2 + s[..., 1, 2] ** 2
        p2 = ((s[..., 0, 0] - q) ** 2 + (s[..., 1, 1] - q) ** 2
              + (s[..., 2, 2] - q) ** 2 + 2.0 * p1)
        p = torch.sqrt(torch.clamp(p2 / 6.0, min=1e-30))
        I = torch.eye(3, dtype=s.dtype, device=s.device)
        B = (s - q[..., None, None] * I) / p[..., None, None]
        # det(B)/2
        detB = (B[..., 0, 0] * (B[..., 1, 1] * B[..., 2, 2] - B[..., 1, 2] * B[..., 2, 1])
                - B[..., 0, 1] * (B[..., 1, 0] * B[..., 2, 2] - B[..., 1, 2] * B[..., 2, 0])
                + B[..., 0, 2] * (B[..., 1, 0] * B[..., 2, 1] - B[..., 1, 1] * B[..., 2, 0]))
        r = torch.clamp(detB / 2.0, -1.0, 1.0)
        phi = torch.acos(r) / 3.0
        eig_max = q + 2.0 * p * torch.cos(phi)
        # degenerate p1==0 -> diagonal, lambda_max = max diagonal
        diagmax = torch.maximum(torch.maximum(s[..., 0, 0], s[..., 1, 1]), s[..., 2, 2])
        return torch.where(p1 > 1e-20, eig_max, diagmax)
    raise ValueError(f"closed-form mu2 only for d<=3, got d={d}")


def mu2_gershgorin(S):
    r"""Upper bound on ``lambda_max`` for any ``d`` (row-sum / Gershgorin):
    ``max_i ( S_ii + sum_{j!=i} |S_ij| )``.  Sound (never underestimates the
    measure), used for ``d >= 4`` where no closed form is available."""
    import torch
    diag = torch.diagonal(S, dim1=-2, dim2=-1)
    absrow = S.abs().sum(dim=-1) - diag.abs()
    return (diag + absrow).amax(dim=-1)


def mu2_torch(J, jacobi_sweeps=4):
    """``mu_2(J) = lambda_max((J+J^T)/2)``: exact closed form for d<=3, and a
    tight SOUND upper bound via ``jacobi_sweeps`` Jacobi sweeps + Gershgorin for
    d>=4 (Gershgorin alone overshoots by ~2-3 on Kuramoto and cripples the
    local bound; Jacobi drives the overshoot to ~0)."""
    S = 0.5 * (J + J.transpose(-1, -2))
    if J.shape[-1] <= 3:
        return mu2_sym_closed(S)
    return mu2_jacobi(S, sweeps=jacobi_sweeps)


def kuramoto_jac_torch(k, n, device=None):
    """Batched analytic Jacobian of the reduced Kuramoto field on device.

    ``jac(X:(N,n-1)) -> (N,n-1,n-1)`` (mirrors
    ``contraction.kuramoto_jac_batched``)."""
    import torch
    from .fused_torch import torch_device
    dev = device or torch_device()
    k, n = float(k), int(n)
    idx = torch.arange(n, device=dev)

    def jac(X):
        N = X.shape[0]
        th = torch.cat([X, torch.zeros(N, 1, dtype=X.dtype, device=X.device)], 1)
        C = torch.cos(th[:, None, :] - th[:, :, None])       # [b,m,l]=cos(th_l-th_m)
        D = (k / n) * C
        offdiag = (k / n) * (C.sum(dim=2) - 1.0)             # sum_j cos - cos0
        D[:, idx, idx] = -offdiag
        return D[:, :n - 1, :n - 1] - D[:, n - 1:n, :n - 1]

    return jac


def make_local_roa_kernel(f_torch, jac_torch, norm_key, M, device=None):
    r"""Fused Theorem-8 RoA kernel with the TRAJECTORY-LOCAL bound.

    Like :func:`fused_torch.make_torch_roa_kernel` but the inflation term is
    ``min(r*exp(L*t), r*exp(A)/(1-r*(M/2)*I))`` with ``A(t)=int mu_2(J)`` and
    ``I(t)=int exp(A)`` accumulated along the centre trajectory (float32, A
    clipped at ``L*t``).  Returns ``kernel(centers,halfs,tau,n_steps,L,c) ->
    (alpha, alpha_lookahead)`` as numpy, matching the reference alpha_max_local.
    """
    import torch
    from .fused_torch import torch_device
    dev = device or torch_device()
    b_rem = 0.5 * float(M)                          # remainder coefficient M/2

    if norm_key == "2":
        def vnorm(x):
            return torch.linalg.vector_norm(x, dim=-1)
    elif norm_key == "inf":
        def vnorm(x):
            return x.abs().amax(dim=-1)
    else:
        def vnorm(x):
            return x.abs().sum(dim=-1)

    @torch.no_grad()
    def kernel(centers, halfs, tau, n_steps, L, c):
        x = torch.as_tensor(np.ascontiguousarray(centers), dtype=torch.float32,
                            device=dev)
        h = torch.as_tensor(np.ascontiguousarray(halfs), dtype=torch.float32,
                            device=dev)
        rads = torch.stack([c * h, c * h / 3.0])           # (2,N): r and r/3
        dt = tau / n_steps
        Vx = vnorm(x)
        marg = Vx.unsqueeze(0) - rads                      # (2,N)
        neg = torch.full_like(marg, -float("inf"))
        acc = neg.clone()
        A = torch.zeros_like(Vx)                           # A(0)=0
        I = torch.zeros_like(Vx)                           # I(0)=0
        a_prev = mu2_torch(jac_torch(x))                   # a(0)
        for k in range(1, n_steps + 1):
            k1 = f_torch(x); k2 = f_torch(x + 0.5 * dt * k1)
            k3 = f_torch(x + 0.5 * dt * k2); k4 = f_torch(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tk = dt * k
            a_k = mu2_torch(jac_torch(x))
            A = torch.minimum(A + torch.maximum(a_prev, a_k) * dt,
                              torch.as_tensor(L * tk, dtype=A.dtype, device=dev))
            I = I + torch.exp(A) * dt
            a_prev = a_k
            Vphi = vnorm(x).unsqueeze(0)                   # (1,N)
            old = rads * float(np.exp(L * tk))
            guard = (rads * b_rem * I.unsqueeze(0)) < 1.0
            denom_g = 1.0 - rads * b_rem * I.unsqueeze(0)
            rho = torch.where(guard, rads * torch.exp(A).unsqueeze(0)
                              / denom_g, torch.full_like(old, float("inf")))
            bnd = torch.minimum(old, rho)
            denom = Vphi + bnd
            valid = (marg > 0) & (denom > 0)
            at = torch.where(valid, (marg / denom).clamp_min(1e-30).log() / tk, neg)
            torch.maximum(acc, at, out=acc)
        out = acc.cpu().numpy().astype(float)
        return out[0], out[1]

    return kernel


def mu2_jacobi(S, sweeps=6):
    r"""Tight SOUND upper bound on ``lambda_max`` of symmetric ``S (...,d,d)``
    for any ``d``: a few cyclic-Jacobi rotation sweeps nearly diagonalize ``S``
    (an orthogonal similarity, so eigenvalues are preserved), then Gershgorin
    on the transformed matrix.  Gershgorin is always an upper bound; after the
    off-diagonal is driven ~0 it is tight.  All elementwise/gather ops -> MPS.
    """
    import torch
    S = S.clone()
    d = S.shape[-1]
    for _ in range(sweeps):
        for p in range(d - 1):
            for q in range(p + 1, d):
                app = S[..., p, p]; aqq = S[..., q, q]; apq = S[..., p, q]
                phi = 0.5 * torch.atan2(2.0 * apq, app - aqq)
                c = torch.cos(phi)[..., None]; s = -torch.sin(phi)[..., None]
                Cp = S[..., :, p].clone(); Cq = S[..., :, q].clone()
                S[..., :, p] = c * Cp - s * Cq          # column update (N,d)
                S[..., :, q] = s * Cp + c * Cq
                Rp = S[..., p, :].clone(); Rq = S[..., q, :].clone()
                S[..., p, :] = c * Rp - s * Rq          # row update (N,d)
                S[..., q, :] = s * Rp + c * Rq
    return mu2_gershgorin(S)
