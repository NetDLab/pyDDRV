r"""Torch (Apple-GPU / CUDA) backend for the fused Theorem-8 RoA kernel.

Mirrors ``_make_jax_roa_kernel`` (32a-only path): batched RK4 rollout of the
cube centres, running-max alpha at radius r = c*h and the one-split lookahead
radius r/3, single pass, float32. Device auto-selects mps > cuda > cpu.

The OpenMP duplicate-runtime guard is required when torch and conda's libomp
coexist; safe on arm64 (both LLVM). Set before importing torch.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def _maybe_compile(fn):
    """torch.compile with eager fallback -- measured 5.6x on MPS (2026-07-06,
    verdict-identical); falls back silently if the backend can't compile."""
    import torch
    try:
        return torch.compile(fn)
    except Exception:
        return fn


def torch_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_torch_roa_kernel(f_torch, norm_key: str, device=None):
    """Return kernel(centers, halfs, tau, n_steps, L, c) -> (a, a3) numpy.

    ``f_torch``: batched field (N, d) tensor -> (N, d) tensor (same device).
    """
    import torch

    dev = device or torch_device()

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
    def kernel(centers, halfs, tau, n_steps, L, c, dmap=None, lo=None,
               cellw=None):
        # optional region path (Trim / (32b) vs the EXACT union): dmap is the
        # rasterized inner inf-distance map, flattened row-major on the device
        x = torch.as_tensor(np.ascontiguousarray(centers),
                            dtype=torch.float32, device=dev)
        h = torch.as_tensor(np.ascontiguousarray(halfs),
                            dtype=torch.float32, device=dev)
        use_region = dmap is not None
        if use_region:
            n_cells = dmap.shape[0]
            dflat = torch.as_tensor(np.ascontiguousarray(dmap.reshape(-1)),
                                    dtype=torch.float32, device=dev)
            _st.update(lo=float(lo), cellw=float(cellw), n_cells=int(n_cells))
        rads = torch.stack([c * h, c * h / 3.0])        # (2, N): r and r/3
        dt = tau / n_steps
        Vx = vnorm(x)
        marg = Vx.unsqueeze(0) - rads                   # (2, N)
        acc = torch.full_like(marg, -float("inf"))
        for k in range(1, n_steps + 1):
            x, acc = _step(x, acc, marg, rads, dt, dt * k,
                           float(np.exp(L * dt * k)),
                           dflat if use_region else None)
        out = acc.cpu().numpy().astype(float)
        return out[0], out[1]

    def _step_impl(x, acc, marg, rads, dt, tk, eLt, dflat):
        k1 = f_torch(x)
        k2 = f_torch(x + (0.5 * dt) * k1)
        k3 = f_torch(x + (0.5 * dt) * k2)
        k4 = f_torch(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Vphi = vnorm(x).unsqueeze(0)                    # (1, N)
        denom = Vphi + rads * eLt
        valid = (marg > 0) & (denom > 0)
        if dflat is not None:
            idx = torch.floor((x - _st["lo"]) / _st["cellw"]).long()
            inb = ((idx >= 0) & (idx < _st["n_cells"])).all(dim=-1)
            idx = idx.clamp(0, _st["n_cells"] - 1)
            flat = idx[:, 0]
            for j in range(1, x.shape[1]):
                flat = flat * _st["n_cells"] + idx[:, j]
            rdist = torch.where(inb, dflat[flat], torch.zeros_like(Vphi[0]))
            valid = valid & (rdist.unsqueeze(0) >= rads * eLt)
        neg = torch.full_like(marg, -float("inf"))
        at = torch.where(valid,
                         (marg / denom).clamp_min(1e-30).log() / tk, neg)
        return x, torch.maximum(acc, at)

    _st = {}
    _step = _maybe_compile(_step_impl)

    return kernel


def make_torch_span_kernel(f_torch, norm_key: str, enforce_S: bool,
                           device=None):
    """Algorithm-1 span kernel on GPU: (a_lo at r=c*h, a_up at r=0, x_end)
    over the time span (t_lo, t_lo + n_steps*dt], with Vx = ||x(0)|| of the
    ORIGINAL centres (span continuation semantics, mirrors _make_jax_kernel)."""
    import torch

    dev = device or torch_device()

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
    def kernel(x_init, Vx, halfs, t_lo, dt, n_steps, L, R, c):
        x = torch.as_tensor(np.ascontiguousarray(x_init),
                            dtype=torch.float32, device=dev)
        Vx_t = torch.as_tensor(np.ascontiguousarray(Vx),
                               dtype=torch.float32, device=dev)
        h = torch.as_tensor(np.ascontiguousarray(halfs),
                            dtype=torch.float32, device=dev)
        rads = torch.stack([c * h, torch.zeros_like(h)])      # (2, N): lo, up
        marg = Vx_t.unsqueeze(0) - rads                       # (2, N)
        acc = torch.full_like(marg, -float("inf"))
        for k in range(1, n_steps + 1):
            tk = t_lo + dt * k
            x, acc = _step(x, acc, marg, rads, dt, tk,
                           float(np.exp(L * tk)), R)
        out = acc.cpu().numpy().astype(float)
        return out[0], out[1], x.cpu().numpy().astype(float)

    def _step_impl(x, acc, marg, rads, dt, tk, eLt, R):
        k1 = f_torch(x)
        k2 = f_torch(x + (0.5 * dt) * k1)
        k3 = f_torch(x + (0.5 * dt) * k2)
        k4 = f_torch(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Vphi = vnorm(x).unsqueeze(0)                          # (1, N)
        denom = Vphi + rads * eLt
        valid = (marg > 0) & (denom > 0)
        if enforce_S:
            phi_inf = x.abs().amax(dim=-1).unsqueeze(0)       # (1, N)
            valid = valid & (phi_inf <= R - rads * eLt)       # (32b), same t
        neg = torch.full_like(marg, -float("inf"))
        at = torch.where(valid,
                         (marg / denom).clamp_min(1e-30).log() / tk, neg)
        return x, torch.maximum(acc, at)

    _step = _maybe_compile(_step_impl)

    return kernel


def kuramoto_torch(k, n, device=None):
    """O(n) form: sum_j sin(th_j - th_i) = cos(th_i)*S - sin(th_i)*C."""
    import torch
    kk = float(k)

    def f(X):
        phi = torch.cat([X, torch.zeros(X.shape[0], 1, dtype=X.dtype,
                                        device=X.device)], dim=1)
        s, co = torch.sin(phi), torch.cos(phi)
        S = s.sum(dim=1, keepdim=True)
        C = co.sum(dim=1, keepdim=True)
        dtheta = (kk / n) * (co * S - s * C)
        return dtheta[:, :n - 1] - dtheta[:, n - 1:n]
    return f
