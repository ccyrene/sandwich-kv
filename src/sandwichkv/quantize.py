"""Core quantizers: random-rotation Lloyd-Max (TurboQuant primitive) + RVQ.

Both routines operate on tensors of shape (..., d) where the trailing axis is
the head dimension. Norms are factored out before rotation so the marginal of
the rotated unit vector matches the codebook distribution.
"""
from __future__ import annotations

import math

import torch


def make_rotations(
    n_layers: int, d: int, seed: int, device: str | torch.device = "cuda:0"
) -> list[torch.Tensor]:
    """One Haar-distributed orthogonal matrix per layer (deterministic from seed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rots: list[torch.Tensor] = []
    for _ in range(n_layers):
        G = torch.randn(d, d, generator=gen)
        Q, _ = torch.linalg.qr(G)
        rots.append(Q.to(device).to(torch.float32))
    return rots


def q_lloyd(
    k: torch.Tensor,
    bits: int,
    R: torch.Tensor,
    codebooks: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Single-pass random-rotation Lloyd-Max quantizer at `bits` bits/coord.

    This is the TurboQuant primitive on which SandwichKV builds.
    """
    cents, bnd = codebooks[bits]
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    idx = torch.searchsorted(bnd, y.contiguous())
    return (cents[idx] @ R) * norms


def q_rvq_same_R(
    k: torch.Tensor,
    R: torch.Tensor,
    codebooks: dict[int, tuple[torch.Tensor, torch.Tensor]],
    bits_per_pass: list[int],
) -> torch.Tensor:
    """Multi-pass Residual Vector Quantization, sharing one rotation R across passes.

    Pass 0 quantizes directly. Passes ≥1 normalize the residual by its
    per-batch standard deviation against the target marginal std (1/sqrt(d))
    before lookup, then rescale the centroid back. This per-batch SD scaling
    is what keeps the pipeline calibration-free.
    """
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    accumulator = torch.zeros_like(y)
    cur_residual = y
    target_std = 1.0 / math.sqrt(cur_residual.shape[-1])
    for i, b in enumerate(bits_per_pass):
        if b >= 16 or b < 1:
            continue
        cents, bnd = codebooks[b]
        if i == 0:
            idx = torch.searchsorted(bnd, cur_residual.contiguous())
            q_hat = cents[idx]
        else:
            sd_dims = tuple(range(cur_residual.dim() - 1))
            res_std = cur_residual.std(dim=sd_dims, keepdim=True).clamp(min=1e-6)
            scale = res_std / target_std
            scaled = cur_residual / scale
            idx = torch.searchsorted(bnd, scaled.contiguous())
            q_hat = cents[idx] * scale
        accumulator = accumulator + q_hat
        cur_residual = cur_residual - q_hat
    return (accumulator @ R) * norms
