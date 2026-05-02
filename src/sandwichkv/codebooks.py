"""Lloyd-Max scalar codebooks for the rotated, normalized K (and V) marginal.

After random rotation R and unit-normalization, each coordinate of k_unit @ R^T
converges (CLT-style at d=128) to a Beta((d-1)/2, (d-1)/2) marginal on [-1, 1].
We fit a 1D Lloyd-Max quantizer on samples drawn from S^{d-1} once, offline,
and the codebook is reused across models, layers, and batches — calibration-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def fit_lloyd_1d(
    samples: torch.Tensor,
    n_levels: int,
    iters: int = 50,
    n_seeds: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a Lloyd-Max scalar quantizer with `n_levels` cells.

    Returns (centroids, full_boundaries) where full_boundaries has ±1e9 sentinels
    at both ends, ready for torch.searchsorted.
    """
    samples = samples.float()
    if samples.numel() < n_levels * 4:
        qs = torch.linspace(0, 1, n_levels + 1)[1:-1]
        boundaries = torch.quantile(samples, qs)
        cents = torch.zeros(n_levels)
        for j in range(n_levels):
            lo = -1e9 if j == 0 else boundaries[j - 1].item()
            hi = 1e9 if j == n_levels - 1 else boundaries[j].item()
            mask = (samples >= lo) & (samples < hi)
            cents[j] = samples[mask].mean() if mask.any() else (lo + hi) / 2
        full_bnd = torch.cat([torch.tensor([-1e9]), boundaries, torch.tensor([1e9])])
        return cents, full_bnd

    best_sse = float("inf")
    best_cents: torch.Tensor | None = None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        sorted_s, _ = samples.sort()
        n = sorted_s.shape[0]
        cents = torch.tensor(
            [sorted_s[(2 * j + 1) * n // (2 * n_levels)].item() for j in range(n_levels)]
        )
        cents = cents + torch.randn_like(cents) * 0.001
        cents, _ = cents.sort()
        for _ in range(iters):
            boundaries = (cents[:-1] + cents[1:]) / 2
            idx = torch.searchsorted(boundaries, samples)
            new_cents = torch.zeros(n_levels)
            for j in range(n_levels):
                mask = idx == j
                new_cents[j] = samples[mask].mean() if mask.any() else cents[j]
            if (new_cents - cents).abs().max() < 1e-6:
                cents = new_cents
                break
            cents = new_cents
        boundaries = (cents[:-1] + cents[1:]) / 2
        idx = torch.searchsorted(boundaries, samples)
        sse = ((samples - cents[idx]) ** 2).sum().item()
        if sse < best_sse:
            best_sse = sse
            best_cents = cents.clone()

    assert best_cents is not None
    boundaries = (best_cents[:-1] + best_cents[1:]) / 2
    full_bnd = torch.cat([torch.tensor([-1e9]), boundaries, torch.tensor([1e9])])
    return best_cents, full_bnd


def _sphere_marginal_samples(d: int, n: int = 500_000, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return x[:, 0]


def build_codebooks(
    d: int,
    bits: tuple[int, ...] = (1, 2, 3, 4),
    out_dir: str | Path = "codebooks",
    n_samples: int = 500_000,
) -> None:
    """Generate Lloyd-Max codebooks for head dimension `d` and the given bit widths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _sphere_marginal_samples(d, n_samples)
    for b in bits:
        n_levels = 2 ** b
        cents, bnd = fit_lloyd_1d(samples, n_levels)
        path = out_dir / f"codebook_d{d}_b{b}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "d": d,
                    "b": b,
                    "n_levels": n_levels,
                    "centroids": cents.tolist(),
                    "boundaries": bnd.tolist(),
                },
                f,
                indent=2,
            )


def load_codebook(
    d: int, b: int, codebook_dir: str | Path = "codebooks"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (centroids, interior_boundaries) for head dim `d` and `b` bits.

    The returned boundaries strip the ±1e9 sentinels so they can be used directly
    with `torch.searchsorted` against rotated coordinates already on [-1, 1].
    """
    path = Path(codebook_dir) / f"codebook_d{d}_b{b}.json"
    with open(path) as f:
        cb = json.load(f)
    cents = torch.tensor(cb["centroids"], dtype=torch.float32)
    bnd = torch.tensor(cb["boundaries"], dtype=torch.float32)[1:-1]
    return cents, bnd
