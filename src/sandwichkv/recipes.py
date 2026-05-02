"""Pre-baked layer-bits callbacks for the most common configurations.

These factories close over a per-layer bits map (`bm`) or a flat bit width and
return functions compatible with `KVHook`'s k_fn / v_fn slot.
"""
from __future__ import annotations

import torch

from .quantize import q_lloyd, q_rvq_same_R


def make_fn_uniform(b: int):
    """TurboQuant uniform Lloyd-Max at `b` bits/coord."""
    def fn(hook, k: torch.Tensor, R: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if b >= 16:
            return k
        return q_lloyd(k, b, R, hook.codebooks)
    return fn


def make_fn_sand_lloyd(bm: dict[int, int]):
    """Sandwich Lloyd-Max: per-layer bit width from `bm` (default 2)."""
    def fn(hook, k: torch.Tensor, R: torch.Tensor, layer_idx: int) -> torch.Tensor:
        bb = bm.get(layer_idx, 2)
        if bb >= 16:
            return k
        return q_lloyd(k, bb, R, hook.codebooks)
    return fn


def make_fn_sand_rvq(bm: dict[int, int], b_outer: int, bits_per_pass: list[int]):
    """Sandwich + RVQ in middle layers, uniform Lloyd-Max in outer layers."""
    def fn(hook, k: torch.Tensor, R: torch.Tensor, layer_idx: int) -> torch.Tensor:
        bb = bm.get(layer_idx, b_outer)
        if bb >= 16:
            return k
        if bb == b_outer:
            return q_lloyd(k, b_outer, R, hook.codebooks)
        return q_rvq_same_R(k, R, hook.codebooks, bits_per_pass)
    return fn
