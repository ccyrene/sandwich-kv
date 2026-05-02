"""Forward-hook wrapper that intercepts each layer's k_proj / v_proj output and
quantizes it through user-supplied K and V functions.

The hook works on Llama-style architectures that expose
`model.model.layers[l].self_attn.{k_proj, v_proj}`. Function callbacks receive
(self, tensor, R, layer_idx) and return a quantized tensor with the same shape.
"""
from __future__ import annotations

from typing import Callable

import torch

from .codebooks import load_codebook
from .quantize import make_rotations


KFn = Callable[["KVHook", torch.Tensor, torch.Tensor, int], torch.Tensor]


class KVHook:
    def __init__(
        self,
        head_dim: int,
        n_layers: int,
        k_fn: KFn | None,
        v_fn: KFn | None,
        device: str | torch.device = "cuda:0",
        codebook_dir: str = "codebooks",
        bits: tuple[int, ...] = (1, 2, 3, 4),
        seed_k: int = 42,
        seed_v: int = 43,
    ) -> None:
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.k_fn = k_fn
        self.v_fn = v_fn
        self.device = device

        self.codebooks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for b in bits:
            cents, bnd = load_codebook(head_dim, b, codebook_dir)
            self.codebooks[b] = (cents.to(device), bnd.to(device))

        self.R_k = make_rotations(n_layers, head_dim, seed_k, device)
        self.R_v = make_rotations(n_layers, head_dim, seed_v, device)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, model) -> None:
        head_dim = self.head_dim
        for l in range(self.n_layers):

            def make_k_hook(layer_idx: int):
                def hook(module, input, output):
                    if self.k_fn is None:
                        return output
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = self.R_k[layer_idx]
                    out_q = self.k_fn(self, out, R, layer_idx)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook

            def make_v_hook(layer_idx: int):
                def hook(module, input, output):
                    if self.v_fn is None:
                        return output
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = self.R_v[layer_idx]
                    out_q = self.v_fn(self, out, R, layer_idx)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook

            attn = model.model.layers[l].self_attn
            self.handles.append(attn.k_proj.register_forward_hook(make_k_hook(l)))
            self.handles.append(attn.v_proj.register_forward_hook(make_v_hook(l)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
