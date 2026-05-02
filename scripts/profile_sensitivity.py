"""Run the per-layer sensitivity profile and apply the SandwichKV/TurboQuant decision rule.

Usage:
    python scripts/profile_sensitivity.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --codebook-dir codebooks \
        --out profile.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sandwichkv import KVHook, decision_rule, profile_sensitivity, make_fn_sand_lloyd


def compute_perplexity(model, tok, chunks, max_len: int = 1024) -> tuple[float, int]:
    device = model.device
    total_loss = 0.0
    total_tokens = 0
    for chunk_text in chunks:
        ids = tok(chunk_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 32:
            continue
        with torch.inference_mode():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens), total_tokens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--codebook-dir", default="codebooks")
    p.add_argument("--out", required=True)
    p.add_argument("--n-chunks", type=int, default=6)
    p.add_argument("--chunk-len", type=int, default=1024)
    args = p.parse_args()

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        try:
            login(token=os.environ["HF_TOKEN"])
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    all_ids = tok(full_text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(args.n_chunks):
        s = i * args.chunk_len
        if s + args.chunk_len > all_ids.shape[0]:
            break
        chunks.append(tok.decode(all_ids[s:s + args.chunk_len], skip_special_tokens=True))

    sample_layers = sorted(set(
        list(range(0, min(8, n_layers))) +
        list(range(8, n_layers - 8, 2)) +
        list(range(max(0, n_layers - 8), n_layers))
    ))

    def make_hook_factory(bm: dict[int, int]) -> KVHook:
        return KVHook(
            head_dim, n_layers,
            make_fn_sand_lloyd(bm), make_fn_sand_lloyd(bm),
            codebook_dir=args.codebook_dir,
        )

    sensitivity, ppl_b2 = profile_sensitivity(
        model, tok, chunks, head_dim, n_layers,
        compute_perplexity_fn=lambda m, t, c: compute_perplexity(m, t, c, args.chunk_len),
        make_hook_factory=make_hook_factory,
        sample_layers=sample_layers,
    )

    decision = decision_rule(sensitivity, n_layers, sample_layers=sample_layers)
    print(f"\n=== Decision: {decision.method} ===")
    print(f"  ratio = {decision.ratio:.2f}  ({decision.verdict})")
    print(f"  boundary lift = {decision.boundary_lift:.3f}")
    print(f"  middle mean  = {decision.middle_lift:.3f}")

    out = {
        "model": args.model,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "all_b2_KV_PPL": float(ppl_b2),
        "per_layer_lift": {str(L): float(v) for L, v in sensitivity.items()},
        "decision": {
            "method": decision.method,
            "ratio": decision.ratio,
            "boundary_lift": decision.boundary_lift,
            "middle_lift": decision.middle_lift,
            "verdict": decision.verdict,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
