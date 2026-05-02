"""Run the WikiText-2 perplexity benchmark for one model across the standard configs.

Compares FP16 / TurboQuant uniform b∈{2,3,4} / Sandwich Lloyd / Sandwich+RVQ.
Designed to reproduce the multi-model table in the paper.

Usage:
    python scripts/run_benchmark.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --codebook-dir codebooks \
        --out results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sandwichkv import (
    KVHook,
    sandwich_bm,
    make_fn_uniform,
    make_fn_sand_lloyd,
    make_fn_sand_rvq,
)


def compute_perplexity(model, tok, chunks, max_len: int) -> tuple[float, int]:
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
    print(f"n_layers={n_layers}, head_dim={head_dim}")

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

    ppl_fp16, _ = compute_perplexity(model, tok, chunks, args.chunk_len)
    print(f"FP16 PPL = {ppl_fp16:.4f}")

    fn_count = max(2, n_layers // 4)
    sand = sandwich_bm(n_layers, fn_count, fn_count, hi=4, lo=2)

    configs = [
        ("FP16", None, None, 16.0, 16.0),
        ("TQ_K=V=b2", make_fn_uniform(2), make_fn_uniform(2), 2.0, 2.0),
        ("TQ_K=V=b3", make_fn_uniform(3), make_fn_uniform(3), 3.0, 3.0),
        ("TQ_K=V=b4", make_fn_uniform(4), make_fn_uniform(4), 4.0, 4.0),
        ("Sand_K=V_3.0", make_fn_sand_lloyd(sand), make_fn_sand_lloyd(sand), 3.0, 3.0),
        ("Sand+RVQ_K=V_3.5",
            make_fn_sand_rvq(sand, 4, [1, 1, 1]),
            make_fn_sand_rvq(sand, 4, [1, 1, 1]), 3.5, 3.5),
        ("Sand+RVQ_K=V_4.0",
            make_fn_sand_rvq(sand, 4, [1, 1, 1, 1]),
            make_fn_sand_rvq(sand, 4, [1, 1, 1, 1]), 4.0, 4.0),
    ]

    results: dict = {
        "model": args.model,
        "FP16_PPL": float(ppl_fp16),
        "n_layers": n_layers,
        "head_dim": head_dim,
        "sandwich_shape": f"{fn_count}-{fn_count}",
    }
    print(f"\n{'config':<25} {'PPL':<10} {'Δppl':<10}")
    print("-" * 50)
    for name, k_fn, v_fn, kb, vb in configs:
        if k_fn is None and v_fn is None:
            print(f"{name:<25} {ppl_fp16:<10.4f} {0.0:<10.4f}")
            results[name] = {"K_bits": kb, "V_bits": vb, "ppl": float(ppl_fp16), "delta_ppl": 0.0}
            continue
        with KVHook(head_dim, n_layers, k_fn, v_fn, codebook_dir=args.codebook_dir) as h:
            h.install(model)
            ppl, _ = compute_perplexity(model, tok, chunks, args.chunk_len)
        delta = ppl - ppl_fp16
        print(f"{name:<25} {ppl:<10.4f} {delta:+.4f}")
        results[name] = {"K_bits": kb, "V_bits": vb, "ppl": float(ppl), "delta_ppl": float(delta)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
