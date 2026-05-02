"""
Iter 35: Debug WHY SandwichKV doesn't work on Mistral / Llama-3.

Step 1: Per-layer leave-one-out sensitivity profile (each layer at b=4, rest at b=2)
Step 2: Compare profiles between R1-Distill (works) vs Mistral/Llama-3 (doesn't work)
Step 3: Build adapted sandwich based on top-K sensitive layers
Step 4: Test adapted sandwich vs uniform TurboQuant
"""
import json, time, math, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent / "perplexity_iter35_profile_debug.json"
N_CHUNKS = 4  # fewer chunks since profiling is many runs
CHUNK_LEN = 1024


def load_lloyd(d, b):
    with open(Path("/workspace") / f"codebook_d{d}_b{b}.json") as f:
        cb = json.load(f)
    return torch.tensor(cb["centroids"], dtype=torch.float32), torch.tensor(cb["boundaries"], dtype=torch.float32)


def q_lloyd(k, bits, R, codebooks):
    cents, bnd = codebooks[bits]
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    idx = torch.searchsorted(bnd, y.contiguous())
    return (cents[idx] @ R) * norms


class KVHook:
    def __init__(self, head_dim, n_layers, k_bits_map, v_bits_map, device="cuda:0"):
        self.head_dim, self.n_layers = head_dim, n_layers
        self.k_bits_map = k_bits_map  # dict layer_idx → bits
        self.v_bits_map = v_bits_map
        self.device = device
        self.codebooks = {}
        for b in [1, 2, 3, 4]:
            cents, bnd = load_lloyd(head_dim, b)
            self.codebooks[b] = (cents.to(device), bnd[1:-1].to(device))
        gen = torch.Generator(device="cpu").manual_seed(42)
        self.R_k = []
        for _ in range(n_layers):
            G = torch.randn(head_dim, head_dim, generator=gen)
            Q, _ = torch.linalg.qr(G)
            self.R_k.append(Q.to(device).to(torch.float32))
        gen_v = torch.Generator(device="cpu").manual_seed(43)
        self.R_v = []
        for _ in range(n_layers):
            G = torch.randn(head_dim, head_dim, generator=gen_v)
            Q, _ = torch.linalg.qr(G)
            self.R_v.append(Q.to(device).to(torch.float32))
        self.handles = []

    def install(self, model):
        head_dim = self.head_dim
        for l in range(self.n_layers):
            k_b = self.k_bits_map.get(l, 16)
            v_b = self.v_bits_map.get(l, 16)
            def make_hook(layer_idx, bits, R_list):
                def hook(module, input, output):
                    if bits >= 16: return output
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = R_list[layer_idx]
                    out_q = q_lloyd(out, bits, R, self.codebooks)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook
            if k_b < 16:
                h = model.model.layers[l].self_attn.k_proj.register_forward_hook(
                    make_hook(l, k_b, self.R_k))
                self.handles.append(h)
            if v_b < 16:
                h = model.model.layers[l].self_attn.v_proj.register_forward_hook(
                    make_hook(l, v_b, self.R_v))
                self.handles.append(h)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def compute_perplexity(model, tok, chunks):
    device = model.device
    total_loss = 0.0; total_tokens = 0
    for chunk_text in chunks:
        ids = tok(chunk_text, return_tensors="pt", truncation=True, max_length=CHUNK_LEN).input_ids.to(device)
        if ids.shape[1] < 32: continue
        with torch.inference_mode():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens), total_tokens


def profile_sensitivity(model_id, name):
    """For each layer: set ALL layers at b=2 (K+V), EXCEPT layer L at b=4. Measure ppl drop."""
    print(f"\n{'='*100}\n=== Profiling {name} ({model_id}) ===\n{'='*100}")
    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        try: login(token=os.environ["HF_TOKEN"])
        except: pass

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    if not Path(f"/workspace/codebook_d{head_dim}_b2.json").exists():
        print(f"  ⚠️ no codebook for d={head_dim}, skip")
        del model; torch.cuda.empty_cache()
        return None

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    all_ids = tok(full_text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(N_CHUNKS):
        s = i * CHUNK_LEN
        if s + CHUNK_LEN > all_ids.shape[0]: break
        chunks.append(tok.decode(all_ids[s:s+CHUNK_LEN], skip_special_tokens=True))

    # Baseline FP16
    print("[FP16]")
    ppl_fp16, _ = compute_perplexity(model, tok, chunks)
    print(f"  PPL = {ppl_fp16:.4f}")

    # All-b=2 K+V baseline
    bm_all2 = {i: 2 for i in range(n_layers)}
    h = KVHook(head_dim, n_layers, bm_all2, bm_all2)
    h.install(model)
    ppl_b2, _ = compute_perplexity(model, tok, chunks)
    h.remove()
    print(f"[all-b=2 K+V] PPL = {ppl_b2:.4f}  Δ=+{ppl_b2 - ppl_fp16:.3f}")

    # Per-layer profile: each layer L at b=4, rest at b=2
    print(f"\n[per-layer leave-one-out (set layer L to b=4, rest b=2)]")
    sensitivity = {}
    for L in range(n_layers):
        bm = {i: 2 for i in range(n_layers)}
        bm[L] = 4
        h = KVHook(head_dim, n_layers, bm, bm)
        h.install(model)
        try:
            ppl, _ = compute_perplexity(model, tok, chunks)
            lift = ppl_b2 - ppl  # positive = upgrading layer L helps
            sensitivity[L] = {
                "ppl_at_b4": float(ppl),
                "lift_over_b2": float(lift),
            }
            if L < 5 or L >= n_layers - 5 or L % 4 == 0:
                print(f"  L{L:02d}: ppl={ppl:.3f}  lift={lift:+.3f}")
        except Exception as e:
            print(f"  L{L}: ERROR {e}")
        finally:
            h.remove()

    # Sort by lift, find top sensitive layers
    sorted_L = sorted(sensitivity.keys(), key=lambda L: sensitivity[L]["lift_over_b2"], reverse=True)
    print(f"\nTop 12 most sensitive layers:")
    for r, L in enumerate(sorted_L[:12]):
        print(f"  {r+1:2d}. L{L:02d}  lift={sensitivity[L]['lift_over_b2']:+.3f}")

    # Test ADAPTED sandwich: top-16 layers at b=4, rest at b=2 (avg 3.0)
    top16 = sorted_L[:16]
    bm_adapt = {i: (4 if i in top16 else 2) for i in range(n_layers)}
    h = KVHook(head_dim, n_layers, bm_adapt, bm_adapt)
    h.install(model)
    ppl_adapt, _ = compute_perplexity(model, tok, chunks)
    h.remove()
    print(f"\n[Adapted sandwich (top-16 sensitive at b=4, rest b=2)]")
    print(f"  PPL = {ppl_adapt:.4f}  Δ vs FP16 = +{ppl_adapt - ppl_fp16:.3f}")

    # Test STANDARD sandwich-8-8 (first 8 + last 8)
    fn_count = max(2, n_layers // 4)
    bm_std = {i: (4 if (i < fn_count or i >= n_layers - fn_count) else 2) for i in range(n_layers)}
    h = KVHook(head_dim, n_layers, bm_std, bm_std)
    h.install(model)
    ppl_std, _ = compute_perplexity(model, tok, chunks)
    h.remove()
    print(f"\n[Standard sandwich {fn_count}-{fn_count} (first/last at b=4)]")
    print(f"  PPL = {ppl_std:.4f}  Δ vs FP16 = +{ppl_std - ppl_fp16:.3f}")

    # Test TurboQuant K=V=b=3 (matched memory)
    bm_tq = {i: 3 for i in range(n_layers)}
    h = KVHook(head_dim, n_layers, bm_tq, bm_tq)
    h.install(model)
    ppl_tq, _ = compute_perplexity(model, tok, chunks)
    h.remove()
    print(f"\n[TurboQuant uniform b=3 (matched memory)]")
    print(f"  PPL = {ppl_tq:.4f}  Δ vs FP16 = +{ppl_tq - ppl_fp16:.3f}")

    result = {
        "model": name,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "FP16_PPL": float(ppl_fp16),
        "all_b2_KV_PPL": float(ppl_b2),
        "all_b2_delta": float(ppl_b2 - ppl_fp16),
        "sensitivity_per_layer": sensitivity,
        "ranking_by_lift": sorted_L,
        "top12_sensitive": sorted_L[:12],
        "adapted_sandwich_top16": {
            "ppl": float(ppl_adapt),
            "delta_vs_fp16": float(ppl_adapt - ppl_fp16),
        },
        "standard_sandwich": {
            "shape": f"{fn_count}-{fn_count}",
            "ppl": float(ppl_std),
            "delta_vs_fp16": float(ppl_std - ppl_fp16),
        },
        "turboquant_b3": {
            "ppl": float(ppl_tq),
            "delta_vs_fp16": float(ppl_tq - ppl_fp16),
        },
    }

    del model; torch.cuda.empty_cache()
    return result


def main():
    models = [
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "R1-Distill-Llama-8B"),
        ("mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3"),
        ("NousResearch/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct"),
    ]
    all_results = {}
    for mid, name in models:
        try:
            r = profile_sensitivity(mid, name)
            if r is not None:
                all_results[name] = r
        except Exception as e:
            print(f"\n  ⚠️ Failed {name}: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}
        # Save incrementally
        with open(OUT, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*100}\nSUMMARY: Adapted vs Standard sandwich vs TurboQuant\n{'='*100}")
    print(f"{'Model':<30} {'FP16':<10} {'TQ b=3':<10} {'Std-Sand':<12} {'Adapted':<12} {'Best non-TQ?':<14}")
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<30} ERR")
            continue
        fp = r["FP16_PPL"]
        tq = r["turboquant_b3"]["delta_vs_fp16"]
        std = r["standard_sandwich"]["delta_vs_fp16"]
        adapt = r["adapted_sandwich_top16"]["delta_vs_fp16"]
        best = "ours wins" if min(std, adapt) < tq else "TQ wins"
        print(f"{name:<30} {fp:<10.3f} {tq:<+10.3f} {std:<+12.3f} {adapt:<+12.3f} {best:<14}")

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
