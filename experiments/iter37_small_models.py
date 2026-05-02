"""
Iter 37: Test SandwichKV vs TurboQuant on small models <7B (head_dim=128).

Models to test:
1. R1-Distill-Qwen-1.5B (R1 family, small)
2. Qwen-2.5-1.5B-Instruct (base counterpart)
3. Qwen-2.5-3B-Instruct (mid-size)
4. R1-Distill-Llama-8B (already known, sanity check)

For each model:
- FP16 baseline
- TurboQuant K=V uniform b=2/3/4
- SandwichKV K=V at 3.0/3.5/4.0 bits

Output: full Pareto + decision rule check (L0/L_middle ratio)
"""
import json, time, math, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path("/workspace/iter37_small_models.json")
N_CHUNKS = 6
CHUNK_LEN = 1024


def load_lloyd(d, b):
    with open(Path("/workspace") / f"codebook_d{d}_b{b}.json") as f:
        cb = json.load(f)
    cents = torch.tensor(cb["centroids"], dtype=torch.float32)
    bnd = torch.tensor(cb["boundaries"], dtype=torch.float32)
    # Strip the ±inf edges since searchsorted needs only interior boundaries
    bnd = bnd[1:-1]
    return cents, bnd


def q_lloyd(k, bits, R, codebooks):
    cents, bnd = codebooks[bits]
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    idx = torch.searchsorted(bnd, y.contiguous())
    return (cents[idx] @ R) * norms


def q_rvq_same_R(k, R, codebooks, bits_per_pass):
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    accumulator = torch.zeros_like(y)
    cur_residual = y
    target_std = 1.0 / math.sqrt(cur_residual.shape[-1])
    for i, b in enumerate(bits_per_pass):
        if b >= 16 or b < 1: continue
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


def make_sandwich_bm(n_layers, first_n, last_n, hi, lo):
    return {i: hi if (i < first_n or i >= n_layers - last_n) else lo for i in range(n_layers)}


class KVHook:
    def __init__(self, head_dim, n_layers, k_fn, v_fn, device="cuda:0"):
        self.head_dim, self.n_layers = head_dim, n_layers
        self.k_fn, self.v_fn = k_fn, v_fn
        self.device = device
        self.codebooks = {}
        for b in [1, 2, 3, 4]:
            cents, bnd = load_lloyd(head_dim, b)
            self.codebooks[b] = (cents.to(device), bnd.to(device))
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
            def make_k_hook(layer_idx):
                def hook(module, input, output):
                    if self.k_fn is None: return output
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = self.R_k[layer_idx]
                    out_q = self.k_fn(self, out, R, layer_idx)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook
            def make_v_hook(layer_idx):
                def hook(module, input, output):
                    if self.v_fn is None: return output
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = self.R_v[layer_idx]
                    out_q = self.v_fn(self, out, R, layer_idx)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook
            h_k = model.model.layers[l].self_attn.k_proj.register_forward_hook(make_k_hook(l))
            self.handles.append(h_k)
            h_v = model.model.layers[l].self_attn.v_proj.register_forward_hook(make_v_hook(l))
            self.handles.append(h_v)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def make_fn_uniform(b):
    def fn(self, k, R, layer_idx):
        if b >= 16: return k
        return q_lloyd(k, b, R, self.codebooks)
    return fn


def make_fn_sand_lloyd(bm):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        return q_lloyd(k, bb, R, self.codebooks)
    return fn


def make_fn_sand_rvq(bm, b_outer, bits_per_pass):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, b_outer)
        if bb >= 16: return k
        if bb == b_outer:
            return q_lloyd(k, b_outer, R, self.codebooks)
        return q_rvq_same_R(k, R, self.codebooks, bits_per_pass)
    return fn


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


def profile_sensitivity(model, tok, chunks, head_dim, n_layers, ppl_fp16):
    """Quick leave-one-out: each layer L at b=4 (rest at b=2 K+V).
    Return: dict {layer_idx: lift over all-b=2}."""
    bm_all2 = {i: 2 for i in range(n_layers)}
    h = KVHook(head_dim, n_layers, make_fn_sand_lloyd(bm_all2), make_fn_sand_lloyd(bm_all2))
    h.install(model)
    ppl_b2, _ = compute_perplexity(model, tok, chunks)
    h.remove()

    sensitivity = {}
    for L in range(n_layers):
        bm = {i: 2 for i in range(n_layers)}
        bm[L] = 4
        h = KVHook(head_dim, n_layers, make_fn_sand_lloyd(bm), make_fn_sand_lloyd(bm))
        h.install(model)
        try:
            ppl, _ = compute_perplexity(model, tok, chunks)
            sensitivity[L] = ppl_b2 - ppl  # positive = upgrading L helps
        except Exception:
            sensitivity[L] = 0.0
        finally:
            h.remove()
    return sensitivity, ppl_b2


def benchmark_one(model_id, name):
    print(f"\n{'='*100}\n=== {name} ({model_id}) ===\n{'='*100}")
    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        try: login(token=os.environ["HF_TOKEN"])
        except: pass

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="eager",
        )
    except Exception as e:
        print(f"  ⚠️ Load failed: {e}")
        return {"error": str(e)}

    model.eval()
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    n_q_heads = model.config.num_attention_heads
    print(f"  n_layers={n_layers}, head_dim={head_dim}, n_kv={n_kv_heads}, n_q={n_q_heads}")

    if not Path(f"/workspace/codebook_d{head_dim}_b2.json").exists():
        print(f"  ⚠️ no codebook for d={head_dim}")
        del model; torch.cuda.empty_cache()
        return {"error": f"no codebook for d={head_dim}"}

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    all_ids = tok(full_text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(N_CHUNKS):
        s = i * CHUNK_LEN
        if s + CHUNK_LEN > all_ids.shape[0]: break
        chunks.append(tok.decode(all_ids[s:s+CHUNK_LEN], skip_special_tokens=True))

    print("[FP16]")
    ppl_fp16, _ = compute_perplexity(model, tok, chunks)
    print(f"  PPL = {ppl_fp16:.4f}")
    if ppl_fp16 > 1000:
        print(f"  ⚠️ FP16 baseline broken (>1000), skip")
        del model; torch.cuda.empty_cache()
        return {"error": f"FP16 baseline broken: {ppl_fp16}"}

    fn_count = max(2, n_layers // 4)
    sand = make_sandwich_bm(n_layers, fn_count, fn_count, 4, 2)

    configs = [
        ("FP16", None, None, 16.0, 16.0),
        ("TQ_K=V=b2", make_fn_uniform(2), make_fn_uniform(2), 2.0, 2.0),
        ("TQ_K=V=b3", make_fn_uniform(3), make_fn_uniform(3), 3.0, 3.0),
        ("TQ_K=V=b4", make_fn_uniform(4), make_fn_uniform(4), 4.0, 4.0),
        ("Sand_K=V_3.0", make_fn_sand_lloyd(sand), make_fn_sand_lloyd(sand), 3.0, 3.0),
        ("Sand+RVQ_K=V_3.5", make_fn_sand_rvq(sand, 4, [1,1,1]),
                              make_fn_sand_rvq(sand, 4, [1,1,1]), 3.5, 3.5),
        ("Sand+RVQ_K=V_4.0", make_fn_sand_rvq(sand, 4, [1,1,1,1]),
                              make_fn_sand_rvq(sand, 4, [1,1,1,1]), 4.0, 4.0),
    ]

    results = {"FP16_PPL": float(ppl_fp16), "n_layers": n_layers,
               "head_dim": head_dim, "n_kv_heads": n_kv_heads,
               "sandwich_shape": f"{fn_count}-{fn_count}"}
    print(f"\n{'config':<25} {'PPL':<10} {'Δppl':<10}")
    print("-" * 50)
    for name_, k_fn, v_fn, kb, vb in configs:
        if k_fn is None and v_fn is None:
            print(f"{name_:<25} {ppl_fp16:<10.4f} {0.0:<10.4f}")
            results[name_] = {"K_bits": kb, "V_bits": vb, "ppl": float(ppl_fp16), "delta_ppl": 0.0}
            continue
        h = KVHook(head_dim, n_layers, k_fn, v_fn)
        h.install(model)
        try:
            ppl, _ = compute_perplexity(model, tok, chunks)
            delta = ppl - ppl_fp16
            print(f"{name_:<25} {ppl:<10.4f} {delta:+.4f}")
            results[name_] = {"K_bits": kb, "V_bits": vb, "ppl": float(ppl), "delta_ppl": float(delta)}
        except Exception as e:
            print(f"{name_:<25}: ERROR {e}")
            results[name_] = {"error": str(e)}
        finally:
            h.remove()

    # Sensitivity profile (decision rule)
    print("\n[Per-layer sensitivity profile]")
    sensitivity, ppl_b2 = profile_sensitivity(model, tok, chunks, head_dim, n_layers, ppl_fp16)
    sorted_L = sorted(sensitivity.keys(), key=lambda L: sensitivity[L], reverse=True)
    top5 = [(L, sensitivity[L]) for L in sorted_L[:5]]
    boundary_lift = max(sensitivity.get(0, 0), sensitivity.get(1, 0))
    middle_layers = list(range(n_layers // 4, 3 * n_layers // 4))
    middle_lift = np.mean([sensitivity.get(L, 0) for L in middle_layers])
    ratio = boundary_lift / max(middle_lift, 0.01)
    print(f"  Top-5 sensitive: {top5}")
    print(f"  Boundary lift: {boundary_lift:.3f}, middle mean lift: {middle_lift:.3f}")
    print(f"  → L0/L_middle ratio = {ratio:.2f} {'(boundary-dominant)' if ratio > 5 else '(flat)'}")
    results["sensitivity_profile"] = {
        "all_b2_KV_PPL": float(ppl_b2),
        "all_b2_delta": float(ppl_b2 - ppl_fp16),
        "per_layer_lift": {str(L): float(v) for L, v in sensitivity.items()},
        "top5_sensitive_layers": [(L, float(v)) for L, v in top5],
        "boundary_lift": float(boundary_lift),
        "middle_mean_lift": float(middle_lift),
        "ratio": float(ratio),
        "verdict": "boundary-dominant" if ratio > 5 else "flat",
    }

    del model; torch.cuda.empty_cache()
    return results


def main():
    models = [
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "R1-Distill-Qwen-1.5B"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-2.5-1.5B-Instruct"),
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen-2.5-3B-Instruct"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "R1-Distill-Llama-8B"),
    ]
    all_results = {}
    for mid, name in models:
        try:
            r = benchmark_one(mid, name)
            all_results[name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}
        with open(OUT, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*120}\nCROSS-MODEL SUMMARY\n{'='*120}")
    print(f"{'Method':<25}", end="")
    for _, name in models:
        if name in all_results and "error" not in all_results[name]:
            print(f"{name[:18]:<20}", end="")
    print()
    for cfg in ["FP16_PPL_baseline", "TQ_K=V=b2", "TQ_K=V=b3", "TQ_K=V=b4",
                "Sand_K=V_3.0", "Sand+RVQ_K=V_3.5", "Sand+RVQ_K=V_4.0", "ratio"]:
        label = cfg
        if cfg == "FP16_PPL_baseline":
            print(f"{'FP16 PPL':<25}", end="")
            for _, name in models:
                r = all_results.get(name, {})
                if "error" in r:
                    print(f"{'ERR':<20}", end="")
                else:
                    print(f"{r.get('FP16_PPL', 0):<20.3f}", end="")
        elif cfg == "ratio":
            print(f"{'L0/L_mid ratio':<25}", end="")
            for _, name in models:
                r = all_results.get(name, {})
                if "error" in r:
                    print(f"{'ERR':<20}", end="")
                else:
                    sp = r.get("sensitivity_profile", {})
                    rt = sp.get("ratio", 0)
                    verdict = sp.get("verdict", "—")
                    print(f"{rt:<7.1f} ({verdict[:9]:<9})", end="")
        else:
            print(f"{cfg:<25}", end="")
            for _, name in models:
                r = all_results.get(name, {})
                if "error" in r:
                    print(f"{'ERR':<20}", end="")
                else:
                    v = r.get(cfg, {})
                    if "delta_ppl" in v:
                        d = v["delta_ppl"]
                        print(f"{('+' if d >= 0 else '') + f'{d:.3f}':<20}", end="")
                    else:
                        print(f"{'—':<20}", end="")
        print()


if __name__ == "__main__":
    main()
