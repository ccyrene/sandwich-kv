"""
Iter 33 (Day 1): K+V joint quantization with sandwich + RVQ.

Critical missing piece: previously only K was quantized. All major KV-quant papers
(KIVI, KVQuant, GEAR, Atom) compress BOTH K and V. We must validate K+V joint.

Test recipe at TRUE bit budgets:
  K = sandwich-8-8 outer b=4 + middle RVQ depth (2B-4) all-b=1 same-R
  V = same recipe (symmetric K+V) AND asymmetric variants

Targets:
  - Match TurboQuant K+V uniform b=4 (+0.23) at same memory
  - Match TurboQuant K+V uniform b=3 (+2.78) at 3-bit budget
  - Find best K vs V allocation
"""
import json, time, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT = Path(__file__).parent / "perplexity_iter33_kv_joint.json"
N_CHUNKS = 6
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


def q_rvq_same_R(k, R, codebooks, bits_per_pass):
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


def make_sandwich_bm(n_layers, first_n, last_n, hi, lo):
    return {i: hi if (i < first_n or i >= n_layers - last_n) else lo for i in range(n_layers)}


def true_avg_bits(n_layers, first_n, last_n, b_outer, bits_per_pass_middle):
    n_outer = first_n + last_n
    n_middle = n_layers - n_outer
    middle_bits = sum(bits_per_pass_middle)
    return (n_outer * b_outer + n_middle * middle_bits) / n_layers


class KVHook:
    """Hook BOTH K and V projections."""
    def __init__(self, head_dim, n_layers, k_fn, v_fn, device="cuda:0"):
        self.head_dim, self.n_layers = head_dim, n_layers
        self.k_fn, self.v_fn = k_fn, v_fn
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
        # Different rotation seed for V
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


def make_fn_uniform_lloyd(b):
    def fn(self, k, R, layer_idx):
        if b >= 16: return k
        return q_lloyd(k, b, R, self.codebooks)
    return fn


def make_fn_sandwich_rvq(bm, b_outer, bits_per_pass_middle):
    """Sandwich outer scalar + middle RVQ same-R."""
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, b_outer)
        if bb >= 16: return k
        if bb == b_outer:
            return q_lloyd(k, b_outer, R, self.codebooks)
        # middle: RVQ
        return q_rvq_same_R(k, R, self.codebooks, bits_per_pass_middle)
    return fn


def make_fn_sandwich_lloyd(bm):
    """Sandwich without RVQ (just per-layer scalar Lloyd-Max)."""
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        return q_lloyd(k, bb, R, self.codebooks)
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


def memory_bytes(avg_bits, n_layers=32, head_dim=128, n_kv_heads=8, with_norm=True):
    """Bytes per token for ONE side (K or V) across all layers and KV heads."""
    norm_bits = 16 if with_norm and avg_bits < 16 else 0
    return (avg_bits * head_dim + norm_bits) * n_layers * n_kv_heads / 8


def main():
    print(f"[load] {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
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
    for i in range(N_CHUNKS):
        s = i * CHUNK_LEN
        if s + CHUNK_LEN > all_ids.shape[0]: break
        chunks.append(tok.decode(all_ids[s:s+CHUNK_LEN], skip_special_tokens=True))

    print("\n=== FP16 baseline ===")
    ppl_baseline, _ = compute_perplexity(model, tok, chunks)
    print(f"  ppl = {ppl_baseline:.4f}")

    sand_88 = make_sandwich_bm(n_layers, 8, 8, 4, 2)

    # Recipe library
    def sandwich_rvq_at_target(target_bits):
        """Returns (function, true_bits) for SandwichKV at target avg bits."""
        depth = max(0, round(2 * target_bits - 4))
        if depth == 0:
            return make_fn_sandwich_lloyd(sand_88), 3.0  # canonical sand
        passes = [1] * depth
        true_b = true_avg_bits(n_layers, 8, 8, 4, passes)
        return make_fn_sandwich_rvq(sand_88, 4, passes), true_b

    # Configs: (name, k_fn, v_fn, K_bits, V_bits)
    configs = [
        # ===== References =====
        ("01_FP16_baseline (no quant)", None, None, 16.0, 16.0),
        ("02_K=V=uniform_b3 (TurboQuant)", make_fn_uniform_lloyd(3), make_fn_uniform_lloyd(3), 3.0, 3.0),
        ("03_K=V=uniform_b4 (TurboQuant)", make_fn_uniform_lloyd(4), make_fn_uniform_lloyd(4), 4.0, 4.0),

        # ===== K-only (V at FP16) — for reference =====
        ("04_K=sand_8-8_b4_b2 / V=FP16", make_fn_sandwich_lloyd(sand_88), None, 3.0, 16.0),
        ("05_K=sand+RVQ_d4 / V=FP16", make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]), None, 4.0, 16.0),

        # ===== K+V symmetric (same recipe) =====
        # K=V at SandwichKV 3.0 bit each
        ("06_K=V=sand_8-8 (no RVQ) [3.0 each]",
         make_fn_sandwich_lloyd(sand_88), make_fn_sandwich_lloyd(sand_88), 3.0, 3.0),
        # K=V at SandwichKV 3.5 bit each (depth 3 RVQ)
        ("07_K=V=sand+RVQ_depth3 [3.5 each]",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1]),
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1]), 3.5, 3.5),
        # K=V at SandwichKV 4.0 bit each (depth 4 RVQ)
        ("08_K=V=sand+RVQ_depth4 [4.0 each]",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]),
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]), 4.0, 4.0),

        # ===== K+V asymmetric: K compressed, V less =====
        ("09_K=sand+RVQ_d4 / V=uniform_b4",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]), make_fn_uniform_lloyd(4), 4.0, 4.0),
        ("10_K=sand+RVQ_d4 / V=uniform_b3",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]), make_fn_uniform_lloyd(3), 4.0, 3.0),
        ("11_K=sand+RVQ_d3 / V=uniform_b3",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1]), make_fn_uniform_lloyd(3), 3.5, 3.0),
        ("12_K=sand_8-8 / V=uniform_b3",
         make_fn_sandwich_lloyd(sand_88), make_fn_uniform_lloyd(3), 3.0, 3.0),
        ("13_K=sand_8-8 / V=uniform_b4",
         make_fn_sandwich_lloyd(sand_88), make_fn_uniform_lloyd(4), 3.0, 4.0),

        # KIVI-inspired asymmetric: K-sand, V at b=4 (per-token in KIVI, here scalar):
        ("14_K=V=uniform_b4 + RVQ_d4 K-side only",
         make_fn_sandwich_rvq(sand_88, 4, [1, 1, 1, 1]), make_fn_uniform_lloyd(4), 4.0, 4.0),
    ]

    print(f"\n{'config':<55} {'K bits':<8} {'V bits':<8} {'total':<8} {'mem(K+V)':<12} {'ppl':<10} {'Δppl':<10}")
    print("-" * 130)

    results = {}
    for name, k_fn, v_fn, kb, vb in configs:
        h = KVHook(head_dim, n_layers, k_fn, v_fn)
        h.install(model)
        try:
            t0 = time.time()
            ppl, _ = compute_perplexity(model, tok, chunks)
            elapsed = time.time() - t0
            delta = ppl - ppl_baseline
            mem_k = memory_bytes(kb)
            mem_v = memory_bytes(vb)
            mem_total = mem_k + mem_v
            mark = ""
            if delta < 0.20: mark = "🌟"
            elif delta < 0.50: mark = "🏆"
            elif delta < 2.00: mark = "✓"
            tot = (kb + vb) / 2
            print(f"{name:<55} {kb:<8.2f} {vb:<8.2f} {tot:<8.2f} {mem_total:<12.0f} {ppl:<10.3f} +{delta:<9.3f} {mark}")
            results[name] = {
                "K_bits": kb, "V_bits": vb,
                "K_bytes_per_tok": float(mem_k),
                "V_bytes_per_tok": float(mem_v),
                "total_KV_bytes": float(mem_total),
                "avg_bits_KV": (kb + vb) / 2,
                "ppl": float(ppl), "delta_ppl": float(delta), "time_s": elapsed,
            }
        except Exception as e:
            print(f"{name:<55}: ERROR {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
        finally:
            h.remove()

    # Pareto analysis
    print(f"\n{'='*100}")
    print(f"K+V JOINT PARETO (sorted by total memory)")
    print(f"{'='*100}")
    valid = [(n, r) for n, r in results.items() if "error" not in r]
    valid.sort(key=lambda x: x[1]["total_KV_bytes"])
    print(f"{'name':<55} {'KV mem (B/tok)':<15} {'compress vs FP16':<18} {'Δppl':<10}")
    fp16_mem = memory_bytes(16) * 2
    for name, r in valid:
        compress = fp16_mem / r["total_KV_bytes"] if r["total_KV_bytes"] > 0 else 0
        print(f"{name:<55} {r['total_KV_bytes']:<15.0f} {compress:<18.2f}× {r['delta_ppl']:<10.3f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
