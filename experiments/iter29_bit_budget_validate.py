"""
Iter 29: HONEST validation + latency + memory accounting.

DOUBLE-CHECK breakthrough numbers:
  - Compute TRUE bit budget (not mislabeled)
  - Re-verify perplexity
  - Measure latency (prefill + decode)
  - Compute memory bytes/token vs FP16 and TurboQuant

KEY ACCOUNTING:
  For sandwich shape sand_F-L (first F + last L outer at b=4, middle at RVQ):
    avg_bits = (F+L) × b_outer + (32 - F - L) × sum(bits_per_pass_middle)
                                 ÷ 32

  For RVQ recipe (passes=[2, 1]): middle stores 3 bits/coord
  For passes=[2, 1, 1]: middle stores 4 bits/coord
  For passes=[1, 1]: middle stores 2 bits/coord

  Norm overhead: 16 bits/token/layer/head extra (separate FP16 storage)

Memory per (token, layer, head) at d=128:
  FP16 K:           128 × 16 = 2048 bits = 256 bytes
  TurboQuant b=3:    128 × 3 + 16 (norm) = 400 bits = 50 bytes (5.12× compress)
  TurboQuant b=4:    128 × 4 + 16        = 528 bits = 66 bytes (3.88× compress)
"""
import json, time, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT = Path(__file__).parent / "perplexity_iter29_validate.json"
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


def q_rvq_fixed(k, R, codebooks, bits_per_pass, R_residual=None):
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
            if R_residual is not None and (i - 1) < len(R_residual) and R_residual[i - 1] is not None:
                R_pass = R_residual[i - 1]
                rotated = cur_residual @ R_pass.T
            else:
                R_pass = None
                rotated = cur_residual
            sd_dims = tuple(range(rotated.dim() - 1))
            res_std = rotated.std(dim=sd_dims, keepdim=True).clamp(min=1e-6)
            scale = res_std / target_std
            scaled = rotated / scale
            idx = torch.searchsorted(bnd, scaled.contiguous())
            q_hat_scaled = cents[idx] * scale
            if R_pass is not None:
                q_hat = q_hat_scaled @ R_pass
            else:
                q_hat = q_hat_scaled
        accumulator = accumulator + q_hat
        cur_residual = cur_residual - q_hat
    return (accumulator @ R) * norms


def make_sandwich_bm(n_layers, first_n, last_n, hi, lo):
    return {i: hi if (i < first_n or i >= n_layers - last_n) else lo for i in range(n_layers)}


def compute_TRUE_avg_bits(n_layers, first_n, last_n, b_outer, bits_per_pass_middle, recent_n=0, recent_b=0, T_avg=600):
    """Compute TRUE avg bits/coord considering RVQ multi-pass + recent boost."""
    n_outer = first_n + last_n
    n_middle = n_layers - n_outer
    middle_base_bits = sum([b for b in bits_per_pass_middle if b < 16])
    if recent_n > 0:
        # Recent N tokens in middle layers upgraded to recent_b
        recent_frac = min(1.0, recent_n / T_avg)
        # Bits per middle layer per token = recent_frac * recent_b + (1 - recent_frac) * middle_base_bits
        avg_middle_bits = recent_frac * recent_b + (1 - recent_frac) * middle_base_bits
    else:
        avg_middle_bits = middle_base_bits
    avg = (n_outer * b_outer + n_middle * avg_middle_bits) / n_layers
    return avg


def memory_bytes(avg_bits_coord, n_layers, head_dim=128, n_heads=8, with_norm=True):
    """Bytes per token per K-cache (across all layers, all KV heads)."""
    norm_bits = 16 if with_norm else 0
    bits_per_layer_per_head = avg_bits_coord * head_dim + norm_bits
    total_bits = bits_per_layer_per_head * n_layers * n_heads
    return total_bits / 8  # bytes


class KVHook:
    def __init__(self, head_dim, n_layers, fn, device="cuda:0"):
        self.head_dim, self.n_layers, self.fn, self.device = head_dim, n_layers, fn, device
        self.codebooks = {}
        for b in [1, 2, 3, 4]:
            cents, bnd = load_lloyd(head_dim, b)
            self.codebooks[b] = (cents.to(device), bnd[1:-1].to(device))
        gen = torch.Generator(device="cpu").manual_seed(42)
        self.R = []
        for _ in range(n_layers):
            G = torch.randn(head_dim, head_dim, generator=gen)
            Q, _ = torch.linalg.qr(G)
            self.R.append(Q.to(device).to(torch.float32))
        self.R_residual = []
        for l in range(n_layers):
            extras = []
            for r_idx in range(4):
                gen2 = torch.Generator(device="cpu").manual_seed(1000 + l * 10 + r_idx)
                G = torch.randn(head_dim, head_dim, generator=gen2)
                Q, _ = torch.linalg.qr(G)
                extras.append(Q.to(device).to(torch.float32))
            self.R_residual.append(extras)
        self.handles = []

    def install(self, model):
        head_dim = self.head_dim
        for l in range(self.n_layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    B, T, _ = output.shape
                    out = output.reshape(B, T, -1, head_dim).to(torch.float32)
                    R = self.R[layer_idx]
                    out_q = self.fn(self, out, R, layer_idx)
                    return out_q.to(output.dtype).reshape(B, T, -1)
                return hook
            h = model.model.layers[l].self_attn.k_proj.register_forward_hook(make_hook(l))
            self.handles.append(h)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def make_fn_uniform(b):
    def fn(self, k, R, layer_idx):
        if b >= 16: return k
        return q_lloyd(k, b, R, self.codebooks)
    return fn


def make_fn_sandwich(bm):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        return q_lloyd(k, bb, R, self.codebooks)
    return fn


def make_fn_rvq_sand(bm, bits_per_pass, use_diff_R=False, recent_n=0, recent_b=0):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        if bb == 4:
            return q_lloyd(k, 4, R, self.codebooks)
        R_res = self.R_residual[layer_idx] if use_diff_R else None
        out = q_rvq_fixed(k, R, self.codebooks, bits_per_pass, R_residual=R_res)
        if recent_n > 0 and k.shape[1] > recent_n:
            out = out.clone()
            out[:, -recent_n:, :, :] = q_lloyd(k[:, -recent_n:, :, :], recent_b, R, self.codebooks)
        return out
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


def benchmark_latency(model, tok, text_prefill, text_decode, n_runs_pre=5, n_runs_dec=3, n_new=32):
    """Measure prefill (1024 tok) and decode (32 new tok) latency."""
    ids_pre = tok(text_prefill, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    ids_dec = tok(text_decode, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
    # Warmup
    with torch.inference_mode():
        for _ in range(2):
            model(ids_pre)
            model.generate(ids_dec, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    pre_times = []
    for _ in range(n_runs_pre):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(ids_pre)
        torch.cuda.synchronize()
        pre_times.append(time.perf_counter() - t0)
    dec_times = []
    for _ in range(n_runs_dec):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(ids_dec, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dec_times.append(time.perf_counter() - t0)
    return {
        "prefill_ms_mean": float(np.mean(pre_times) * 1000),
        "prefill_ms_std": float(np.std(pre_times) * 1000),
        "decode_total_s": float(np.mean(dec_times)),
        "decode_ms_per_tok": float(np.mean(dec_times) / n_new * 1000),
        "decode_tok_per_s": float(n_new / np.mean(dec_times)),
    }


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
    n_kv_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else 8

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    all_ids = tok(full_text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(N_CHUNKS):
        s = i * CHUNK_LEN
        if s + CHUNK_LEN > all_ids.shape[0]: break
        chunks.append(tok.decode(all_ids[s:s+CHUNK_LEN], skip_special_tokens=True))

    text_decode = tok.decode(all_ids[:128], skip_special_tokens=True)

    print("\n=== FP16 baseline ===")
    ppl_baseline, _ = compute_perplexity(model, tok, chunks)
    print(f"  ppl = {ppl_baseline:.4f}")
    fp16_lat = benchmark_latency(model, tok, chunks[0], text_decode)
    print(f"  prefill: {fp16_lat['prefill_ms_mean']:.1f}±{fp16_lat['prefill_ms_std']:.1f} ms, decode: {fp16_lat['decode_ms_per_tok']:.2f} ms/tok")

    sand_88 = make_sandwich_bm(n_layers, 8, 8, 4, 2)
    sand_12_12 = make_sandwich_bm(n_layers, 12, 12, 4, 2)

    # CONFIGS with TRUE bit budget
    configs = [
        # (name, fn, true_avg_bits)
        ("FP16", None, 16.0),
        ("TurboQuant_uniform_b2", make_fn_uniform(2), 2.0),
        ("TurboQuant_uniform_b3", make_fn_uniform(3), 3.0),
        ("TurboQuant_uniform_b4", make_fn_uniform(4), 4.0),
        # Sandwich (no RVQ): bm = {outer:4, middle:2} → avg = 3.0
        ("sand_8-8 (no RVQ)", make_fn_sandwich(sand_88),
         compute_TRUE_avg_bits(32, 8, 8, 4, [2])),  # middle uses scalar b=2 = 1 pass at 2 bits
        # Sandwich + RVQ b=2+b=1 in middle: middle = 3 bits → TRUE avg = 3.5
        ("sand_8-8 + RVQ_b2+b1 middle (TRUE 3.5bit)",
         make_fn_rvq_sand(sand_88, [2, 1]),
         compute_TRUE_avg_bits(32, 8, 8, 4, [2, 1])),
        # Sandwich + RVQ b=2+b=1 with diff-R: same TRUE 3.5
        ("sand_8-8 + RVQ_b2+b1_diffR (TRUE 3.5bit)",
         make_fn_rvq_sand(sand_88, [2, 1], use_diff_R=True),
         compute_TRUE_avg_bits(32, 8, 8, 4, [2, 1])),
        # Sandwich + RVQ_b2+b1+recent256_b4: middle 3 + recent overhead
        ("sand+RVQ_b2+b1+recent256_b4 (TRUE ~3.52bit)",
         make_fn_rvq_sand(sand_88, [2, 1], use_diff_R=True, recent_n=256, recent_b=4),
         compute_TRUE_avg_bits(32, 8, 8, 4, [2, 1], recent_n=256, recent_b=4)),
        # Sandwich + RVQ_b2+b1+b1: middle 4 bits → TRUE 4.0
        ("sand_8-8 + RVQ_b2+b1+b1 middle (TRUE 4.0bit)",
         make_fn_rvq_sand(sand_88, [2, 1, 1]),
         compute_TRUE_avg_bits(32, 8, 8, 4, [2, 1, 1])),
        # Sand 12-12 + RVQ_b2+b1: outer 24 layers b=4, middle 8 layers RVQ-3 → TRUE 3.75
        ("sand_12-12 + RVQ_b2+b1 (TRUE 3.75bit)",
         make_fn_rvq_sand(sand_12_12, [2, 1]),
         compute_TRUE_avg_bits(32, 12, 12, 4, [2, 1])),
        # Sand 8-8 + RVQ_b1+b1 middle: middle 2 bits → TRUE 3.0!
        ("sand_8-8 + RVQ_b1+b1 middle (TRUE 3.0bit)",
         make_fn_rvq_sand(sand_88, [1, 1]),
         compute_TRUE_avg_bits(32, 8, 8, 4, [1, 1])),
        # Sand 8-8 + RVQ_b1+b1_diffR middle: TRUE 3.0
        ("sand_8-8 + RVQ_b1+b1_diffR (TRUE 3.0bit)",
         make_fn_rvq_sand(sand_88, [1, 1], use_diff_R=True),
         compute_TRUE_avg_bits(32, 8, 8, 4, [1, 1])),
    ]

    print(f"\n{'='*100}")
    print(f"VALIDATION: TRUE bit budgets + ppl + latency + memory")
    print(f"{'='*100}")
    print(f"{'name':<55} {'true_bits':<10} {'ppl':<10} {'Δppl':<10} {'pre_ms':<10} {'dec_ms/tok':<12}")
    print(f"{'-'*120}")

    results = {}
    for name, fn, true_bits in configs:
        h = None
        if fn is not None:
            h = KVHook(head_dim, n_layers, fn)
            h.install(model)
        try:
            ppl, _ = compute_perplexity(model, tok, chunks)
            lat = benchmark_latency(model, tok, chunks[0], text_decode)
            delta = ppl - ppl_baseline
            print(f"{name:<55} {true_bits:<10.3f} {ppl:<10.4f} +{delta:<9.4f} {lat['prefill_ms_mean']:<10.1f} {lat['decode_ms_per_tok']:<12.2f}")
            # Memory bytes per token (across all layers and KV heads)
            mem_bytes = memory_bytes(true_bits, n_layers, head_dim, n_kv_heads, with_norm=(true_bits < 16))
            results[name] = {
                "true_bits": float(true_bits),
                "ppl": float(ppl),
                "delta_ppl": float(delta),
                "prefill_ms": lat['prefill_ms_mean'],
                "decode_ms_per_tok": lat['decode_ms_per_tok'],
                "mem_bytes_per_token_K": float(mem_bytes),
            }
        except Exception as e:
            print(f"{name:<55}: ERROR {e}")
            results[name] = {"error": str(e), "true_bits": float(true_bits)}
        finally:
            if h is not None: h.remove()

    # Memory comparison
    print(f"\n{'='*100}")
    print(f"MEMORY COMPRESSION (K-cache only, all layers, all KV heads, per token)")
    print(f"{'='*100}")
    fp16_bytes = results["FP16"]["mem_bytes_per_token_K"]
    tq3_bytes = results["TurboQuant_uniform_b3"]["mem_bytes_per_token_K"]
    tq4_bytes = results["TurboQuant_uniform_b4"]["mem_bytes_per_token_K"]
    print(f"{'name':<55} {'bytes':<10} {'vs_FP16':<12} {'vs_TQ_b3':<12} {'vs_TQ_b4':<12}")
    print(f"{'-'*100}")
    for name in [c[0] for c in configs]:
        r = results.get(name, {})
        if "error" in r: continue
        b = r["mem_bytes_per_token_K"]
        compress_fp = fp16_bytes / b if b > 0 else 0
        compress_tq3 = tq3_bytes / b if b > 0 else 0
        compress_tq4 = tq4_bytes / b if b > 0 else 0
        print(f"{name:<55} {b:<10.0f} {compress_fp:<12.2f}× {compress_tq3:<12.2f}× {compress_tq4:<12.2f}×")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
