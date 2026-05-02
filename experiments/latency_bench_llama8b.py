"""
Latency benchmark: compare sandwich, VQ, recent-boost vs FP16 + uniform Lloyd-Max baselines.

Measures:
  - Per-token decode latency (single-token autoregressive generation)
  - Prefill throughput (tok/s on 1024-token chunk)
  - Hook overhead per layer

Three settings: prefill, decode (single-token), and pure quantization-time.
"""
import json, time, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT = Path(__file__).parent / "latency_bench.json"


def load_lloyd(d, b):
    with open(Path("/workspace") / f"codebook_d{d}_b{b}.json") as f:
        cb = json.load(f)
    return torch.tensor(cb["centroids"], dtype=torch.float32), torch.tensor(cb["boundaries"], dtype=torch.float32)


def kmeans_fast(X, k, iters=20, n_seeds=3, device="cuda:0"):
    X = X.to(device); N, D = X.shape
    best_cents = None; best_sse = float("inf")
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        idx = torch.randperm(N, device=device)[:k]
        cents = X[idx].clone()
        for _ in range(iters):
            d2 = torch.cdist(X, cents)
            a = d2.argmin(dim=1)
            new_cents = torch.zeros_like(cents)
            counts = torch.zeros(k, device=device)
            new_cents.index_add_(0, a, X)
            counts.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
            mask = counts > 0
            new_cents[mask] = new_cents[mask] / counts[mask].unsqueeze(-1)
            new_cents[~mask] = cents[~mask]
            if (new_cents - cents).abs().max() < 1e-5:
                cents = new_cents; break
            cents = new_cents
        d2 = torch.cdist(X, cents)
        sse = d2.min(dim=1).values.sum().item()
        if sse < best_sse:
            best_sse = sse; best_cents = cents.clone()
    return best_cents


def vq_lookup(X, cents):
    orig = X.shape
    flat = X.reshape(-1, orig[-1])
    d2 = torch.cdist(flat, cents)
    return cents[d2.argmin(dim=1)].reshape(orig)


def q_lloyd(k, bits, R, codebooks):
    cents, bnd = codebooks[bits]
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    idx = torch.searchsorted(bnd, y.contiguous())
    return (cents[idx] @ R) * norms


def q_vq2d(k, R, vq_2d):
    norms = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-12)
    k_unit = k / norms
    y = k_unit @ R.T
    D = y.shape[-1]
    pairs = y.reshape(*y.shape[:-1], D // 2, 2)
    return (vq_lookup(pairs, vq_2d).reshape(*y.shape[:-1], D) @ R) * norms


def make_sandwich_bm(n_layers, first_n, last_n, hi, lo):
    return {i: hi if (i < first_n or i >= n_layers - last_n) else lo for i in range(n_layers)}


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
        self.vq_2d = None
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
    def fn(self, k, R, layer_idx): return q_lloyd(k, b, R, self.codebooks)
    return fn


def make_fn_sandwich(bm):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        return q_lloyd(k, bb, R, self.codebooks)
    return fn


def make_fn_recent_boost(bm, recent_n, recent_b):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        out = q_lloyd(k, bb, R, self.codebooks)
        if bb == 2 and k.shape[1] > recent_n:
            out = out.clone()
            out[:, -recent_n:, :, :] = q_lloyd(k[:, -recent_n:, :, :], recent_b, R, self.codebooks)
        return out
    return fn


def make_fn_vq_middle(bm):
    def fn(self, k, R, layer_idx):
        bb = bm.get(layer_idx, 2)
        if bb >= 16: return k
        if bb == 4: return q_lloyd(k, bb, R, self.codebooks)
        return q_vq2d(k, R, self.vq_2d)
    return fn


def benchmark_prefill(model, tok, text, n_runs=5):
    """Measure full forward pass on 1024-token sequence."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    # Warmup
    with torch.inference_mode():
        for _ in range(2): model(ids)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def benchmark_decode(model, tok, prompt_text, n_new=32, n_runs=3):
    """Measure greedy decode of N new tokens."""
    ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
    with torch.inference_mode():
        for _ in range(2):
            model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times)


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
    full_text = "\n\n".join([t for t in ds["text"][:200] if len(t.strip()) > 50])
    text_long = tok.decode(tok(full_text, return_tensors="pt").input_ids[0][:1024], skip_special_tokens=True)
    text_short = tok.decode(tok(full_text, return_tensors="pt").input_ids[0][:128], skip_special_tokens=True)

    # Pre-fit 2D VQ codebook (need calibration)
    print("[calib] collecting K for 2D VQ fit...")
    layer_K_mid = []
    handles_calib = []
    def make_h(l):
        def h(module, input, output):
            B, T, _ = output.shape
            out = output.reshape(B, T, -1, head_dim).to(torch.float32)
            layer_K_mid.append(out.detach().cpu().reshape(-1, head_dim))
            return None
        return h
    h_calib = model.model.layers[16].self_attn.k_proj.register_forward_hook(make_h(16))
    ids = tok(text_long, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode(): model(ids)
    h_calib.remove()
    K_mid = torch.cat(layer_K_mid, dim=0)
    norms = torch.norm(K_mid, dim=-1, keepdim=True).clamp(min=1e-12)
    unit = (K_mid / norms).to("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(42)
    G = torch.randn(head_dim, head_dim, generator=gen)
    Q, _ = torch.linalg.qr(G)
    R_mid = Q.to("cuda:0").to(torch.float32)
    pairs = (unit @ R_mid.T).reshape(-1, 2)
    print("[calib] fitting vq_2d_64...")
    vq_2d_64 = kmeans_fast(pairs, 64, iters=30)

    sand_88 = make_sandwich_bm(n_layers, 8, 8, 4, 2)

    methods = [
        ("FP16_baseline", None, None, 16.0),
        ("uniform_b2", make_fn_uniform(2), None, 2.0),
        ("uniform_b3", make_fn_uniform(3), None, 3.0),
        ("uniform_b4", make_fn_uniform(4), None, 4.0),
        ("sandwich_8-8_b4_b2", make_fn_sandwich(sand_88), None, 3.0),
        ("sand+recent256_b4", make_fn_recent_boost(sand_88, 256, 4), None, 3.05),
        ("sand+vq2d_64_middle", make_fn_vq_middle(sand_88), vq_2d_64, 3.5),
    ]

    results = {}

    for name, fn, vq, avg_b in methods:
        print(f"\n=== {name} (avg_bits={avg_b}) ===")
        h = None
        if fn is not None:
            h = KVHook(head_dim, n_layers, fn)
            if vq is not None: h.vq_2d = vq
            h.install(model)
        try:
            prefill_t = benchmark_prefill(model, tok, text_long, n_runs=5)
            decode_t = benchmark_decode(model, tok, text_short, n_new=32, n_runs=3)
            prefill_ms = prefill_t.mean() * 1000
            prefill_std = prefill_t.std() * 1000
            decode_total_s = decode_t.mean()
            decode_per_tok_ms = (decode_total_s / 32) * 1000
            tok_per_s = 32 / decode_total_s
            print(f"  prefill 1024-tok: {prefill_ms:.1f} ± {prefill_std:.1f} ms")
            print(f"  decode/token:     {decode_per_tok_ms:.2f} ms ({tok_per_s:.1f} tok/s)")
            results[name] = {
                "avg_bits": avg_b,
                "prefill_ms_1024tok_mean": float(prefill_ms),
                "prefill_ms_1024tok_std": float(prefill_std),
                "decode_total_s_32tok": float(decode_total_s),
                "decode_ms_per_tok": float(decode_per_tok_ms),
                "decode_tok_per_s": float(tok_per_s),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
        finally:
            if h is not None: h.remove()

    # Compute relative overhead vs FP16
    fp_pre = results["FP16_baseline"]["prefill_ms_1024tok_mean"]
    fp_dec = results["FP16_baseline"]["decode_ms_per_tok"]
    print("\n" + "=" * 90)
    print(f"{'method':<35} {'avg_b':<8} {'prefill_ms':<14} {'overhead':<12} {'decode_ms':<12} {'overhead':<10}")
    print("=" * 90)
    for name in [m[0] for m in methods]:
        v = results.get(name)
        if v is None or "error" in v: continue
        pre = v["prefill_ms_1024tok_mean"]
        dec = v["decode_ms_per_tok"]
        pre_ovh = (pre / fp_pre - 1) * 100
        dec_ovh = (dec / fp_dec - 1) * 100
        print(f"  {name:<33} {v['avg_bits']:<8.2f} {pre:<14.1f} {pre_ovh:+.1f}%       {dec:<12.2f} {dec_ovh:+.1f}%")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT.name}")


if __name__ == "__main__":
    main()
