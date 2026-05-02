# Experiments — frozen scripts, reproduce paper numbers

These scripts are kept as-is, in the form that produced the numbers reported in
the paper. They are intentionally self-contained (each script reimplements the
quantizer inline) so that a reviewer can pin a single file without depending on
the library code in `src/sandwichkv/`.

For new work, use the library API in `src/sandwichkv/` instead — it is the
cleaned-up, deduplicated version of the routines vendored here.

| Script | Reproduces |
|---|---|
| `iter38_llama_family.py` | Multi-model Llama-family table (R1-Distill-Llama-8B, Llama-3.1-8B, Llama-2-7B, Mistral-7B) + per-layer sensitivity profile + decision rule |
| `iter37_small_models.py` | Sub-7B model sweep including R1-Distill-Qwen-1.5B (documents Qwen architecture incompatibility) |
| `iter33_kv_joint.py` | K+V joint quantization at b=4 each on R1-Distill-Llama-8B (Δppl = −0.84) |
| `iter29_bit_budget_validate.py` | True bit-budget accounting validator (caught the 3-bit→3.5-bit mislabeling earlier in the project) |
| `iter35_profile_debug.py` | Debugging-pass per-layer sensitivity profiler used while iterating on the decision rule |
| `latency_bench_llama8b.py` | Latency measurement on Llama-8B (eager Python; CUDA-kernel work is future) |

## Reproducing the headline table (paper §6.2)

```bash
# 1. Generate codebooks once (head_dim=128 for the Llama family)
python ../scripts/build_codebooks.py --d 128 --bits 1 2 3 4 --out /workspace

# 2. Set HF token if you want gated repos
export HF_TOKEN=hf_...

# 3. Run the four-model benchmark
python iter38_llama_family.py
# → /workspace/iter38_llama_family.json
```

The script writes intermediate JSON after each model so partial runs are
recoverable. Full sweep takes ~45 min on a 24 GB GPU.

## Negative results, kept for honesty

The Qwen-family runs in `iter37_small_models.py` all break — even the
TurboQuant baseline returns degenerate perplexity. We hypothesize this is a
hook-framework / Qwen2 architecture interaction (likely `k_norm` or RoPE
phasing). The paper's scope is therefore Llama-derived families only.
