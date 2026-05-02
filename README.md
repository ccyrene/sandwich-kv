# SandwichKV

> Layer-asymmetric mixed precision + residual vector quantization for KV-cache
> compression in reasoning-distilled LLMs.

This repository releases the code, codebooks, and reproduction scripts behind
the arXiv tech report **"SandwichKV: Layer-Asymmetric Mixed Precision and
Residual Vector Quantization for KV-Cache Compression in Reasoning-Distilled
LLMs"** (paper PDF kept in the companion paper repo).

## What this is

A calibration-free K+V quantization pipeline that extends [TurboQuant]'s
random-rotation Lloyd-Max primitive with two structural levers:

1. **Sandwich layer-precision allocation.** Outer layers (first / last *N*) use
   the higher bit width; the middle layers use the lower bit width.
2. **Residual Vector Quantization (RVQ) in the middle layers.** Multi-pass
   quantization of the rotated residual, using a single shared rotation across
   passes, with per-batch standard-deviation scaling so no calibration data is
   required.

We additionally provide a one-pass per-layer **sensitivity profiler** and a
deterministic **decision rule** that tells you whether SandwichKV or vanilla
TurboQuant uniform Lloyd-Max is the right choice for a given model.

## What this is *not*

- Not a universal win over TurboQuant. The advantage tracks per-layer
  sensitivity structure: SandwichKV dominates on **boundary-dominant** models
  (R1-Distill-Llama-8B, ratio ≈ 786) and ties or loses on flat-sensitivity
  models (Llama-3.1-8B, ratio ≈ 2.7).
- Not production-ready. The current implementation is eager Python via
  `torch` forward hooks; CUDA-kernel work is future.
- Not peer-reviewed. The companion paper is an arXiv tech report.

## Headline numbers (WikiText-2, FP16 baseline)

| Model | FP16 ppl | TQ uniform b=4 | SandwichKV b=4 | L0/L_mid ratio |
|---|---:|---:|---:|---:|
| R1-Distill-Llama-8B | 34.73 | +0.69 | **−0.54** | 786 |
| Llama-2-7B-Chat | 9.34 | +0.16 | **+0.13** | 14 |
| Mistral-7B-Instruct-v0.3 | 8.08 | **+0.02** | +0.05 | 8.4 |
| Llama-3.1-8B-Instruct | 10.10 | **+0.16** | +0.40 | 2.7 |

K and V are quantized symmetrically at the same recipe. Δppl is reported
relative to the FP16 baseline (lower is better; negative = strictly better
than FP16 within evaluation noise).

## Repository layout

```
sandwich-kv/
├── src/sandwichkv/        clean library API (use this for new work)
│   ├── codebooks.py       Lloyd-Max fitter + JSON I/O
│   ├── quantize.py        q_lloyd, q_rvq_same_R, make_rotations
│   ├── schedule.py        sandwich_bm, profile_sensitivity, decision_rule
│   ├── recipes.py         pre-baked layer-bits callbacks
│   └── hook.py            KVHook — forward-hook installer for HF Llama-style models
├── scripts/               thin CLIs over the library
│   ├── build_codebooks.py
│   ├── run_benchmark.py
│   └── profile_sensitivity.py
├── experiments/           frozen iter scripts — reproduce the paper numbers as-is
├── data/                  archived JSON outputs from the iter runs
├── figures/               figures referenced in the paper
└── tests/                 pytest sanity checks
```

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Generate Lloyd-Max codebooks once (head_dim=128 covers the Llama family)
python scripts/build_codebooks.py --d 128 --bits 1 2 3 4 --out codebooks

# 3. Run the benchmark on one model
export HF_TOKEN=hf_...   # only needed for gated repos
python scripts/run_benchmark.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --codebook-dir codebooks \
    --out results/r1_distill.json
```

To decide whether SandwichKV is right for a *new* model:

```bash
python scripts/profile_sensitivity.py \
    --model your-org/your-model \
    --codebook-dir codebooks \
    --out results/profile.json
```

The output prints the boundary-to-middle sensitivity ratio and the recommended
method (`sandwichkv` vs `turboquant_uniform`).

## Library API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sandwichkv import KVHook, sandwich_bm, make_fn_sand_rvq

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    torch_dtype=torch.bfloat16, device_map="cuda:0",
    attn_implementation="eager",
)
n_layers = model.config.num_hidden_layers
head_dim = model.config.hidden_size // model.config.num_attention_heads

bm = sandwich_bm(n_layers, first_n=8, last_n=8, hi=4, lo=2)
k_fn = make_fn_sand_rvq(bm, b_outer=4, bits_per_pass=[1, 1, 1, 1])
v_fn = make_fn_sand_rvq(bm, b_outer=4, bits_per_pass=[1, 1, 1, 1])

with KVHook(head_dim, n_layers, k_fn, v_fn, codebook_dir="codebooks") as h:
    h.install(model)
    # ... run inference, generation, perplexity, etc.
```

## Scope and known limitations

- **Llama-style architectures only.** Tested on Llama-2/3.1, Mistral-7B-v0.3,
  and DeepSeek-R1-Distill-Llama. The Qwen family interacts badly with our
  hook framework (even the TurboQuant baseline degenerates) — likely a
  `k_norm` or RoPE-phasing issue we did not chase down.
- **head_dim = 128.** The shipped codebooks are fitted for d=128; rebuild with
  `scripts/build_codebooks.py --d <other>` for non-128 dimensions.
- **24 GB GPU envelope.** All experiments fit on a single 24 GB device.
  Models above ~9 B were not evaluated.

## Reproducing the paper

The exact scripts that produced each table in the paper live in
[`experiments/`](experiments/), with their JSON outputs in [`data/`](data/).
See [`experiments/README.md`](experiments/README.md) for a per-script index.

## Citation

```bibtex
@misc{sandwichkv2026,
  title  = {SandwichKV: Layer-Asymmetric Mixed Precision and Residual Vector Quantization for KV-Cache Compression in Reasoning-Distilled LLMs},
  author = {Rungrod Thongjampa},
  year   = {2026},
  eprint = {TBD},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

This work builds directly on [TurboQuant][TurboQuant]'s random-rotation +
Lloyd-Max primitive. The sandwich-layer schedule and RVQ extension are our
contribution; the underlying scalar quantizer is theirs.

[TurboQuant]: https://arxiv.org/abs/2504.16131
