"""Regenerate fig_hero.pdf and fig_sensitivity.pdf from iter38 multi-model data."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "iter38_llama_family.json"
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    data = json.load(f)

# Pretty model labels in order of decreasing boundary-to-middle ratio
order = [
    ("R1-Distill-Llama-8B", "R1-Distill\nLlama-8B"),
    ("Llama-2-7B-Chat", "Llama-2\n7B-Chat"),
    ("Mistral-7B-Instruct", "Mistral-7B\nInstruct-v0.3"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1\n8B-Instruct"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_TQ = "#4C72B0"      # blue
C_SAND = "#C44E52"    # red
C_FP16 = "#999999"


def fig_hero():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})

    # ── LEFT: Δppl at b=4 K=V across 4 models ──
    labels = [lbl for _, lbl in order]
    tq_vals = [data[mid]["TQ_K=V=b4"]["delta_ppl"] for mid, _ in order]
    sand_vals = [data[mid]["Sand+RVQ_K=V_4.0"]["delta_ppl"] for mid, _ in order]
    ratios = [data[mid]["sensitivity_profile"]["ratio"] for mid, _ in order]

    x = np.arange(len(labels))
    w = 0.36
    axL.axhline(0.0, color=C_FP16, lw=0.8, ls="--", zorder=1)
    bL = axL.bar(x - w/2, tq_vals, w, label="TurboQuant uniform $b{=}4$",
                 color=C_TQ, edgecolor="black", lw=0.4)
    bR = axL.bar(x + w/2, sand_vals, w, label="SandwichKV+RVQ $b{=}4$ (ours)",
                 color=C_SAND, edgecolor="black", lw=0.4)

    ymin = min(min(tq_vals), min(sand_vals))
    ymax = max(max(tq_vals), max(sand_vals))
    axL.set_ylim(ymin - 0.45, ymax + 0.85)

    for bars, vals in [(bL, tq_vals), (bR, sand_vals)]:
        for b, v in zip(bars, vals):
            ypos = v + (0.05 if v >= 0 else -0.05)
            va = "bottom" if v >= 0 else "top"
            axL.text(b.get_x() + b.get_width() / 2, ypos, f"{v:+.2f}",
                     ha="center", va=va, fontsize=8, color="black")

    # Annotate boundary-to-middle ratio under each model
    for xi, r in zip(x, ratios):
        axL.text(xi, -0.16, f"ratio = {r:.0f}" if r >= 10 else f"ratio = {r:.1f}",
                 ha="center", va="top", fontsize=8.5, color="#444444",
                 transform=axL.get_xaxis_transform(), clip_on=False)

    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_ylabel(r"$\Delta$ perplexity vs FP16 (lower is better)")
    axL.set_title("(a) Joint K+V at 4 bits/coord — multi-model comparison",
                  loc="left", pad=10, fontsize=11)
    axL.legend(loc="upper right", frameon=False)
    axL.grid(axis="y", alpha=0.25, lw=0.5)
    axL.tick_params(axis="x", pad=24)

    # ── RIGHT: per-layer sensitivity profile, R1-Distill-Llama-8B ──
    sp = data["R1-Distill-Llama-8B"]["sensitivity_profile"]
    layers = sorted(int(k) for k in sp["per_layer_lift"].keys())
    lifts = [sp["per_layer_lift"][str(L)] for L in layers]

    n_layers = data["R1-Distill-Llama-8B"]["n_layers"]
    boundary = [L for L in layers if L < 8 or L >= n_layers - 8]
    colors = [C_SAND if L in boundary else C_TQ for L in layers]

    axR.bar(layers, lifts, color=colors, edgecolor="black", lw=0.3, width=0.85)
    axR.axhline(0, color="black", lw=0.5)

    yhi = max(lifts) * 1.18
    ylo = min(min(lifts) * 1.4, -0.5)
    axR.set_ylim(ylo, yhi)

    axR.axvspan(-0.5, 7.5, color=C_SAND, alpha=0.07, lw=0)
    axR.axvspan(n_layers - 8.5, n_layers - 0.5, color=C_SAND, alpha=0.07, lw=0)

    axR.text(3.5, yhi * 0.93, "outer (b=4)", ha="center", fontsize=9,
             color=C_SAND, fontweight="bold")
    axR.text(n_layers / 2, yhi * 0.93, "middle (b=2 + RVQ)", ha="center",
             fontsize=9, color=C_TQ, fontweight="bold")
    axR.text(n_layers - 4.5, yhi * 0.93, "outer (b=4)", ha="center",
             fontsize=9, color=C_SAND, fontweight="bold")

    axR.set_xlabel("Layer index")
    axR.set_ylabel(r"PPL lift when layer $L$ upgraded $b{=}2 \to b{=}4$")
    ratio_r1 = sp["ratio"]
    axR.set_title(f"(b) R1-Distill-Llama-8B — boundary-to-middle ratio = {ratio_r1:.0f}",
                  loc="left", pad=10, fontsize=11)
    axR.set_xlim(-0.6, n_layers - 0.4)
    axR.grid(axis="y", alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_hero.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_hero.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote fig_hero.{pdf,png}")


def fig_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes = axes.flatten()

    for ax, (mid, lbl) in zip(axes, order):
        sp = data[mid]["sensitivity_profile"]
        n_layers = data[mid]["n_layers"]
        layers = sorted(int(k) for k in sp["per_layer_lift"].keys())
        lifts = [sp["per_layer_lift"][str(L)] for L in layers]
        boundary = [L for L in layers if L < 8 or L >= n_layers - 8]
        colors = [C_SAND if L in boundary else C_TQ for L in layers]
        ax.bar(layers, lifts, color=colors, edgecolor="black", lw=0.3, width=0.85)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvspan(-0.5, 7.5, color=C_SAND, alpha=0.06, lw=0)
        ax.axvspan(n_layers - 8.5, n_layers - 0.5, color=C_SAND, alpha=0.06, lw=0)

        title = lbl.replace("\n", " ")
        ratio = sp["ratio"]
        verdict = sp["verdict"]
        ax.set_title(f"{title}   ratio = {ratio:.1f}  ({verdict})", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    # Common labels
    for ax in axes[2:]:
        ax.set_xlabel("Layer index")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"PPL lift ($b{=}2 \to b{=}4$ at layer $L$)")

    # Custom legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_SAND, edgecolor="black", lw=0.3, label="boundary layer (sandwich-outer)"),
        Patch(facecolor=C_TQ, edgecolor="black", lw=0.3, label="middle layer"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-layer sensitivity profile across the Llama family (WikiText-2)",
                 y=1.06, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_sensitivity.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote fig_sensitivity.{pdf,png}")


if __name__ == "__main__":
    fig_hero()
    fig_sensitivity()
