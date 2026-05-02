"""Sandwich layer schedule, sensitivity profiling, and the deployment decision rule."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sandwich_bm(n_layers: int, first_n: int, last_n: int, hi: int, lo: int) -> dict[int, int]:
    """Sandwich precision map: outer `first_n + last_n` layers at `hi` bits, middle at `lo`."""
    return {
        i: hi if (i < first_n or i >= n_layers - last_n) else lo
        for i in range(n_layers)
    }


@dataclass
class DecisionResult:
    method: str  # "sandwichkv" or "turboquant_uniform"
    ratio: float
    boundary_lift: float
    middle_lift: float
    verdict: str  # "boundary-dominant" | "moderate" | "flat"


def decision_rule(
    sensitivity: dict[int, float],
    n_layers: int,
    *,
    sample_layers: list[int] | None = None,
    threshold_strong: float = 100.0,
    threshold_weak: float = 10.0,
) -> DecisionResult:
    """Deterministic method selector based on the boundary-to-middle sensitivity ratio.

    `sensitivity[L]` is the perplexity *lift* observed when layer L is upgraded
    from the all-b2 baseline to b=4 (leave-one-out). The ratio of L0/L1 lift to
    middle-layer mean lift predicts whether SandwichKV beats TurboQuant uniform:
      ratio ≳ 100  →  boundary-dominant   →  use SandwichKV
      10 ≲ ratio   →  moderate            →  either method is fine
      ratio < 10   →  flat                →  use TurboQuant uniform Lloyd-Max
    """
    sample = sample_layers or list(sensitivity.keys())
    boundary_lift = max(sensitivity.get(0, 0.0), sensitivity.get(1, 0.0))
    middle_layers = [L for L in sample if n_layers // 4 <= L < 3 * n_layers // 4]
    middle_lift = (
        float(np.mean([sensitivity.get(L, 0.0) for L in middle_layers]))
        if middle_layers
        else 0.0
    )
    ratio = boundary_lift / max(middle_lift, 0.01)
    if ratio >= threshold_strong:
        verdict, method = "boundary-dominant", "sandwichkv"
    elif ratio >= threshold_weak:
        verdict, method = "moderate", "sandwichkv"
    else:
        verdict, method = "flat", "turboquant_uniform"
    return DecisionResult(
        method=method,
        ratio=float(ratio),
        boundary_lift=float(boundary_lift),
        middle_lift=float(middle_lift),
        verdict=verdict,
    )


def profile_sensitivity(
    model,
    tok,
    chunks,
    head_dim: int,
    n_layers: int,
    compute_perplexity_fn,
    make_hook_factory,
    sample_layers: list[int] | None = None,
) -> tuple[dict[int, float], float]:
    """Leave-one-out per-layer sensitivity profile.

    Each layer L is set to b=4 with all others at b=2 (K and V both). The lift
    over the all-b2 baseline measures how much that layer benefits from extra
    precision. `make_hook_factory(bm)` should return an installed-and-removable
    hook object given a layer→bits map.
    """
    bm_all2 = {i: 2 for i in range(n_layers)}
    h = make_hook_factory(bm_all2)
    h.install(model)
    ppl_b2, _ = compute_perplexity_fn(model, tok, chunks)
    h.remove()

    sensitivity: dict[int, float] = {}
    layers = sample_layers if sample_layers is not None else list(range(n_layers))
    for L in layers:
        bm = {i: 2 for i in range(n_layers)}
        bm[L] = 4
        h = make_hook_factory(bm)
        h.install(model)
        try:
            ppl, _ = compute_perplexity_fn(model, tok, chunks)
            sensitivity[L] = ppl_b2 - ppl
        except Exception:
            sensitivity[L] = 0.0
        finally:
            h.remove()
    return sensitivity, ppl_b2
