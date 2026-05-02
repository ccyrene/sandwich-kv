"""Sanity checks for the quantizer primitives. Runs on CPU, no model needed."""
import math

import torch

from sandwichkv import (
    fit_lloyd_1d,
    make_rotations,
    q_lloyd,
    q_rvq_same_R,
    sandwich_bm,
    decision_rule,
)


def _toy_codebooks(d: int = 16, bits=(1, 2, 3, 4)):
    torch.manual_seed(0)
    x = torch.randn(50_000, d)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    samples = x[:, 0]
    codebooks = {}
    for b in bits:
        cents, bnd = fit_lloyd_1d(samples, 2 ** b)
        codebooks[b] = (cents, bnd[1:-1])
    return codebooks


def test_lloyd_codebook_centroids_are_sorted():
    cents, bnd = fit_lloyd_1d(torch.randn(10_000) * 0.3, 8)
    assert torch.all(cents[1:] >= cents[:-1])
    assert torch.all(bnd[1:] >= bnd[:-1])


def test_q_lloyd_preserves_norm_approximately():
    d = 16
    codebooks = _toy_codebooks(d)
    rots = make_rotations(1, d, seed=7, device="cpu")
    k = torch.randn(2, 5, 4, d) * 3.0
    k_q = q_lloyd(k, bits=4, R=rots[0], codebooks=codebooks)
    n_in = k.norm(dim=-1)
    n_out = k_q.norm(dim=-1)
    rel_err = ((n_in - n_out).abs() / n_in.clamp(min=1e-6)).mean().item()
    assert rel_err < 0.05, rel_err


def test_q_lloyd_higher_bits_lower_error():
    d = 16
    codebooks = _toy_codebooks(d)
    rots = make_rotations(1, d, seed=7, device="cpu")
    k = torch.randn(8, 32, 4, d)
    errors = []
    for b in (1, 2, 3, 4):
        k_q = q_lloyd(k, bits=b, R=rots[0], codebooks=codebooks)
        errors.append((k - k_q).pow(2).mean().item())
    assert errors == sorted(errors, reverse=True), errors


def test_rvq_reduces_error_over_single_pass():
    d = 16
    codebooks = _toy_codebooks(d)
    rots = make_rotations(1, d, seed=11, device="cpu")
    k = torch.randn(4, 16, 4, d)
    one_pass = q_lloyd(k, bits=1, R=rots[0], codebooks=codebooks)
    three_pass = q_rvq_same_R(k, R=rots[0], codebooks=codebooks, bits_per_pass=[1, 1, 1])
    e1 = (k - one_pass).pow(2).mean().item()
    e3 = (k - three_pass).pow(2).mean().item()
    assert e3 < e1, (e1, e3)


def test_sandwich_bm_layout():
    bm = sandwich_bm(n_layers=10, first_n=2, last_n=2, hi=4, lo=2)
    assert bm[0] == 4 and bm[1] == 4
    assert bm[8] == 4 and bm[9] == 4
    assert all(bm[i] == 2 for i in range(2, 8))


def test_decision_rule_picks_sandwichkv_for_boundary_dominant():
    sens = {0: 5.0, 1: 4.5, 5: 0.01, 10: 0.005, 15: 0.02, 30: 0.01, 31: 4.2}
    res = decision_rule(sens, n_layers=32, sample_layers=list(sens.keys()))
    assert res.method == "sandwichkv"
    assert res.verdict == "boundary-dominant"
    assert res.ratio > 100


def test_decision_rule_picks_turboquant_for_flat():
    sens = {i: 0.3 for i in range(32)}
    res = decision_rule(sens, n_layers=32, sample_layers=list(sens.keys()))
    assert res.method == "turboquant_uniform"
    assert res.verdict == "flat"
