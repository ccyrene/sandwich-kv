"""SandwichKV: Layer-Asymmetric Mixed Precision + Residual Vector Quantization.

A calibration-free K+V quantization pipeline that extends TurboQuant's
random-rotation Lloyd-Max primitive with two structural levers:
  (i) sandwich layer-precision allocation (boundary high, middle low)
 (ii) multi-pass Residual Vector Quantization in the middle layers
"""
from .codebooks import fit_lloyd_1d, build_codebooks, load_codebook
from .quantize import q_lloyd, q_rvq_same_R, make_rotations
from .schedule import sandwich_bm, decision_rule, profile_sensitivity
from .hook import KVHook
from .recipes import make_fn_uniform, make_fn_sand_lloyd, make_fn_sand_rvq

__version__ = "0.1.0"
__all__ = [
    "fit_lloyd_1d",
    "build_codebooks",
    "load_codebook",
    "q_lloyd",
    "q_rvq_same_R",
    "make_rotations",
    "sandwich_bm",
    "decision_rule",
    "profile_sensitivity",
    "KVHook",
    "make_fn_uniform",
    "make_fn_sand_lloyd",
    "make_fn_sand_rvq",
]
