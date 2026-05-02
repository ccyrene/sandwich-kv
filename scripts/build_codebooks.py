"""Generate Lloyd-Max codebooks for one or more head dimensions.

Usage:
    python scripts/build_codebooks.py --d 128 --bits 1 2 3 4 --out codebooks
"""
from __future__ import annotations

import argparse

from sandwichkv import build_codebooks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, nargs="+", default=[128],
                   help="head dimension(s) to fit")
    p.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--out", default="codebooks", help="output directory")
    p.add_argument("--n-samples", type=int, default=500_000)
    args = p.parse_args()
    for d in args.d:
        print(f"=== d={d} ===")
        build_codebooks(d=d, bits=tuple(args.bits), out_dir=args.out,
                        n_samples=args.n_samples)
        print(f"  → wrote codebooks to {args.out}/codebook_d{d}_b{{{','.join(map(str, args.bits))}}}.json")


if __name__ == "__main__":
    main()
