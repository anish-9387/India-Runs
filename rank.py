#!/usr/bin/env python3
"""
Single-command entry point for the Redrob candidate ranker.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs CPU-only, no network, and well within the 5-minute / 16 GB budget on the
full 100k pool. Supports .jsonl and .jsonl.gz inputs.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from redrob_ranker.pipeline import rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", default="submission.csv",
                    help="Output CSV path (default: submission.csv)")
    ap.add_argument("--dense", default="artifacts/dense_embeddings.npz",
                    help="Optional precomputed dense-embedding artifact "
                         "(used if present; falls back to TF-IDF otherwise)")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit(f"error: candidates file not found: {args.candidates}")

    rank(args.candidates, args.out, dense_path=args.dense, top_n=args.top_n)


if __name__ == "__main__":
    main()
