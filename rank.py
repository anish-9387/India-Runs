#!/usr/bin/env python3
"""
Single-command entry point for the candidate ranker.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs CPU-only, no network, and well within the 5-minute / 16 GB budget on the
full 100k pool. Supports .jsonl and .jsonl.gz inputs.

New in v2.0:
  - BM25 + TF-IDF + optional dense embeddings fused via RRF
  - Skill knowledge graph for related-skill reinforcement
  - JD parser for generalizable JD adaptation
  - Promotion trajectory detection
  - Company quality scoring (startup / growth / research-lab)
  - Recency-weighted AI experience
  - Project diversity scoring
  - Recruiter confidence scores
  - Rejection reasons for non-top candidates
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Top-100 candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", default="submission.csv",
                    help="Output CSV path (default: submission.csv)")
    ap.add_argument("--dense", default="artifacts/dense_embeddings.npz",
                    help="Optional precomputed dense-embedding artifact "
                         "(used if present; falls back to BM25+TF-IDF otherwise)")
    ap.add_argument("--top-n", type=int, default=100,
                    help="Number of top candidates to output (default: 100)")
    ap.add_argument("--jd", default="",
                    help="Path to a custom JD text file for parsing "
                         "(uses built-in JD if not provided)")
    ap.add_argument("--rejection-out", default="",
                    help="Output path for rejection-reasons JSONL "
                         "(optional; for non-top candidates)")
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit(f"error: candidates file not found: {args.candidates}")

    # Load custom JD text if specified
    jd_text = ""
    if args.jd:
        if not os.path.exists(args.jd):
            sys.exit(f"error: JD file not found: {args.jd}")
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read()

    # Default rejection output path
    rejection_out = args.rejection_out or args.out.replace(".csv", "_rejected.jsonl")

    rank(args.candidates, args.out, dense_path=args.dense, top_n=args.top_n,
         jd_text=jd_text, rejection_out=rejection_out)


if __name__ == "__main__":
    main()
