#!/usr/bin/env python3
"""
Exploratory data analysis used to calibrate the ranker.

Two things this script establishes (and that the design relies on):
  1. Honeypot thresholds: for genuine candidates, (years_of_experience -
     career_span) and (years_of_experience - sum_of_role_durations) are tightly
     bounded, while ~25 honeypots sit in a cleanly-separated tail; and
     zero-duration "expert/advanced" skills are normally 0 but spike for ~21
     honeypots. These gaps are what features.detect_honeypot() keys on.
  2. The role landscape: most of the 100k pool is non-AI (the keyword-stuffer
     trap), so title-based scoring must dominate.

Usage:
    python scripts/eda.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dates import months_between, parse_date  # noqa: E402
from loading import (dataset_reference_date,  # noqa: E402
                                   iter_candidates)


def _pct(a, ps=(50, 90, 99, 99.9, 99.95, 100)):
    a = np.asarray(a, float)
    return {p: round(float(np.percentile(a, p)), 2) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ref = dataset_reference_date(args.candidates)
    print(f"dataset reference date: {ref}\n")

    yoe_span, yoe_sum, zero_expert, titles = [], [], [], Counter()
    n = 0
    for c in iter_candidates(args.candidates):
        n += 1
        p = c["profile"]
        yoe = p.get("years_of_experience", 0) or 0
        hist = c.get("career_history", []) or []
        sk = c.get("skills", []) or []
        titles[p.get("current_title", "?")] += 1
        starts = [parse_date(h.get("start_date")) for h in hist]
        starts = [s for s in starts if s]
        if starts:
            span = (months_between(min(starts), ref) or 0) / 12.0
            yoe_span.append(yoe - span)
        yoe_sum.append(yoe - sum(h.get("duration_months", 0) or 0 for h in hist) / 12.0)
        zero_expert.append(sum(
            1 for s in sk if s.get("proficiency") in ("advanced", "expert")
            and s.get("duration_months") == 0))
        if args.limit and n >= args.limit:
            break

    sp, su, ze = map(np.array, (yoe_span, yoe_sum, zero_expert))
    print(f"N = {n:,}\n")
    print("HONEYPOT SIGNAL 1 — years_of_experience minus career span (years)")
    print("  percentiles:", _pct(sp))
    print(f"  candidates with gap > 1.5y : {int((sp > 1.5).sum())}\n")
    print("HONEYPOT SIGNAL 2 — years_of_experience minus sum of role durations")
    print("  percentiles:", _pct(su))
    print(f"  candidates with gap > 2.0y : {int((su > 2.0).sum())}\n")
    print("HONEYPOT SIGNAL 3 — count of zero-duration advanced/expert skills")
    print("  percentiles:", _pct(ze))
    print(f"  candidates with count >= 3 : {int((ze >= 3).sum())}\n")
    print("TOP 25 current titles in the pool (note how few are AI roles):")
    for t, k in titles.most_common(25):
        print(f"  {k:5}  {t}")


if __name__ == "__main__":
    main()
