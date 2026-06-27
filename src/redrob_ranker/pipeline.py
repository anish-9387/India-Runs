"""
End-to-end ranking pipeline: candidates.jsonl -> top-100 submission.csv.

  1. reference date  (fast scan for the dataset's "now")
  2. feature extraction  (structured fit + behavioral modifier + honeypot flag)
  3. semantic channel  (TF-IDF cosine to the JD, optionally RRF-fused w/ dense)
  4. fusion            final = (blend of structured + semantic) * behavior, gated
  5. rank + grounded reasoning + spec-compliant CSV
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from . import config as C
from .features import extract
from .loading import dataset_reference_date, iter_candidates
from .reasoning import build_reasoning
from .text_channel import semantic_channel

TOP_N = 100


def _log(msg: str):
    print(f"[redrob-ranker] {msg}", file=sys.stderr, flush=True)


def rank(candidates_path: str, out_path: str, dense_path: str | None = None,
         top_n: int = TOP_N) -> dict:
    t0 = time.time()

    ref_date = dataset_reference_date(candidates_path)
    _log(f"reference date (dataset 'now'): {ref_date}")

    feats = [extract(c, ref_date) for c in iter_candidates(candidates_path)]
    n = len(feats)
    _log(f"extracted features for {n:,} candidates in {time.time()-t0:.1f}s")

    candidate_ids = [f["candidate_id"] for f in feats]
    texts = [f["semantic_text"] for f in feats]

    sem, sem_meta = semantic_channel(texts, candidate_ids, dense_path)
    _log(f"semantic channel: {sem_meta['channel']}")

    structured = np.array([f["structured_fit"] for f in feats], dtype=np.float32)
    modifier = np.array([f["modifier"] for f in feats], dtype=np.float32)
    honeypot = np.array([f["is_honeypot"] for f in feats], dtype=bool)

    pre = C.STRUCTURED_BLEND * structured + C.SEMANTIC_BLEND * sem
    final = pre * modifier
    final = np.where(honeypot, final * C.HONEYPOT_GATE, final)
    final = final / max(float(final.max()), 1e-9)  # normalize to [0, 1]

    # stable order: score desc, then candidate_id asc (matches the validator)
    order = sorted(range(n), key=lambda i: (-float(final[i]), candidate_ids[i]))
    top = order[:top_n]

    rows = []
    prev_score = float("inf")
    for pos, i in enumerate(top, start=1):
        # tone follows the ABSOLUTE fit, not the rank position, so reasoning
        # stays honest when the top-100 contains only weak matches.
        s_i = float(final[i])
        band = "top" if s_i >= 0.55 else ("mid" if s_i >= 0.35 else "low")
        reasoning = build_reasoning(feats[i], band)
        score = round(float(final[i]), 6)
        if score >= prev_score:                 # force strictly decreasing
            score = round(prev_score - 1e-6, 6)
        prev_score = score
        rows.append((candidate_ids[i], pos, score, reasoning))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rnk, score, reasoning in rows:
            w.writerow([cid, rnk, f"{score:.6f}", reasoning])

    stats = {
        "candidates": n,
        "honeypots_flagged": int(honeypot.sum()),
        "honeypots_in_top": int(sum(feats[i]["is_honeypot"] for i in top)),
        "semantic_channel": sem_meta["channel"],
        "elapsed_sec": round(time.time() - t0, 1),
        "out": out_path,
    }
    _log(f"wrote {len(rows)} rows -> {out_path}  ({stats['elapsed_sec']}s, "
         f"{stats['honeypots_flagged']} honeypots flagged, "
         f"{stats['honeypots_in_top']} in top-{top_n})")
    return stats
