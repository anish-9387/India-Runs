"""
End-to-end ranking pipeline: candidates.jsonl -> top-100 submission.csv.

  1. reference date  (fast scan for the dataset's "now")
  2. skill graph     (build co-occurrence graph from the candidate pool)
  3. feature extraction  (structured fit + behavioral modifier + honeypot flag)
  4. semantic channel  (BM25 + TF-IDF + dense, RRF-fused)
  5. fusion            final = (blend of structured + semantic) * behavior, gated
  6. confidence        recruiter confidence per candidate
  7. rank + grounded reasoning + spec-compliant CSV
  8. rejection reasons (stored for non-top candidates for recruiter insight)
"""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

import config as C
from features import extract
from loading import dataset_reference_date, iter_candidates
from reasoning import build_reasoning, build_confidence_reasoning, build_rejection_reason
from skill_graph import SkillGraph
from text_channel import semantic_channel

TOP_N = 100


def _log(msg: str):
    print(f"[ranker] {msg}", file=sys.stderr, flush=True)


def _build_skill_graph(candidates_path: str) -> SkillGraph | None:
    """Build a skill co-occurrence graph from the candidate pool."""
    try:
        pool = list(iter_candidates(candidates_path))
        _log(f"building skill graph from {len(pool):,} candidates...")
        graph = SkillGraph(min_cooccurrence=5)
        graph.build(pool)
        _log(f"skill graph built: {len(graph.skill_to_idx)} unique skills")
        return graph
    except Exception as e:
        _log(f"skill graph build failed (non-fatal): {e}")
        return None


def rank(candidates_path: str, out_path: str, dense_path: str | None = None,
         top_n: int = TOP_N, jd_text: str = "",
         rejection_out: str | None = None) -> dict:
    t0 = time.time()

    ref_date = dataset_reference_date(candidates_path)
    _log(f"reference date (dataset 'now'): {ref_date}")

    # Build skill graph
    skill_graph = _build_skill_graph(candidates_path)

    # Parse JD if provided
    if jd_text:
        from jd_parser import parse_jd
        jd_reqs = parse_jd(jd_text)
        _log(f"parsed JD: {len(jd_reqs['required_skills'])} required, "
             f"{len(jd_reqs['preferred_skills'])} preferred skills")

    # Re-iterate candidates for feature extraction
    pool = list(iter_candidates(candidates_path))
    n = len(pool)
    feats = [extract(c, ref_date, skill_graph) for c in pool]
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
    final = final / max(float(final.max()), 1e-9)

    # stable order: score desc, then candidate_id asc
    order = sorted(range(n), key=lambda i: (-float(final[i]), candidate_ids[i]))
    top = order[:top_n]

    # Confidence scores for all candidates
    confidences = [
        build_confidence_reasoning(feats[i], float(sem[i]), pos, n)
        for pos, i in enumerate(order)
    ]

    rows = []
    prev_score = float("inf")
    for pos, i in enumerate(top, start=1):
        s_i = float(pre[i])
        band = "top" if s_i >= 0.55 else ("mid" if s_i >= 0.35 else "low")
        reasoning = build_reasoning(feats[i], band)
        score = round(float(final[i]), 6)
        if score >= prev_score:
            score = round(prev_score - 1e-6, 6)
        prev_score = score
        rows.append((candidate_ids[i], pos, score, reasoning))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rnk, score, reasoning in rows:
            w.writerow([cid, rnk, f"{score:.6f}", reasoning])

    # Optionally write rejection reasons for non-top candidates
    if rejection_out:
        _write_rejection_reasons(rejection_out, feats, sem, order, top_n, n)

    # Optionally write debug data
    debug_out = out_path.replace(".csv", "_debug.json")
    _write_debug_metadata(debug_out, feats, top, order, top_n, n, confidences, sem_meta)

    stats = {
        "candidates": n,
        "honeypots_flagged": int(honeypot.sum()),
        "honeypots_in_top": int(sum(feats[i]["is_honeypot"] for i in top)),
        "semantic_channel": sem_meta["channel"],
        "elapsed_sec": round(time.time() - t0, 1),
        "out": out_path,
        "skill_graph_nodes": len(skill_graph.skill_to_idx) if skill_graph else 0,
    }
    _log(f"wrote {len(rows)} rows -> {out_path}  ({stats['elapsed_sec']}s, "
         f"{stats['honeypots_flagged']} honeypots flagged, "
         f"{stats['honeypots_in_top']} in top-{top_n})")
    return stats


def _write_rejection_reasons(path: str, feats: list, sem: np.ndarray,
                              order: list, top_n: int, n: int):
    """Write rejection reasons for all candidates below the top-N cutoff."""
    rejected = []
    for pos, i in enumerate(order):
        if pos >= top_n:
            reason = build_rejection_reason(feats[i])
            confidence = build_confidence_reasoning(feats[i], float(sem[i]), pos, n)
            rejected.append({
                "candidate_id": feats[i]["candidate_id"],
                "rank_if_ranked": pos + 1,
                "score": round(float(feats[i]["structured_fit"]), 4),
                "semantic_score": round(float(sem[i]), 4),
                "reason": reason,
                "confidence": confidence["score"],
                "title": feats[i]["title"],
                "yoe": feats[i]["yoe"],
                "current_company": feats[i]["current_company"],
            })

    with open(path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    _log(f"wrote {len(rejected)} rejection reasons -> {path}")


def _write_debug_metadata(path: str, feats: list, top: list, order: list,
                           top_n: int, n: int, confidences: list,
                           sem_meta: dict):
    """Write a debug JSON with confidence scores, weight breakdowns, etc."""
    top_data = []
    for pos, i in enumerate(top, start=1):
        f = feats[i]
        parts = f["parts"]
        top_data.append({
            "candidate_id": f["candidate_id"],
            "rank": pos,
            "confidence": confidences[order.index(i)]["score"] if i in order else 0.5,
            "weight_breakdown": {
                "role": round(parts.get("role", 0), 4),
                "domain": round(parts.get("domain", 0), 4),
                "product": round(parts.get("product", 0), 4),
                "experience": round(parts.get("experience", 0), 4),
                "external": round(parts.get("external", 0), 4),
                "location": round(parts.get("location", 0), 4),
                "promotion": round(parts.get("promotion", 0), 4),
                "diversity": round(parts.get("diversity_score", 0), 4),
                "company_quality": round(parts.get("company_quality", 0), 4),
            },
            "role_category": parts.get("role_cat", ""),
            "domain_counts": parts.get("domain_counts", {}),
            "promotion_trajectory": parts.get("promotion_traj", ""),
            "diversity_coverage": parts.get("diversity_count", 0),
            "company_category": parts.get("best_company_cat", ""),
            "behavioral": {
                "modifier": round(float(f["modifier"]), 4),
                "availability": f["behavior"].get("availability", 0),
            },
            "is_honeypot": f["is_honeypot"],
        })

    debug = {
        "metadata": {
            "total_candidates": n,
            "top_n": top_n,
            "semantic_channel": sem_meta["channel"],
            "weights": {
                "structured_blend": C.STRUCTURED_BLEND,
                "semantic_blend": C.SEMANTIC_BLEND,
                "structured_weights": C.STRUCTURED_WEIGHTS,
            },
        },
        "top_candidates": top_data,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)
    except Exception:
        pass  # debug file is optional
