#!/usr/bin/env python3
"""
OPTIONAL: precompute a dense-embedding artifact to enrich the semantic channel.

The default ranker runs fully offline on TF-IDF and needs none of this. If you
want extra semantic depth, run this ONCE offline (it may download a small static
model and may exceed the 5-minute ranking window — that is allowed for
pre-computation). It writes artifacts/dense_embeddings.npz, which rank.py picks
up automatically; the timed ranking step only loads the precomputed vectors.

Why model2vec: static embeddings are ~500x faster than transformer encoders on
CPU and only ~30 MB on disk, so the artifact regenerates quickly and the model
bundles into the repo with no network needed at ranking time.

Usage:
    pip install model2vec
    python scripts/precompute_embeddings.py --candidates ./candidates.jsonl \
        --out artifacts/dense_embeddings.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as C  # noqa: E402
from dates import parse_date  # noqa: E402
from loading import (dataset_reference_date,  # noqa: E402
                                   iter_candidates)
from features import extract  # noqa: E402

DEFAULT_MODEL = "minishlab/potion-base-8M"  # static, CPU-friendly, ~30 MB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="artifacts/dense_embeddings.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    try:
        from model2vec import StaticModel
    except ImportError:
        sys.exit("model2vec not installed. Run: pip install model2vec")

    print(f"loading static model: {args.model}")
    model = StaticModel.from_pretrained(args.model)

    ref = dataset_reference_date(args.candidates)
    ids, texts = [], []
    for c in iter_candidates(args.candidates):
        f = extract(c, ref)
        ids.append(f["candidate_id"])
        texts.append(f["semantic_text"])
    print(f"embedding {len(texts):,} candidates + JD query ...")

    cand_vecs = model.encode(texts, show_progress_bar=True).astype(np.float32)
    jd_vec = model.encode([C.JD_QUERY_TEXT]).astype(np.float32)[0]

    # cosine similarity to the JD ideal-profile query
    cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
    jd_sim = (cand_norm @ jd_norm).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, candidate_ids=np.array(ids),
                        jd_similarity=jd_sim)
    print(f"wrote {args.out}  ({len(ids):,} candidates)")


if __name__ == "__main__":
    main()
