"""
Semantic / retrieval channel.

Default: a TF-IDF cosine similarity between each candidate's text and a JD
"ideal-profile" query. Fully offline, no model downloads, fast on CPU. This is
what catches "plain-language Tier-5s" who describe building a recommender
without using buzzwords.

Optional: if a precomputed dense-embedding artifact is present (see
scripts/precompute_embeddings.py), its cosine ranking is fused with the TF-IDF
ranking via Reciprocal Rank Fusion (RRF) for extra semantic depth.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config as C


def tfidf_similarity(texts: list[str], query: str) -> np.ndarray:
    """Cosine similarity of every candidate text to the JD query, in [0, 1]."""
    vec = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_df=0.6,
        max_features=200_000, stop_words="english", dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)          # (N, V), L2-normalized rows
    q = vec.transform([query])                 # (1, V), L2-normalized
    sims = linear_kernel(q, matrix).ravel()    # == cosine for normalized tf-idf
    return sims.astype(np.float32)


def rrf_fuse(*rank_arrays: np.ndarray, k: int = C.RRF_K,
             weights: Optional[list[float]] = None) -> np.ndarray:
    """
    Weighted Reciprocal Rank Fusion. Each input is an array of 0-based ranks
    (rank 0 = best) aligned by candidate index. Returns a fused score per
    candidate (higher = better). Scale-free by construction.
    """
    n_lists = len(rank_arrays)
    weights = weights or [1.0] * n_lists
    fused = np.zeros_like(rank_arrays[0], dtype=np.float64)
    for w, ranks in zip(weights, rank_arrays):
        fused += w / (k + ranks.astype(np.float64))
    return fused


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """0-based ranks where the highest score gets rank 0."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def load_dense_artifact(path: Optional[str], candidate_ids: list[str]):
    """
    Load a precomputed dense-embedding artifact if present and aligned with the
    current candidate order. Returns a per-candidate cosine-to-JD array or None.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        ids = list(data["candidate_ids"])
        sims = data["jd_similarity"].astype(np.float32)
    except Exception:
        return None
    if len(ids) != len(candidate_ids):
        return None
    index = {cid: i for i, cid in enumerate(ids)}
    try:
        reorder = np.array([index[cid] for cid in candidate_ids])
    except KeyError:
        return None
    return sims[reorder]


def semantic_channel(texts, candidate_ids, dense_path: Optional[str] = None):
    """
    Produce a normalized [0,1] semantic score per candidate. Uses TF-IDF alone,
    or RRF(TF-IDF, dense) if a dense artifact is available.
    """
    tfidf = tfidf_similarity(texts, C.JD_QUERY_TEXT)
    dense = load_dense_artifact(dense_path, candidate_ids)
    if dense is None:
        return minmax(tfidf), {"channel": "tfidf"}
    fused = rrf_fuse(
        ranks_from_scores(tfidf), ranks_from_scores(dense),
        weights=[1.0, 1.0],
    )
    return minmax(fused.astype(np.float32)), {"channel": "tfidf+dense(RRF)"}
