"""
Semantic / retrieval channel.

Multi-channel sparse + dense retrieval fused via Reciprocal Rank Fusion:

  BM25 (primary sparse)  ─┐
  TF-IDF (secondary)     ─┤── RRF ──► semantic score
  Dense (optional)       ─┘

BM25 consistently outperforms vanilla TF-IDF on long recruiter-style
documents. RRF merges all rankings into a single robust score without
score-calibration issues.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Optional

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# BM25 (Okapi BM25) — standalone implementation, no external deps.
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 ranking model."""

    def __init__(self, k1: float = C.BM25_K1, b: float = C.BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.vocab: set[str] = set()
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit BM25 on a corpus of documents."""
        self.doc_freqs = []
        self.doc_len = []
        df: dict[str, int] = {}
        n = len(texts)

        for text in texts:
            tokens = self._tokenize(text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_len.append(len(tokens))
            for token in freq:
                df[token] = df.get(token, 0) + 1
                self.vocab.add(token)

        self.avgdl = sum(self.doc_len) / max(n, 1)

        # Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            token: math.log((n - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            for token in df
        }
        self._fitted = True

    def transform(self, query: str) -> np.ndarray:
        """Score all documents against the query. Returns per-document scores."""
        if not self._fitted:
            raise RuntimeError("BM25 not fitted yet")

        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.doc_freqs), dtype=np.float64)

        for q in query_tokens:
            if q not in self.idf:
                continue
            idf_q = self.idf[q]
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(q, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf_q * (numerator / denominator)

        return scores.astype(np.float32)

    def fit_transform(self, texts: list[str], query: str) -> np.ndarray:
        """Convenience: fit + transform in one call."""
        self.fit(texts)
        return self.transform(query)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alpha."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9+#]+(?:[-.][a-z0-9+#]+)*", text)
        return [t for t in tokens if len(t) > 1 or t.isdigit()]


# ---------------------------------------------------------------------------
# TF-IDF (existing)
# ---------------------------------------------------------------------------

def tfidf_similarity(texts: list[str], query: str) -> np.ndarray:
    """Cosine similarity of every candidate text to the JD query, in [0, 1]."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_df=0.6,
        max_features=200_000, stop_words="english", dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)
    q = vec.transform([query])
    sims = linear_kernel(q, matrix).ravel()
    return sims.astype(np.float32)


# ---------------------------------------------------------------------------
# RRF utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dense artifact loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main semantic channel
# ---------------------------------------------------------------------------

def semantic_channel(texts, candidate_ids, dense_path: Optional[str] = None):
    """
    Produce a normalized [0,1] semantic score per candidate.

    Pipeline: BM25 (primary) + TF-IDF (secondary) + Dense (optional)
    fused via weighted Reciprocal Rank Fusion.

    Returns (scores, metadata dict).
    """
    # 1. BM25 (primary sparse retriever)
    bm25 = BM25()
    bm25_scores = bm25.fit_transform(texts, C.JD_QUERY_TEXT)

    # 2. TF-IDF (secondary sparse retriever)
    tfidf_scores = tfidf_similarity(texts, C.JD_QUERY_TEXT)

    # 3. Dense (optional)
    dense_scores = load_dense_artifact(dense_path, candidate_ids)

    # 4. RRF fusion
    ranks_list = [ranks_from_scores(bm25_scores), ranks_from_scores(tfidf_scores)]
    weights = [C.BM25_RRF_WEIGHT, C.TFIDF_RRF_WEIGHT]
    channel_name = "bm25+tfidf"

    if dense_scores is not None:
        ranks_list.append(ranks_from_scores(dense_scores))
        weights.append(C.DENSE_RRF_WEIGHT)
        channel_name = "bm25+tfidf+dense(RRF)"

    fused = rrf_fuse(*ranks_list, weights=weights)
    return minmax(fused.astype(np.float32)), {"channel": channel_name}
