"""
Skill Knowledge Graph — related skills reinforce one another.

Builds a co-occurrence graph from the candidate pool: if two skills
frequently appear together in the same candidate's skill list, they are
considered related. When scoring domain evidence, a candidate's skill
that is related to a JD-relevant skill gets a partial boost.

This mimics how production retrieval systems use knowledge graphs to
propagate relevance through related entities.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

import numpy as np


class SkillGraph:
    """
    Lightweight skill co-occurrence graph.

    Builds adjacency from the full candidate pool, then provides
    `reinforce(skill_names, base_score)` to boost scores when a
    candidate's skill list contains related skills.
    """

    def __init__(self, min_cooccurrence: int = 5):
        self.min_cooccurrence = min_cooccurrence
        self.skill_to_idx: dict[str, int] = {}
        self.idx_to_skill: dict[int, str] = {}
        self.cooccurrence: np.ndarray | None = None
        self.similarity: np.ndarray | None = None
        self._built = False

    def build(self, candidates: list[dict]) -> None:
        """
        Build the co-occurrence graph from the candidate pool.

        Reads the `skills` array from each candidate, normalizes
        skill names, and counts pairwise co-occurrences.
        """
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        skill_set: set[str] = set()

        for cand in candidates:
            skill_names = set()
            for s in cand.get("skills", []):
                name = self._normalize(s.get("name", ""))
                if name:
                    skill_names.add(name)
                    skill_set.add(name)

            for s1 in skill_names:
                for s2 in skill_names:
                    if s1 < s2:
                        pair_counts[(s1, s2)] += 1

        # Filter by min co-occurrence
        filtered_pairs = {
            pair: count
            for pair, count in pair_counts.items()
            if count >= self.min_cooccurrence
        }

        # Build index
        all_skills = sorted(skill_set)
        self.skill_to_idx = {s: i for i, s in enumerate(all_skills)}
        self.idx_to_skill = {i: s for i, s in enumerate(all_skills)}
        n = len(all_skills)

        # Build co-occurrence matrix
        cooc = np.zeros((n, n), dtype=np.float32)
        for (s1, s2), count in filtered_pairs.items():
            i, j = self.skill_to_idx[s1], self.skill_to_idx[s2]
            cooc[i, j] = count
            cooc[j, i] = count

        # Convert to Jaccard similarity: |A ∩ B| / |A ∪ B|
        diag = cooc.diagonal().copy()
        sim = np.zeros_like(cooc)
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim[i, j] = 1.0
                elif cooc[i, j] > 0:
                    union = diag[i] + diag[j] - cooc[i, j]
                    sim[i, j] = cooc[i, j] / max(union, 1.0)

        self.cooccurrence = cooc
        self.similarity = sim
        self._built = True

    def reinforce(self, skill_names: list[str], base_score: float,
                  jd_terms: Optional[list[str]] = None) -> float:
        """
        Boost base_score when candidate skills are related to JD-relevant terms.

        For each candidate skill that has a strong co-occurrence with
        any JD-relevant term, add a proportional boost.

        Args:
            skill_names: List of skill names from the candidate.
            base_score: The domain score before reinforcement.
            jd_terms: JD-relevant terms to reinforce against (e.g. RETRIEVAL_RANKING_TERMS).

        Returns:
            Boosted score in [0, 1].
        """
        if not self._built or not skill_names:
            return base_score

        jd_terms = jd_terms or []
        jd_set = {self._normalize(t) for t in jd_terms}

        boost = 0.0
        seen = set()

        for raw_name in skill_names:
            name = self._normalize(raw_name)
            if not name or name in seen:
                continue
            seen.add(name)

            if name not in self.skill_to_idx:
                continue

            i = self.skill_to_idx[name]
            for jd_term in jd_set:
                if jd_term in self.skill_to_idx:
                    j = self.skill_to_idx[jd_term]
                    sim_val = float(self.similarity[i, j])
                    if sim_val > 0.15:
                        boost = max(boost, sim_val * 0.3)

        if boost > 0:
            return min(1.0, base_score + boost)
        return base_score

    def neighbors(self, skill_name: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get the top-k most similar skills to a given skill."""
        if not self._built:
            return []
        name = self._normalize(skill_name)
        if name not in self.skill_to_idx:
            return []
        i = self.skill_to_idx[name]
        sims = [(self.idx_to_skill[j], float(self.similarity[i, j]))
                for j in range(len(self.idx_to_skill)) if j != i]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a skill name to a canonical form."""
        n = name.lower().strip()
        n = re.sub(r"[^a-z0-9+#_.]", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n
