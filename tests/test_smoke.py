"""
Smoke tests with hand-crafted candidates that exercise the core behaviours:

  * a genuine AI/retrieval engineer outranks a keyword-stuffer,
  * an off-role profile with stuffed AI skills scores low,
  * a timeline-impossible profile is flagged as a honeypot,
  * promotion trajectory detection,
  * skill graph reinforcement,
  * diversity scoring,
  * BM25 semantic channel behaviour.

Run:  python -m pytest tests/  -q     (or)     python tests/test_smoke.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import extract, detect_honeypot  # noqa: E402
from text_channel import BM25  # noqa: E402
from skill_graph import SkillGraph  # noqa: E402

REF = date(2026, 6, 1)


def _cand(cid, title, yoe, desc, skills, signals=None, hist_company="Swiggy",
          hist_industry="Food Delivery", start="2019-01-01", history=None):
    if history is None:
        history = [{"company": hist_company, "title": title,
                    "start_date": start, "end_date": None,
                    "duration_months": int(yoe * 12), "is_current": True,
                    "industry": hist_industry, "company_size": "1001-5000",
                    "description": desc}]
    return {
        "candidate_id": cid,
        "profile": {"current_title": title, "years_of_experience": yoe,
                    "headline": title, "summary": desc, "location": "Pune",
                    "country": "India", "current_company": hist_company,
                    "current_company_size": "1001-5000",
                    "current_industry": hist_industry},
        "career_history": history,
        "education": [], "skills": skills,
        "redrob_signals": _signals(signals),
    }


def _signals(overrides=None):
    s = {"profile_completeness_score": 90, "signup_date": "2024-01-01",
         "last_active_date": "2026-05-25", "open_to_work_flag": True,
         "profile_views_received_30d": 20, "applications_submitted_30d": 3,
         "recruiter_response_rate": 0.8, "avg_response_time_hours": 5,
         "skill_assessment_scores": {}, "connection_count": 100,
         "endorsements_received": 50, "notice_period_days": 30,
         "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
         "preferred_work_mode": "hybrid", "willing_to_relocate": True,
         "github_activity_score": 40, "search_appearance_30d": 30,
         "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.8,
         "offer_acceptance_rate": 0.5, "verified_email": True,
         "verified_phone": True, "linkedin_connected": True}
    if overrides:
        s.update(overrides)
    return s


def _skill(name, prof="advanced", dm=24):
    return {"name": name, "proficiency": prof, "endorsements": 10,
            "duration_months": dm}


# --- fixed candidates ---
GENUINE = _cand(
    "CAND_0000001", "Recommendation Systems Engineer", 6.0,
    "Built and shipped learning-to-rank models and embedding-based retrieval "
    "for our recommendation and search product in production at scale; owned "
    "offline-online evaluation with NDCG and A/B testing.",
    [_skill("FAISS"), _skill("Embeddings"), _skill("Learning to Rank")])

STUFFER = _cand(
    "CAND_0000002", "Marketing Manager", 7.0,
    "Led marketing campaigns, brand strategy, and content calendars for our "
    "consumer brand; managed budgets and agency relationships.",
    [_skill("RAG"), _skill("Pinecone"), _skill("LLM"), _skill("Embeddings"),
     _skill("Vector Database")])

HONEYPOT = _cand(
    "CAND_0000003", "Machine Learning Engineer", 8.0,
    "Built ranking and retrieval systems in production.",
    [_skill("FAISS")], start="2025-01-01")

PROMOTED = _cand(
    "CAND_0000004", "Lead ML Engineer", 8.0,
    "Designed and deployed large-scale ML ranking systems in production.",
    [_skill("FAISS"), _skill("PyTorch")],
    history=[
        {"company": "TechCo", "title": "ML Engineer",
         "start_date": "2018-01-01", "end_date": "2020-06-01",
         "duration_months": 29, "is_current": False,
         "description": "Built ML models for ranking."},
        {"company": "TechCo", "title": "Senior ML Engineer",
         "start_date": "2020-06-01", "end_date": "2023-01-01",
         "duration_months": 31, "is_current": False,
         "description": "Led team building retrieval systems."},
        {"company": "TechCo", "title": "Lead ML Engineer",
         "start_date": "2023-01-01", "end_date": None,
         "duration_months": 41, "is_current": True,
         "description": "Lead ML ranking systems at scale."},
    ])

DIVERSE = _cand(
    "CAND_0000005", "Senior ML Engineer", 7.0,
    "Built retrieval systems for search ranking, fine-tuned LLMs for RAG, "
    "ran offline evaluation with NDCG and A/B testing, and deployed ranking "
    "models to production at scale handling millions of requests.",
    [_skill("FAISS"), _skill("PyTorch"), _skill("Elasticsearch")])


# =====================================================================
# Tests
# =====================================================================

def test_genuine_outranks_stuffer():
    g = extract(GENUINE, REF)
    s = extract(STUFFER, REF)
    assert g["structured_fit"] > s["structured_fit"]
    assert g["structured_fit"] > 0.6, g["structured_fit"]
    assert s["structured_fit"] < 0.25, s["structured_fit"]


def test_offrole_role_score_zero():
    s = extract(STUFFER, REF)
    assert s["parts"]["role_cat"] == "off_role"
    assert s["parts"]["role"] == 0.0


def test_honeypot_flagged():
    is_hp, reason = detect_honeypot(HONEYPOT, REF)
    assert is_hp, "timeline-impossible profile should be flagged"
    assert "career began" in reason


def test_genuine_not_honeypot():
    is_hp, _ = detect_honeypot(GENUINE, REF)
    assert not is_hp


def test_reasoning_is_grounded():
    from reasoning import build_reasoning
    g = extract(GENUINE, REF)
    r = build_reasoning(g, "top")
    assert "Recommendation Systems Engineer" in r
    assert "6 yrs" in r


def test_offrole_not_top_band_in_small_pool():
    import config as C
    import numpy as np

    pool = [
        STUFFER,
        _cand("CAND_0010", "Junior Clerk", 2.0, "Clerical work.",
              [_skill("Excel")], hist_company="OfficeCo", hist_industry="Admin"),
        _cand("CAND_0011", "Accountant", 3.0, "Bookkeeping.",
              [_skill("QuickBooks")], hist_company="FinCorp", hist_industry="Finance"),
        _cand("CAND_0012", "Driver", 4.0, "Delivery driving.",
              [_skill("Navigation")], hist_company="LogiCo", hist_industry="Transport"),
    ]
    feats = [extract(c, REF) for c in pool]
    n = len(feats)

    structured = np.array([f["structured_fit"] for f in feats], dtype=np.float32)
    modifier = np.array([f["modifier"] for f in feats], dtype=np.float32)

    sem = np.zeros(n, dtype=np.float32)
    stuffer_idx = next(
        i for i, f in enumerate(feats) if f["parts"]["role_cat"] == "off_role"
    )
    sem[stuffer_idx] = 0.20

    pre = C.STRUCTURED_BLEND * structured + C.SEMANTIC_BLEND * sem
    s_pre = float(pre[stuffer_idx])

    band = "top" if s_pre >= 0.55 else ("mid" if s_pre >= 0.35 else "low")
    assert band in ("mid", "low"), (
        f"Off-role candidate got '{band}' band (pre={s_pre:.3f}, "
        f"structured_fit={feats[stuffer_idx]['structured_fit']:.3f})."
    )


def test_bm25_basic():
    """BM25 should produce reasonable scores."""
    bm25 = BM25()
    texts = [
        "senior ai engineer building retrieval systems",
        "marketing manager running campaigns",
        "search engineer working on ranking and relevance",
    ]
    scores = bm25.fit_transform(texts, "ai retrieval ranking engineer")
    assert len(scores) == 3
    assert scores[1] < scores[0]  # BM25 should rank AI engineer > marketing


def test_bm25_ranks_retrieval_higher():
    """BM25 should score retrieval/ranking text higher than generic text."""
    bm25 = BM25()
    texts = [
        "I built retrieval and ranking systems with FAISS and Elasticsearch",
        "I managed marketing campaigns and social media",
        "I wrote Python scripts for data analysis",
    ]
    scores = bm25.fit_transform(texts, "retrieval ranking faiss elasticsearch")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_promotion_trajectory():
    """Promoted candidates should have higher promotion scores."""
    p = extract(PROMOTED, REF)
    assert p["parts"]["promotion_traj"] in ("strong_upward", "upward")
    assert p["parts"]["promotion"] >= 0.7


def test_genuine_has_domain_evidence():
    """Genuine candidate should have strong domain evidence."""
    g = extract(GENUINE, REF)
    assert g["parts"]["domain"] > 0.3
    assert g["parts"]["domain_counts"]["retrieval_ranking"] > 0


def test_diversity_scoring():
    """Diverse candidates should get diversity bonus."""
    d = extract(DIVERSE, REF)
    assert d["parts"]["diversity_count"] >= 3
    assert d["parts"]["diversity_score"] >= 0.6


def test_skill_graph_basic():
    """Skill graph should build and provide neighbors."""
    candidates = [GENUINE, STUFFER, HONEYPOT, PROMOTED, DIVERSE]
    graph = SkillGraph(min_cooccurrence=1)
    graph.build(candidates)
    assert len(graph.skill_to_idx) > 0
    assert graph._built

    # FAISS should have neighbors from co-occurrence
    neighbors = graph.neighbors("FAISS")
    assert len(neighbors) > 0


def test_confidence_score():
    """Confidence should be in [0, 1] range."""
    from reasoning import build_confidence_reasoning
    g = extract(GENUINE, REF)
    c = build_confidence_reasoning(g, 0.8, 0, 100)
    assert 0 <= c["score"] <= 1.0


def test_rejection_reason():
    """Stuffer should get a rejection reason."""
    from reasoning import build_rejection_reason
    s = extract(STUFFER, REF)
    reason = build_rejection_reason(s)
    assert reason is not None
    assert len(reason) > 0


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nAll smoke tests passed.")
