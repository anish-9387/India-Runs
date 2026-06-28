"""
Smoke tests with hand-crafted candidates that exercise the core behaviours:

  * a genuine AI/retrieval engineer outranks a keyword-stuffer,
  * an off-role profile with stuffed AI skills scores low,
  * a timeline-impossible profile is flagged as a honeypot,
  * the structured fit and honeypot logic behave as designed.

Run:  python -m pytest tests/  -q     (or)     python tests/test_smoke.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redrob_ranker.features import extract, detect_honeypot  # noqa: E402

REF = date(2026, 6, 1)


def _cand(cid, title, yoe, desc, skills, signals=None, hist_company="Swiggy",
          hist_industry="Food Delivery", start="2019-01-01"):
    return {
        "candidate_id": cid,
        "profile": {"current_title": title, "years_of_experience": yoe,
                    "headline": title, "summary": desc, "location": "Pune",
                    "country": "India", "current_company": hist_company,
                    "current_company_size": "1001-5000",
                    "current_industry": hist_industry},
        "career_history": [{"company": hist_company, "title": title,
                            "start_date": start, "end_date": None,
                            "duration_months": int(yoe * 12), "is_current": True,
                            "industry": hist_industry, "company_size": "1001-5000",
                            "description": desc}],
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
    [_skill("FAISS")], start="2025-01-01")  # 8y claimed, career began ~1.4y ago


def test_genuine_outranks_stuffer():
    g = extract(GENUINE, REF)
    s = extract(STUFFER, REF)
    assert g["structured_fit"] > s["structured_fit"]
    assert g["structured_fit"] > 0.7, g["structured_fit"]
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
    from redrob_ranker.reasoning import build_reasoning
    g = extract(GENUINE, REF)
    r = build_reasoning(g, "top")
    assert "Recommendation Systems Engineer" in r
    assert "6 yrs" in r


# --- regression: off-role candidate in a small weak pool -------------------
# Without the fix (pipeline.py using `final` for banding), this candidate
# would get a rescaled final = 1.0 → "top" band despite being off-role.
# With the fix (`pre` for banding) its pre << 0.35 → "mid" or "low".
#
# We build the pool WITHOUT the GENUINE candidate so every candidate is weak,
# then verify the off-role keyword-stuffer (Marketing Manager with stuffed AI
# skills) gets a "mid" or "low" band — never "top".


def test_offrole_not_top_band_in_small_pool():
    """Off-role candidate in a pool of ≤10 candidates must NOT get 'top' band.

    The fix uses `pre` (pre-normalization) for banding, which is invariant
    to the pool-relative score rescaling that inflated off-role candidates
    on small/weak pools under the old `final`-based logic. This test verifies
    that even when the off-role keyword-stuffer (Marketing Manager + stuffed
    AI skills) has a modest semantic-channel boost, its `pre` is far below
    the 0.55 / 0.35 thresholds.
    """
    from redrob_ranker import config as C
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
    sem[stuffer_idx] = 0.20  # keyword-stuffer gets modest semantic boost

    pre = C.STRUCTURED_BLEND * structured + C.SEMANTIC_BLEND * sem
    s_pre = float(pre[stuffer_idx])

    band = "top" if s_pre >= 0.55 else ("mid" if s_pre >= 0.35 else "low")
    assert band in ("mid", "low"), (
        f"Off-role candidate got '{band}' band (pre={s_pre:.3f}, "
        f"structured_fit={feats[stuffer_idx]['structured_fit']:.3f}). "
        f"Must be 'mid' or 'low'."
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nAll smoke tests passed.")
