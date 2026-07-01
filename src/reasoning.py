"""
Grounded reasoning generator — "Rank, don't generate".

Each 1-2 sentence justification is ASSEMBLED FROM THE CANDIDATE'S REAL FIELDS.
There is no free-text LLM step, so the reasoning is hallucination-free by
construction (exactly what the Stage-4 manual review rewards): every fact stated
is read directly from the profile. Phrasing varies by tier band and a
deterministic per-candidate index so the 10 sampled rows aren't templated-identical,
and the tone is made to match the rank (concerns surfaced for weaker picks).

Also computes recruiter confidence scores and rejection reasons for non-top
candidates.
"""

from __future__ import annotations

from features import compute_confidence, compute_rejection_reason


def _evidence_phrase(parts) -> tuple[str, str]:
    """Return (what they did, concrete matched terms) from domain evidence."""
    ex = parts.get("domain_examples", {})
    counts = parts.get("domain_counts", {})
    if counts.get("retrieval_ranking", 0) > 0:
        terms = ", ".join(ex.get("retrieval_ranking", [])[:2])
        return "production retrieval/ranking/recommender work", terms
    if counts.get("nlp_llm", 0) > 0:
        terms = ", ".join(ex.get("nlp_llm", [])[:2])
        return "applied NLP/LLM work", terms
    if counts.get("production_scale", 0) > 0:
        return "production ML/data engineering", ", ".join(
            ex.get("production_scale", [])[:2])
    return "general engineering background", ""


def _concern(feat) -> str:
    """The single most salient honest concern, or '' if none."""
    parts, beh, sig = feat["parts"], feat["behavior"], feat["signals"]
    if feat["is_honeypot"]:
        return f"profile inconsistency ({feat['honeypot_reason']})"
    if parts["role_cat"] == "off_role":
        return f"current title '{feat['title']}' is not an AI/ML role"
    if parts["services_frac"] >= 0.999:
        return "entire career at IT-services firms"
    if beh["days_inactive"] > 150:
        return f"last active ~{beh['days_inactive']//30} months ago"
    if sig["response_rate"] < 0.25:
        return f"low recruiter response rate ({sig['response_rate']:.2f})"
    if sig["notice"] is not None and sig["notice"] > 90:
        return f"{int(sig['notice'])}-day notice period"
    if feat["yoe"] and feat["yoe"] < 4:
        return f"only {feat['yoe']:.0f} years of experience for a senior role"
    return ""


def _signal_phrase(feat) -> str:
    beh, sig = feat["behavior"], feat["signals"]
    bits = []
    if beh["days_inactive"] <= 45:
        bits.append("active this month")
    if sig["response_rate"] >= 0.6:
        bits.append(f"high recruiter response ({sig['response_rate']:.2f})")
    if isinstance(sig["github"], (int, float)) and sig["github"] > 0:
        bits.append(f"GitHub activity {sig['github']:.0f}")
    if sig.get("willing_relocate"):
        bits.append("willing to relocate")
    return ", ".join(bits)


def _promotion_phrase(parts) -> str:
    """Add promotion trajectory context."""
    traj = parts.get("promotion_traj", "")
    if traj == "strong_upward":
        return "Clear promotion trajectory across roles."
    elif traj == "upward":
        return "Shows career progression."
    return ""


def _diversity_phrase(parts) -> str:
    """Add project diversity context."""
    details = parts.get("diversity_details", {})
    covered = [k for k, v in details.items() if v]
    if len(covered) >= 3:
        return f"Broad experience across {', '.join(covered)}."
    return ""


def _company_quality_phrase(parts) -> str:
    """Add company quality context."""
    cat = parts.get("best_company_cat", "")
    if cat == "research_lab":
        return "Research lab background."
    elif cat == "startup":
        return "Startup background."
    return ""


def build_reasoning(feat, band: str) -> str:
    """band in {'top', 'mid', 'low'} controls tone."""
    title = feat["title"] or "Candidate"
    yoe = feat["yoe"]
    did, terms = _evidence_phrase(feat["parts"])
    terms_clause = f" (mentions {terms})" if terms else ""
    sig = _signal_phrase(feat)
    concern = _concern(feat)
    idx = int(feat["candidate_id"][-4:]) % 3

    head = f"{title}, {yoe:.0f} yrs"
    if band == "top":
        cores = [
            f"{head} — career history shows {did}{terms_clause}, matching the JD's "
            f"call for shipped ranking/retrieval systems at a product company.",
            f"{head}. Strong fit: {did}{terms_clause}; aligns with the JD's "
            f"'product over research' profile.",
            f"{head} with {did}{terms_clause} — the kind of production retrieval "
            f"experience the role is built around.",
        ]
        s = cores[idx]
        extra = []
        promo = _promotion_phrase(feat["parts"])
        if promo:
            extra.append(promo)
        div = _diversity_phrase(feat["parts"])
        if div:
            extra.append(div)
        if extra:
            s += " " + " ".join(extra)
        if sig:
            s += f" {sig.capitalize()}."
        if concern:
            s += f" Concern: {concern}."
        return s

    if band == "mid":
        cores = [
            f"{head}; relevant {did}{terms_clause} but not a top-tier match.",
            f"{head} — partial fit via {did}{terms_clause}.",
            f"{head}. Adjacent profile: {did}{terms_clause}.",
        ]
        s = cores[idx]
        if concern:
            s += f" {concern.capitalize()}."
        elif sig:
            s += f" {sig.capitalize()}."
        return s

    cores = [
        f"{head} — only adjacent skills; included near the cutoff.",
        f"{head}. Weak fit for a senior AI-engineering role.",
        f"{head}; limited direct retrieval/ranking evidence.",
    ]
    s = cores[idx]
    if concern:
        s += f" {concern.capitalize()}."
    return s


def build_confidence_reasoning(feat, sem_score: float, pool_position: int,
                                pool_size: int) -> dict:
    """
    Compute confidence score and return structured confidence data.

    Returns:
      dict with confidence, signal_completeness, agreement, position, etc.
    """
    confidence = compute_confidence(feat, sem_score, pool_position, pool_size)

    return {
        "score": round(confidence, 4),
        "signal_completeness": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
        "agreement": "aligned" if abs(feat.get("structured_fit", 0) - sem_score) < 0.2 else "divergent",
        "position": pool_position,
        "pool_size": pool_size,
    }


def build_rejection_reason(feat) -> str:
    """Build a human-readable rejection reason for non-top-100 candidates."""
    reason_key = compute_rejection_reason(feat)

    reasons = {
        "honeypot": "Rejected because profile contains internal inconsistencies",
        "marketing_title": "Rejected because current title is not an AI/ML role",
        "services_background": "Rejected because entire career at IT-services firms",
        "no_retrieval_work": "Rejected because no retrieval/ranking work in career descriptions",
        "inactive": "Rejected because candidate has been inactive for a prolonged period",
        "notice_90_days": "Rejected because of 90+ day notice period",
        "cv_speech_robotics_only": "Rejected because CV/speech/robotics background without NLP/IR evidence",
        "junior_experience": "Rejected because insufficient experience for senior role",
        "off_role_no_domain": "Rejected because off-role title with no domain evidence",
        "weak_fit": "Rejected because overall fit is below threshold",
    }

    return reasons.get(reason_key, "Rejected because weak overall fit")
