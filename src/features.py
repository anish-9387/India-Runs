"""
Per-candidate feature extraction and the structured ("rules") scoring channel.

This is the heart of the ranker. Every sub-score encodes a specific line of the
JD, and the design choices here are what you defend at the Stage-5 interview:

  * role_score reads the TITLE -> an HR Manager with 9 AI skills scores ~0.
  * domain_score reads free-text CAREER DESCRIPTIONS (not the skills array) ->
    keyword-stuffing the skills list earns nothing here.
  * promotion_trajectory detects Engineer -> Senior -> Lead progression.
  * company_quality_score classifies firms beyond services/product.
  * diversity_score measures breadth across retrieval/ranking/LLM/eval/production.
  * recency_weighted_domain discounts older descriptions.
  * behavioral_modifier down-weights "great on paper, unreachable in practice".
  * detect_honeypot flags internally-impossible profiles (forced to sink).
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Optional

import config as C
from dates import parse_date, months_between


# --- compile domain-term group regexes once -------------------------------
_SHORT_TOKENS = {
    "ltr", "ann", "mrr", "ndcg", "bm25", "bge", "e5", "ctr", "nlp", "llm",
    "rag", "bert", "lora", "qlora", "peft", "rrf", "ner", "sre", "sde", "ros",
    "cv",
}


def _term_regex(t: str) -> str:
    t = t.strip()
    esc = re.escape(t)
    if t in _SHORT_TOKENS:
        return r"\b" + esc + r"\b"
    return r"\b" + esc


_CV_TERMS = C.CV_SPEECH_ROBOTICS_TITLES + [
    "opencv", "object detection", "image classification", "segmentation",
    "lidar", "ros",
]
_DOMAIN_GROUPS = {
    "retrieval_ranking": C.RETRIEVAL_RANKING_TERMS,
    "nlp_llm": C.NLP_LLM_TERMS,
    "production_scale": C.PRODUCTION_SCALE_TERMS,
    "eval": C.EVAL_TERMS,
    "cv": _CV_TERMS,
}
_TERM_TO_GROUPS: dict[str, set] = {}
for _g, _terms in _DOMAIN_GROUPS.items():
    for _t in _terms:
        _TERM_TO_GROUPS.setdefault(_t.strip().lower(), set()).add(_g)
_ALL_TERMS = sorted(_TERM_TO_GROUPS, key=len, reverse=True)
_MASTER_RE = re.compile("|".join(_term_regex(t) for t in _ALL_TERMS))


# --- diversity category regexes (single combined pass) ----------------------
_DIVERSITY_TERM_TO_CAT: dict[str, str] = {}
for _cat, _terms in C.DIVERSITY_CATEGORIES.items():
    for _t in _terms:
        _DIVERSITY_TERM_TO_CAT.setdefault(_t.lower(), _cat)
_ALL_DIV_TERMS = sorted(_DIVERSITY_TERM_TO_CAT, key=len, reverse=True)
_DIVERSITY_RE = re.compile("|".join(r"\b" + re.escape(t) + r"\b"
                                     for t in _ALL_DIV_TERMS), re.IGNORECASE)


def _scan(text: str):
    """One pass: per-group hit counts + a few example phrases per group."""
    counts = {g: 0 for g in _DOMAIN_GROUPS}
    examples: dict[str, list] = {g: [] for g in _DOMAIN_GROUPS}
    for m in _MASTER_RE.finditer(text):
        tok = m.group(0).lower()
        for g in _TERM_TO_GROUPS.get(tok, ()):
            counts[g] += 1
            ex = examples[g]
            if len(ex) < 3 and tok not in ex:
                ex.append(tok)
    return counts, examples


# --- role -----------------------------------------------------------------
def _title_category(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in C.OFF_ROLE_TITLES) and not any(
        k in t for k in C.CORE_AI_TITLES
    ):
        if not any(k in t for k in ("machine learning", "ml ", "ai engineer",
                                    "data scien", "nlp")):
            return "off_role"
    if any(k in t for k in C.CORE_AI_TITLES):
        return "core_ai"
    if any(k in t for k in C.DATA_SCIENCE_TITLES):
        return "data_science"
    if any(k in t for k in C.CV_SPEECH_ROBOTICS_TITLES):
        return "cv_speech_robotics"
    if any(k in t for k in C.ADJACENT_ML_TITLES):
        return "adjacent_ml"
    if any(k in t for k in C.OTHER_ENG_TITLES):
        return "other_eng"
    if any(k in t for k in C.OFF_ROLE_TITLES):
        return "off_role"
    return "unknown"


def _role_score(current_title: str, history_titles) -> tuple[float, str]:
    """Best role category across titles. Current title at full weight,
    historical titles at a small recency discount."""
    cur_cat = _title_category(current_title)
    cur = C.ROLE_SCORES[cur_cat]
    best, best_cat = cur, cur_cat
    for ht in history_titles:
        cat = _title_category(ht)
        val = C.ROLE_SCORES[cat] * 0.9
        if val > best:
            best, best_cat = val, cat
    return best, best_cat


# --- promotion trajectory --------------------------------------------------
def _promotion_score(history) -> tuple[float, str]:
    """
    Detect career progression: Engineer -> Senior -> Lead.

    Single-pass: extracts titles in chronological order while
    checking for level upgrades and promotion hints simultaneously.
    """
    if not history:
        return 0.5, "no_history"

    titled_entries = []
    promo_hints = 0

    for h in history:
        title = (h.get("title") or "").lower()
        start = parse_date(h.get("start_date"))
        if title and start:
            titled_entries.append((start, title))
        desc = (h.get("description") or "").lower()
        if any(k in desc for k in ("promot", "advance", "elevat", "upgrade")):
            promo_hints += 1

    if len(titled_entries) < 2:
        return 0.5, "single_role"

    titled_entries.sort(key=lambda x: x[0])
    titles_only = [t for _, t in titled_entries]

    upgrades = 0
    prev_lvl = _title_level(titles_only[0])
    for curr in titles_only[1:]:
        curr_lvl = _title_level(curr)
        if curr_lvl > prev_lvl:
            upgrades += 1
        prev_lvl = curr_lvl

    total_upgrades = upgrades + promo_hints
    rate = total_upgrades / max(len(titles_only) - 1 + len(history), 1)
    score = min(1.0, 0.5 + rate * 2.0)

    if upgrades >= 2:
        trajectory = "strong_upward"
    elif upgrades >= 1:
        trajectory = "upward"
    elif len(titles_only) >= 3:
        trajectory = "flat"
    else:
        trajectory = "stable"

    return score, trajectory


def _title_level(title: str) -> int:
    """Map a title to a numeric level for progression detection."""
    t = title.lower()
    if any(k in t for k in ("junior", "jr", "trainee", "intern")):
        return 1
    if any(k in t for k in ("principal", "distinguished", "fellow")):
        return 6
    if any(k in t for k in ("staff", "architect")):
        return 5
    if any(k in t for k in ("lead", "manager", "head", "director")):
        return 4
    if any(k in t for k in ("senior", "sr", "ii", "iii", "2", "3")):
        return 3
    if any(k in t for k in ("engineer", "scientist", "developer", "analyst")):
        return 2
    return 2  # default to engineer level


# --- domain evidence in free text -----------------------------------------
def _domain_score(counts: dict) -> float:
    """Weighted, saturated evidence of real retrieval/ranking/NLP work."""
    raw = sum(counts[g] * w for g, w in C.DOMAIN_GROUP_WEIGHTS.items())
    return 1.0 - math.exp(-raw / C.DOMAIN_SATURATION)


# --- recency-weighted domain evidence -------------------------------------
def _recency_weighted_domain(history, base_domain: float) -> float:
    """
    Apply recency decay to domain score based on career timeline.

    Uses career span (earliest start to now) to compute a decay factor.
    No re-scanning needed — just multiplies the base domain score.
    """
    if not history:
        return base_domain

    # Find earliest start date
    earliest = None
    for h in history:
        start = parse_date(h.get("start_date"))
        if start and (earliest is None or start < earliest):
            earliest = start

    if earliest is None:
        return base_domain

    # Career span in years
    now = date(2026, 6, 1)
    span_years = max((now.year - earliest.year) * 12 + (now.month - earliest.month), 1) / 12.0

    # Recency multiplier: newer careers get full weight, older careers decay
    # If they have recent experience (< 2 years), no decay
    latest = None
    for h in history:
        end = parse_date(h.get("end_date")) or now
        if latest is None or end > latest:
            latest = end

    recent_gap = (now - latest).days / 365.0 if latest else 0

    if recent_gap < 2.0:
        return base_domain

    decay = math.exp(-recent_gap / 5.0)
    return base_domain * max(0.5, decay)


# --- project diversity -----------------------------------------------------
def _diversity_score(text: str) -> tuple[float, int, dict]:
    """
    Measure how many distinct project categories the candidate has touched.
    Uses a single regex pass over the text.

    Categories: retrieval, ranking, LLM, evaluation, production.
    Getting >= 3 categories earns a bonus.
    """
    text = text.lower()
    cat_details = {cat: False for cat in C.DIVERSITY_CATEGORIES}

    for m in _DIVERSITY_RE.finditer(text):
        tok = m.group(0).lower()
        cat = _DIVERSITY_TERM_TO_CAT.get(tok)
        if cat:
            cat_details[cat] = True

    covered = sum(1 for v in cat_details.values() if v)
    total_cats = len(C.DIVERSITY_CATEGORIES)
    base = covered / total_cats

    if covered >= C.DIVERSITY_BONUS_THRESHOLD:
        base = min(1.0, base + C.DIVERSITY_BONUS)

    return base, covered, cat_details


# --- product vs services + company quality ---------------------------------
def _company_quality_score(career_history) -> tuple[float, float, str]:
    """
    Classify each company and score based on quality tier.

    Scores: research_lab > startup > growth > product > enterprise > services.
    Uses pre-compiled keyword lookups for speed.
    """
    if not career_history:
        return 0.5, 0.5, "unknown"

    quality_total = 0.0
    services = 0
    total = 0
    cat_count: dict[str, int] = {}

    for h in career_history:
        total += 1
        comp = (h.get("company") or "").lower()
        ind = (h.get("industry") or "").lower()
        combined = f"{comp} {ind}"

        is_services = any(f in comp for f in C.SERVICES_FIRMS)
        if not is_services:
            is_services = "it services" in ind or "consulting" in ind

        if is_services:
            services += 1
            quality_total += C.COMPANY_QUALITY_SCORES["services"]
            cat_count["services"] = cat_count.get("services", 0) + 1
            continue

        category = "product"
        if any(kw in combined for kw in C.RESEARCH_LAB_KEYWORDS):
            category = "research_lab"
        elif any(kw in combined for kw in C.STARTUP_KEYWORDS):
            category = "startup"
        elif any(kw in combined for kw in C.GROWTH_KEYWORDS):
            category = "growth"
        elif "enterprise" in ind or "enterprise" in comp:
            category = "enterprise"

        quality_total += C.COMPANY_QUALITY_SCORES[category]
        cat_count[category] = cat_count.get(category, 0) + 1

    services_frac = services / total if total else 0.0
    avg_quality = quality_total / total if total else 0.5
    best_category = max(cat_count, key=cat_count.get) if cat_count else "unknown"

    return avg_quality, services_frac, best_category


def _product_score(career_history):
    """Legacy product score (kept for backward compat)."""
    if not career_history:
        return 0.5, 1.0
    services = 0
    total = 0
    for h in career_history:
        total += 1
        comp = (h.get("company") or "").lower()
        ind = (h.get("industry") or "").lower()
        is_services = any(f in comp for f in C.SERVICES_FIRMS) or (
            "it services" in ind or "consulting" in ind
        )
        if is_services:
            services += 1
    services_frac = services / total if total else 0.0
    return 1.0 - services_frac, services_frac


# --- experience band ------------------------------------------------------
def _experience_score(yoe: float) -> float:
    if yoe is None:
        return 0.4
    g = math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))
    return max(0.25, g)


# --- external validation --------------------------------------------------
def _external_score(signals, text: str) -> float:
    gh = signals.get("github_activity_score", -1)
    base = min(1.0, gh / 50.0) if isinstance(gh, (int, float)) and gh > 0 else 0.0
    if "open source" in text or "open-source" in text:
        base = min(1.0, base + 0.15)
    return base


# --- location -------------------------------------------------------------
def _location_score(profile, signals) -> float:
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate"))
    in_india = ("india" in country) or any(h in loc for h in C.INDIA_HINTS)
    if any(c in loc for c in C.TOP_LOCATIONS):
        s = 1.0
    elif any(c in loc for c in C.TIER1_INDIA):
        s = 0.8
    elif in_india:
        s = 0.6
    elif relocate:
        s = 0.4
    else:
        s = 0.15
    if relocate:
        s = min(1.0, s + 0.15)
    return s


# --- behavioral availability modifier -------------------------------------
def behavioral_modifier(signals, ref_date: date):
    last = parse_date(signals.get("last_active_date"))
    days_inactive = (ref_date - last).days if last else 365
    recency = _clamp((210 - days_inactive) / 180.0)

    rr = _num(signals.get("recruiter_response_rate"), 0.0)
    otw = 1.0 if signals.get("open_to_work_flag") else 0.0
    icr = _num(signals.get("interview_completion_rate"), 0.0)
    oar = max(0.0, _num(signals.get("offer_acceptance_rate"), 0.0))
    notice = _num(signals.get("notice_period_days"), 90)
    notice_s = _clamp((120 - notice) / 90.0)
    saved = _num(signals.get("saved_by_recruiters_30d"), 0)
    views = _num(signals.get("profile_views_received_30d"), 0)
    engagement = _clamp(math.log1p(saved + views) / math.log1p(60))

    a = (0.30 * recency + 0.25 * rr + 0.15 * otw + 0.10 * icr +
         0.05 * oar + 0.10 * notice_s + 0.05 * engagement)
    modifier = C.BEHAVIOR_MIN + (C.BEHAVIOR_MAX - C.BEHAVIOR_MIN) * a
    return modifier, {"days_inactive": days_inactive, "response_rate": rr,
                      "open_to_work": bool(otw), "notice": notice,
                      "availability": round(a, 3)}


# --- honeypot detection (internal-consistency) ----------------------------
def detect_honeypot(cand, ref_date: date):
    profile = cand.get("profile", {})
    yoe = _num(profile.get("years_of_experience"), 0.0)
    history = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []

    starts = [parse_date(h.get("start_date")) for h in history]
    starts = [s for s in starts if s]
    if starts:
        span_y = (months_between(min(starts), ref_date) or 0) / 12.0
        if yoe - span_y > 1.5:
            return True, (f"claims {yoe:.1f}y experience but career began only "
                          f"{span_y:.1f}y ago")

    sum_y = sum(h.get("duration_months", 0) or 0 for h in history) / 12.0
    if history and yoe - sum_y > 2.0:
        return True, (f"claims {yoe:.1f}y experience but roles sum to only "
                      f"{sum_y:.1f}y")

    zero_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months") == 0
    )
    if zero_expert >= 3:
        return True, f"{zero_expert} advanced/expert skills with 0 months of use"

    return False, ""


# --- rejection reasons for non-top candidates -----------------------------
def compute_rejection_reason(feat) -> str:
    """Determine why a candidate would be rejected (for non-top-100)."""
    parts = feat["parts"]
    beh = feat["behavior"]
    sig = feat["signals"]

    if feat["is_honeypot"]:
        return "honeypot"
    if parts["role_cat"] == "off_role":
        return "marketing_title"
    if parts["services_frac"] >= 0.999 and feat.get("career_history"):
        return "services_background"
    if parts.get("domain", 0) < 0.15:
        return "no_retrieval_work"
    if beh["days_inactive"] > 200:
        return "inactive"
    if sig.get("notice_period") and sig["notice_period"] > 90:
        return "notice_90_days"
    if parts["role_cat"] == "cv_speech_robotics" and parts.get("domain", 0) < 0.2:
        return "cv_speech_robotics_only"
    if feat.get("yoe", 0) < 3:
        return "junior_experience"
    if parts["role_cat"] == "off_role" and parts.get("domain", 0) < 0.3:
        return "off_role_no_domain"
    return "weak_fit"


# --- recruiter confidence score -------------------------------------------
def compute_confidence(feat, sem_score: float, pool_position: int,
                       pool_size: int) -> float:
    """
    Compute recruiter confidence in the score.

    Factors:
      - Signal completeness (how many recruiter signals are available)
      - Semantic/structured agreement (coherence between channels)
      - Pool position (where they sit relative to neighbors)
      - Behavioral certainty (how many behavioral signals exist)
    """
    # Signal completeness: how many of the 23 signals are non-null
    signals = feat.get("signals", {})
    signal_count = sum(1 for v in signals.values() if v is not None and v != -1)
    signal_total = max(len(signals), 1)
    completeness = signal_count / signal_total

    # Semantic-structured agreement
    structured = feat.get("structured_fit", 0)
    agreement = 1.0 - abs(structured - sem_score)

    # Pool position: top 1% -> high confidence, bottom -> low
    position = 1.0 - (pool_position / max(pool_size, 1))

    # Behavioral certainty
    beh = feat.get("behavior", {})
    beh_keys = ["days_inactive", "response_rate", "open_to_work", "notice"]
    beh_present = sum(1 for k in beh_keys if k in beh)
    behav_certainty = beh_present / len(beh_keys)

    w = C.CONFIDENCE_WEIGHTS
    confidence = (w["signal_completeness"] * completeness +
                  w["semantic_structured_agreement"] * agreement +
                  w["pool_position"] * position +
                  w["behavioral_certainty"] * behav_certainty)

    return _clamp(confidence)


# --- skill graph reinforcement -------------------------------------------
def apply_skill_graph_boost(cand_skills: list[dict], domain_score: float,
                            skill_graph) -> float:
    """Apply knowledge-graph-based reinforcement to domain score."""
    if skill_graph is None:
        return domain_score
    skill_names = [s.get("name", "") for s in cand_skills if s.get("name")]
    jd_terms = (C.RETRIEVAL_RANKING_TERMS + C.NLP_LLM_TERMS +
                C.PRODUCTION_SCALE_TERMS + C.EVAL_TERMS)
    return skill_graph.reinforce(skill_names, domain_score, jd_terms)


# --- the full structured fit ----------------------------------------------
def structured_fit(cand, deep_text: str, skill_graph=None):
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    signals = cand.get("redrob_signals", {})

    history_titles = [h.get("title", "") for h in history]
    role, role_cat = _role_score(profile.get("current_title", ""), history_titles)
    dom_counts, dom_examples = _scan(deep_text)

    # Base domain score with recency weighting
    domain = _domain_score(dom_counts)
    domain = _recency_weighted_domain(history, domain)

    # Skill graph reinforcement
    cand_skills = cand.get("skills", [])
    domain = apply_skill_graph_boost(cand_skills, domain, skill_graph)

    product, services_frac = _product_score(history)
    company_quality, _, best_company_cat = _company_quality_score(history)
    exp = _experience_score(_num(profile.get("years_of_experience"), None))
    external = _external_score(signals, deep_text)
    location = _location_score(profile, signals)
    promotion, promotion_traj = _promotion_score(history)

    # Project diversity
    div_score, div_count, div_details = _diversity_score(deep_text)

    w = C.STRUCTURED_WEIGHTS
    raw = (w["role"] * role + w["domain"] * domain +
           w["product"] * product +
           w["experience"] * exp + w["external"] * external +
           w["location"] * location +
           w["promotion"] * promotion +
           w["diversity"] * div_score +
           w["company_quality"] * company_quality)

    # penalties (the JD's explicit "do NOT want")
    penalty = 0.0
    cv_hits = dom_counts["cv"]
    nlp_ir_hits = dom_counts["retrieval_ranking"] + dom_counts["nlp_llm"]
    if cv_hits >= 3 and nlp_ir_hits == 0:
        penalty += 0.30
    if services_frac >= 0.999 and history:
        penalty += 0.20
    if role_cat == "off_role" and domain < 0.30:
        penalty += 0.25

    fit = _clamp(raw - penalty)
    parts = {"role": role, "role_cat": role_cat, "domain": domain,
             "product": product, "experience": exp, "external": external,
             "location": location, "services_frac": services_frac,
             "penalty": penalty, "domain_counts": dom_counts,
             "domain_examples": dom_examples,
             "promotion": promotion, "promotion_traj": promotion_traj,
             "diversity_score": div_score, "diversity_count": div_count,
             "diversity_details": div_details,
             "company_quality": company_quality,
             "best_company_cat": best_company_cat}
    return fit, parts


# --- assemble everything for one candidate --------------------------------
def extract(cand, ref_date: date, skill_graph=None):
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    signals = cand.get("redrob_signals", {})

    descs, seen = [], set()
    for h in history:
        d = (h.get("description") or "").strip()
        if d and d not in seen:
            seen.add(d)
            descs.append(d)

    domain_text = " ".join([
        profile.get("headline", ""), profile.get("summary", ""), " ".join(descs)
    ]).lower()

    fit, parts = structured_fit(cand, domain_text, skill_graph)
    modifier, beh = behavioral_modifier(signals, ref_date)
    is_hp, hp_reason = detect_honeypot(cand, ref_date)

    skill_names = " ".join(s.get("name", "") for s in cand.get("skills", []))
    semantic_text = " ".join([
        profile.get("headline", ""), profile.get("current_title", ""),
        profile.get("summary", ""), " ".join(descs), skill_names,
    ])

    return {
        "candidate_id": cand["candidate_id"],
        "structured_fit": fit,
        "modifier": modifier,
        "is_honeypot": is_hp,
        "honeypot_reason": hp_reason,
        "semantic_text": semantic_text,
        "parts": parts,
        "behavior": beh,
        "title": profile.get("current_title", ""),
        "yoe": _num(profile.get("years_of_experience"), 0.0),
        "current_company": profile.get("current_company", ""),
        "signals": {
            "response_rate": _num(signals.get("recruiter_response_rate"), 0.0),
            "github": signals.get("github_activity_score", -1),
            "notice": _num(signals.get("notice_period_days"), None),
            "notice_period": _num(signals.get("notice_period_days"), None),
            "open_to_work": bool(signals.get("open_to_work_flag")),
            "views": _num(signals.get("profile_views_received_30d"), 0),
            "saved": _num(signals.get("saved_by_recruiters_30d"), 0),
            "completeness": _num(signals.get("profile_completeness_score"), 50),
            "willing_relocate": bool(signals.get("willing_to_relocate")),
        },
        "career_history": history,
    }


# --- small numeric helpers ------------------------------------------------
def _num(v, default):
    return v if isinstance(v, (int, float)) else default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
