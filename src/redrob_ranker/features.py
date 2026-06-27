"""
Per-candidate feature extraction and the structured ("rules") scoring channel.

This is the heart of the ranker. Every sub-score encodes a specific line of the
JD, and the design choices here are what you defend at the Stage-5 interview:

  * role_score reads the TITLE -> an HR Manager with 9 AI skills scores ~0.
  * domain_score reads free-text CAREER DESCRIPTIONS (not the skills array) ->
    keyword-stuffing the skills list earns nothing here.
  * behavioral_modifier down-weights "great on paper, unreachable in practice".
  * detect_honeypot flags internally-impossible profiles (forced to sink).
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Optional

from . import config as C
from .dates import parse_date, months_between


# --- compile domain-term group regexes once -------------------------------
# Short acronyms must match as WHOLE words (so "rag" doesn't fire inside
# "ave-rag-e" / "sto-rag-e"). Everything else is prefix-anchored at a word
# boundary so stems still match ("embedding" -> "embeddings",
# "fine-tun" -> "fine-tuning") without matching mid-word.
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


# Single combined scanner: all domain vocabularies are matched in ONE finditer
# pass per candidate (instead of 6 separate findall calls), then each hit is
# attributed to its group(s) via a term->groups map. This is the hot path for
# 100k candidates, so it is deliberately built once at import time.
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
# longest terms first so phrase matches win over their sub-tokens
_ALL_TERMS = sorted(_TERM_TO_GROUPS, key=len, reverse=True)
_MASTER_RE = re.compile("|".join(_term_regex(t) for t in _ALL_TERMS))


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
        # off-role unless it also literally contains an AI-role phrase
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
        val = C.ROLE_SCORES[cat] * 0.9  # historical discount
        if val > best:
            best, best_cat = val, cat
    return best, best_cat


# --- domain evidence in free text -----------------------------------------
def _domain_score(counts: dict) -> float:
    """Weighted, saturated evidence of real retrieval/ranking/NLP work."""
    raw = sum(counts[g] * w for g, w in C.DOMAIN_GROUP_WEIGHTS.items())
    return 1.0 - math.exp(-raw / C.DOMAIN_SATURATION)


# --- product vs services --------------------------------------------------
def _product_score(career_history):
    if not career_history:
        return 0.5, 1.0  # neutral, unknown
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
    g = math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))  # peak at 7, band ~5-9
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
    recency = _clamp((210 - days_inactive) / 180.0)  # 1 at <=30d, 0 at >=210d

    rr = _num(signals.get("recruiter_response_rate"), 0.0)
    otw = 1.0 if signals.get("open_to_work_flag") else 0.0
    icr = _num(signals.get("interview_completion_rate"), 0.0)
    oar = max(0.0, _num(signals.get("offer_acceptance_rate"), 0.0))
    notice = _num(signals.get("notice_period_days"), 90)
    notice_s = _clamp((120 - notice) / 90.0)  # 1 at <=30d, ~0.3 at 90, 0 at >=120
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
    """
    Flag the dataset's ~80 subtly-impossible "honeypot" profiles.

    Thresholds are calibrated against the full 100k distribution (see
    scripts/eda.py): for genuine candidates these quantities are tightly
    bounded (99.95th pct gap <= 0.5y; zero-duration expert skills == 0), while
    honeypots sit in a cleanly-separated tail. The checks therefore fire with
    essentially zero false positives, which is exactly what we need to stay far
    under the Stage-3 ">10% of top-100" disqualification rule.
    """
    profile = cand.get("profile", {})
    yoe = _num(profile.get("years_of_experience"), 0.0)
    history = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []

    # (1) Claims much more experience than the career timeline allows.
    #     ("8 years of experience at a company founded 3 years ago")
    starts = [parse_date(h.get("start_date")) for h in history]
    starts = [s for s in starts if s]
    if starts:
        span_y = (months_between(min(starts), ref_date) or 0) / 12.0
        if yoe - span_y > 1.5:
            return True, (f"claims {yoe:.1f}y experience but career began only "
                          f"{span_y:.1f}y ago")

    # (2) Stated experience far exceeds the sum of role durations.
    sum_y = sum(h.get("duration_months", 0) or 0 for h in history) / 12.0
    if history and yoe - sum_y > 2.0:
        return True, (f"claims {yoe:.1f}y experience but roles sum to only "
                      f"{sum_y:.1f}y")

    # (3) Multiple 'advanced/expert' skills with zero months of use.
    #     ("'expert' proficiency in 10 skills with 0 years used")
    zero_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months") == 0
    )
    if zero_expert >= 3:
        return True, f"{zero_expert} advanced/expert skills with 0 months of use"

    return False, ""


# --- the full structured fit ----------------------------------------------
def structured_fit(cand, deep_text: str):
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    signals = cand.get("redrob_signals", {})

    history_titles = [h.get("title", "") for h in history]
    role, role_cat = _role_score(profile.get("current_title", ""), history_titles)
    dom_counts, dom_examples = _scan(deep_text)
    domain = _domain_score(dom_counts)
    product, services_frac = _product_score(history)
    exp = _experience_score(_num(profile.get("years_of_experience"), None))
    external = _external_score(signals, deep_text)
    location = _location_score(profile, signals)

    w = C.STRUCTURED_WEIGHTS
    raw = (w["role"] * role + w["domain"] * domain + w["product"] * product +
           w["experience"] * exp + w["external"] * external +
           w["location"] * location)

    # penalties (the JD's explicit "do NOT want")
    penalty = 0.0
    cv_hits = dom_counts["cv"]
    nlp_ir_hits = dom_counts["retrieval_ranking"] + dom_counts["nlp_llm"]
    if cv_hits >= 3 and nlp_ir_hits == 0:
        penalty += 0.30  # CV/speech/robotics primary, no NLP/IR
    if services_frac >= 0.999 and history:
        penalty += 0.20  # entire career at services firms
    if role_cat == "off_role" and domain < 0.30:
        # JD: an off-role title with a stuffed skill list "is not a fit, no
        # matter how perfect their skill list looks". Genuine career-changers
        # (real retrieval/ranking work in their descriptions -> domain >= 0.30)
        # are spared.
        penalty += 0.25

    fit = _clamp(raw - penalty)
    parts = {"role": role, "role_cat": role_cat, "domain": domain,
             "product": product, "experience": exp, "external": external,
             "location": location, "services_frac": services_frac,
             "penalty": penalty, "domain_counts": dom_counts,
             "domain_examples": dom_examples}
    return fit, parts


# --- assemble everything for one candidate --------------------------------
def extract(cand, ref_date: date):
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    signals = cand.get("redrob_signals", {})

    # de-duplicate career descriptions (synthetic data repeats them)
    descs, seen = [], set()
    for h in history:
        d = (h.get("description") or "").strip()
        if d and d not in seen:
            seen.add(d)
            descs.append(d)

    domain_text = " ".join([
        profile.get("headline", ""), profile.get("summary", ""), " ".join(descs)
    ]).lower()

    fit, parts = structured_fit(cand, domain_text)
    modifier, beh = behavioral_modifier(signals, ref_date)
    is_hp, hp_reason = detect_honeypot(cand, ref_date)

    # text for the semantic channel (skills included but diluted in a long doc)
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
        # facts for grounded reasoning
        "title": profile.get("current_title", ""),
        "yoe": _num(profile.get("years_of_experience"), 0.0),
        "current_company": profile.get("current_company", ""),
        "signals": {
            "response_rate": _num(signals.get("recruiter_response_rate"), 0.0),
            "github": signals.get("github_activity_score", -1),
            "notice": _num(signals.get("notice_period_days"), None),
            "open_to_work": bool(signals.get("open_to_work_flag")),
        },
    }


# --- small numeric helpers ------------------------------------------------
def _num(v, default):
    return v if isinstance(v, (int, float)) else default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
