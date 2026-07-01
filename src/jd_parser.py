"""
JD Parser — extracts structured requirements from any JD text.

Turns a raw job description into machine-readable fields:
  - Required skills         → must-match list for semantic scoring
  - Preferred skills        → bonus matches for semantic scoring
  - Disqualifiers           → title/skills that auto-reject
  - Preferred locations     → geo preferences
  - Experience range        → min/ideal/max years
  - Behavioral traits       → recruiter preference signals

Makes the pipeline adaptable to ANY JD instead of only the hard-coded one.
"""

from __future__ import annotations

import re
from typing import Any

import config as C


def parse_jd(jd_text: str = "") -> dict[str, Any]:
    """
    Parse JD text into structured requirements. If jd_text is empty,
    falls back to the default JD_REQUIREMENTS in config.py (the
    "Senior AI Engineer — Founding Team" role).

    Returns a dict with keys:
      required_skills, preferred_skills, disqualifiers,
      preferred_locations, tier1_locations,
      experience_min, experience_ideal, experience_max,
      behavioral_traits
    """
    if not jd_text:
        return dict(C.JD_REQUIREMENTS)

    text = jd_text.lower()

    reqs = {
        "required_skills": _extract_skills(text, "required"),
        "preferred_skills": _extract_skills(text, "preferred"),
        "disqualifiers": _extract_disqualifiers(text),
        "preferred_locations": _extract_locations(text),
        "tier1_locations": set(C.JD_REQUIREMENTS["tier1_locations"]),
        "experience_min": _extract_experience(text, "min"),
        "experience_ideal": _extract_experience(text, "ideal"),
        "experience_max": _extract_experience(text, "max"),
        "behavioral_traits": _extract_traits(text),
    }

    # Merge with defaults for any missing fields
    for k, default_val in C.JD_REQUIREMENTS.items():
        if k not in reqs or not reqs[k]:
            reqs[k] = default_val

    return reqs


def _extract_skills(text: str, kind: str) -> list[str]:
    """Extract skill mentions from JD text."""
    skills = set()

    # Common tech skill patterns
    tech_patterns = [
        r"(?:^|\s)(python|java|scala|go|rust|c\+\+|typescript|javascript)\b",
        r"(?:^|\s)(pytorch|tensorflow|jax|keras|scikit-learn|sklearn)\b",
        r"(?:^|\s)(faiss|pinecone|weaviate|qdrant|milvus|elasticsearch)\b",
        r"(?:^|\s)(spark|hadoop|flink|kafka|airflow|docker|kubernetes)\b",
        r"(?:^|\s)(rag|llm|bert|transformer|attention|embedding)\b",
        r"(?:^|\s)(ndcg|mrr|map|recall|precision|auc)\b",
        r"(?:^|\s)(sql|nosql|postgres|redis|mongodb|neo4j)\b",
        r"(?:^|\s)(aws|gcp|azure|mlflow|wandb)\b",
    ]

    for pat in tech_patterns:
        for m in re.finditer(pat, text):
            skills.add(m.group(1).lower())

    return list(skills)


def _extract_disqualifiers(text: str) -> list[str]:
    """Extract explicit 'do not want' signals from JD text."""
    disqualifiers = []

    patterns = [
        (r"(?:not|no|without)\s+(?:a\s+)?(?:degree|bachelor|master|phd)", "no_degree"),
        (r"(?:not|no)\s+(?:prior\s+)?experience\s+(?:in\s+)?(cv|computer vision|frontend|ui|design)", "wrong_background"),
    ]

    for pat, tag in patterns:
        if re.search(pat, text):
            disqualifiers.append(tag)

    # Always include the default disqualifiers from config
    disqualifiers.extend(C.JD_REQUIREMENTS["disqualifiers"])

    return list(set(disqualifiers))


def _extract_locations(text: str) -> set[str]:
    """Extract preferred locations from JD text."""
    locations = set()
    india_cities = {
        "pune", "noida", "bangalore", "bengaluru", "hyderabad", "mumbai",
        "delhi", "gurgaon", "gurugram", "chennai", "kolkata", "ahmedabad",
    }

    for city in india_cities:
        if city in text:
            locations.add(city)

    if not locations:
        return set(C.JD_REQUIREMENTS["preferred_locations"])

    return locations


def _extract_experience(text: str, kind: str) -> int:
    """Extract experience requirements."""
    # Patterns like "5+ years", "5-8 years", "at least 5 years"
    ranges = re.findall(r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years|yrs)", text)
    singles = re.findall(r"(\d+)\s*[+]\s*(?:years|yrs)", text)
    at_least = re.findall(r"(?:at least|minimum|min)\s+(\d+)\s*(?:years|yrs)", text)

    if kind == "min":
        if at_least:
            return int(at_least[0])
        if ranges:
            return int(ranges[0][0])
        return C.JD_REQUIREMENTS["experience_min"]
    elif kind == "ideal":
        if ranges:
            return (int(ranges[0][0]) + int(ranges[0][1])) // 2
        if singles:
            return int(singles[0])
        return C.JD_REQUIREMENTS["experience_ideal"]
    else:  # max
        if ranges:
            return int(ranges[0][1])
        if singles:
            return int(singles[0]) + 3
        return C.JD_REQUIREMENTS["experience_max"]


def _extract_traits(text: str) -> dict[str, float]:
    """Extract behavioral trait preferences from JD."""
    traits = dict(C.JD_REQUIREMENTS["behavioral_traits"])

    if "onsite" in text or "in-office" in text:
        traits["willing_to_relocate"] = 1.5
    if "remote" in text or "wfh" in text or "work from home" in text:
        traits["willing_to_relocate"] = 0.5

    return traits
