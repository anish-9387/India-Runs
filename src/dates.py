"""Tiny date helpers (ISO 'YYYY-MM-DD' parsing without external deps)."""

from __future__ import annotations

from datetime import date
from typing import Optional


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    try:
        y, m, d = value[:10].split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def months_between(start: Optional[date], end: Optional[date]) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month)
