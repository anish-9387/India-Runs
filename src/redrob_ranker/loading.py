"""Streaming loaders for the candidate pool (.jsonl or .jsonl.gz)."""

from __future__ import annotations

import gzip
import json
import re
from datetime import date
from typing import Iterator

from .dates import parse_date

_LAST_ACTIVE_RE = re.compile(r'"last_active_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')


def open_candidates(path: str):
    """Open a .jsonl or .jsonl.gz file as text, transparently."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per non-empty line."""
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dataset_reference_date(path: str) -> date:
    """
    The "now" used for recency calculations: the most recent last_active_date
    in the pool. Reproducible (no wall-clock dependency) and robust to whatever
    window the synthetic data was generated in. Uses a fast regex scan rather
    than parsing every record.
    """
    latest = ""
    with open_candidates(path) as f:
        for line in f:
            m = _LAST_ACTIVE_RE.search(line)
            if m and m.group(1) > latest:
                latest = m.group(1)
    return parse_date(latest) or date(2026, 1, 1)
