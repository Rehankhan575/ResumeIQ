"""
app/extraction/text_utils.py
============================
Text tokenization and normalization utilities for Stage 2 extraction.
Mirrors scripts/taxonomy_stats.py logic to ensure extraction and taxonomy
statistics never drift.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Token pattern: matches words, dotted acronyms, c++/c#, .net, &, etc.
TOKEN_PATTERN = re.compile(
    r"\.net\b|[a-z0-9]+(?:#|\+\+)?(?:[/\'\&.-][a-z0-9]+)*(?:#|\+\+)?|\&"
)

# Case-insensitive token pattern for preserving original text spans and casing
TOKEN_PATTERN_CI = re.compile(
    r"\.net\b|[a-zA-Z0-9]+(?:#|\+\+)?(?:[/\'\&.-][a-zA-Z0-9]+)*(?:#|\+\+)?|\&",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize unicode characters and lowercase."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Tokenize text into normalized tokens. Identical to taxonomy_stats.py."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def find_token_spans(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (token, start, end) offsets over normalize_text(text)."""
    norm = normalize_text(text)
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(norm)]


def find_raw_token_spans(text: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (raw_token, norm_token, start_char, end_char) over raw text."""
    spans = []
    for m in TOKEN_PATTERN_CI.finditer(text):
        raw = m.group()
        norm = normalize_text(raw)
        spans.append((raw, norm, m.start(), m.end()))
    return spans
