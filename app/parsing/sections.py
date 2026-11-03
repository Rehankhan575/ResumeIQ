"""
app/parsing/sections.py
=======================
Splits cleaned document text into canonical sections using heading taxonomies
and fuzzy string matching.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional, Set

from rapidfuzz import fuzz, process

from app.config import settings

# Canonical sections defined in Stage 1 specification
CANONICAL_SECTIONS: Set[str] = {
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "awards",
    "publications",
    "languages",
    "interests",
    "references",
}

# Regex to strip trailing colons and punctuation
TRAILING_COLON_RE = re.compile(r"[:：]+$")
SPACES_RE = re.compile(r"[ \t]+")


def load_heading_taxonomy(csv_path: Optional[Path] = None) -> Dict[str, str]:
    """Load the heading synonym to canonical section mapping from CSV.

    Args:
        csv_path: Optional path to section_headings.csv. If None, uses settings.taxonomies_dir.

    Returns:
        Dictionary mapping normalized heading synonym to canonical section name.
    """
    path = csv_path or (settings.taxonomies_dir / "section_headings.csv")
    taxonomy: Dict[str, str] = {}

    if not path.exists():
        # Fallback to canonical identity mapping if file missing
        return {s: s for s in CANONICAL_SECTIONS}

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            heading = row.get("heading", "").strip().lower()
            canonical = row.get("canonical", "").strip().lower()
            if heading and canonical in CANONICAL_SECTIONS:
                taxonomy[heading] = canonical

    # Ensure all canonical sections map to themselves
    for sec in CANONICAL_SECTIONS:
        taxonomy.setdefault(sec, sec)

    return taxonomy


# Module-level cached taxonomy
_HEADING_TAXONOMY: Optional[Dict[str, str]] = None


def get_heading_taxonomy() -> Dict[str, str]:
    """Return the cached heading taxonomy mapping."""
    global _HEADING_TAXONOMY
    if _HEADING_TAXONOMY is None:
        _HEADING_TAXONOMY = load_heading_taxonomy()
    return _HEADING_TAXONOMY


def match_heading(line: str, taxonomy: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Determine if a line is a section heading and return its canonical name.

    Rules:
        - 4 words or fewer.
        - No trailing period.
        - Normalize case, trailing colons, whitespace, and "&".
        - Exact match first against taxonomy / canonical names.
        - Rapidfuzz fuzzy match (ratio >= 92) fallback.

    Args:
        line: The candidate line from the document.
        taxonomy: Optional dictionary of heading synonyms to canonical names.

    Returns:
        Canonical section string if matched, None otherwise.
    """
    raw = line.strip()
    if not raw or raw.endswith("."):
        return None

    words = raw.split()
    if len(words) > 4:
        return None

    # Normalization
    norm = TRAILING_COLON_RE.sub("", raw).strip().lower()
    norm = SPACES_RE.sub(" ", norm)

    if not norm:
        return None

    tax = taxonomy if taxonomy is not None else get_heading_taxonomy()

    # Exact match: normalized line
    if norm in CANONICAL_SECTIONS:
        return norm
    if norm in tax:
        return tax[norm]

    # Exact match: normalized line with "&" expanded to "and"
    norm_and = norm.replace("&", "and")
    norm_and = SPACES_RE.sub(" ", norm_and).strip()
    if norm_and in tax:
        return tax[norm_and]

    # Exact match: normalized line with "and" collapsed to "&"
    norm_amp = re.sub(r"\band\b", "&", norm)
    norm_amp = SPACES_RE.sub(" ", norm_amp).strip()
    if norm_amp in tax:
        return tax[norm_amp]

    # Fuzzy match with rapidfuzz (ratio >= 92)
    choices = list(tax.keys())
    match = process.extractOne(norm, choices, scorer=fuzz.ratio, score_cutoff=92)
    if match:
        return tax[match[0]]

    if norm != norm_and:
        match_and = process.extractOne(norm_and, choices, scorer=fuzz.ratio, score_cutoff=92)
        if match_and:
            return tax[match_and[0]]

    return None


def _split_single_stream(text: str, tax: Dict[str, str]) -> Dict[str, str]:
    """Split a single contiguous stream of text into sections."""
    sections: Dict[str, list[str]] = {}
    current_section = "header"

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            if current_section in sections and sections[current_section]:
                sections[current_section].append("")
            continue

        canonical_heading = match_heading(line_stripped, tax)
        if canonical_heading:
            current_section = canonical_heading
            if current_section not in sections:
                sections[current_section] = []
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line_stripped)

    result: Dict[str, str] = {}
    for sec_name, sec_lines in sections.items():
        joined = "\n".join(sec_lines).strip()
        if joined:
            result[sec_name] = joined

    return result


def split_sections(text: str, taxonomy: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Split cleaned document text into canonical sections.

    Text before the first detected heading is stored under the "header" key.
    Duplicate headings are concatenated with newlines.

    Args:
        text: Cleaned full text of the résumé.
        taxonomy: Optional heading synonym dictionary.

    Returns:
        Dictionary mapping canonical section names (plus optional "header") to text.
    """
    if not text:
        return {}

    tax = taxonomy if taxonomy is not None else get_heading_taxonomy()

    # If document has column break markers, split per column and concatenate duplicate sections
    if "---column-break---" in text:
        chunks = text.split("---column-break---")
        combined_sections: Dict[str, str] = {}
        for chunk in chunks:
            chunk_sections = _split_single_stream(chunk, tax)
            for sec_name, sec_text in chunk_sections.items():
                if sec_name in combined_sections:
                    combined_sections[sec_name] = f"{combined_sections[sec_name]}\n\n{sec_text}"
                else:
                    combined_sections[sec_name] = sec_text
        return combined_sections

    return _split_single_stream(text, tax)
