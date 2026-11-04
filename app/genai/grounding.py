"""
app/genai/grounding.py
======================
Grounding Guard for GenAI bullet and text rewrites.

Enforces strict factual hallucination prevention:
Flags any number, capitalised organisation-like token, or taxonomy skill
present in the rewritten text that is not in the source text.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Iterator, List, Set, Tuple

import pandas as pd

from app.config import ACTION_VERBS

logger = logging.getLogger(__name__)

TAXONOMIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "taxonomies"
SKILLS_CSV = TAXONOMIES_DIR / "skills.csv"

# Global cached taxonomy skill terms
_CACHED_TAXONOMY_SKILLS: Set[str] | None = None

# Number regex: matches digits, percentages, currencies, multipliers, etc.
NUMBER_RE = re.compile(
    r"(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?%?|\b\d+\s*(?:k|m|b|x|percent)\b|[€£¥]\s*\d+)",
    re.IGNORECASE,
)

# Common words to ignore when scanning for capitalized organization tokens
COMMON_CAPITALIZED_WORDS: Set[str] = {
    "I", "A", "An", "The", "In", "On", "At", "By", "With", "For", "To", "From",
    "And", "Or", "Nor", "But", "So", "Yet", "If", "Then", "Else", "When",
    "Where", "While", "After", "Before", "Both", "Each", "All", "Every",
    "Using", "Through", "Across", "Over", "Under", "Within", "During",
    "Throughout", "Between", "Among", "Upon", "Into", "Onto", "Toward", "Towards",
    "Add", "Metric", "Placeholder", "Star", "Task", "Result", "Action", "Situation",
    "Key", "Core", "New", "High", "Low", "Full", "Deep", "Top", "Senior", "Lead",
    "Principal", "Staff", "Junior", "Primary", "Main", "Major", "Critical",
}


def _get_taxonomy_skills() -> Set[str]:
    """Load canonical skills and aliases from data/taxonomies/skills.csv."""
    global _CACHED_TAXONOMY_SKILLS
    if _CACHED_TAXONOMY_SKILLS is not None:
        return _CACHED_TAXONOMY_SKILLS

    skills: Set[str] = set()
    if SKILLS_CSV.exists():
        try:
            df = pd.read_csv(SKILLS_CSV, keep_default_na=False)
            for _, row in df.iterrows():
                can = str(row.get("skill", "")).strip().lower()
                if can:
                    skills.add(can)
                aliases = str(row.get("aliases", "")).split("|")
                for a in aliases:
                    a_clean = a.strip().lower()
                    if a_clean:
                        skills.add(a_clean)
        except Exception as exc:
            logger.debug("Failed loading skills.csv for grounding: %s", exc)

    _CACHED_TAXONOMY_SKILLS = skills
    return _CACHED_TAXONOMY_SKILLS


class GroundingResult:
    """Result object of check_grounding.

    Behaves as a boolean (True if grounded/clean, False if violations flagged),
    unpacks as a (is_grounded, flags) tuple, and exposes flags and violations.
    """

    def __init__(self, is_grounded: bool, flags: List[str]):
        self.is_grounded = is_grounded
        self.flags = flags
        self.violations = flags

    def __bool__(self) -> bool:
        return self.is_grounded

    def __iter__(self) -> Iterator:
        yield self.is_grounded
        yield self.flags

    def __len__(self) -> int:
        return len(self.flags)

    def __contains__(self, item: str) -> bool:
        return any(item.lower() in f.lower() for f in self.flags)

    def __repr__(self) -> str:
        return f"GroundingResult(is_grounded={self.is_grounded}, flags={self.flags})"


def _clean_placeholders(text: str) -> str:
    """Remove explicit placeholder tokens like [add metric] before checking."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def _extract_numbers(text: str) -> Set[str]:
    """Extract normalized numbers from text."""
    clean = _clean_placeholders(text)
    raw_matches = NUMBER_RE.findall(clean)
    normalized = set()
    for m in raw_matches:
        norm = re.sub(r"[,\s\$€£¥]", "", m).lower()
        if norm:
            normalized.add(norm)
    return normalized


def _extract_organization_tokens(text: str) -> Set[str]:
    """Extract capitalized organization-like tokens and acronyms."""
    clean = _clean_placeholders(text)
    orgs = set()

    # Split into sentences or lines to distinguish sentence-initial capitalization
    sentences = [s.strip() for s in re.split(r"[.\n;]+", clean) if s.strip()]
    for sent in sentences:
        words = re.findall(r"\b[A-Za-z0-9&.-]+\b", sent)
        for idx, tok in enumerate(words):
            if len(tok) < 2 or not tok[0].isupper():
                continue
            tok_lower = tok.lower()
            if tok in COMMON_CAPITALIZED_WORDS or tok_lower in ACTION_VERBS:
                continue

            # All-caps acronyms (e.g. AWS, IBM, GCP, NASA) are flagged anywhere
            if tok.isupper() and len(tok) >= 2 and tok.isalpha():
                orgs.add(tok)
            # Sentence-initial words (index 0) are capitalized by standard grammar
            elif idx == 0:
                continue
            # Capitalized words inside sentence (index > 0) are organizations or proper nouns
            elif tok[0].isupper():
                orgs.add(tok)

    return orgs


def _extract_skills(text: str, taxonomy: Set[str]) -> Set[str]:
    """Extract taxonomy skills mentioned in text."""
    clean = _clean_placeholders(text).lower()
    found = set()
    # Check multi-word and single-word skills via word boundaries
    for skill in taxonomy:
        # Avoid matching single-character skills blindly
        if len(skill) <= 2 and skill not in {"r", "c", "go", "ai", "ml", "ui", "ux", "qa"}:
            continue
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, clean):
            found.add(skill)
    return found


def check_grounding(original: str, rewritten: str) -> GroundingResult:
    """Check whether a rewritten bullet contains ungrounded factual inventions.

    Flags:
      1. Numbers in rewrite not in source.
      2. Capitalised organisation-like tokens in rewrite not in source.
      3. Taxonomy skills in rewrite not in source.

    Args:
        original: Original source bullet or context.
        rewritten: Generated rewritten bullet.

    Returns:
        :class:`GroundingResult` indicating if clean (is_grounded=True) or flagged (is_grounded=False).
    """
    if not rewritten.strip():
        return GroundingResult(True, [])

    flags: List[str] = []

    # 1. Check numbers
    orig_numbers = _extract_numbers(original)
    new_numbers = _extract_numbers(rewritten)
    invented_numbers = new_numbers - orig_numbers
    for num in sorted(invented_numbers):
        flags.append(f"invented_number: {num}")

    # 2. Check organization tokens / acronyms
    orig_tokens = {t.lower() for t in re.findall(r"\b[A-Za-z0-9&.-]+\b", original)}
    new_orgs = _extract_organization_tokens(rewritten)
    # Organization is ungrounded if not present in original tokens (case-insensitive)
    for org in sorted(new_orgs):
        if org.lower() not in orig_tokens:
            flags.append(f"invented_organization: {org}")

    # 3. Check taxonomy skills
    taxonomy = _get_taxonomy_skills()
    orig_skills = _extract_skills(original, taxonomy)
    new_skills = _extract_skills(rewritten, taxonomy)
    invented_skills = new_skills - orig_skills
    for skill in sorted(invented_skills):
        flags.append(f"invented_skill: {skill}")

    is_grounded = len(flags) == 0
    return GroundingResult(is_grounded, flags)
