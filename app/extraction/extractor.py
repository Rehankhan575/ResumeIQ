"""
app/extraction/extractor.py
===========================
Stage 2 — Orchestrator for Structured Extraction.

Enriches a :class:`ParsedResume` with:
- Canonical skills (via TaxonomySkillExtractor)
- Entity spans (rules-based SKILL entities)
- Normalized EducationEntry items
- Date-anchored ExperienceEntry items
- Non-overlapping total_years_experience

Falls back gracefully on missing or empty sections by searching full raw text.
Keeps the function signature that app/pipeline.py calls.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.extraction.entries import (
    compute_total_years_experience,
    extract_education,
    extract_experience,
)
from app.extraction.skills import TaxonomySkillExtractor
from app.schemas import Entity, ParsedResume

logger = logging.getLogger(__name__)

# Module-level cached extractor singleton
_SKILL_EXTRACTOR: Optional[TaxonomySkillExtractor] = None


def get_skill_extractor() -> TaxonomySkillExtractor:
    """Return cached TaxonomySkillExtractor instance."""
    global _SKILL_EXTRACTOR
    if _SKILL_EXTRACTOR is None:
        _SKILL_EXTRACTOR = TaxonomySkillExtractor()
    return _SKILL_EXTRACTOR


def extract_entities(
    parsed: ParsedResume,
    as_of: Optional[str] = None,
) -> ParsedResume:
    """Enrich a :class:`ParsedResume` with structured entities and fields.

    Extracts skills, entities, education, experience, and total_years_experience.
    Maintains exact backward-compatible signature for app/pipeline.py.

    Args:
        parsed: Output of :func:`app.parsing.parser.parse_resume`.
        as_of: Deterministic reference date string (YYYY-MM-DD). Defaults to
            settings.default_as_of_date ("2026-11-01").

    Returns:
        The enriched :class:`ParsedResume`.
    """
    as_of_date = as_of or settings.default_as_of_date
    extractor = get_skill_extractor()

    # 1. Skills extraction with section fallback
    extracted_skills = extractor.extract_skills_from_resume_sections(
        sections=parsed.sections,
        raw_text=parsed.raw_text,
    )
    parsed.skills = extracted_skills

    # Populate entities list with rules-based skill spans
    parsed.entities = [
        Entity(text=s, label="SKILL", source="rules", score=1.0)
        for s in extracted_skills
    ]

    # 2. Education extraction
    edu_text = parsed.sections.get("education", "")
    if not edu_text.strip() and parsed.raw_text:
        edu_text = parsed.raw_text

    parsed.education = extract_education(edu_text)

    # 3. Experience extraction
    exp_text = (
        parsed.sections.get("work_experience")
        or parsed.sections.get("experience")
        or ""
    )
    if not exp_text.strip() and parsed.raw_text:
        exp_text = parsed.raw_text

    parsed.experience = extract_experience(exp_text, as_of=as_of_date)

    # 4. Total years of experience calculation
    parsed.total_years_experience = compute_total_years_experience(
        parsed.experience,
        as_of=as_of_date,
    )

    return parsed
