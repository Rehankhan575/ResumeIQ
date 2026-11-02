"""
app/extraction/extractor.py
===========================
Stage 2 — NER & Structured Extraction.

Extracts skills, entities, education, experience, projects, and contact info
from a :class:`ParsedResume`. Uses rule-based matching + a fine-tuned DistilBERT NER.
"""

from __future__ import annotations

import logging

from app.schemas import (
    ContactInfo,
    EducationEntry,
    Entity,
    ExperienceEntry,
    ParsedResume,
    ProjectEntry,
)

logger = logging.getLogger(__name__)


def extract_entities(parsed: ParsedResume) -> ParsedResume:
    """Enrich a :class:`ParsedResume` with structured entities and fields.

    This is currently a **stub** that returns valid dummy data.
    Real implementation: rule-based patterns + HuggingFace NER pipeline
    (``dslim/distilbert-NER`` fine-tuned on resume NER datasets).

    Args:
        parsed: Output of :func:`app.parsing.parser.parse_document`.

    Returns:
        The same :class:`ParsedResume` with ``contact``, ``skills``,
        ``entities``, ``education``, ``experience``, ``projects``,
        ``certifications``, and ``total_years_experience`` populated.
    """
    logger.debug("extract_entities called")

    parsed.contact = ContactInfo(
        email="john@example.com",
        phone="+1-555-0100",
        linkedin="linkedin.com/in/johndoe",
        github="github.com/johndoe",
        other_links=[],
    )

    parsed.skills = ["Python", "PyTorch", "SQL", "Machine Learning"]

    parsed.entities = [
        Entity(text="Python", label="SKILL", start=0, end=6, score=0.98, source="ner"),
        Entity(text="MIT", label="ORG", start=10, end=13, score=0.95, source="ner"),
    ]

    parsed.education = [
        EducationEntry(
            degree="B.Sc",
            field="Computer Science",
            institution="MIT",
            start_year=2018,
            end_year=2022,
            gpa=3.8,
        )
    ]

    parsed.experience = [
        ExperienceEntry(
            title="Software Engineer",
            company="ACME Corp",
            start_date="2022-06",
            end_date="2024-06",
            is_current=False,
            duration_months=24,
            bullets=[
                "Built ML pipeline reducing inference latency by 40%.",
                "Led team of 3 engineers to deliver feature on schedule.",
            ],
        )
    ]

    parsed.projects = [
        ProjectEntry(
            name="Resume Analyzer",
            description="End-to-end NLP pipeline for resume scoring.",
            technologies=["Python", "PyTorch", "Streamlit"],
        )
    ]

    parsed.certifications = ["AWS Certified Developer"]
    parsed.total_years_experience = 2.0

    return parsed
