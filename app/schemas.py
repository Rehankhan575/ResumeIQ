"""
app/schemas.py
==============
All Pydantic v2 data models for the AI Resume Analyzer pipeline.

SCHEMA_VERSION must be bumped on any change to this file.
Any change also requires:
  1. Updating tests/test_schemas.py
  2. Updating PROJECT_CONTEXT.md (Decisions Log + CHANGELOG)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION: str = "1.2.0"


# ---------------------------------------------------------------------------
# ContactInfo
# ---------------------------------------------------------------------------

class ContactInfo(BaseModel):
    """Structured contact details extracted from a résumé."""

    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    other_links: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LayoutMeta
# ---------------------------------------------------------------------------

class LayoutMeta(BaseModel):
    """Metadata about the physical layout and parsing method used."""

    file_type: str = ""          # e.g. "pdf", "docx", "image"
    page_count: int = 0
    word_count: int = 0
    has_tables: bool = False
    has_images: bool = False
    is_multi_column: bool = False
    font_count: int = 0
    ocr_used: bool = False
    needs_ocr: bool = False
    extraction_method: str = ""


# ---------------------------------------------------------------------------
# EducationEntry
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    """One education record (degree, institution, dates, GPA)."""

    degree: str = ""
    field: str = ""
    institution: str = ""
    start_year: int | None = None
    end_year: int | None = None
    graduation_year: int | None = None
    gpa: float | None = None


# ---------------------------------------------------------------------------
# ExperienceEntry
# ---------------------------------------------------------------------------

class ExperienceEntry(BaseModel):
    """One work-experience record."""

    title: str = ""
    company: str = ""
    start_date: str = ""         # ISO date string or free-form, e.g. "2021-06"
    end_date: str = ""
    start: str = ""              # Format-compatible alias e.g. "2021-06"
    end: str = ""                # Format-compatible alias e.g. "Present"
    is_current: bool = False
    duration_months: int = 0
    bullets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ProjectEntry
# ---------------------------------------------------------------------------

class ProjectEntry(BaseModel):
    """One project record."""

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity span identified by rules or a NER model."""

    text: str = ""
    label: str = ""              # e.g. "SKILL", "ORG", "DATE"
    start: int = 0               # character offset in raw_text
    end: int = 0
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["rules", "ner"] = "rules"

    @field_validator("score")
    @classmethod
    def score_range(cls, v: float) -> float:
        """Confidence must be in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Entity.score must be in [0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# ParsedResume
# ---------------------------------------------------------------------------

class ParsedResume(BaseModel):
    """Complete structured output of the parsing + extraction stages."""

    raw_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    contact: ContactInfo = Field(default_factory=ContactInfo)
    layout: LayoutMeta = Field(default_factory=LayoutMeta)
    skills: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_years_experience: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# JobDescription
# ---------------------------------------------------------------------------

class JobDescription(BaseModel):
    """Structured representation of a job description."""

    title: str = ""
    text: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years: float | None = None
    min_years_experience: float | None = None


# ---------------------------------------------------------------------------
# CategoryPrediction
# ---------------------------------------------------------------------------

class CategoryPrediction(BaseModel):
    """A single category label with associated probability."""

    category: str = ""
    probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("probability")
    @classmethod
    def probability_range(cls, v: float) -> float:
        """Probability must be in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"CategoryPrediction.probability must be in [0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

class MatchResult(BaseModel):
    """Résumé–JD matching output combining semantic and keyword signals."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    section_scores: dict[str, float] = Field(default_factory=dict)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    method: str = ""             # e.g. "sbert_cosine", "tfidf_cosine"

    @field_validator("score")
    @classmethod
    def score_range(cls, v: float) -> float:
        """Overall match score must be in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"MatchResult.score must be in [0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    """A single ATS or quality issue found in the résumé."""

    component: str = ""
    message: str = ""
    severity: Literal["low", "medium", "high"] = "low"
    estimated_gain: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------

class Highlight(BaseModel):
    """A character-span highlight for the explainability overlay."""

    start: int = 0
    end: int = 0
    text: str = ""
    weight: float = 0.0          # positive = helps score, negative = hurts
    type: Literal[
        "positive", "negative", "keyword_match", "keyword_missing"
    ] = "positive"


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------

class Suggestion(BaseModel):
    """A generative improvement suggestion."""

    kind: Literal[
        "bullet_rewrite", "cover_letter", "improvement_plan"
    ] = "improvement_plan"
    original: str = ""
    suggested: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# ScoreResult  (top-level output of run_pipeline)
# ---------------------------------------------------------------------------

class ScoreResult(BaseModel):
    """Complete output of the full AI Resume Analyzer pipeline."""

    total: float = Field(default=0.0, ge=0.0, le=100.0)
    grade: str = ""              # e.g. "B+", "C", "A"
    components: dict[str, float] = Field(default_factory=dict)
    missing_skills: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
    predicted_categories: list[CategoryPrediction] = Field(default_factory=list)
    match: MatchResult | None = None
    suggestions: list[Suggestion] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @field_validator("total")
    @classmethod
    def total_range(cls, v: float) -> float:
        """ATS total score must be in [0, 100]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"ScoreResult.total must be in [0, 100], got {v}")
        return v

    @model_validator(mode="after")
    def grade_auto_fill(self) -> "ScoreResult":
        """Auto-assign a letter grade if none was supplied."""
        if not self.grade:
            t = self.total
            if t >= 90:
                self.grade = "A"
            elif t >= 80:
                self.grade = "A-"
            elif t >= 70:
                self.grade = "B"
            elif t >= 60:
                self.grade = "C"
            else:
                self.grade = "D"
        return self
