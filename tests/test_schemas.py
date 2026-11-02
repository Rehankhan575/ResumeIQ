"""
tests/test_schemas.py
=====================
Tests for app/schemas.py.

Coverage:
  - Every schema can be instantiated with valid dummy data.
  - Range validators reject bad values:
      * ScoreResult.total = 150  → ValidationError
      * CategoryPrediction.probability = 1.5 → ValidationError
      * MatchResult.score = -0.1 → ValidationError
      * Entity.score = 2.0 → ValidationError
  - Default values are correct types.
  - Nested models validate correctly.
  - ScoreResult auto-fills grade when not provided.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    SCHEMA_VERSION,
    CategoryPrediction,
    ContactInfo,
    EducationEntry,
    Entity,
    ExperienceEntry,
    Highlight,
    Issue,
    JobDescription,
    LayoutMeta,
    MatchResult,
    ParsedResume,
    ProjectEntry,
    ScoreResult,
    Suggestion,
)


# ---------------------------------------------------------------------------
# SCHEMA_VERSION
# ---------------------------------------------------------------------------

def test_schema_version_is_string() -> None:
    """SCHEMA_VERSION must be a non-empty string."""
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0


# ---------------------------------------------------------------------------
# ContactInfo
# ---------------------------------------------------------------------------

def test_contact_info_defaults() -> None:
    c = ContactInfo()
    assert c.email == ""
    assert c.other_links == []


def test_contact_info_valid() -> None:
    c = ContactInfo(email="a@b.com", phone="123", linkedin="li", github="gh", other_links=["x"])
    assert c.email == "a@b.com"
    assert c.other_links == ["x"]


# ---------------------------------------------------------------------------
# LayoutMeta
# ---------------------------------------------------------------------------

def test_layout_meta_defaults() -> None:
    lm = LayoutMeta()
    assert lm.file_type == ""
    assert lm.page_count == 0
    assert lm.ocr_used is False


# ---------------------------------------------------------------------------
# EducationEntry
# ---------------------------------------------------------------------------

def test_education_entry_valid() -> None:
    e = EducationEntry(degree="B.Sc", field="CS", institution="MIT", start_year=2018, end_year=2022, gpa=3.9)
    assert e.gpa == 3.9


def test_education_entry_nullable_fields() -> None:
    e = EducationEntry()
    assert e.start_year is None
    assert e.gpa is None


# ---------------------------------------------------------------------------
# ExperienceEntry
# ---------------------------------------------------------------------------

def test_experience_entry_defaults() -> None:
    ex = ExperienceEntry()
    assert ex.is_current is False
    assert ex.bullets == []


def test_experience_entry_valid() -> None:
    ex = ExperienceEntry(title="SWE", company="ACME", duration_months=24, bullets=["Did X"])
    assert len(ex.bullets) == 1


# ---------------------------------------------------------------------------
# ProjectEntry
# ---------------------------------------------------------------------------

def test_project_entry_valid() -> None:
    p = ProjectEntry(name="X", description="Y", technologies=["Python"])
    assert p.technologies == ["Python"]


# ---------------------------------------------------------------------------
# Entity — valid + range rejection
# ---------------------------------------------------------------------------

def test_entity_valid() -> None:
    e = Entity(text="Python", label="SKILL", start=0, end=6, score=0.95, source="ner")
    assert e.score == 0.95


def test_entity_score_boundary_low() -> None:
    e = Entity(score=0.0)
    assert e.score == 0.0


def test_entity_score_boundary_high() -> None:
    e = Entity(score=1.0)
    assert e.score == 1.0


def test_entity_score_above_1_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(score=2.0)


def test_entity_score_below_0_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(score=-0.1)


# ---------------------------------------------------------------------------
# ParsedResume
# ---------------------------------------------------------------------------

def test_parsed_resume_defaults() -> None:
    pr = ParsedResume()
    assert pr.raw_text == ""
    assert pr.skills == []
    assert pr.warnings == []


def test_parsed_resume_with_nested_models() -> None:
    pr = ParsedResume(
        raw_text="hello",
        contact=ContactInfo(email="a@b.com"),
        layout=LayoutMeta(file_type="pdf", page_count=2),
    )
    assert pr.contact.email == "a@b.com"
    assert pr.layout.page_count == 2


# ---------------------------------------------------------------------------
# JobDescription
# ---------------------------------------------------------------------------

def test_job_description_defaults() -> None:
    jd = JobDescription()
    assert jd.min_years_experience is None


def test_job_description_valid() -> None:
    jd = JobDescription(title="MLE", text="We need Python", required_skills=["Python"], min_years_experience=3.0)
    assert jd.min_years_experience == 3.0


# ---------------------------------------------------------------------------
# CategoryPrediction — valid + range rejection
# ---------------------------------------------------------------------------

def test_category_prediction_valid() -> None:
    cp = CategoryPrediction(category="Data Science", probability=0.85)
    assert cp.probability == 0.85


def test_category_prediction_probability_zero() -> None:
    cp = CategoryPrediction(probability=0.0)
    assert cp.probability == 0.0


def test_category_prediction_probability_one() -> None:
    cp = CategoryPrediction(probability=1.0)
    assert cp.probability == 1.0


def test_category_prediction_probability_above_1_rejected() -> None:
    """probability=1.5 must raise ValidationError."""
    with pytest.raises(ValidationError):
        CategoryPrediction(category="X", probability=1.5)


def test_category_prediction_probability_below_0_rejected() -> None:
    with pytest.raises(ValidationError):
        CategoryPrediction(category="X", probability=-0.01)


# ---------------------------------------------------------------------------
# MatchResult — valid + range rejection
# ---------------------------------------------------------------------------

def test_match_result_valid() -> None:
    mr = MatchResult(score=0.73, method="sbert_cosine")
    assert mr.score == 0.73


def test_match_result_score_negative_rejected() -> None:
    """score=-0.1 must raise ValidationError."""
    with pytest.raises(ValidationError):
        MatchResult(score=-0.1)


def test_match_result_score_above_1_rejected() -> None:
    with pytest.raises(ValidationError):
        MatchResult(score=1.1)


def test_match_result_score_boundary_zero() -> None:
    mr = MatchResult(score=0.0)
    assert mr.score == 0.0


def test_match_result_score_boundary_one() -> None:
    mr = MatchResult(score=1.0)
    assert mr.score == 1.0


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

def test_issue_valid() -> None:
    i = Issue(component="formatting", message="Tables break ATS.", severity="high", estimated_gain=5.0)
    assert i.severity == "high"


def test_issue_defaults() -> None:
    i = Issue()
    assert i.severity == "low"
    assert i.estimated_gain == 0.0


# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------

def test_highlight_valid() -> None:
    h = Highlight(start=0, end=6, text="Python", weight=0.15, type="keyword_match")
    assert h.type == "keyword_match"


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------

def test_suggestion_valid() -> None:
    s = Suggestion(kind="bullet_rewrite", original="Did stuff", suggested="Achieved X by doing Y")
    assert s.kind == "bullet_rewrite"


# ---------------------------------------------------------------------------
# ScoreResult — valid + range rejection + grade auto-fill
# ---------------------------------------------------------------------------

def test_score_result_valid() -> None:
    sr = ScoreResult(total=75.0, grade="B")
    assert sr.total == 75.0
    assert sr.grade == "B"


def test_score_result_total_above_100_rejected() -> None:
    """total=150 must raise ValidationError."""
    with pytest.raises(ValidationError):
        ScoreResult(total=150.0)


def test_score_result_total_below_0_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoreResult(total=-1.0)


def test_score_result_total_boundary_zero() -> None:
    sr = ScoreResult(total=0.0)
    assert sr.total == 0.0


def test_score_result_total_boundary_100() -> None:
    sr = ScoreResult(total=100.0)
    assert sr.total == 100.0


def test_score_result_grade_auto_filled_a() -> None:
    sr = ScoreResult(total=95.0)
    assert sr.grade == "A"


def test_score_result_grade_auto_filled_b() -> None:
    sr = ScoreResult(total=75.0)
    assert sr.grade == "B"


def test_score_result_grade_auto_filled_d() -> None:
    sr = ScoreResult(total=45.0)
    assert sr.grade == "D"


def test_score_result_grade_manual_not_overridden() -> None:
    sr = ScoreResult(total=95.0, grade="S+")
    assert sr.grade == "S+"


def test_score_result_with_match_none() -> None:
    sr = ScoreResult(total=60.0, match=None)
    assert sr.match is None


def test_score_result_with_nested_match() -> None:
    mr = MatchResult(score=0.5)
    sr = ScoreResult(total=60.0, match=mr)
    assert sr.match is not None
    assert sr.match.score == 0.5


def test_score_result_defaults_are_empty_lists() -> None:
    sr = ScoreResult()
    assert sr.issues == []
    assert sr.highlights == []
    assert sr.suggestions == []
    assert sr.predicted_categories == []
