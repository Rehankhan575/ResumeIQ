"""
tests/test_pipeline.py
======================
End-to-end stub pipeline tests.

Coverage:
  - run_pipeline returns a valid ScoreResult (schema validation).
  - All required fields are present and correctly typed.
  - Pipeline works with and without a JD.
  - Stage outputs (categories, highlights, suggestions) are non-empty.
  - timings_ms records all 7 stages.
  - ScoreResult.total is within [0, 100].
  - ScoreResult.match.score is within [0, 1] when JD is provided.
"""

from __future__ import annotations

import pytest

from app.pipeline import run_pipeline
from app.schemas import ScoreResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_BYTES = b"%PDF-1.4 fake pdf content for testing"
DUMMY_FILENAME = "test_resume.pdf"
DUMMY_JD = "We are looking for a Python developer with experience in PyTorch and SQL."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_pipeline_returns_score_result() -> None:
    """run_pipeline must return a ScoreResult instance."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result, ScoreResult)


def test_run_pipeline_score_result_is_valid_schema() -> None:
    """ScoreResult returned by pipeline must pass Pydantic validation."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    # Re-validate by round-tripping through model_validate
    validated = ScoreResult.model_validate(result.model_dump())
    assert validated.total == result.total


def test_run_pipeline_total_in_range() -> None:
    """ScoreResult.total must be in [0, 100]."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert 0.0 <= result.total <= 100.0


def test_run_pipeline_grade_is_string() -> None:
    """ScoreResult.grade must be a non-empty string."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result.grade, str)
    assert len(result.grade) > 0


def test_run_pipeline_without_jd_match_score_zero() -> None:
    """Without a JD, match score must be 0.0 (neutral)."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME, jd_text="")
    assert result.match is not None
    assert result.match.score == 0.0


def test_run_pipeline_with_jd_match_score_positive() -> None:
    """With a JD provided, match score must be > 0."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME, jd_text=DUMMY_JD)
    assert result.match is not None
    assert result.match.score > 0.0


def test_run_pipeline_with_jd_match_score_in_range() -> None:
    """match.score must be in [0, 1]."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME, jd_text=DUMMY_JD)
    assert result.match is not None
    assert 0.0 <= result.match.score <= 1.0


def test_run_pipeline_categories_non_empty() -> None:
    """Pipeline must return at least one category prediction."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert len(result.predicted_categories) > 0


def test_run_pipeline_category_probabilities_in_range() -> None:
    """All category probabilities must be in [0, 1]."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    for cat in result.predicted_categories:
        assert 0.0 <= cat.probability <= 1.0


def test_run_pipeline_highlights_non_empty() -> None:
    """Explainability stage must produce at least one highlight."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert len(result.highlights) > 0


def test_run_pipeline_suggestions_non_empty() -> None:
    """Generative stage must produce at least one suggestion."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert len(result.suggestions) > 0


def test_run_pipeline_issues_are_valid() -> None:
    """All Issue objects must have valid severity literals."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    valid_severities = {"low", "medium", "high"}
    for issue in result.issues:
        assert issue.severity in valid_severities


def test_run_pipeline_timings_recorded() -> None:
    """timings_ms must contain keys for all 7 pipeline stages."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    expected_keys = {
        "parsing_ms", "extraction_ms", "classification_ms",
        "matching_ms", "scoring_ms", "explain_ms", "genai_ms",
    }
    assert expected_keys.issubset(result.timings_ms.keys())


def test_run_pipeline_timings_are_positive() -> None:
    """All timing values must be non-negative floats."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    for key, val in result.timings_ms.items():
        assert val >= 0.0, f"Negative timing for {key}: {val}"


def test_run_pipeline_components_dict_present() -> None:
    """ScoreResult.components must be a non-empty dict."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result.components, dict)
    assert len(result.components) > 0


def test_run_pipeline_idempotent() -> None:
    """Calling run_pipeline twice with the same input returns consistent total."""
    r1 = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    r2 = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert r1.total == r2.total
    assert r1.grade == r2.grade
