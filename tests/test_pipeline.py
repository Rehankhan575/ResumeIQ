"""
tests/test_pipeline.py
======================
Pipeline tests covering:
  - Existing smoke tests (run_pipeline with/without JD)
  - Item 0 fix: ds_clean vs jd_chef < ds_clean vs jd_data_scientist − 20 pts
  - Item 0: chef produces non-empty missing_skills
  - Item 0b: "rn" rule in ambiguous_terms.csv (test only)
  - Item 1: new signature (jd_title, use_genai), graceful degradation, bad-file errors
  - Privacy: logs contain no resume text
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.pipeline import run_pipeline, warm_up
from app.schemas import ScoreResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DS_CLEAN = FIXTURES_DIR / "ds_clean.pdf"
CHEF_JD = FIXTURES_DIR / "jd_chef.txt"
DS_JD = FIXTURES_DIR / "jd_data_scientist.txt"

DUMMY_BYTES = b"%PDF-1.4 fake pdf content for testing"
DUMMY_FILENAME = "test_resume.pdf"
DUMMY_JD = "We are looking for a Python developer with experience in PyTorch and SQL."

# ---------------------------------------------------------------------------
# Existing smoke tests (backward-compatible)
# ---------------------------------------------------------------------------

def test_run_pipeline_returns_score_result() -> None:
    """run_pipeline must return a ScoreResult instance."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result, ScoreResult)


def test_run_pipeline_score_result_is_valid_schema() -> None:
    """ScoreResult returned by pipeline must pass Pydantic validation."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
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


# ---------------------------------------------------------------------------
# Item 0: ranking fix — ds_clean vs jd_chef vs jd_data_scientist
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DS_CLEAN.exists(), reason="ds_clean.pdf fixture not found")
def test_ranking_chef_below_data_scientist_by_20pts() -> None:
    """ds_clean vs jd_chef must score at least 20 pts below ds_clean vs jd_data_scientist.

    Before the fix: jd_chef scored 75.50 (empty required_skills → coverage=1.0).
    After the fix : jd_chef scores ~55.50 (required_skills extracted from JD text).
    """
    from app.parsing.parser import parse_resume
    from app.extraction.extractor import extract_entities
    from app.extras.ranking import rank_jds
    from app.schemas import JobDescription

    parsed = extract_entities(parse_resume(DS_CLEAN.read_bytes(), DS_CLEAN.name))

    jd_ds = JobDescription(text=DS_JD.read_text(encoding="utf-8"), title="Data Scientist")
    jd_chef = JobDescription(text=CHEF_JD.read_text(encoding="utf-8"), title="Executive Chef")

    ranked = rank_jds(parsed, [jd_ds, jd_chef])
    assert len(ranked) == 2

    scores = {r["jd_title"]: r["score"] for r in ranked}
    assert scores["Data Scientist"] > scores["Executive Chef"], (
        f"Expected Data Scientist ({scores['Data Scientist']:.2f}) > Chef ({scores['Executive Chef']:.2f})"
    )
    gap = scores["Data Scientist"] - scores["Executive Chef"]
    assert gap >= 20, (
        f"Expected gap >= 20 pts between Data Scientist and Chef, got {gap:.2f}"
    )


@pytest.mark.skipif(not DS_CLEAN.exists(), reason="ds_clean.pdf fixture not found")
def test_ranking_chef_has_non_empty_missing_skills() -> None:
    """ds_clean vs jd_chef must list non-empty missing skills (culinary skills)."""
    from app.parsing.parser import parse_resume
    from app.extraction.extractor import extract_entities
    from app.extras.ranking import rank_jds
    from app.schemas import JobDescription

    parsed = extract_entities(parse_resume(DS_CLEAN.read_bytes(), DS_CLEAN.name))
    jd_chef = JobDescription(text=CHEF_JD.read_text(encoding="utf-8"), title="Executive Chef")

    ranked = rank_jds(parsed, [jd_chef])
    assert len(ranked) == 1
    chef_result = ranked[0]
    assert chef_result["missing_skills"], (
        f"Expected non-empty missing_skills for ds_clean vs jd_chef, got: {chef_result['missing_skills']}"
    )


# ---------------------------------------------------------------------------
# Item 0b: "rn" rule in ambiguous_terms.csv
# ---------------------------------------------------------------------------

def test_ambiguous_rn_rule_exists_in_csv() -> None:
    """ambiguous_terms.csv must contain the 'rn' rule with require_context."""
    from app.config import settings
    csv_path = settings.ambiguous_terms_csv_path
    assert csv_path.exists(), f"ambiguous_terms.csv not found at {csv_path}"
    content = csv_path.read_text(encoding="utf-8")
    assert "rn,require_context" in content, (
        "'rn,require_context' not found in ambiguous_terms.csv"
    )


# ---------------------------------------------------------------------------
# Item 1: new run_pipeline signature
# ---------------------------------------------------------------------------

def test_run_pipeline_new_signature_jd_title() -> None:
    """run_pipeline(file_bytes, filename, jd_text, jd_title='', use_genai=True) works."""
    result = run_pipeline(
        DUMMY_BYTES, DUMMY_FILENAME,
        jd_text=DUMMY_JD,
        jd_title="Test Engineer",
        use_genai=False,
    )
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.total <= 100.0


def test_run_pipeline_use_genai_false_skips_suggestions() -> None:
    """use_genai=False must still produce a valid result (suggestions may be empty)."""
    result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME, use_genai=False)
    assert isinstance(result, ScoreResult)
    assert result.timings_ms.get("genai_ms", -1) == 0.0  # skipped


def test_run_pipeline_warm_up_is_idempotent() -> None:
    """Calling warm_up() multiple times must not raise."""
    warm_up()
    warm_up()


# ---------------------------------------------------------------------------
# Item 1: file size and type validation
# ---------------------------------------------------------------------------

def test_run_pipeline_oversized_file_returns_error_result() -> None:
    """Files over 5 MB must return an error result, not raise an exception."""
    big_bytes = b"X" * (5 * 1024 * 1024 + 1)
    result = run_pipeline(big_bytes, "big_resume.pdf")
    assert isinstance(result, ScoreResult)
    assert result.total == 0.0
    assert result.grade == "F"
    assert any("5 MB" in issue.message or "MB" in issue.message for issue in result.issues), (
        f"Expected size-limit message in issues, got: {result.issues}"
    )


def test_run_pipeline_unsupported_type_returns_error_result() -> None:
    """Files with unsupported extension must return an error result, not raise."""
    result = run_pipeline(b"some bytes", "resume.exe")
    assert isinstance(result, ScoreResult)
    assert result.total == 0.0
    assert result.grade == "F"
    assert any("not supported" in issue.message.lower() or "type" in issue.message.lower()
               for issue in result.issues)


def test_run_pipeline_corrupted_file_returns_error_result() -> None:
    """Corrupted file content must return a graceful error result."""
    result = run_pipeline(b"\x00\x01\x02corrupted", "resume.pdf")
    assert isinstance(result, ScoreResult)
    # Either parse fails gracefully or scores low; must not raise
    assert 0.0 <= result.total <= 100.0


# ---------------------------------------------------------------------------
# Item 1: stage-failure degradation
# ---------------------------------------------------------------------------

def test_run_pipeline_extraction_failure_degrades_gracefully() -> None:
    """If extraction stage raises, pipeline must still return a ScoreResult with warnings."""
    with patch("app.pipeline.extract_entities", side_effect=RuntimeError("extraction mock failure")):
        result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result, ScoreResult)
    # Should have recorded a warning
    assert any("extraction" in w.lower() for w in getattr(result, "warnings", [])), (
        f"Expected extraction warning in result.warnings; got: {result.warnings}"
    )


def test_run_pipeline_classification_failure_degrades_gracefully() -> None:
    """If classification stage raises, pipeline must still return a ScoreResult."""
    with patch("app.pipeline.classify_resume", side_effect=RuntimeError("cls mock failure")):
        result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)
    assert isinstance(result, ScoreResult)
    # categories may be empty but result is valid
    assert 0.0 <= result.total <= 100.0


def test_run_pipeline_genai_failure_degrades_gracefully() -> None:
    """If GenAI stage raises, pipeline must still return a ScoreResult."""
    with patch("app.pipeline.generate_suggestions", side_effect=RuntimeError("genai mock failure")):
        result = run_pipeline(DUMMY_BYTES, DUMMY_FILENAME, use_genai=True)
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.total <= 100.0


# ---------------------------------------------------------------------------
# Privacy: logs must not contain resume text
# ---------------------------------------------------------------------------

def test_pipeline_logs_no_resume_text(caplog) -> None:
    """Pipeline logger must not emit any resume text content."""
    resume_text_marker = "fake pdf content for testing"  # unique string in DUMMY_BYTES
    with caplog.at_level(logging.DEBUG, logger="app.pipeline"):
        run_pipeline(DUMMY_BYTES, DUMMY_FILENAME)

    for record in caplog.records:
        assert resume_text_marker not in record.message, (
            f"Resume text leaked into log: {record.message!r}"
        )


def test_pipeline_logs_no_pii_from_real_fixture(caplog) -> None:
    """Pipeline logs must not contain contact-info markers from real fixture."""
    if not DS_CLEAN.exists():
        pytest.skip("ds_clean.pdf not available")

    file_bytes = DS_CLEAN.read_bytes()
    # We only check that log lines don't contain the PII markers we know are in ds_clean
    pii_markers = ["@example.com", "example.com"]

    with caplog.at_level(logging.DEBUG, logger="app.pipeline"):
        run_pipeline(file_bytes, DS_CLEAN.name, use_genai=False)

    for record in caplog.records:
        for marker in pii_markers:
            assert marker not in record.message, (
                f"PII marker '{marker}' found in log: {record.message!r}"
            )
