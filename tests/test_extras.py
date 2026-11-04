"""
tests/test_extras.py
====================
Test suite for Stage 7 Extras:
  1. Multi-JD ranking (rank_jds)
  2. Skill gap mapping (skill_gap)
  3. Unconscious bias auditing (bias_check)
  4. Local SQLite history logging (record_history)
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from app.extraction.extractor import extract_entities
from app.extras.bias import bias_check
from app.extras.history import get_history, record_history
from app.extras.ranking import rank_jds
from app.extras.skill_gap import skill_gap
from app.parsing.parser import parse_resume
from app.schemas import JobDescription, ParsedResume, ScoreResult

ROOT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_resume() -> ParsedResume:
    p = FIXTURES_DIR / "ds_clean.pdf"
    return extract_entities(parse_resume(p.read_bytes(), str(p)))


@pytest.fixture
def sample_jds() -> list[JobDescription]:
    jds = []
    for fname, title in [
        ("jd_data_scientist.txt", "Data Scientist"),
        ("jd_software_engineer.txt", "Software Engineer"),
        ("jd_chef.txt", "Executive Chef"),
    ]:
        p = FIXTURES_DIR / fname
        jds.append(JobDescription(text=p.read_text(encoding="utf-8"), title=title))
    return jds


def test_rank_jds_order_matches_individual_scores(sample_resume, sample_jds) -> None:
    """rank_jds sorts JDs descending by score and matches individual score evaluation."""
    from app.scoring.ats import score_resume

    # Compute individual scores
    individual_scores = {}
    for jd in sample_jds:
        res = score_resume(sample_resume, jd)
        individual_scores[jd.title] = res.total

    # Rank with rank_jds
    ranked = rank_jds(sample_resume, sample_jds)
    assert len(ranked) == 3

    # Check rank order matches sorted individual scores
    scores_in_ranked = [item["score"] for item in ranked]
    assert scores_in_ranked == sorted(scores_in_ranked, reverse=True)
    assert ranked[0]["jd_title"] == "Data Scientist"

    for item in ranked:
        expected_score = individual_scores[item["jd_title"]]
        assert item["score"] == pytest.approx(expected_score, abs=0.01)
        assert "missing_skills" in item


def test_bias_check_accountant_table_and_ds_clean() -> None:
    """bias_check flags accountant_table for DOB, marital status, graduation year, and gendered wording, and passes ds_clean cleanly."""
    acc_path = FIXTURES_DIR / "accountant_table.pdf"
    acc_parsed = extract_entities(parse_resume(acc_path.read_bytes(), str(acc_path)))
    findings_acc = bias_check(acc_parsed)

    assert len(findings_acc) >= 4
    evidence_labels = " ".join(f["evidence_label"].lower() for f in findings_acc)
    types = {f["type"] for f in findings_acc}

    # 1. Date of birth
    assert "birth" in evidence_labels
    # 2. Marital status
    assert "marital" in evidence_labels
    # 3. Graduation year (>15 years)
    assert "graduation" in evidence_labels
    # 4. Gendered wording
    assert "gender" in evidence_labels or "gender_bias" in types

    # Ensure sensitive raw text (like 1988-04-12 or Married) is NOT leaked in evidence_labels
    assert "1988-04-12" not in evidence_labels
    assert "married" not in [f["evidence_label"].lower() for f in findings_acc]

    # ds_clean must pass cleanly
    ds_path = FIXTURES_DIR / "ds_clean.pdf"
    ds_parsed = extract_entities(parse_resume(ds_path.read_bytes(), str(ds_path)))
    findings_ds = bias_check(ds_parsed)
    assert findings_ds == []


def test_skill_gap_mapping() -> None:
    """skill_gap maps missing skills to curated courses without URLs."""
    resume_skills = ["Python", "SQL", "Pandas"]
    jd_required = ["Python", "Kubernetes", "Docker", "AWS", "Apache Spark"]

    gap = skill_gap(resume_skills, jd_required)

    # Python is present in resume; must not be in gap
    assert "Python" not in gap
    assert "python" not in gap

    # Missing skills must be mapped
    assert "Kubernetes" in gap
    assert "Docker" in gap
    assert "AWS" in gap
    assert "Apache Spark" in gap

    for skill, courses in gap.items():
        assert len(courses) >= 1
        for c in courses:
            assert "title" in c
            assert "provider" in c
            assert "level" in c
            assert "estimated_hours" in c
            # Hard requirement: NO URLs in course catalog
            for val in c.values():
                assert "http://" not in str(val)
                assert "https://" not in str(val)
                assert ".com" not in str(val)


def test_sqlite_history_opt_in_and_no_text_fields(tmp_path) -> None:
    """SQLite history requires explicit opt-in and contains no resume or candidate text fields."""
    db_file = tmp_path / "test_history.db"

    mock_score = ScoreResult(
        total=84.5,
        grade="A-",
        components={
            "semantic_match": 0.88,
            "keyword_coverage": 0.80,
            "section_completeness": 1.0,
            "formatting": 0.90,
            "experience_fit": 0.85,
            "quantified_achievements": 0.75,
            "action_verbs": 0.80,
            "readability": 0.85,
            "contact_completeness": 0.90,
        },
    )

    # 1. When opt_in is False (default) -> Nothing written
    res_false = record_history(mock_score, jd_title="Data Scientist", db_path=db_file, opt_in=False)
    assert res_false is False
    assert not db_file.exists() or len(get_history(db_file)) == 0

    # 2. When opt_in is True -> Record written
    res_true = record_history(mock_score, jd_title="Data Scientist", db_path=db_file, opt_in=True)
    assert res_true is True
    records = get_history(db_file)
    assert len(records) == 1
    assert records[0]["total_score"] == pytest.approx(84.5)
    assert records[0]["jd_title"] == "Data Scientist"

    # 3. Check table schema via PRAGMA table_info
    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(score_history);")
        columns = [row[1] for row in cursor.fetchall()]

    # Required columns
    assert "timestamp" in columns
    assert "jd_title" in columns
    assert "total_score" in columns
    assert "grade" in columns
    assert "semantic_match" in columns

    # Hard requirement: NO text fields for resume or candidate details
    forbidden = ["resume_text", "raw_text", "text", "name", "email", "phone", "contact", "summary", "bullets"]
    for col in columns:
        assert col not in forbidden, f"Forbidden personal/resume text column detected in SQLite: {col}"


def test_empty_inputs_handled_in_extras() -> None:
    """Extras functions handle empty or None inputs without raising exceptions."""
    assert rank_jds("", []) == []
    assert skill_gap([], []) == {}
    assert bias_check("") == []
    assert bias_check(ParsedResume(raw_text="")) == []
    assert record_history(None, opt_in=True) is False
    assert get_history("nonexistent_db_12345.db") == []
