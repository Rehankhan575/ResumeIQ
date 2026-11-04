"""
tests/test_scoring.py
=====================
Unit and integration tests for ATS scoring (Stage 5) and Explainability (Stage 6).

Verifies:
  1. Ambiguous terms: "rn" rule present with require_context.
  2. Weights: SCORE_WEIGHTS sum to 1.0 and cover all 9 mandatory components.
  3. Formatting: layout metadata correctly penalizes multi-column, tables, bad page/word counts.
  4. Scoring properties:
     - Total score within [0, 100]
     - Deterministic output
     - Monotonicity: adding missing required keyword never lowers score
     - Empty/corrupted input yields low score and clear issue without crashing
     - Issues sorted strictly descending by estimated point gain
  5. Explainability:
     - Highlight character offsets match the raw text exactly
     - render_html correctly escapes raw HTML/script tags
"""

from __future__ import annotations

from pathlib import Path
import html
import pandas as pd
import pytest

from app.config import ACTION_VERBS, SCORE_WEIGHTS, settings
from app.extraction.extractor import extract_entities
from app.parsing.parser import parse_resume
from app.schemas import ContactInfo, Highlight, Issue, JobDescription, LayoutMeta, ParsedResume, ScoreResult
from app.scoring.ats import score_resume
from app.scoring.explain import add_highlights, compute_sentence_contributions, render_html
from app.scoring.formatting import score_formatting

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TAXONOMIES_DIR = Path(__file__).resolve().parent.parent / "data" / "taxonomies"


# ---------------------------------------------------------------------------
# 0. Ambiguous Terms ("rn" Rule)
# ---------------------------------------------------------------------------

def test_rn_rule_in_ambiguous_terms_csv() -> None:
    """ambiguous_terms.csv must contain 'rn' with rule 'require_context'."""
    csv_path = TAXONOMIES_DIR / "ambiguous_terms.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    df = pd.read_csv(csv_path, keep_default_na=False)

    df["term_clean"] = df["term"].astype(str).str.strip().str.lower()
    df["rule_clean"] = df["rule"].astype(str).str.strip().str.lower()

    rn_rows = df[df["term_clean"] == "rn"]
    assert not rn_rows.empty, "Term 'rn' not found in ambiguous_terms.csv"
    assert rn_rows.iloc[0]["rule_clean"] == "require_context", (
        f"Expected rule 'require_context' for 'rn', got {rn_rows.iloc[0]['rule_clean']}"
    )


# ---------------------------------------------------------------------------
# 1. Config Weights
# ---------------------------------------------------------------------------

def test_score_weights_sum_to_one() -> None:
    """All 9 components must be present in SCORE_WEIGHTS and sum to 1.0."""
    expected_components = {
        "semantic_match",
        "keyword_coverage",
        "section_completeness",
        "formatting",
        "experience_fit",
        "quantified_achievements",
        "action_verbs",
        "readability",
        "contact_completeness",
    }
    assert set(SCORE_WEIGHTS.keys()) == expected_components
    total_weight = sum(SCORE_WEIGHTS.values())
    assert total_weight == pytest.approx(1.0, rel=1e-5)


def test_action_verbs_non_empty() -> None:
    """ACTION_VERBS must contain common resume power verbs."""
    assert len(ACTION_VERBS) >= 50
    assert "developed" in ACTION_VERBS
    assert "led" in ACTION_VERBS
    assert "built" in ACTION_VERBS


# ---------------------------------------------------------------------------
# 2. Formatting Scorer
# ---------------------------------------------------------------------------

def test_formatting_clean_layout() -> None:
    """Clean single-column layout without tables or images gets high score."""
    clean_layout = LayoutMeta(
        file_type="pdf",
        page_count=1,
        word_count=500,
        has_tables=False,
        has_images=False,
        is_multi_column=False,
    )
    score, issues = score_formatting(clean_layout)
    assert score >= 0.90
    assert len(issues) == 0


def test_formatting_penalties() -> None:
    """Multi-column, tables, images, and long page count incur deductions."""
    messy_layout = LayoutMeta(
        file_type="pdf",
        page_count=5,
        word_count=2000,
        has_tables=True,
        has_images=True,
        is_multi_column=True,
    )
    score, issues = score_formatting(messy_layout)
    assert score < 0.50
    assert len(issues) >= 4
    issue_msgs = [i.message.lower() for i in issues]
    assert any("multi-column" in m for m in issue_msgs)
    assert any("tables" in m for m in issue_msgs)


# ---------------------------------------------------------------------------
# 3. Score Properties
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_resume() -> ParsedResume:
    fpath = FIXTURES_DIR / "ds_clean.pdf"
    parsed = parse_resume(fpath.read_bytes(), str(fpath))
    return extract_entities(parsed)


@pytest.fixture
def sample_jd() -> JobDescription:
    jpath = FIXTURES_DIR / "jd_data_scientist.txt"
    return JobDescription(
        text=jpath.read_text(encoding="utf-8"),
        title="Data Scientist",
        required_skills=["Python", "Machine Learning", "SQL", "Pandas"],
        min_years=3.0,
    )


def test_score_within_range(sample_resume: ParsedResume, sample_jd: JobDescription) -> None:
    """ScoreResult.total must be in [0, 100] and components in [0, 1]."""
    result = score_resume(sample_resume, sample_jd)
    assert 0.0 <= result.total <= 100.0
    assert result.grade in {"A", "B", "C", "D", "F"}
    for comp, val in result.components.items():
        assert 0.0 <= val <= 1.0, f"Component {comp} out of bounds: {val}"


def test_score_deterministic(sample_resume: ParsedResume, sample_jd: JobDescription) -> None:
    """Calling score_resume repeatedly with same inputs yields identical results."""
    r1 = score_resume(sample_resume, sample_jd)
    r2 = score_resume(sample_resume, sample_jd)
    assert r1.total == r2.total
    assert r1.grade == r2.grade
    assert r1.components == r2.components


def test_adding_missing_required_keyword_never_lowers_score(
    sample_resume: ParsedResume, sample_jd: JobDescription
) -> None:
    """Adding a missing required keyword must never lower total score or keyword coverage."""
    initial = score_resume(sample_resume, sample_jd)
    if not initial.missing_skills:
        pytest.skip("No missing skills to test in sample")

    missing_kw = initial.missing_skills[0]

    # Add missing keyword to resume skills and raw_text
    modified_resume = sample_resume.model_copy(deep=True)
    modified_resume.skills.append(missing_kw)
    modified_resume.raw_text += f"\nProficient in {missing_kw}."

    modified_res = score_resume(modified_resume, sample_jd)
    assert modified_res.components["keyword_coverage"] >= initial.components["keyword_coverage"]
    assert modified_res.total >= initial.total


def test_empty_or_corrupt_resume_low_score_and_issue() -> None:
    """Empty or corrupt resume must produce total 0.0, grade F, and clear issue without crash."""
    empty_resume = ParsedResume(raw_text="", sections={}, contact=ContactInfo(), layout=LayoutMeta())
    jd = JobDescription(text="Need Python developer", required_skills=["Python"])

    res = score_resume(empty_resume, jd)
    assert res.total == 0.0
    assert res.grade == "F"
    assert len(res.issues) >= 1
    assert any("empty" in i.message.lower() for i in res.issues)


def test_issues_sorted_by_estimated_gain(sample_resume: ParsedResume, sample_jd: JobDescription) -> None:
    """Issues must be sorted strictly descending by estimated_gain."""
    res = score_resume(sample_resume, sample_jd)
    gains = [i.estimated_gain for i in res.issues]
    assert gains == sorted(gains, reverse=True)


# ---------------------------------------------------------------------------
# 4. Explainability & Highlights
# ---------------------------------------------------------------------------

def test_highlight_offsets_match_text(sample_resume: ParsedResume, sample_jd: JobDescription) -> None:
    """Character offsets of highlights must accurately slice the original raw_text."""
    res = score_resume(sample_resume, sample_jd)
    res = add_highlights(res, sample_resume, sample_jd)

    assert len(res.highlights) > 0
    raw_text = sample_resume.raw_text

    for h in res.highlights:
        if h.type == "keyword_missing":
            assert h.start == 0 and h.end == 0
            continue
        assert 0 <= h.start < h.end <= len(raw_text)
        assert raw_text[h.start:h.end] == h.text


def test_render_html_escapes_content() -> None:
    """render_html must escape raw HTML tags and entities."""
    malicious_text = 'Senior Engineer <script>alert("XSS")</script> & Co. Python expert.'
    py_start = malicious_text.index("Python")
    py_end = py_start + len("Python")
    highlights = [
        Highlight(start=0, end=15, text="Senior Engineer", weight=0.5, type="positive"),
        Highlight(start=py_start, end=py_end, text="Python", weight=1.0, type="keyword_match"),
    ]
    html_output = render_html(highlights, malicious_text)

    # Must NOT contain raw script tag
    assert "<script>" not in html_output
    assert "&lt;script&gt;" in html_output
    assert "&amp;" in html_output
    assert '<mark class="ats-highlight ats-highlight-positive"' in html_output
    assert '<mark class="ats-highlight ats-highlight-keyword_match"' in html_output


# ---------------------------------------------------------------------------
# 5. Unmeasurable Components Handling
# ---------------------------------------------------------------------------

def test_unmeasurable_components_fixture_matrix_unchanged() -> None:
    """The 15 fixture pairs produce scores matching preflight benchmarks to within 0.01."""
    from app.pipeline import run_pipeline

    expected_benchmarks = {
        ("accountant_table.pdf", "jd_chef.txt"): 53.41,
        ("accountant_table.pdf", "jd_data_scientist.txt"): 50.86,
        ("accountant_table.pdf", "jd_software_engineer.txt"): 50.18,
        ("chef_plain.docx", "jd_chef.txt"): 80.29,
        ("chef_plain.docx", "jd_data_scientist.txt"): 57.27,
        ("chef_plain.docx", "jd_software_engineer.txt"): 57.45,
        ("ds_clean.pdf", "jd_chef.txt"): 55.50,
        ("ds_clean.pdf", "jd_data_scientist.txt"): 84.88,
        ("ds_clean.pdf", "jd_software_engineer.txt"): 60.53,
        ("nurse_scanned.pdf", "jd_chef.txt"): 55.84,
        ("nurse_scanned.pdf", "jd_data_scientist.txt"): 55.16,
        ("nurse_scanned.pdf", "jd_software_engineer.txt"): 53.58,
        ("swe_two_column.pdf", "jd_chef.txt"): 49.88,
        ("swe_two_column.pdf", "jd_data_scientist.txt"): 54.42,
        ("swe_two_column.pdf", "jd_software_engineer.txt"): 73.75,
    }

    for (res_name, jd_name), expected_score in expected_benchmarks.items():
        res_path = FIXTURES_DIR / res_name
        jd_path = FIXTURES_DIR / jd_name
        res_bytes = res_path.read_bytes()
        jd_text = jd_path.read_text(encoding="utf-8")
        out = run_pipeline(res_bytes, res_name, jd_text)
        assert abs(out.total - expected_score) <= 0.01, (
            f"Score mismatch for {res_name} + {jd_name}: got {out.total:.2f}, expected {expected_score:.2f}"
        )


def test_unmeasurable_components_junk_jd(sample_resume: ParsedResume) -> None:
    """ds_clean vs junk JD 'n nnvhhv' scores at least 20 points below ds_clean vs jd_data_scientist."""
    jd_good_path = FIXTURES_DIR / "jd_data_scientist.txt"
    jd_good_text = jd_good_path.read_text(encoding="utf-8")
    jd_good = JobDescription(
        text=jd_good_text,
        title="Data Scientist",
        required_skills=["Python", "Machine Learning", "SQL", "Pandas"],
        min_years=3.0,
    )
    score_good = score_resume(sample_resume, jd_good)

    # Junk JD with no detectable skills and no min_years
    jd_junk = JobDescription(text="n nnvhhv", title="Unknown")
    score_junk = score_resume(sample_resume, jd_junk)

    # Must score at least 20 points below good match
    diff = score_good.total - score_junk.total
    assert diff >= 20.0, f"Expected drop >= 20.0 pts, got {diff:.2f} (good={score_good.total}, junk={score_junk.total})"

    # Unmeasurable component warnings must be populated
    assert any("keyword coverage" in w.lower() for w in score_junk.warnings)
    assert any("experience fit" in w.lower() for w in score_junk.warnings)

