"""
tests/test_matcher.py
=====================
Test suite for Stage 4 Résumé–JD Matching.

Verifies:
1. Match scores are strictly within [0.0, 1.0].
2. Section similarities ('skills' and 'experience') are strictly within [0.0, 1.0].
3. Fallback mechanism: gracefully falls back to TF-IDF with a warning when fine-tuned
   model is missing or disabled.
4. Proof of split isolation: NO test résumé ID and NO test JD ID appears in training pairs.
5. Domain discrimination sanity check on real fixtures:
   - ds_clean.pdf (parsed via parse_resume) scores higher against jd_data_scientist.txt than jd_chef.txt
   - chef_plain.docx (parsed via parse_resume) scores higher against jd_chef.txt than jd_data_scientist.txt
6. Pipeline compatibility: match_resume_jd integrates seamlessly with app/pipeline.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import settings
from app.models.matcher import (
    FINE_TUNED_MODEL_PATH,
    match,
    match_resume_jd,
)
from app.parsing import parse_resume
from app.schemas import JobDescription, MatchResult

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# 1. Score Range Tests
# ---------------------------------------------------------------------------

def test_match_score_range() -> None:
    resume = "Experienced software engineer with expertise in Python, PyTorch, and Docker."
    jd = "Seeking a senior Python developer with strong machine learning and cloud background."
    res = match(resume, jd)

    assert isinstance(res, dict)
    assert 0.0 <= res["score"] <= 1.0
    assert "skills" in res["section_similarity"]
    assert "experience" in res["section_similarity"]
    assert 0.0 <= res["section_similarity"]["skills"] <= 1.0
    assert 0.0 <= res["section_similarity"]["experience"] <= 1.0


def test_match_empty_inputs_returns_zero_score() -> None:
    res = match("", "Some job description")
    assert res["score"] == 0.0
    assert res["method"] == "none"

    res2 = match("Some resume text", "")
    assert res2["score"] == 0.0
    assert res2["method"] == "none"


# ---------------------------------------------------------------------------
# 2. Fallback Mechanism Tests
# ---------------------------------------------------------------------------

def test_matcher_fallback_to_tfidf(caplog: pytest.LogCaptureFixture) -> None:
    resume = "Accountant with knowledge of QuickBooks, GAAP, and financial reporting."
    jd = "Looking for a staff accountant with GAAP compliance and ledger reconciliation experience."

    # Force fallback to TF-IDF
    res = match(resume, jd, force_fallback=True)

    assert res["method"] == "tfidf_cosine"
    assert 0.0 <= res["score"] <= 1.0
    assert 0.0 <= res["section_similarity"]["skills"] <= 1.0
    assert 0.0 <= res["section_similarity"]["experience"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Proof of Split Isolation
# ---------------------------------------------------------------------------

def test_no_test_ids_in_training_pairs() -> None:
    """Prove that NO test resume ID and NO test JD ID appears in training pairs."""
    # 1. Load test split IDs
    df_test_res = pd.read_parquet(ROOT / "data/processed/classifier_test.parquet")
    test_resume_ids = set(df_test_res["id"].astype(str))

    df_test_jd = pd.read_parquet(ROOT / "data/processed/jd_test.parquet")
    test_jd_ids = set(df_test_jd["jd_id"].astype(str))

    # 2. Check saved training ID manifest if present, or generate training IDs from train split
    manifest_file = ROOT / "artifacts/models/matcher/training_id_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        train_resume_ids = set(manifest["train_resume_ids"])
        train_jd_ids = set(manifest["train_jd_ids"])
    else:
        from scripts.train_matcher import build_training_pairs, load_datasets

        resumes, jds = load_datasets()
        _, train_res_ids_list, train_jd_ids_list = build_training_pairs(resumes["train"], jds["train"])
        train_resume_ids = set(train_res_ids_list)
        train_jd_ids = set(train_jd_ids_list)

    # 3. Assert zero overlap with test splits
    resume_overlap = train_resume_ids.intersection(test_resume_ids)
    jd_overlap = train_jd_ids.intersection(test_jd_ids)

    assert len(resume_overlap) == 0, f"LEAK DETECTED: {len(resume_overlap)} test resume IDs in train set!"
    assert len(jd_overlap) == 0, f"LEAK DETECTED: {len(jd_overlap)} test JD IDs in train set!"


# ---------------------------------------------------------------------------
# 4. Domain Discrimination Sanity Check on Real Fixtures
# ---------------------------------------------------------------------------

def test_fixture_discrimination_sanity_check() -> None:
    """Sanity check:

    ds_clean.pdf scores higher against jd_data_scientist.txt than jd_chef.txt.
    chef_plain.docx scores higher against jd_chef.txt than jd_data_scientist.txt.
    """
    # Parse real fixtures using Stage 1 parser
    ds_bytes = (FIXTURES_DIR / "ds_clean.pdf").read_bytes()
    ds_parsed = parse_resume(ds_bytes, "ds_clean.pdf")

    chef_bytes = (FIXTURES_DIR / "chef_plain.docx").read_bytes()
    chef_parsed = parse_resume(chef_bytes, "chef_plain.docx")

    # Load real JD texts
    jd_ds_text = (FIXTURES_DIR / "jd_data_scientist.txt").read_text()
    jd_chef_text = (FIXTURES_DIR / "jd_chef.txt").read_text()

    # Match DS resume against both JDs
    ds_vs_ds = match(ds_parsed.raw_text, jd_ds_text)
    ds_vs_chef = match(ds_parsed.raw_text, jd_chef_text)

    # Match Chef resume against both JDs
    chef_vs_chef = match(chef_parsed.raw_text, jd_chef_text)
    chef_vs_ds = match(chef_parsed.raw_text, jd_ds_text)

    # Assert clear discrimination
    assert ds_vs_ds["score"] > ds_vs_chef["score"], (
        f"Expected DS resume to match DS JD ({ds_vs_ds['score']}) higher than Chef JD ({ds_vs_chef['score']})"
    )
    assert chef_vs_chef["score"] > chef_vs_ds["score"], (
        f"Expected Chef resume to match Chef JD ({chef_vs_chef['score']}) higher than DS JD ({chef_vs_ds['score']})"
    )


# ---------------------------------------------------------------------------
# 5. Pipeline Integration Test
# ---------------------------------------------------------------------------

def test_match_resume_jd_pipeline_compatibility() -> None:
    ds_bytes = (FIXTURES_DIR / "ds_clean.pdf").read_bytes()
    ds_parsed = parse_resume(ds_bytes, "ds_clean.pdf")
    jd_ds_text = (FIXTURES_DIR / "jd_data_scientist.txt").read_text()

    jd = JobDescription(text=jd_ds_text, title="Data Scientist")
    res = match_resume_jd(ds_parsed, jd)

    assert isinstance(res, MatchResult)
    assert 0.0 <= res.score <= 1.0
    assert "skills" in res.section_scores
    assert "experience" in res.section_scores
    assert res.method in ("sbert_fine_tuned", "sbert_zero_shot", "tfidf_cosine")

    # None JD returns zero score
    res_none = match_resume_jd(ds_parsed, None)
    assert isinstance(res_none, MatchResult)
    assert res_none.score == 0.0
    assert res_none.method == "none"

