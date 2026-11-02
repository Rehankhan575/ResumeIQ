"""test_audit_helpers.py – Offline unit tests for data audit helper functions.

Strictly offline: all tests run on synthetic/fake data in-memory.
No external network or raw data dependencies.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_data import (
    check_annotation_alignment,
    check_label_leak,
    compute_entity_counts_per_resume,
    compute_overlap_stats,
    compute_tfidf_cosine_near_duplicates,
    count_near_duplicates_by_columns,
    detect_pii,
    normalise_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Alignment Checking
# ─────────────────────────────────────────────────────────────────────────────

def test_check_annotation_alignment_exact_and_case() -> None:
    """Test exact and case-insensitive alignment checks."""
    text = "Proficient in Python and TypeScript development."
    # Python is at offsets [14, 20]
    assert check_annotation_alignment(text, 14, 20, "SKILL: Python") is True
    assert check_annotation_alignment(text, 14, 20, "SKILL: python") is True
    assert check_annotation_alignment(text, 14, 20, "SKILL: PYTHON") is True


def test_check_annotation_alignment_whitespace_and_prefix() -> None:
    """Test stripped whitespace and label without 'SKILL: ' prefix."""
    text = "Skills:   Docker   and Kubernetes."
    # Slice [7, 19] covers "   Docker   "
    assert check_annotation_alignment(text, 7, 19, "SKILL: Docker") is True
    assert check_annotation_alignment(text, 7, 19, "Docker") is True


def test_check_annotation_alignment_mismatch_and_bounds() -> None:
    """Test mismatch and boundary violation conditions."""
    text = "Experienced in Java."
    # Mismatch
    assert check_annotation_alignment(text, 15, 19, "SKILL: Python") is False

    # Out of bounds
    assert check_annotation_alignment(text, -1, 5, "SKILL: Java") is False
    assert check_annotation_alignment(text, 15, 100, "SKILL: Java") is False
    assert check_annotation_alignment(text, 19, 15, "SKILL: Java") is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_normalise_text_standard() -> None:
    """Test standard alphanumeric normalisation and case folding."""
    raw = "Senior Software Engineer / Architect (Backend & API) #1!"
    expected = "seniorsoftwareengineerarchitectbackendapi1"
    assert normalise_text(raw) == expected


def test_normalise_text_empty_and_special() -> None:
    """Test empty string and string with only punctuation."""
    assert normalise_text("") == ""
    assert normalise_text("--- /// !!! @@@") == ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Label Leak Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_check_label_leak_exact_and_main_word() -> None:
    """Test label leak detection for exact category and main words."""
    # Exact category match in leading text
    resume_sales = "Sales Representative with 5+ years driving B2B pipeline growth."
    assert check_label_leak(resume_sales, "SALES", prefix_chars=100) is True

    # Main word match in compound category
    resume_it = "Information Technology Specialist with cloud migration experience."
    assert check_label_leak(resume_it, "INFORMATION-TECHNOLOGY", prefix_chars=100) is True

    # Main word single term match (e.g. Technology only)
    resume_tech = "Technology Analyst supporting global enterprise networks."
    assert check_label_leak(resume_tech, "INFORMATION-TECHNOLOGY", prefix_chars=100) is True

    # Short category acronym
    resume_hr = "HR Generalist managing benefits administration and onboarding."
    assert check_label_leak(resume_hr, "HR", prefix_chars=100) is True


def test_check_label_leak_negative_and_out_of_prefix() -> None:
    """Test that label leak correctly rejects non-matching text and delayed mentions."""
    # Completely unrelated category
    resume_chef = "Executive Chef with Michelin-star kitchen experience."
    assert check_label_leak(resume_chef, "ACCOUNTANT", prefix_chars=100) is False

    # Empty text
    assert check_label_leak("", "SALES", prefix_chars=100) is False

    # Word appears only far down in document (beyond prefix_chars)
    padded = ("x" * 200) + " sales representative"
    assert check_label_leak(padded, "SALES", prefix_chars=100) is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Near-Duplicate Column Counting
# ─────────────────────────────────────────────────────────────────────────────

def test_count_near_duplicates_by_columns_with_dups() -> None:
    """Test grouping and counting duplicates across subsets of columns."""
    df = pd.DataFrame([
        {"title": "Engineer", "desc_prefix": "Build APIs", "id": 1},
        {"title": "Engineer", "desc_prefix": "Build APIs", "id": 2},  # dup group 1
        {"title": "Manager", "desc_prefix": "Lead team", "id": 3},
        {"title": "Manager", "desc_prefix": "Lead team", "id": 4},  # dup group 2
        {"title": "Designer", "desc_prefix": "Design UI", "id": 5},  # unique
    ])
    total_dup_rows, distinct_groups = count_near_duplicates_by_columns(df, ["title", "desc_prefix"])
    assert total_dup_rows == 4
    assert distinct_groups == 2


def test_count_near_duplicates_by_columns_unique_and_empty() -> None:
    """Test duplicate counting with no duplicate rows and empty DataFrame."""
    df_unique = pd.DataFrame([
        {"title": "Engineer", "desc": "A"},
        {"title": "Designer", "desc": "B"},
    ])
    total, groups = count_near_duplicates_by_columns(df_unique, ["title"])
    assert total == 0
    assert groups == 0

    df_empty = pd.DataFrame(columns=["title"])
    total_e, groups_e = count_near_duplicates_by_columns(df_empty, ["title"])
    assert total_e == 0
    assert groups_e == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. TF-IDF Cosine Near-Duplicates
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_tfidf_cosine_near_duplicates_identical_and_orthogonal() -> None:
    """Test pairwise TF-IDF cosine duplicate counting."""
    # 2 identical texts -> cosine sim = 1.0
    texts = [
        "python machine learning pytorch transformer deep neural network",
        "python machine learning pytorch transformer deep neural network",
    ]
    res = compute_tfidf_cosine_near_duplicates(texts, threshold_high=0.95, threshold_low=0.90)
    assert res["count_high"] == 1
    assert res["count_low"] == 1
    assert res["cross_cat_high"] == 0

    # 2 disjoint texts -> cosine sim = 0.0
    disjoint_texts = [
        "quantum physics particle accelerator boson",
        "gardening tomato planting compost fertilizer",
    ]
    res_disjoint = compute_tfidf_cosine_near_duplicates(disjoint_texts)
    assert res_disjoint["count_high"] == 0
    assert res_disjoint["count_low"] == 0


def test_compute_tfidf_cosine_near_duplicates_cross_category() -> None:
    """Test cross-category tracking when high similarity pairs span different categories."""
    texts = [
        "experienced software developer building scalable backend microservices",
        "experienced software developer building scalable backend microservices",
    ]
    categories = ["INFORMATION-TECHNOLOGY", "ENGINEERING"]
    res = compute_tfidf_cosine_near_duplicates(texts, categories=categories)
    assert res["count_high"] == 1
    assert res["cross_cat_high"] == 1


def test_compute_tfidf_cosine_single_text() -> None:
    """Test edge case with fewer than 2 documents."""
    res = compute_tfidf_cosine_near_duplicates(["single document"])
    assert res["count_high"] == 0
    assert res["count_low"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. PII Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_pii_emails_and_phones() -> None:
    """Test detection of emails and various phone formats."""
    text_email = "Reach out via jane.doe@company.org for details."
    res = detect_pii(text_email)
    assert res["has_email"] is True
    assert res["has_phone"] is False

    text_phone = "Direct line: +1 (555) 234-5678 or 555-876-5432."
    res_phone = detect_pii(text_phone)
    assert res_phone["has_email"] is False
    assert res_phone["has_phone"] is True

    text_both = "Contact: john@acme.com, phone: 415-555-0199."
    res_both = detect_pii(text_both)
    assert res_both["has_email"] is True
    assert res_both["has_phone"] is True


def test_detect_pii_clean_text() -> None:
    """Test that text without contact info returns false."""
    clean = "Over 10 years of experience architecting distributed cloud infrastructure."
    res = detect_pii(clean)
    assert res["has_email"] is False
    assert res["has_phone"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entity Counts Per Resume
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_entity_counts_per_resume() -> None:
    """Test min, median, max entity count aggregation per label."""
    resumes = [
        [{"label": "NAME"}, {"label": "SKILL"}, {"label": "SKILL"}],
        [{"label": "NAME"}, {"label": "SKILL"}],
        [{"label": "NAME"}, {"label": "SKILL"}, {"label": "SKILL"}, {"label": "SKILL"}],
    ]
    stats = compute_entity_counts_per_resume(resumes, labels=["NAME", "SKILL", "CERT"])
    # NAME: strictly 1 per resume
    assert stats["NAME"]["min"] == 1.0
    assert stats["NAME"]["median"] == 1.0
    assert stats["NAME"]["max"] == 1.0

    # SKILL: 2, 1, 3 -> min 1, median 2, max 3
    assert stats["SKILL"]["min"] == 1.0
    assert stats["SKILL"]["median"] == 2.0
    assert stats["SKILL"]["max"] == 3.0

    # CERT: 0, 0, 0
    assert stats["CERT"]["min"] == 0.0
    assert stats["CERT"]["median"] == 0.0
    assert stats["CERT"]["max"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Overlap Stats
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_overlap_stats() -> None:
    """Test set overlap and difference calculations."""
    set_a = {"python", "sql", "git"}
    set_b = {"git", "docker", "aws"}

    res = compute_overlap_stats(set_a, set_b)
    assert res["size_a"] == 3
    assert res["size_b"] == 3
    assert res["intersection_size"] == 1
    assert res["only_a_size"] == 2
    assert res["only_b_size"] == 2
