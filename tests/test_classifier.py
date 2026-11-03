"""
tests/test_classifier.py
========================
Unit tests for Stage 3a Resume Category Classifier.

Tests:
1. classify() output shape: exactly top-3 CategoryPrediction items.
2. classify() probabilities sum: top-3 probabilities sum to 1.0 (normalized) and are valid floats in [0, 1].
3. classify() fallback behavior: when model directory/files are missing, falls back cleanly to on-the-fly baseline without crashing.
4. Data leakage check: train, val, and test splits in data/processed have ZERO duplicate texts (text_masked).
5. classify_resume(): integration with ParsedResume schema.
6. Empty / whitespace string handling: returns valid graceful top-3 predictions without crashing.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.models.classifier import classify, classify_resume, clear_classifier_cache
from app.schemas import CategoryPrediction, ParsedResume


def test_classifier_output_shape():
    """Verify classify() returns exactly top-3 CategoryPrediction objects."""
    text = (
        "Experienced Software Engineer with 5 years in Python, FastAPI, Docker, and AWS. "
        "Built distributed microservices, optimized SQL databases, and led agile sprint planning."
    )
    preds = classify(text, top_k=3)
    assert len(preds) == 3
    for p in preds:
        assert isinstance(p, CategoryPrediction)
        assert isinstance(p.category, str)
        assert len(p.category) > 0
        assert 0.0 <= p.probability <= 1.0


def test_classifier_probabilities_sum():
    """Verify classify() probabilities sum to 1.0 when normalized."""
    text = (
        "Registered Nurse with 8 years of clinical patient care in intensive care unit (ICU), "
        "BLS and ACLS certified, medication administration, vitals monitoring, and patient triage."
    )
    preds = classify(text, top_k=3, normalize=True)
    assert len(preds) == 3
    total_prob = sum(p.probability for p in preds)
    assert total_prob == pytest.approx(1.0, abs=1e-3)

    # Probabilities should be in descending order
    for i in range(len(preds) - 1):
        assert preds[i].probability >= preds[i + 1].probability


def test_classifier_fallback_on_missing_model(tmp_path: Path):
    """Verify that when model files are missing, classify falls back cleanly to on-the-fly baseline without crashing."""
    clear_classifier_cache()
    # Mock MODEL_DIR to point to an empty temporary path
    with patch("app.models.classifier.MODEL_DIR", tmp_path):
        text = "Financial analyst with experience in budgeting, forecasting, Excel VBA, and financial reporting."
        preds = classify(text, top_k=3)
        assert len(preds) == 3
        for p in preds:
            assert isinstance(p, CategoryPrediction)
            assert 0.0 <= p.probability <= 1.0
        assert sum(p.probability for p in preds) == pytest.approx(1.0, abs=1e-3)
    clear_classifier_cache()


def test_data_splits_zero_leakage():
    """Strict leakage check: train, val, and test splits must contain ZERO overlapping masked texts."""
    train_path = Path("data/processed/classifier_train.parquet")
    val_path = Path("data/processed/classifier_val.parquet")
    test_path = Path("data/processed/classifier_test.parquet")

    assert train_path.exists(), "classifier_train.parquet not found"
    assert val_path.exists(), "classifier_val.parquet not found"
    assert test_path.exists(), "classifier_test.parquet not found"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    train_texts = set(train_df["text_masked"].dropna().str.strip())
    val_texts = set(val_df["text_masked"].dropna().str.strip())
    test_texts = set(test_df["text_masked"].dropna().str.strip())

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    assert len(train_val_overlap) == 0, f"Found {len(train_val_overlap)} duplicate texts between train and val!"
    assert len(train_test_overlap) == 0, f"Found {len(train_test_overlap)} duplicate texts between train and test!"
    assert len(val_test_overlap) == 0, f"Found {len(val_test_overlap)} duplicate texts between val and test!"


def test_classify_resume_integration():
    """Verify classify_resume() consumes a ParsedResume schema object."""
    parsed = ParsedResume(
        raw_text="Executive Chef with 12 years culinary experience in fine dining, menu design, kitchen management.",
        sections={"experience": "Head Chef at Bistro 2018-Present"},
    )
    preds = classify_resume(parsed)
    assert len(preds) == 3
    assert all(isinstance(p, CategoryPrediction) for p in preds)
    assert sum(p.probability for p in preds) == pytest.approx(1.0, abs=1e-3)


def test_classifier_empty_and_whitespace_input():
    """Verify classify() handles empty or whitespace text gracefully without raising exceptions."""
    preds_empty = classify("")
    assert len(preds_empty) == 3
    assert sum(p.probability for p in preds_empty) == pytest.approx(1.0, abs=1e-3)

    preds_space = classify("   \n\t  ")
    assert len(preds_space) == 3
    assert sum(p.probability for p in preds_space) == pytest.approx(1.0, abs=1e-3)

