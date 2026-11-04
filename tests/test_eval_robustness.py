"""
tests/test_eval_robustness.py
=============================
Tests for scripts/eval_robustness.py and scripts/classifier_ci.py:
  - eval_robustness scans fixtures directory, outputs valid records without privacy leakage.
  - classifier_ci bootstrap logic and contingency matrix correctness.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
from sklearn.metrics import f1_score

from scripts.eval_robustness import evaluate_folder
from scripts.classifier_ci import compute_paired_comparison, run_bootstrap_ci

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_eval_robustness_on_fixtures(capsys) -> None:
    """evaluate_folder must run on fixtures without error and output privacy-safe records."""
    records = evaluate_folder(FIXTURES_DIR)
    assert len(records) >= 5
    captured = capsys.readouterr().out

    # Check headers
    assert "ROBUSTNESS EVALUATION" in captured
    assert "ROBUSTNESS SUMMARY COUNTS" in captured

    # Verify privacy: no candidate email patterns or resume body text in output
    assert "@" not in captured
    assert "resume text" not in captured.lower()
    assert "john doe" not in captured.lower()
    assert "jane doe" not in captured.lower()
    assert "555-" not in captured

    # Verify structure of records
    for rec in records:
        assert "filename" in rec
        assert "pages" in rec
        assert "method" in rec
        assert "seconds" in rec
        assert "sections" in rec
        assert "skills" in rec
        assert "years" in rec
        assert "warnings" in rec
        assert rec["status"] == "success"


def test_eval_robustness_nonexistent_folder(capsys) -> None:
    """evaluate_folder on nonexistent folder returns empty list and handles gracefully."""
    records = evaluate_folder(Path("data/nonexistent_folder_xyz"))
    assert records == []
    captured = capsys.readouterr().out
    assert "does not exist" in captured


def test_classifier_ci_bootstrap_math() -> None:
    """run_bootstrap_ci returns valid 95% intervals containing the point estimate."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=50)
    # create slightly noisy predictions
    y_pred = y_true.copy()
    flip_idx = rng.choice(50, size=10, replace=False)
    y_pred[flip_idx] = (y_pred[flip_idx] + 1) % 3

    ci_res, boot_f1, boot_acc = run_bootstrap_ci(
        y_true,
        {"test_model": y_pred},
        n_resamples=100,
        seed=42,
    )

    stats = ci_res["test_model"]
    assert stats["ci_95_lower"] <= stats["test_macro_f1"] <= stats["ci_95_upper"]
    assert 0.0 <= stats["test_accuracy"] <= 1.0


def test_classifier_ci_paired_comparison() -> None:
    """compute_paired_comparison returns valid contingency table and McNemar statistics."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    preds_lr = np.array([0, 1, 2, 0, 1, 0, 0, 1, 2, 1])   # 8/10 correct
    preds_mlp = np.array([0, 1, 2, 0, 1, 2, 0, 0, 2, 1])  # 8/10 correct

    boot_dist_mlp_f1 = np.array([0.8, 0.82, 0.79])
    boot_dist_lr_f1 = np.array([0.8, 0.80, 0.81])
    boot_dist_mlp_acc = np.array([0.8, 0.8, 0.8])
    boot_dist_lr_acc = np.array([0.8, 0.8, 0.8])

    paired = compute_paired_comparison(
        y_true,
        preds_mlp,
        preds_lr,
        boot_dist_mlp_f1,
        boot_dist_lr_f1,
        boot_dist_mlp_acc,
        boot_dist_lr_acc,
        "MLP",
        "LR",
    )

    assert "macro_f1" in paired
    assert "accuracy" in paired
    assert "mcnemar" in paired
    assert paired["macro_f1"]["diff"] == round(float(f1_score(y_true, preds_mlp, average="macro") - f1_score(y_true, preds_lr, average="macro")), 4)
    tbl = paired["mcnemar"]
    assert tbl["both_correct"] + tbl["both_wrong"] + tbl["b_correct_a_wrong"] + tbl["a_correct_b_wrong"] == 10

