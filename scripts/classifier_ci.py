"""
scripts/classifier_ci.py
========================
Bootstrap 95% confidence intervals (1,000 resamples, seed 42) for test set
macro-F1 across resume classification models, plus paired hypothesis testing
(bootstrap difference and McNemar test) for MLP vs. Logistic Regression.

Usage:
    python scripts/classifier_ci.py [--n-resamples 1000] [--seed 42]

Models evaluated:
    - tfidf_lr: TF-IDF + Logistic Regression
    - tfidf_mlp: TF-IDF + 2-layer MLP (PyTorch)
    - bilstm: Word-level Bidirectional LSTM (PyTorch)
    - distilbert: Fine-tuned DistilBERT (Transformers)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_classifier import (
    BiLSTMClassifier,
    BiLSTMDataset,
    DistilBERTDataset,
    TextMLP,
    WordVocab,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("classifier_ci")

MODEL_DIR = ROOT_DIR / "artifacts" / "models" / "classifier"
METRIC_DIR = ROOT_DIR / "artifacts" / "metrics" / "classifier"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "classifier_test.parquet"


def get_eval_device() -> torch.device:
    """Return CPU device for deterministic, safe inference."""
    return torch.device("cpu")


def load_test_data() -> Tuple[List[str], np.ndarray, List[str], Dict[str, int]]:
    """Load test set texts and labels without touching training pipeline."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    classes_path = MODEL_DIR / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_path}")

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    class2idx = {c: i for i, c in enumerate(classes)}

    df = pd.read_parquet(TEST_DATA_PATH)
    texts = df["text_masked"].tolist()
    labels = np.array([class2idx[c] for c in df["category"]], dtype=int)

    return texts, labels, classes, class2idx


def predict_tfidf_lr(texts: List[str]) -> Optional[np.ndarray]:
    """Generate predictions using TF-IDF + Logistic Regression."""
    vec_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    lr_path = MODEL_DIR / "tfidf_lr.joblib"
    if not vec_path.exists() or not lr_path.exists():
        logger.warning("TF-IDF LR model files missing; skipping tfidf_lr.")
        return None

    try:
        vec = joblib.load(vec_path)
        clf = joblib.load(lr_path)
        X = vec.transform(texts)
        return clf.predict(X)
    except Exception as e:
        logger.warning(f"Error predicting with tfidf_lr: {e}")
        return None


def predict_tfidf_mlp(texts: List[str], num_classes: int) -> Optional[np.ndarray]:
    """Generate predictions using TF-IDF + MLP."""
    vec_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    mlp_path = MODEL_DIR / "tfidf_mlp.pt"
    if not vec_path.exists() or not mlp_path.exists():
        logger.warning("TF-IDF MLP model files missing; skipping tfidf_mlp.")
        return None

    try:
        device = get_eval_device()
        vec = joblib.load(vec_path)
        X = vec.transform(texts)
        mlp = TextMLP(in_features=X.shape[1], num_classes=num_classes, dropout=0.3).to(device)
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp.eval()
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = mlp(X_tensor).argmax(dim=1).cpu().numpy()
        return preds
    except Exception as e:
        logger.warning(f"Error predicting with tfidf_mlp: {e}")
        return None


def predict_bilstm(texts: List[str], labels: np.ndarray, num_classes: int) -> Optional[np.ndarray]:
    """Generate predictions using BiLSTM."""
    vocab_path = MODEL_DIR / "bilstm_vocab.json"
    model_path = MODEL_DIR / "bilstm.pt"
    if not vocab_path.exists() or not model_path.exists():
        logger.warning("BiLSTM model files missing; skipping bilstm.")
        return None

    try:
        device = get_eval_device()
        vocab = WordVocab.load(vocab_path)
        bilstm = BiLSTMClassifier(
            vocab_size=len(vocab.w2i),
            embed_dim=128,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.3,
        ).to(device)
        bilstm.load_state_dict(torch.load(model_path, map_location=device))
        bilstm.eval()

        dataset = BiLSTMDataset(texts, labels, vocab, max_len=400)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds: List[int] = []
        with torch.no_grad():
            for bx, _ in loader:
                preds.extend(bilstm(bx.to(device)).argmax(dim=1).cpu().numpy())
        return np.array(preds)
    except Exception as e:
        logger.warning(f"Error predicting with bilstm: {e}")
        return None


def predict_distilbert(texts: List[str], labels: np.ndarray) -> Optional[np.ndarray]:
    """Generate predictions using fine-tuned DistilBERT."""
    bert_dir = MODEL_DIR / "distilbert"
    if not (bert_dir / "config.json").exists() or not (bert_dir / "model.safetensors").exists():
        logger.warning("DistilBERT checkpoint not found; skipping distilbert.")
        return None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = get_eval_device()
        tok = AutoTokenizer.from_pretrained(bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
        model.eval()

        dataset = DistilBERTDataset(texts, labels, tok, max_len=256)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds: List[int] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        return np.array(preds)
    except Exception as e:
        logger.warning(f"Error predicting with distilbert: {e}")
        return None


def run_bootstrap_ci(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Compute 95% bootstrap confidence intervals for Macro-F1 across models."""
    n_samples = len(y_true)
    rng = np.random.default_rng(seed)

    # Pre-generate bootstrap sample indices for paired consistency
    bootstrap_indices = [
        rng.choice(n_samples, size=n_samples, replace=True)
        for _ in range(n_resamples)
    ]

    boot_results: Dict[str, Dict[str, float]] = {}
    boot_distributions: Dict[str, np.ndarray] = {}

    for model_name, y_pred in model_predictions.items():
        point_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        point_acc = accuracy_score(y_true, y_pred)

        f1_samples = np.zeros(n_resamples, dtype=float)
        for b_idx, idx in enumerate(bootstrap_indices):
            f1_samples[b_idx] = f1_score(
                y_true[idx],
                y_pred[idx],
                average="macro",
                zero_division=0,
            )

        ci_low = float(np.percentile(f1_samples, 2.5))
        ci_high = float(np.percentile(f1_samples, 97.5))
        boot_distributions[model_name] = f1_samples

        boot_results[model_name] = {
            "test_macro_f1": float(point_f1),
            "ci_95_lower": round(ci_low, 4),
            "ci_95_upper": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
            "test_accuracy": round(float(point_acc), 4),
        }

    return boot_results, boot_distributions


def compute_paired_comparison(
    y_true: np.ndarray,
    preds_mlp: np.ndarray,
    preds_lr: np.ndarray,
    boot_dist_mlp: np.ndarray,
    boot_dist_lr: np.ndarray,
) -> Dict[str, Any]:
    """Compute paired McNemar test and bootstrap difference distribution for MLP vs LR."""
    # 1. Bootstrap Difference
    delta_f1 = boot_dist_mlp - boot_dist_lr
    diff_point = float(
        f1_score(y_true, preds_mlp, average="macro", zero_division=0)
        - f1_score(y_true, preds_lr, average="macro", zero_division=0)
    )
    diff_ci_low = float(np.percentile(delta_f1, 2.5))
    diff_ci_high = float(np.percentile(delta_f1, 97.5))

    # Two-sided empirical bootstrap p-value
    p_boot = 2.0 * min(
        float(np.mean(delta_f1 <= 0)),
        float(np.mean(delta_f1 >= 0)),
    )
    p_boot = min(1.0, max(0.0, p_boot))

    # 2. Paired McNemar Test
    correct_lr = (preds_lr == y_true)
    correct_mlp = (preds_mlp == y_true)

    # Contingency table
    # n01: LR correct, MLP wrong
    # n10: LR wrong, MLP correct
    n00 = int(np.sum(~correct_lr & ~correct_mlp))
    n01 = int(np.sum(correct_lr & ~correct_mlp))
    n10 = int(np.sum(~correct_lr & correct_mlp))
    n11 = int(np.sum(correct_lr & correct_mlp))

    discordant = n01 + n10
    if discordant == 0:
        mcnemar_stat = 0.0
        p_mcnemar = 1.0
    elif discordant < 25:
        # Exact binomial test for small discordant counts
        mcnemar_stat = float(abs(n01 - n10))
        p_mcnemar = float(2.0 * stats.binom.cdf(min(n01, n10), discordant, 0.5))
    else:
        # Edwards continuity correction
        mcnemar_stat = float((abs(n01 - n10) - 1.0) ** 2 / discordant)
        p_mcnemar = float(stats.chi2.sf(mcnemar_stat, df=1))

    return {
        "diff_macro_f1": round(diff_point, 4),
        "diff_ci_95_lower": round(diff_ci_low, 4),
        "diff_ci_95_upper": round(diff_ci_high, 4),
        "bootstrap_p_value": round(p_boot, 4),
        "contingency_table": {
            "both_wrong": n00,
            "lr_correct_mlp_wrong": n01,
            "mlp_correct_lr_wrong": n10,
            "both_correct": n11,
        },
        "mcnemar_chi2": round(mcnemar_stat, 4),
        "mcnemar_p_value": round(p_mcnemar, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 95% CIs and hypothesis tests for classifier models.")
    parser.add_argument("--n-resamples", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("RESUME CLASSIFIER TEST SET: 95% BOOTSTRAP CIs & PAIRED TESTS")
    logger.info("=" * 80)
    logger.info("Loading test dataset (read-only without retraining)...")

    texts, labels, classes, _ = load_test_data()
    logger.info(f"Test samples: {len(texts)}, Categories: {len(classes)}")

    model_preds: Dict[str, np.ndarray] = {}

    # 1. TF-IDF + LR
    logger.info("Generating predictions: TF-IDF + Logistic Regression...")
    lr_preds = predict_tfidf_lr(texts)
    if lr_preds is not None:
        model_preds["tfidf_lr"] = lr_preds
    else:
        logger.warning("tfidf_lr could not be evaluated.")

    # 2. TF-IDF + MLP
    logger.info("Generating predictions: TF-IDF + MLP...")
    mlp_preds = predict_tfidf_mlp(texts, num_classes=len(classes))
    if mlp_preds is not None:
        model_preds["tfidf_mlp"] = mlp_preds
    else:
        logger.warning("tfidf_mlp could not be evaluated.")

    # 3. BiLSTM
    logger.info("Generating predictions: BiLSTM...")
    bilstm_preds = predict_bilstm(texts, labels, num_classes=len(classes))
    if bilstm_preds is not None:
        model_preds["bilstm"] = bilstm_preds
    else:
        logger.warning("bilstm could not be evaluated.")

    # 4. DistilBERT
    logger.info("Generating predictions: DistilBERT...")
    bert_preds = predict_distilbert(texts, labels)
    if bert_preds is not None:
        model_preds["distilbert"] = bert_preds
    else:
        logger.warning("distilbert could not be evaluated.")

    if not model_preds:
        logger.error("No models were available for evaluation.")
        return

    logger.info(f"\nRunning {args.n_resamples} bootstrap resamples (seed={args.seed})...")
    ci_results, boot_dists = run_bootstrap_ci(
        labels,
        model_preds,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    # Build CSV rows
    csv_rows = []
    for model_name, stats_dict in ci_results.items():
        csv_rows.append({
            "model": model_name,
            "test_macro_f1": stats_dict["test_macro_f1"],
            "ci_95_lower": stats_dict["ci_95_lower"],
            "ci_95_upper": stats_dict["ci_95_upper"],
            "ci_width": stats_dict["ci_width"],
            "test_accuracy": stats_dict["test_accuracy"],
        })
    ci_df = pd.DataFrame(csv_rows)
    csv_path = REPORT_DIR / "classifier_ci.csv"
    ci_df.to_csv(csv_path, index=False)
    logger.info(f"Saved bootstrap CIs to {csv_path}")

    # Paired test: MLP vs LR
    paired_res: Optional[Dict[str, Any]] = None
    if "tfidf_mlp" in model_preds and "tfidf_lr" in model_preds:
        logger.info("\nComputing paired comparison (MLP vs LR)...")
        paired_res = compute_paired_comparison(
            labels,
            model_preds["tfidf_mlp"],
            model_preds["tfidf_lr"],
            boot_dists["tfidf_mlp"],
            boot_dists["tfidf_lr"],
        )

    # Save full metrics JSON
    full_output = {
        "n_samples": len(texts),
        "n_classes": len(classes),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "models": ci_results,
        "paired_mlp_vs_lr": paired_res,
    }
    json_path = METRIC_DIR / "classifier_ci.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved full CI metrics to {json_path}")

    # Print Table to stdout
    print("\n" + "=" * 80)
    print("CLASSIFIER TEST SET 95% CONFIDENCE INTERVALS (1,000 BOOTSTRAP RESAMPLES)")
    print("=" * 80)
    print(f"{'Model':<16} | {'Test Macro-F1':<14} | {'95% CI Lower':<14} | {'95% CI Upper':<14} | {'Test Acc':<10}")
    print("-" * 80)
    for row in csv_rows:
        print(f"{row['model']:<16} | {row['test_macro_f1']:<14.4f} | {row['ci_95_lower']:<14.4f} | {row['ci_95_upper']:<14.4f} | {row['test_accuracy']:<10.4f}")
    print("=" * 80)

    if paired_res:
        print("\n" + "=" * 80)
        print("PAIRED COMPARISON: TF-IDF + MLP vs. TF-IDF + LR")
        print("=" * 80)
        print(f"Point Macro-F1 Difference (MLP - LR): {paired_res['diff_macro_f1']:+.4f}")
        print(f"Bootstrap 95% CI of Difference:        [{paired_res['diff_ci_95_lower']:+.4f}, {paired_res['diff_ci_95_upper']:+.4f}]")
        print(f"Bootstrap Difference p-value:          {paired_res['bootstrap_p_value']:.4f}")
        tbl = paired_res["contingency_table"]
        print(f"Contingency Table:")
        print(f"  - Both Correct:                      {tbl['both_correct']}")
        print(f"  - Both Incorrect:                    {tbl['both_wrong']}")
        print(f"  - LR Correct, MLP Incorrect:         {tbl['lr_correct_mlp_wrong']}")
        print(f"  - MLP Correct, LR Incorrect:         {tbl['mlp_correct_lr_wrong']}")
        print(f"McNemar Test Chi-Square (df=1):        {paired_res['mcnemar_chi2']:.4f}")
        print(f"McNemar p-value:                       {paired_res['mcnemar_p_value']:.4f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
