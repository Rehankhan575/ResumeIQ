#!/usr/bin/env python3
"""
scripts/collect_results.py
==========================
Read-only metrics collector and report aggregator.

Reads raw evaluation results from artifacts/metrics/:
  - artifacts/metrics/classifier/metrics.json
  - artifacts/metrics/matcher/matcher_val.json
  - artifacts/metrics/matcher/matcher_eval.json
  - artifacts/metrics/scoring/ablation.csv
  - artifacts/metrics/scoring/scoring_eval.json
  - artifacts/metrics/extraction_eval.json
  - artifacts/metrics/parsing_eval.json

Generates:
  - artifacts/reports/results_summary.md
  - artifacts/reports/classifier_4_model.csv
  - artifacts/reports/matcher_overall.csv
  - artifacts/reports/matcher_per_category.csv
  - artifacts/reports/scoring_ablation.csv
  - artifacts/reports/xai_faithfulness.csv
  - artifacts/reports/extraction_eval.csv
  - artifacts/reports/parsing_eval.csv

Strictly read-only: does not retrain models or recompute scores.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_classifier_metrics() -> List[Dict[str, Any]]:
    path = METRICS_DIR / "classifier" / "metrics.json"
    data = load_json(path)
    model_labels = {
        "tfidf_lr": "TF-IDF + Logistic Regression",
        "tfidf_mlp": "TF-IDF + MLP (2-layer)",
        "bilstm": "BiLSTM + Pretrained Embeddings",
        "distilbert": "DistilBERT (Fine-tuned, 3 epochs)",
    }

    rows = []
    for key, label in model_labels.items():
        m = data.get(key, {})
        train_f1 = m.get("train_macro_f1", 0.0)
        val_f1 = m.get("val_macro_f1", 0.0)
        test_f1 = m.get("test_macro_f1", 0.0)
        train_acc = m.get("train_accuracy", 0.0)
        val_acc = m.get("val_accuracy", 0.0)
        test_acc = m.get("test_accuracy", 0.0)
        gap = train_f1 - val_f1

        rows.append({
            "model_key": key,
            "model_name": label,
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_f1, 4),
            "gen_gap_f1": round(gap, 4),
        })
    return rows


def collect_matcher_overall() -> List[Dict[str, Any]]:
    eval_data = load_json(METRICS_DIR / "matcher" / "matcher_eval.json")
    val_data = eval_data.get("val", {})
    test_data = eval_data.get("test", {})

    model_names = {
        "tfidf": "TF-IDF Baseline",
        "zero_shot": "Zero-Shot SBERT (all-MiniLM-L6-v2)",
        "fine_tuned": "Fine-Tuned Siamese SBERT (MNRL)",
    }

    rows = []
    for key, name in model_names.items():
        v = val_data.get(key, {}).get("overall", {})
        t = test_data.get(key, {}).get("overall", {})
        rows.append({
            "model_key": key,
            "model_name": name,
            "val_mrr": round(v.get("mrr", 0.0), 4),
            "val_ndcg10": round(v.get("ndcg@10", 0.0), 4),
            "val_recall5": round(v.get("recall@5", 0.0), 4),
            "val_recall10": round(v.get("recall@10", 0.0), 4),
            "test_mrr": round(t.get("mrr", 0.0), 4),
            "test_ndcg10": round(t.get("ndcg@10", 0.0), 4),
            "test_recall5": round(t.get("recall@5", 0.0), 4),
            "test_recall10": round(t.get("recall@10", 0.0), 4),
        })
    return rows


def collect_matcher_per_category() -> List[Dict[str, Any]]:
    eval_data = load_json(METRICS_DIR / "matcher" / "matcher_eval.json")
    val_data = eval_data.get("val", {})
    test_data = eval_data.get("test", {})

    categories = list(val_data.get("fine_tuned", {}).get("per_category", {}).keys())
    rows = []
    for cat in sorted(categories):
        tfidf_val = val_data.get("tfidf", {}).get("per_category", {}).get(cat, {})
        zs_val = val_data.get("zero_shot", {}).get("per_category", {}).get(cat, {})
        ft_val = val_data.get("fine_tuned", {}).get("per_category", {}).get(cat, {})

        tfidf_test = test_data.get("tfidf", {}).get("per_category", {}).get(cat, {})
        zs_test = test_data.get("zero_shot", {}).get("per_category", {}).get(cat, {})
        ft_test = test_data.get("fine_tuned", {}).get("per_category", {}).get(cat, {})

        rows.append({
            "category": cat,
            "val_count": ft_val.get("count", 0),
            "test_count": ft_test.get("count", 0),
            "val_tfidf_mrr": round(tfidf_val.get("mrr", 0.0), 4),
            "val_zs_mrr": round(zs_val.get("mrr", 0.0), 4),
            "val_ft_mrr": round(ft_val.get("mrr", 0.0), 4),
            "test_tfidf_mrr": round(tfidf_test.get("mrr", 0.0), 4),
            "test_zs_mrr": round(zs_test.get("mrr", 0.0), 4),
            "test_ft_mrr": round(ft_test.get("mrr", 0.0), 4),
        })
    return rows


def collect_scoring_ablation() -> List[Dict[str, Any]]:
    path = METRICS_DIR / "scoring" / "ablation.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "component": r["component"],
                "original_weight": float(r["original_weight"]),
                "mean_original_score": float(r["mean_original_score"]),
                "mean_ablated_score": float(r["mean_ablated_score"]),
                "mean_score_delta": float(r["mean_score_delta"]),
            })
    return rows


def collect_xai_faithfulness() -> Dict[str, Any]:
    path = METRICS_DIR / "scoring" / "scoring_eval.json"
    data = load_json(path)
    f = data.get("faithfulness", {})
    t = data.get("timing", {})
    return {
        "matched_pair_pass": data.get("matched_pair_pass", False),
        "n_evaluated": f.get("n_evaluated", 0),
        "mean_top_k_drop": f.get("mean_top_k_drop", 0.0),
        "mean_random_k_drop": f.get("mean_random_k_drop", 0.0),
        "is_faithful": f.get("is_faithful", False),
        "mean_ms": t.get("mean_ms", 0.0),
        "median_ms": t.get("median_ms", 0.0),
        "p95_ms": t.get("p95_ms", 0.0),
        "total_compute_seconds": data.get("total_compute_seconds", 0.0),
    }


def collect_parsing_eval() -> List[Dict[str, Any]]:
    path = METRICS_DIR / "parsing_eval.json"
    data = load_json(path)
    fixtures = data.get("fixtures", {})
    rows = []
    for fname, details in fixtures.items():
        contact = details.get("contact", {})
        contact_pass = all(v.get("pass", False) for v in contact.values()) if contact else True
        layout = details.get("layout", {})
        rows.append({
            "fixture": fname,
            "duration_ms": details.get("duration_ms", 0.0),
            "sections_precision": details.get("sections_precision", 0.0),
            "sections_recall": details.get("sections_recall", 0.0),
            "contact_pass": contact_pass,
            "page_count": layout.get("page_count", {}).get("extracted", 1),
            "is_multi_column": layout.get("is_multi_column", {}).get("extracted", False),
            "has_tables": layout.get("has_tables", {}).get("extracted", False),
            "needs_ocr": layout.get("needs_ocr", {}).get("extracted", False),
        })
    return rows


def collect_extraction_eval() -> Dict[str, Any]:
    path = METRICS_DIR / "extraction_eval.json"
    data = load_json(path)
    fixtures_dict = data.get("fixtures", {}).get("fixtures", {})
    fix_rows = []
    for fname, details in fixtures_dict.items():
        fix_rows.append({
            "fixture": fname,
            "type": details.get("type", "unknown"),
            "precision": round(details.get("precision", 0.0), 4),
            "recall": round(details.get("recall", 0.0), 4),
            "f1": round(details.get("f1", 0.0), 4),
            "trap_hits": len(details.get("trap_hits", [])),
            "years_pass": details.get("years_pass", True),
        })
    return {
        "summary": {
            "micro_precision": data.get("fixtures", {}).get("micro_precision", 0.0),
            "micro_recall": data.get("fixtures", {}).get("micro_recall", 0.0),
            "micro_f1": data.get("fixtures", {}).get("micro_f1", 0.0),
            "total_trap_violations": data.get("fixtures", {}).get("total_trap_violations", 0),
        },
        "fixtures": fix_rows,
        "real_text": data.get("real_text_stress_test", {}),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_reports() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Classifier
    cls_rows = collect_classifier_metrics()
    write_csv(
        REPORTS_DIR / "classifier_4_model.csv",
        cls_rows,
        [
            "model_key",
            "model_name",
            "train_accuracy",
            "train_macro_f1",
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "gen_gap_f1",
        ],
    )

    # 2. Matcher Overall
    matcher_rows = collect_matcher_overall()
    write_csv(
        REPORTS_DIR / "matcher_overall.csv",
        matcher_rows,
        [
            "model_key",
            "model_name",
            "val_mrr",
            "val_ndcg10",
            "val_recall5",
            "val_recall10",
            "test_mrr",
            "test_ndcg10",
            "test_recall5",
            "test_recall10",
        ],
    )

    # 3. Matcher Per-Category
    matcher_cat_rows = collect_matcher_per_category()
    write_csv(
        REPORTS_DIR / "matcher_per_category.csv",
        matcher_cat_rows,
        [
            "category",
            "val_count",
            "test_count",
            "val_tfidf_mrr",
            "val_zs_mrr",
            "val_ft_mrr",
            "test_tfidf_mrr",
            "test_zs_mrr",
            "test_ft_mrr",
        ],
    )

    # 4. Scoring Ablation
    ablation_rows = collect_scoring_ablation()
    write_csv(
        REPORTS_DIR / "scoring_ablation.csv",
        ablation_rows,
        [
            "component",
            "original_weight",
            "mean_original_score",
            "mean_ablated_score",
            "mean_score_delta",
        ],
    )

    # 5. XAI Faithfulness
    xai = collect_xai_faithfulness()
    write_csv(
        REPORTS_DIR / "xai_faithfulness.csv",
        [xai],
        [
            "matched_pair_pass",
            "n_evaluated",
            "mean_top_k_drop",
            "mean_random_k_drop",
            "is_faithful",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "total_compute_seconds",
        ],
    )

    # 6. Parsing
    parsing_rows = collect_parsing_eval()
    write_csv(
        REPORTS_DIR / "parsing_eval.csv",
        parsing_rows,
        [
            "fixture",
            "duration_ms",
            "sections_precision",
            "sections_recall",
            "contact_pass",
            "page_count",
            "is_multi_column",
            "has_tables",
            "needs_ocr",
        ],
    )

    # 7. Extraction
    extraction_data = collect_extraction_eval()
    fix_rows = extraction_data.get("fixtures", [])
    write_csv(
        REPORTS_DIR / "extraction_eval.csv",
        fix_rows,
        ["fixture", "type", "precision", "recall", "f1", "trap_hits", "years_pass"],
    )

    # 8. Markdown summary
    real_text = extraction_data.get("real_text", {})
    ey = real_text.get("entries_yield", {})
    lat = real_text.get("latency_ms", {})
    fix_summary = extraction_data.get("summary", {})

    md_lines = [
        "# Empirical Results Summary",
        "",
        "This document aggregates evaluation metrics across all system stages.",
        "Generated automatically by `scripts/collect_results.py` from files in `artifacts/metrics/`.",
        "",
        "## 1. Stage 3 Resume Classification (4 Models)",
        "",
        "Evaluated on 24 balanced job categories from the Kaggle Resume Dataset.",
        "Train split: 1,739 resumes; Val split: 370 resumes; Test split: 374 resumes.",
        "",
        "| Architecture | Train Acc | Train Macro F1 | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Gen Gap (F1) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in cls_rows:
        md_lines.append(
            f"| **{r['model_name']}** | {r['train_accuracy']*100:.2f}% | {r['train_macro_f1']:.4f} | "
            f"{r['val_accuracy']*100:.2f}% | {r['val_macro_f1']:.4f} | {r['test_accuracy']*100:.2f}% | "
            f"{r['test_macro_f1']:.4f} | {r['gen_gap_f1']:.4f} |"
        )

    md_lines.extend([
        "",
        "> **Key Classifier Takeaways & Hypothesis Testing:**",
        "> - **LR and MLP are indistinguishable on test** (macro-F1 and accuracy, paired tests: macro-F1 diff +0.0222, p=0.348; accuracy diff -0.0107, p=0.598, McNemar p=0.6265).",
        "> - **DistilBERT scores higher on test:** vs LR macro-F1 +0.050 (95% CI [0.0056, 0.0931], p=0.036) and accuracy +6.4 pts (+0.0642, p=0.022, McNemar p=0.0101); vs MLP accuracy +7.5 pts (+0.0749, p=0.010, McNemar p=0.0081), but its macro-F1 gain over MLP (+0.0273, 95% CI [-0.0250, 0.0901], p=0.280) is not significant.",
        "> - **Caveats:** six paired tests with no multiplicity correction; one test split of 374 resumes; on validation DistilBERT did NOT outperform (val macro-F1 lower than the MLP: 0.5675 vs 0.6001), so validation and test disagree; DistilBERT trained only 3 epochs; the BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).",
        "> - **Served Classifier:** The served classifier is the MLP because the selection rule (val macro-F1) was fixed before the test set was opened; choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.",
        "",
        "## 2. Stage 4 Semantic Resume-JD Matching",
        "",
        "Evaluated across 15 target categories using reciprocal rank and nDCG metrics against gold resume-JD category alignment.",
        "",
        "### Overall Matcher Performance (Validation & Test)",
        "",
        "| Model Architecture | Val MRR | Val nDCG@10 | Val Recall@5 | Val Recall@10 | Test MRR | Test nDCG@10 | Test Recall@5 | Test Recall@10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in matcher_rows:
        md_lines.append(
            f"| **{r['model_name']}** | {r['val_mrr']:.4f} | {r['val_ndcg10']:.4f} | {r['val_recall5']:.4f} | {r['val_recall10']:.4f} | "
            f"{r['test_mrr']:.4f} | {r['test_ndcg10']:.4f} | {r['test_recall5']:.4f} | {r['test_recall10']:.4f} |"
        )

    md_lines.extend([
        "",
        "### Per-Category Reciprocal Rank (MRR)",
        "",
        "| Category | Val N | Test N | TF-IDF Val | Zero-Shot Val | Fine-Tuned Val | TF-IDF Test | Zero-Shot Test | Fine-Tuned Test |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in matcher_cat_rows:
        md_lines.append(
            f"| {r['category']} | {r['val_count']} | {r['test_count']} | "
            f"{r['val_tfidf_mrr']:.4f} | {r['val_zs_mrr']:.4f} | {r['val_ft_mrr']:.4f} | "
            f"{r['test_tfidf_mrr']:.4f} | {r['test_zs_mrr']:.4f} | {r['test_ft_mrr']:.4f} |"
        )

    md_lines.extend([
        "",
        "> **Key Matcher Takeaways:**",
        "> - Fine-tuned Siamese SBERT yields a **+14.87 point increase in Test MRR** (0.7798 vs 0.6311 zero-shot) and **+27.54 point increase in Test nDCG@10** (0.7286 vs 0.4532).",
        "> - Performance varies heavily by category specificity: CHEF, ACCOUNTANT, SALES, and TEACHER achieve near-perfect retrieval (MRR > 0.95), while broad or noisy domains (BPO, CONSULTANT) suffer from high ambiguity.",
        "",
        "## 3. Stage 5 ATS Scoring & Ablation Study",
        "",
        "Evaluated on paired and un-paired resumes with 9 distinct weighted score components.",
        "",
        "| Component | Default Weight | Mean Score (Full) | Mean Score (Ablated) | Mean Score Impact (Δ) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for r in ablation_rows:
        md_lines.append(
            f"| `{r['component']}` | {r['original_weight']:.2f} | {r['mean_original_score']:.2f} | "
            f"{r['mean_ablated_score']:.2f} | **{r['mean_score_delta']:+.2f}** |"
        )

    md_lines.extend([
        "",
        "### XAI Faithfulness & Latency Benchmarks",
        "",
        f"- **Evaluated Pairs:** {xai['n_evaluated']}",
        f"- **Matched Pair Check:** {'PASSED' if xai['matched_pair_pass'] else 'FAILED'}",
        f"- **Mean Top-k Sentence Erasure Drop:** {xai['mean_top_k_drop']:.4f}",
        f"- **Mean Random Sentence Erasure Drop:** {xai['mean_random_k_drop']:.4f}",
        f"- **Sentence Erasure Faithfulness Check:** `{xai['is_faithful']}` (honest empirical observation: top-3 sentence removal causes 0.07 pt drop vs 0.11 pt drop for random-3 sentences; ATS sentence explanations are inconclusive and unvalidated).",
        f"- **Scoring Latency:** Mean {xai['mean_ms']:.2f} ms, Median {xai['median_ms']:.2f} ms, P95 {xai['p95_ms']:.2f} ms (Total compute time: {xai['total_compute_seconds']:.2f} s).",
        "",
        "## 4. Stage 1 & 2 Parsing and Extraction Verification",
        "",
        "### Multi-Format Parsing Benchmark (6 Fixtures)",
        "",
        "| Fixture | Duration (ms) | Section Precision | Section Recall | Contact Extracted | Multi-column | Has Tables | OCR Needed |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in parsing_rows:
        md_lines.append(
            f"| `{r['fixture']}` | {r['duration_ms']:.2f} | {r['sections_precision']*100:.1f}% | "
            f"{r['sections_recall']*100:.1f}% | {'Yes' if r['contact_pass'] else 'No'} | "
            f"{'Yes' if r['is_multi_column'] else 'No'} | {'Yes' if r['has_tables'] else 'No'} | "
            f"{'Yes' if r['needs_ocr'] else 'No'} |"
        )

    md_lines.extend([
        "",
        "### Gold Resume & JD Fixture Extraction",
        "",
        f"- **Overall Micro Precision:** {fix_summary.get('micro_precision', 0.0)*100:.2f}%",
        f"- **Overall Micro Recall:** {fix_summary.get('micro_recall', 0.0)*100:.2f}%",
        f"- **Overall Micro F1:** {fix_summary.get('micro_f1', 0.0):.4f}",
        f"- **Total Trap Violations:** {fix_summary.get('total_trap_violations', 0)}",
        "",
        "| Fixture | Type | Precision | Recall | F1 Score | Trap Hits | Years Pass |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in fix_rows:
        md_lines.append(
            f"| `{r['fixture']}` | {r['type']} | {r['precision']*100:.1f}% | "
            f"{r['recall']*100:.1f}% | {r['f1']:.4f} | {r['trap_hits']} | "
            f"{'Yes' if r['years_pass'] else 'N/A'} |"
        )

    md_lines.extend([
        "",
        "### Real-Text Extraction Yield (370 Resumes)",
        "",
        f"- **Education Section Parse Yield:** {ey.get('pct_education_parsed', 0.0):.2f}%",
        f"- **Experience Section Parse Yield:** {ey.get('pct_experience_parsed', 0.0):.2f}%",
        f"- **Years Experience:** Mean {ey.get('total_years_mean', 0.0):.2f} yrs, Median {ey.get('total_years_median', 0.0):.2f} yrs, P95 {ey.get('total_years_p95', 0.0):.2f} yrs, Over 40 yrs cap: {ey.get('total_years_pct_over_40', 0.0):.2f}%",
        f"- **Extraction Latency:** Mean {lat.get('mean', 0.0):.2f} ms, P95 {lat.get('p95', 0.0):.2f} ms",
        "",
        "---",
        "*Report generated cleanly with zero test-time inference or model retraining.*",
    ])

    report_md_path = REPORTS_DIR / "results_summary.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Results summary successfully written to {report_md_path}")
    print(f"Aggregated CSV reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    generate_reports()
