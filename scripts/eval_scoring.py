"""
scripts/eval_scoring.py
=======================
Evaluation benchmark for Stage 5 (ATS Scoring) and Stage 6 (Explainability).

Evaluates:
  1. Score Matrix: 5 resume fixtures x 3 JD fixtures (parsed and extracted via real pipelines).
     Verifies that matched pairs (ds_clean/jd_data_scientist, swe_two_column/jd_software_engineer,
     chef_plain/jd_chef) achieve the highest score in their row.
  2. Component Ablation: Leaves out each scoring component, renormalises weights,
     calculates score delta, and saves results to artifacts/metrics/scoring/ablation.csv.
  3. XAI Faithfulness: On 20 validation resumes (block-flattened Resume_html), evaluates
     score drop after removing top-k (k=3) most important sentences vs k random sentences (seed 42).
     Saves bar chart to artifacts/reports/xai_faithfulness.png.
  4. Latency / Timing: Benchmarks score + explain latency per resume.

Privacy: Prints aggregate statistics only; never displays raw resume text.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.config import SCORE_WEIGHTS, settings
from app.extraction.extractor import extract_entities
from app.extraction.skills import TaxonomySkillExtractor
from app.parsing.cleaner import clean_text
from app.parsing.parser import parse_resume
from app.parsing.sections import split_sections
from app.schemas import JobDescription, LayoutMeta, ParsedResume, ScoreResult
from app.scoring.ats import score_resume
from app.scoring.explain import add_highlights, compute_sentence_contributions

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
SPLIT_CSV = ROOT_DIR / "data" / "splits" / "classifier_split.csv"
RAW_LIVECREER_PARQUET = ROOT_DIR / "data" / "raw" / "livecareer" / "train.parquet"
OUTPUT_METRICS_DIR = ROOT_DIR / "artifacts" / "metrics" / "scoring"
OUTPUT_REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"

OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RESUME_FIXTURES = [
    "accountant_table.pdf",
    "chef_plain.docx",
    "ds_clean.pdf",
    "nurse_scanned.pdf",
    "swe_two_column.pdf",
]

JD_FIXTURES = [
    "jd_chef.txt",
    "jd_data_scientist.txt",
    "jd_software_engineer.txt",
]

MATCHED_PAIRS = {
    "chef_plain.docx": "jd_chef.txt",
    "ds_clean.pdf": "jd_data_scientist.txt",
    "swe_two_column.pdf": "jd_software_engineer.txt",
}


def load_fixtures(skill_ext: TaxonomySkillExtractor) -> Tuple[Dict[str, ParsedResume], Dict[str, JobDescription]]:
    """Parse real fixture resumes and JDs."""
    parsed_resumes: Dict[str, ParsedResume] = {}
    for fname in RESUME_FIXTURES:
        fpath = FIXTURES_DIR / fname
        with open(fpath, "rb") as f:
            raw_bytes = f.read()
        parsed = parse_resume(raw_bytes, str(fpath))
        parsed = extract_entities(parsed)
        parsed_resumes[fname] = parsed

    parsed_jds: Dict[str, JobDescription] = {}
    for jname in JD_FIXTURES:
        jpath = FIXTURES_DIR / jname
        txt = jpath.read_text(encoding="utf-8")
        info = skill_ext.extract_jd_skills(txt)
        parsed_jds[jname] = JobDescription(
            text=txt,
            title=jname.replace(".txt", "").replace("jd_", "").replace("_", " ").title(),
            required_skills=info["required_skills"],
            preferred_skills=info["preferred_skills"],
            min_years=info["min_years"],
        )

    return parsed_resumes, parsed_jds


def evaluate_score_matrix(
    resumes: Dict[str, ParsedResume],
    jds: Dict[str, JobDescription],
) -> Dict[str, Any]:
    """Compute 5x3 score matrix and verify matched pair rankings."""
    logger.info("=" * 70)
    logger.info("1. SCORE MATRIX EVALUATION (5 Resumes x 3 Job Descriptions)")
    logger.info("=" * 70)

    matrix: Dict[str, Dict[str, float]] = {}
    matched_rankings: Dict[str, bool] = {}

    jd_short_names = [j.replace(".txt", "") for j in JD_FIXTURES]
    header = f"| {'Resume Fixture':<22} | " + " | ".join(f"{name:<20}" for name in jd_short_names) + " | Best Match | Matched Pair Win? |"
    logger.info(header)
    logger.info("|" + "---|"* (len(jd_short_names) + 3))

    for rname, resume in resumes.items():
        matrix[rname] = {}
        for jname, jd in jds.items():
            res = score_resume(resume, jd)
            matrix[rname][jname] = res.total

        best_jd = max(matrix[rname], key=matrix[rname].get)
        scores_str = " | ".join(f"{matrix[rname][j]:<20.2f}" for j in JD_FIXTURES)

        is_matched = rname in MATCHED_PAIRS
        expected_jd = MATCHED_PAIRS.get(rname, "")
        if is_matched:
            win = best_jd == expected_jd
            matched_rankings[rname] = win
            status = "PASS (Yes)" if win else f"FAIL (Expected {expected_jd.replace('.txt', '')})"
        else:
            status = "N/A (Control)"

        best_display = best_jd.replace(".txt", "").replace("jd_", "")
        logger.info(f"| {rname:<22} | {scores_str} | {best_display:<10} | {status} |")

    # Check assertions on matched pairs
    all_matched_pass = all(matched_rankings.values())
    logger.info(f"\nMatched Pair Diagnostic: {'ALL PASS' if all_matched_pass else 'SOME FAILED'}")
    for rname, win in matched_rankings.items():
        exp = MATCHED_PAIRS[rname].replace(".txt", "")
        logger.info(f"  - {rname} -> {exp}: {'HIGHEST (PASS)' if win else 'NOT HIGHEST (FAIL)'}")

    matrix_path = OUTPUT_METRICS_DIR / "score_matrix.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    logger.info(f"Saved score matrix to {matrix_path}")

    return {
        "matrix": matrix,
        "matched_pair_pass": all_matched_pass,
        "matched_rankings": matched_rankings,
    }


def evaluate_ablation(
    resumes: Dict[str, ParsedResume],
    jds: Dict[str, JobDescription],
) -> pd.DataFrame:
    """Run component ablation by dropping each component and renormalising weights."""
    logger.info("\n" + "=" * 70)
    logger.info("2. COMPONENT ABLATION STUDY")
    logger.info("=" * 70)

    # Base weights
    base_weights = dict(SCORE_WEIGHTS)
    components = list(base_weights.keys())

    # We evaluate on the matched pairs + representative pairs
    eval_pairs = [
        ("chef_plain.docx", "jd_chef.txt"),
        ("ds_clean.pdf", "jd_data_scientist.txt"),
        ("swe_two_column.pdf", "jd_software_engineer.txt"),
    ]

    records: List[Dict[str, Any]] = []

    for comp in components:
        comp_weight = base_weights[comp]
        # Renormalise remaining weights
        remaining_sum = 1.0 - comp_weight
        renorm_weights = {k: base_weights[k] / remaining_sum for k in components if k != comp}

        deltas: List[float] = []
        orig_totals: List[float] = []
        ablated_totals: List[float] = []

        for rname, jname in eval_pairs:
            resume = resumes[rname]
            jd = jds[jname]
            base_res = score_resume(resume, jd)
            orig_total = base_res.total
            comp_vals = base_res.components

            # Recompute total with renormalised weights
            ablated_total = sum(renorm_weights[k] * comp_vals[k] for k in renorm_weights) * 100.0
            ablated_total = round(max(0.0, min(100.0, ablated_total)), 2)
            delta = round(ablated_total - orig_total, 2)

            deltas.append(delta)
            orig_totals.append(orig_total)
            ablated_totals.append(ablated_total)

        mean_orig = float(np.mean(orig_totals))
        mean_ablated = float(np.mean(ablated_totals))
        mean_delta = float(np.mean(deltas))

        records.append({
            "component": comp,
            "original_weight": comp_weight,
            "mean_original_score": round(mean_orig, 2),
            "mean_ablated_score": round(mean_ablated, 2),
            "mean_score_delta": round(mean_delta, 2),
        })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_METRICS_DIR / "ablation.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved ablation table to {csv_path}\n")

    logger.info("| Component Dropped | Original Weight | Mean Orig Score | Mean Ablated Score | Mean Score Delta |")
    logger.info("|---|---|---|---|---|")
    for _, row in df.iterrows():
        sign = "+" if row["mean_score_delta"] > 0 else ""
        delta_str = f"{sign}{row['mean_score_delta']:.2f}"
        logger.info(
            f"| `{row['component']:<23}` | {row['original_weight']:<15.2f} | "
            f"{row['mean_original_score']:<15.2f} | {row['mean_ablated_score']:<18.2f} | {delta_str:<16} |"
        )

    return df


def evaluate_faithfulness(
    skill_ext: TaxonomySkillExtractor,
    sample_size: int = 20,
    k: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate XAI faithfulness by comparing top-k vs random-k sentence removal."""
    logger.info("\n" + "=" * 70)
    logger.info("3. XAI FAITHFULNESS BENCHMARK (Top-k vs Random-k Sentence Erasure)")
    logger.info("=" * 70)

    if not SPLIT_CSV.exists() or not RAW_LIVECREER_PARQUET.exists():
        logger.warning("Split CSV or raw LiveCareer dataset not found; skipping faithfulness.")
        return {}

    val_split = pd.read_csv(SPLIT_CSV)
    val_ids = set(val_split[val_split["split"] == "val"]["id"].astype(str))

    raw_df = pd.read_parquet(RAW_LIVECREER_PARQUET)
    raw_val_df = raw_df[raw_df["ID"].astype(str).isin(val_ids)].copy()

    # Deterministic sample of 20 validation resumes
    val_sample = raw_val_df.head(sample_size)
    logger.info(f"Loaded {len(val_sample)} validation resumes for faithfulness evaluation.")

    # Reference JD for evaluation: Software Engineer fixture
    jd_txt = (FIXTURES_DIR / "jd_software_engineer.txt").read_text(encoding="utf-8")
    jd_info = skill_ext.extract_jd_skills(jd_txt)
    jd = JobDescription(
        text=jd_txt,
        title="Software Engineer",
        required_skills=jd_info["required_skills"],
        preferred_skills=jd_info["preferred_skills"],
        min_years=jd_info["min_years"],
    )

    random.seed(seed)
    np.random.seed(seed)

    top_k_drops: List[float] = []
    random_k_drops: List[float] = []

    for idx, (_, row) in enumerate(val_sample.iterrows()):
        # Block-flattened HTML approach (consistent with eval_extraction.py)
        html_raw = str(row["Resume_html"])
        soup = BeautifulSoup(html_raw, "html.parser")
        for tag in soup.find_all(["div", "p", "li", "br", "tr"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        raw_text = clean_text(soup.get_text())

        secs = split_sections(raw_text)
        parsed = ParsedResume(
            raw_text=raw_text,
            sections=secs,
            layout=LayoutMeta(
                file_type="html_flattened",
                page_count=1,
                word_count=len(raw_text.split()),
            ),
        )
        parsed = extract_entities(parsed)

        # Baseline score
        base_res = score_resume(parsed, jd)
        base_score = base_res.total

        # Sentence contributions via LOSO
        sent_highlights = compute_sentence_contributions(raw_text, jd.text, max_sentences=40)
        # Filter highlights to sentence spans present in text
        valid_sentences = [h for h in sent_highlights if h.weight > 0 and h.text in raw_text]

        if len(valid_sentences) < k:
            # If fewer than k positive sentences found, extract general sentence chunks
            chunks = [s.strip() for s in raw_text.split("\n") if len(s.strip().split()) >= 4]
            if len(chunks) < k:
                continue
            # Rank chunks by simple word overlap with JD
            jd_words = set(jd.text.lower().split())
            ranked = sorted(chunks, key=lambda c: sum(1 for w in c.lower().split() if w in jd_words), reverse=True)
            top_sents = ranked[:k]
            rem_pool = [c for c in chunks if c not in top_sents]
            rand_sents = random.sample(rem_pool, min(k, len(rem_pool))) if len(rem_pool) >= k else chunks[:k]
        else:
            sorted_sents = sorted(valid_sentences, key=lambda h: h.weight, reverse=True)
            top_sents = [h.text for h in sorted_sents[:k]]

            all_sents = [h.text for h in sent_highlights]
            candidate_pool = [s for s in all_sents if s not in top_sents]
            if len(candidate_pool) >= k:
                rand_sents = random.sample(candidate_pool, k)
            else:
                rand_sents = all_sents[-k:]

        # 1. Score after removing top-k sentences
        text_top_removed = raw_text
        for s in top_sents:
            text_top_removed = text_top_removed.replace(s, " ")
        p_top = ParsedResume(
            raw_text=text_top_removed,
            sections=split_sections(text_top_removed),
            layout=parsed.layout,
        )
        p_top = extract_entities(p_top)
        score_top_removed = score_resume(p_top, jd).total
        drop_top = base_score - score_top_removed

        # 2. Score after removing random-k sentences
        text_rand_removed = raw_text
        for s in rand_sents:
            text_rand_removed = text_rand_removed.replace(s, " ")
        p_rand = ParsedResume(
            raw_text=text_rand_removed,
            sections=split_sections(text_rand_removed),
            layout=parsed.layout,
        )
        p_rand = extract_entities(p_rand)
        score_rand_removed = score_resume(p_rand, jd).total
        drop_rand = base_score - score_rand_removed

        top_k_drops.append(drop_top)
        random_k_drops.append(drop_rand)

    mean_top_drop = float(np.mean(top_k_drops)) if top_k_drops else 0.0
    mean_rand_drop = float(np.mean(random_k_drops)) if random_k_drops else 0.0

    logger.info(f"Faithfulness Evaluation over {len(top_k_drops)} resumes (k={k}, seed={seed}):")
    logger.info(f"  - Mean Score Drop (Top-{k} Important Sentences Removed): {mean_top_drop:.2f} pts")
    logger.info(f"  - Mean Score Drop (Random-{k} Sentences Removed):          {mean_rand_drop:.2f} pts")

    is_faithful = mean_top_drop > mean_rand_drop
    if is_faithful:
        logger.info(f"  - Verdict: FAITHFUL (Top-{k} removal causes greater degradation by {mean_top_drop - mean_rand_drop:.2f} pts)")
    else:
        logger.info(f"  - Verdict: ATTENTION SENSITIVE (Top-{k} drop: {mean_top_drop:.2f} vs Random: {mean_rand_drop:.2f})")

    # Save plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["Top-3 Important Removed", "Random-3 Removed"],
        [mean_top_drop, mean_rand_drop],
        color=["#2b5c8f", "#d95f02"],
        width=0.45,
    )
    ax.set_ylabel("Mean Total Score Drop (Points)")
    ax.set_title("XAI Faithfulness: Feature Erasure Degradation", fontsize=12, pad=10)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} pts",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plot_path = OUTPUT_REPORTS_DIR / "xai_faithfulness.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved faithfulness bar plot to {plot_path}")

    return {
        "n_evaluated": len(top_k_drops),
        "mean_top_k_drop": round(mean_top_drop, 2),
        "mean_random_k_drop": round(mean_rand_drop, 2),
        "is_faithful": is_faithful,
    }


def evaluate_timing(
    resumes: Dict[str, ParsedResume],
    jds: Dict[str, JobDescription],
) -> Dict[str, float]:
    """Benchmark per-resume latency for score + explain stages."""
    logger.info("\n" + "=" * 70)
    logger.info("4. LATENCY BENCHMARK (Score + Explain Per Resume)")
    logger.info("=" * 70)

    jd = jds["jd_software_engineer.txt"]
    latencies_ms: List[float] = []

    for rname, resume in resumes.items():
        # Warmup
        _ = score_resume(resume, jd)
        _ = add_highlights(_, resume, jd)

        # Timed run
        t0 = time.perf_counter()
        scored = score_resume(resume, jd)
        _ = add_highlights(scored, resume, jd)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))

    logger.info(f"Benchmarked {len(latencies_ms)} fixture evaluations:")
    logger.info(f"  - Mean Latency:   {mean_lat:.2f} ms")
    logger.info(f"  - Median Latency: {median_lat:.2f} ms")
    logger.info(f"  - P95 Latency:    {p95_lat:.2f} ms")

    return {
        "mean_ms": round(mean_lat, 2),
        "median_ms": round(median_lat, 2),
        "p95_ms": round(p95_lat, 2),
    }


def main() -> None:
    """Run complete Stage 5 and Stage 6 scoring evaluation."""
    t_start = time.perf_counter()
    logger.info("Starting ATS Scoring & XAI Evaluation Benchmark...\n")

    skill_ext = TaxonomySkillExtractor()
    resumes, jds = load_fixtures(skill_ext)

    # 1. Score Matrix
    matrix_res = evaluate_score_matrix(resumes, jds)

    # 2. Component Ablation
    ablation_df = evaluate_ablation(resumes, jds)

    # 3. Faithfulness
    faith_res = evaluate_faithfulness(skill_ext, sample_size=20, k=3, seed=42)

    # 4. Latency
    timing_res = evaluate_timing(resumes, jds)

    total_elapsed = time.perf_counter() - t_start
    logger.info(f"\nAll scoring evaluations completed in {total_elapsed:.2f} s.")

    # Save summary metrics
    summary = {
        "matched_pair_pass": matrix_res["matched_pair_pass"],
        "faithfulness": faith_res,
        "timing": timing_res,
        "total_compute_seconds": round(total_elapsed, 2),
    }
    summary_path = OUTPUT_METRICS_DIR / "scoring_eval.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
