"""
scripts/eval_extraction.py
==========================
Evaluation benchmark for Stage 2 (Structured Extraction).

Evaluates:
1. Fixture precision, recall, and F1 across 5 resume fixtures and 3 JD fixtures.
   Checks trap strings (must be 0 hits), education/experience field matches,
   and total_years_experience tolerance (0.25 years).
2. Real-text stress test on data/processed/classifier_val.parquet:
   - Section detection rates overall and for IT, HEALTHCARE, ACCOUNTANT, CHEF, SALES.
   - Skills per resume distributions (mean, median, p5, % < 5 hits).
   - Top 20 skills and suspect high-frequency skills (>10% not in ambiguous_terms.csv).
   - Education and experience parse yield, total_years distribution, and latency.

Privacy: Never prints raw resume or JD text; outputs only aggregate metrics and skill names.
Saves metrics to artifacts/metrics/extraction_eval.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.extraction.entries import (
    compute_total_years_experience,
    extract_education,
    extract_experience,
)
from app.extraction.extractor import extract_entities
from app.extraction.skills import TaxonomySkillExtractor
from app.parsing.cleaner import clean_text
from app.parsing.parser import parse_resume
from app.parsing.sections import split_sections
from app.schemas import ParsedResume

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = ROOT / "tests" / "fixtures"
METRICS_PATH = settings.metrics_dir / "extraction_eval.json"
VAL_PARQUET = ROOT / "data" / "processed" / "classifier_val.parquet"
AMBIGUOUS_CSV = ROOT / "data" / "taxonomies" / "ambiguous_terms.csv"


def evaluate_fixtures(extractor: TaxonomySkillExtractor) -> Dict[str, Any]:
    """Run extraction benchmark over all fixtures."""
    logger.info("=" * 70)
    logger.info("STAGE 2: FIXTURE EVALUATION")
    logger.info("=" * 70)

    resume_fixtures = [
        "accountant_table.pdf",
        "chef_plain.docx",
        "ds_clean.pdf",
        "nurse_scanned.pdf",
        "swe_two_column.pdf",
    ]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    trap_violations = 0

    fixture_results: Dict[str, Any] = {}

    for fname in resume_fixtures:
        fpath = FIXTURES_DIR / fname
        expected_path = FIXTURES_DIR / (fname.rsplit(".", 1)[0] + ".expected.json")

        with open(fpath, "rb") as f:
            raw_bytes = f.read()
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        as_of = expected.get("as_of", "2026-11-01")
        parsed = parse_resume(raw_bytes, str(fpath))
        parsed = extract_entities(parsed, as_of=as_of)

        got_skills = set(s.lower() for s in parsed.skills)
        exp_skills = set(s.lower() for s in expected.get("skills", []))

        tp = len(got_skills & exp_skills)
        fp = len(got_skills - exp_skills)
        fn = len(exp_skills - got_skills)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0

        # Check traps
        traps = expected.get("traps", [])
        trap_hits = [t for t in traps if t.lower() in got_skills]
        trap_violations += len(trap_hits)

        # Compare education
        got_edu = [
            {
                "degree": e.degree,
                "field": e.field,
                "institution": e.institution,
                "graduation_year": e.graduation_year,
            }
            for e in parsed.education
        ]
        exp_edu = expected.get("education", [])
        edu_match = got_edu == exp_edu

        # Compare experience
        got_exp = [
            {
                "title": ex.title,
                "company": ex.company,
                "start": ex.start,
                "end": ex.end,
                "is_current": ex.is_current,
            }
            for ex in parsed.experience
        ]
        exp_exp = expected.get("experience", [])
        exp_match = True
        if len(got_exp) != len(exp_exp):
            exp_match = False
        else:
            for g, x in zip(got_exp, exp_exp):
                if (
                    g["title"].lower() != x["title"].lower()
                    or g["company"].lower() != x["company"].lower()
                    or g["start"] != x["start"]
                    or g["end"] != x["end"]
                    or g["is_current"] != x["is_current"]
                ):
                    exp_match = False
                    break

        # Total years diff
        exp_years = float(expected.get("total_years_experience", 0.0))
        got_years = float(parsed.total_years_experience)
        years_diff = round(abs(got_years - exp_years), 2)
        years_pass = years_diff <= 0.25

        fixture_results[fname] = {
            "type": "resume",
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "false_positives": sorted(list(got_skills - exp_skills)),
            "misses": sorted(list(exp_skills - got_skills)),
            "trap_hits": trap_hits,
            "education_match": edu_match,
            "experience_match": exp_match,
            "got_years": got_years,
            "expected_years": exp_years,
            "years_diff": years_diff,
            "years_pass": years_pass,
        }

    # Evaluate JD fixtures
    jd_fixtures = ["jd_data_scientist.txt", "jd_software_engineer.txt", "jd_chef.txt"]

    for fname in jd_fixtures:
        fpath = FIXTURES_DIR / fname
        expected_path = FIXTURES_DIR / (fname.rsplit(".", 1)[0] + ".expected.json")

        with open(fpath, "r", encoding="utf-8") as f:
            jd_text = f.read()
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        jd_res = extractor.extract_jd_skills(jd_text)

        # Required skills P/R
        got_req = set(s.lower() for s in jd_res["required_skills"])
        exp_req = set(s.lower() for s in expected.get("required_skills", []))
        tp_r = len(got_req & exp_req)
        fp_r = len(got_req - exp_req)
        fn_r = len(exp_req - got_req)

        p_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) else 1.0
        r_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) else 1.0
        f1_r = (2 * p_r * r_r) / (p_r + r_r) if (p_r + r_r) else 0.0

        # Preferred skills P/R
        got_pref = set(s.lower() for s in jd_res["preferred_skills"])
        exp_pref = set(s.lower() for s in expected.get("preferred_skills", []))
        tp_p = len(got_pref & exp_pref)
        fp_p = len(got_pref - exp_pref)
        fn_p = len(exp_pref - got_pref)

        p_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) else 1.0
        r_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) else 1.0
        f1_p = (2 * p_p * r_p) / (p_p + r_p) if (p_p + r_p) else 0.0

        # Micro counts
        total_tp += tp_r + tp_p
        total_fp += fp_r + fp_p
        total_fn += fn_r + fn_p

        min_years_pass = jd_res["min_years"] == expected.get("min_years")

        fixture_results[fname] = {
            "type": "job_description",
            "req_precision": round(p_r, 4),
            "req_recall": round(r_r, 4),
            "req_f1": round(f1_r, 4),
            "req_misses": sorted(list(exp_req - got_req)),
            "req_fp": sorted(list(got_req - exp_req)),
            "pref_precision": round(p_p, 4),
            "pref_recall": round(r_p, 4),
            "pref_f1": round(f1_p, 4),
            "pref_misses": sorted(list(exp_pref - got_pref)),
            "pref_fp": sorted(list(got_pref - exp_pref)),
            "min_years_got": jd_res["min_years"],
            "min_years_expected": expected.get("min_years"),
            "min_years_pass": min_years_pass,
        }

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    summary = {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "total_trap_violations": trap_violations,
        "fixtures": fixture_results,
    }

    # Print markdown table of fixture results
    logger.info("\n### Fixture Benchmark Results")
    logger.info("| Fixture | Type | Precision | Recall | F1 | Trap Hits | Fields / Years Pass |")
    logger.info("|---|---|---|---|---|---|---|")
    for name, r in fixture_results.items():
        if r["type"] == "resume":
            p_val = r["precision"]
            r_val = r["recall"]
            f1_val = r["f1"]
            th = len(r["trap_hits"])
            edu_status = "Pass" if r["education_match"] else "Fail"
            exp_status = "Pass" if r["experience_match"] else "Fail"
            yr_status = f"Pass (diff={r['years_diff']})" if r["years_pass"] else f"Fail ({r['years_diff']})"
            logger.info(f"| `{name}` | Resume | {p_val:.2f} | {r_val:.2f} | {f1_val:.2f} | {th} | Edu: {edu_status}, Exp: {exp_status}, Yrs: {yr_status} |")
        else:
            p_val = r["req_precision"]
            r_val = r["req_recall"]
            f1_val = r["req_f1"]
            yr_status = f"Pass (val={r['min_years_got']})" if r["min_years_pass"] else f"Fail (got {r['min_years_got']})"
            logger.info(f"| `{name}` | JD | Req P:{p_val:.2f} | Req R:{r_val:.2f} | Req F1:{f1_val:.2f} | 0 | Min Years: {yr_status} |")

    logger.info(f"\nMicro-Averaged Skill Precision: {micro_p:.4f}, Recall: {micro_r:.4f}, F1: {micro_f1:.4f}")
    logger.info(f"Trap Violations: {trap_violations} (Requirement: 0)")

    return summary


def run_real_text_stress_test(extractor: TaxonomySkillExtractor) -> Dict[str, Any]:
    """Run extraction stress test on validation parquet."""
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: REAL-TEXT STRESS TEST (CLASSIFIER VAL PARQUET)")
    logger.info("=" * 70)

    if not VAL_PARQUET.exists():
        logger.error(f"Val parquet not found at {VAL_PARQUET}")
        return {}

    val_df = pd.read_parquet(VAL_PARQUET)
    n_resumes = len(val_df)
    logger.info(f"Loaded {n_resumes} validation resumes.")

    ambig_df = pd.read_csv(AMBIGUOUS_CSV, keep_default_na=False)
    ambig_terms = set(ambig_df["term"].str.lower())

    # Metrics trackers
    target_categories = [
        "INFORMATION-TECHNOLOGY",
        "HEALTHCARE",
        "ACCOUNTANT",
        "CHEF",
        "SALES",
    ]

    sec_detected: Dict[str, Dict[str, int]] = {
        "OVERALL": {"exp": 0, "edu": 0, "skills": 0, "total": 0}
    }
    for cat in target_categories:
        sec_detected[cat] = {"exp": 0, "edu": 0, "skills": 0, "total": 0}

    skill_counts: List[int] = []
    skill_freq: Dict[str, int] = {}
    edu_parsed_count = 0
    exp_parsed_count = 0
    total_years_list: List[float] = []
    latencies_ms: List[float] = []

    for _, row in val_df.iterrows():
        raw_text = str(row["text"])
        cat = str(row.get("category", "")).upper()

        t0 = time.perf_counter()

        # Step 2 check: clean_text + split_sections
        c_text = clean_text(raw_text)
        sections = split_sections(c_text)

        has_exp = any(
            k in sections
            for k in ["work_experience", "experience", "employment", "professional_experience"]
        )
        has_edu = "education" in sections
        has_skills = any(
            k in sections
            for k in ["skills", "technical_skills", "core_competencies", "areas_of_expertise"]
        )

        sec_detected["OVERALL"]["total"] += 1
        if has_exp:
            sec_detected["OVERALL"]["exp"] += 1
        if has_edu:
            sec_detected["OVERALL"]["edu"] += 1
        if has_skills:
            sec_detected["OVERALL"]["skills"] += 1

        if cat in sec_detected:
            sec_detected[cat]["total"] += 1
            if has_exp:
                sec_detected[cat]["exp"] += 1
            if has_edu:
                sec_detected[cat]["edu"] += 1
            if has_skills:
                sec_detected[cat]["skills"] += 1

        # Extract entities
        parsed = ParsedResume(raw_text=c_text, sections=sections)
        parsed = extract_entities(parsed, as_of="2026-11-01")

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_elapsed)

        # Track skills
        s_count = len(parsed.skills)
        skill_counts.append(s_count)
        for s in parsed.skills:
            skill_freq[s] = skill_freq.get(s, 0) + 1

        # Track education & experience yield
        if len(parsed.education) > 0:
            edu_parsed_count += 1
        if len(parsed.experience) > 0:
            exp_parsed_count += 1
            total_years_list.append(parsed.total_years_experience)

    # 1. Section detection rates table
    logger.info("\n### Section Detection Rates (Step 2 Check)")
    logger.info("| Category | Count | Experience % | Education % | Skills % |")
    logger.info("|---|---|---|---|---|")
    sec_stats = {}
    for cat_name, d in sec_detected.items():
        tot = d["total"]
        if tot == 0:
            continue
        pct_exp = round((d["exp"] / tot) * 100, 1)
        pct_edu = round((d["edu"] / tot) * 100, 1)
        pct_skills = round((d["skills"] / tot) * 100, 1)
        logger.info(f"| {cat_name} | {tot} | {pct_exp}% | {pct_edu}% | {pct_skills}% |")
        sec_stats[cat_name] = {
            "total": tot,
            "pct_exp": pct_exp,
            "pct_edu": pct_edu,
            "pct_skills": pct_skills,
        }

    # 2. Skills per resume distribution
    mean_skills = round(float(np.mean(skill_counts)), 2)
    median_skills = round(float(np.median(skill_counts)), 2)
    p5_skills = round(float(np.percentile(skill_counts, 5)), 2)
    pct_under_5 = round(float(np.mean(np.array(skill_counts) < 5) * 100), 2)

    logger.info("\n### Skills Extracted Per Resume")
    logger.info(f"- Mean: {mean_skills}")
    logger.info(f"- Median: {median_skills}")
    logger.info(f"- 5th Percentile (p5): {p5_skills}")
    logger.info(f"- Resumes with < 5 skills: {pct_under_5}% (vs train baseline 23.97%)")

    # 3. Top 20 skills & suspects
    sorted_skills = sorted(skill_freq.items(), key=lambda x: -x[1])
    top_20 = [
        {"skill": s, "count": c, "pct": round((c / n_resumes) * 100, 2)}
        for s, c in sorted_skills[:20]
    ]

    suspects = [
        {"skill": s, "count": c, "pct": round((c / n_resumes) * 100, 2)}
        for s, c in sorted_skills
        if (c / n_resumes) > 0.10 and s.lower() not in ambig_terms
    ]

    logger.info("\n### Top 20 Extracted Skills")
    logger.info("| Rank | Skill | Count | % of Resumes |")
    logger.info("|---|---|---|---|")
    for rank, item in enumerate(top_20, 1):
        logger.info(f"| {rank} | `{item['skill']}` | {item['count']} | {item['pct']}% |")

    logger.info("\n### Suspect Skills (>10% Resumes, not in ambiguous_terms.csv)")
    if suspects:
        logger.info("| Skill | Count | % of Resumes |")
        logger.info("|---|---|---|")
        for item in suspects:
            logger.info(f"| `{item['skill']}` | {item['count']} | {item['pct']}% |")
    else:
        logger.info("None detected.")

    # 4. Education & Experience yield and total_years distribution
    pct_edu_parsed = round((edu_parsed_count / n_resumes) * 100, 2)
    pct_exp_parsed = round((exp_parsed_count / n_resumes) * 100, 2)

    mean_years = round(float(np.mean(total_years_list)), 2) if total_years_list else 0.0
    median_years = round(float(np.median(total_years_list)), 2) if total_years_list else 0.0
    p95_years = round(float(np.percentile(total_years_list, 95)), 2) if total_years_list else 0.0
    pct_over_40 = (
        round(float(np.mean(np.array(total_years_list) > 40.0) * 100), 2)
        if total_years_list
        else 0.0
    )

    logger.info("\n### Structured Entries Yield & Experience Years")
    logger.info(f"- Resumes with >= 1 Education Entry: {pct_edu_parsed}%")
    logger.info(f"- Resumes with >= 1 Experience Entry: {pct_exp_parsed}%")
    logger.info(f"- Total Years Mean: {mean_years}")
    logger.info(f"- Total Years Median: {median_years}")
    logger.info(f"- Total Years p95: {p95_years}")
    logger.info(f"- Total Years > 40 Sanity Flag: {pct_over_40}%")

    # 5. Extraction Latency
    mean_lat = round(float(np.mean(latencies_ms)), 2)
    p95_lat = round(float(np.percentile(latencies_ms, 95)), 2)
    logger.info(f"\nExtraction Latency per Resume: Mean {mean_lat} ms, p95 {p95_lat} ms")

    return {
        "section_detection": sec_stats,
        "skills_distribution": {
            "mean": mean_skills,
            "median": median_skills,
            "p5": p5_skills,
            "pct_under_5": pct_under_5,
        },
        "top_20_skills": top_20,
        "suspect_skills": suspects,
        "entries_yield": {
            "pct_education_parsed": pct_edu_parsed,
            "pct_experience_parsed": pct_exp_parsed,
            "total_years_mean": mean_years,
            "total_years_median": median_years,
            "total_years_p95": p95_years,
            "total_years_pct_over_40": pct_over_40,
        },
        "latency_ms": {
            "mean": mean_lat,
            "p95": p95_lat,
        },
    }


def main() -> None:
    extractor = TaxonomySkillExtractor()
    fixture_summary = evaluate_fixtures(extractor)
    stress_summary = run_real_text_stress_test(extractor)

    full_eval = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixtures": fixture_summary,
        "real_text_stress_test": stress_summary,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_eval, f, indent=2)

    logger.info(f"\nAll extraction metrics exported to {METRICS_PATH}")


if __name__ == "__main__":
    main()
