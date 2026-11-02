#!/usr/bin/env python3
"""verify_data.py – Validate downloaded raw dataset integrity, schemas, and row counts.

Expected checks:
----------------
1. LiveCareer:       2,484 rows, 24 job categories.
2. NER Synthetic:    9,600 train / 1,200 val / 1,200 test rows.
3. NER Real Skills:  5,029 annotated CV rows.
4. Job-Resume Fit:   2,385 rows, 23 job categories.
5. LinkedIn Sample:  5,000 postings with non-empty title & desc >= 200 chars.
6. O*NET:            software_skills.csv (~32,773 rows; reports actual) & occupation_data.csv (1,016 rows).
7. ESCO (Optional):  skills_en.csv (~13,890 rows) if present.

Privacy Rule:
-------------
Never print resume or job description text. Print only counts, column names,
and aggregate statistical metrics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "docs" / "data_manifest.json"


def _rel_path(path: Path) -> str:
    """Safely format relative path to ROOT, or return path string if outside ROOT."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class CheckResult:
    dataset: str
    file_rel: str
    expected: str
    actual: str
    status: str  # "PASS", "FAIL", "OPTIONAL"


def format_table(results: list[CheckResult]) -> str:
    """Format check results as a clean ASCII table."""
    headers = ["Dataset", "File", "Expected", "Actual", "Status"]
    rows = [
        [r.dataset, r.file_rel, r.expected, r.actual, r.status]
        for r in results
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def verify_livecareer() -> list[CheckResult]:
    p = DATA_RAW / "livecareer" / "train.parquet"
    if not p.exists():
        return [CheckResult("LiveCareer", _rel_path(p), "2,484 rows, 24 cats", "FILE MISSING", "FAIL")]

    df = pd.read_parquet(p)
    rows = len(df)
    cats = df["Category"].nunique() if "Category" in df.columns else 0
    cols_ok = all(c in df.columns for c in ["ID", "Resume_str", "Resume_html", "Category"])

    status = "PASS" if (rows == 2484 and cats == 24 and cols_ok) else "FAIL"
    actual = f"{rows:,} rows, {cats} categories"
    return [CheckResult("LiveCareer", _rel_path(p), "2,484 rows, 24 cats", actual, status)]


def verify_ner_synthetic() -> list[CheckResult]:
    base = DATA_RAW / "ner_synthetic"
    specs = [
        ("train.parquet", 9600),
        ("validation.parquet", 1200),
        ("test.parquet", 1200),
    ]
    results = []
    for fname, exp_rows in specs:
        p = base / fname
        if not p.exists():
            results.append(CheckResult("NER Synthetic", _rel_path(p), f"{exp_rows:,} rows", "FILE MISSING", "FAIL"))
            continue
        df = pd.read_parquet(p)
        rows = len(df)
        cols_ok = all(c in df.columns for c in ["id", "text", "bio_labels", "tokens"])
        status = "PASS" if (rows == exp_rows and cols_ok) else "FAIL"
        results.append(CheckResult("NER Synthetic", _rel_path(p), f"{exp_rows:,} rows", f"{rows:,} rows", status))
    return results


def verify_ner_real_skills() -> list[CheckResult]:
    p = DATA_RAW / "ner_real_skills" / "ner_real_skills.parquet"
    if not p.exists():
        return [CheckResult("NER Real Skills", _rel_path(p), "5,029 rows", "FILE MISSING", "FAIL")]

    df = pd.read_parquet(p)
    rows = len(df)
    cols_ok = all(c in df.columns for c in ["id", "text", "annotations"])
    status = "PASS" if (rows == 5029 and cols_ok) else "FAIL"
    return [CheckResult("NER Real Skills", _rel_path(p), "5,029 rows", f"{rows:,} rows", status)]


def verify_job_resume_fit() -> list[CheckResult]:
    p = DATA_RAW / "job_resume_fit" / "train.parquet"
    if not p.exists():
        return [CheckResult("Job-Resume Fit", _rel_path(p), "2,385 rows, 23 cats", "FILE MISSING", "FAIL")]

    df = pd.read_parquet(p)
    rows = len(df)
    cats = df["category"].nunique() if "category" in df.columns else 0
    cols_ok = all(c in df.columns for c in ["ID", "resume_text", "job_text", "category", "ai_match_score"])
    status = "PASS" if (rows == 2385 and cats == 23 and cols_ok) else "FAIL"
    actual = f"{rows:,} rows, {cats} categories"
    return [CheckResult("Job-Resume Fit", _rel_path(p), "2,385 rows, 23 cats", actual, status)]


def verify_linkedin_sample() -> list[CheckResult]:
    p = DATA_RAW / "linkedin_sample" / "postings_5k.parquet"
    if not p.exists():
        return [CheckResult("LinkedIn Sample", _rel_path(p), "5,000 rows", "FILE MISSING", "FAIL")]

    df = pd.read_parquet(p)
    rows = len(df)
    cols_ok = all(c in df.columns for c in ["title", "description"])
    min_desc_len = df["description"].astype(str).str.len().min() if "description" in df.columns and len(df) > 0 else 0
    has_empty_title = df["title"].astype(str).str.strip().eq("").any() if "title" in df.columns else True

    status = "PASS" if (rows == 5000 and cols_ok and min_desc_len >= 200 and not has_empty_title) else "FAIL"
    actual = f"{rows:,} rows (min desc: {min_desc_len} chars)"
    return [CheckResult("LinkedIn Sample", _rel_path(p), "5,000 rows (desc>=200)", actual, status)]


def verify_onet() -> list[CheckResult]:
    base = DATA_RAW / "onet"
    results = []

    # Software skills
    sw_path = base / "software_skills.csv"
    if not sw_path.exists():
        results.append(CheckResult("O*NET Software", _rel_path(sw_path), "~32,773 rows", "FILE MISSING", "FAIL"))
    else:
        df_sw = pd.read_csv(sw_path)
        rows_sw = len(df_sw)
        cols_ok = all(c in df_sw.columns for c in ["O*NET-SOC Code", "Title", "Workplace Example", "Hot Technology", "In Demand"])
        # Expected around 32,773; O*NET 31.0 contains 31,821 rows
        status = "PASS" if (30_000 <= rows_sw <= 35_000 and cols_ok) else "FAIL"
        results.append(CheckResult("O*NET Software", _rel_path(sw_path), "~32,773 rows", f"{rows_sw:,} rows", status))

    # Occupation data
    occ_path = base / "occupation_data.csv"
    if not occ_path.exists():
        results.append(CheckResult("O*NET Occupations", _rel_path(occ_path), "1,016 rows", "FILE MISSING", "FAIL"))
    else:
        df_occ = pd.read_csv(occ_path)
        rows_occ = len(df_occ)
        cols_ok = all(c in df_occ.columns for c in ["O*NET-SOC Code", "Title", "Description"])
        status = "PASS" if (rows_occ == 1016 and cols_ok) else "FAIL"
        results.append(CheckResult("O*NET Occupations", _rel_path(occ_path), "1,016 rows", f"{rows_occ:,} rows", status))

    return results


def verify_esco() -> list[CheckResult]:
    p = DATA_RAW / "esco" / "skills_en.csv"
    if not p.exists():
        return [
            CheckResult(
                "ESCO Taxonomy",
                _rel_path(p),
                "~13,890 concepts (optional)",
                "NOT DOWNLOADED (Manual EU portal export required)",
                "OPTIONAL",
            )
        ]

    df = pd.read_csv(p)
    rows = len(df)
    cols_ok = "preferredLabel" in df.columns
    status = "PASS" if (rows >= 10000 and cols_ok) else "FAIL"
    return [CheckResult("ESCO Taxonomy", _rel_path(p), "~13,890 concepts", f"{rows:,} rows", status)]


def verify_manifest() -> CheckResult:
    if not MANIFEST_PATH.exists():
        return CheckResult("Manifest", _rel_path(MANIFEST_PATH), "Valid JSON manifest", "MISSING", "FAIL")

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries_count = len(data)
        status = "PASS" if entries_count >= 5 else "FAIL"
        return CheckResult("Manifest", _rel_path(MANIFEST_PATH), ">= 5 dataset entries", f"{entries_count} entries", status)
    except Exception as e:
        return CheckResult("Manifest", _rel_path(MANIFEST_PATH), "Valid JSON manifest", f"ERROR ({e})", "FAIL")


def main() -> None:
    logger.info("Starting dataset verification audit...")
    results: list[CheckResult] = []

    results.extend(verify_livecareer())
    results.extend(verify_ner_synthetic())
    results.extend(verify_ner_real_skills())
    results.extend(verify_job_resume_fit())
    results.extend(verify_linkedin_sample())
    results.extend(verify_onet())
    results.extend(verify_esco())
    results.append(verify_manifest())

    table_str = format_table(results)
    print("\n" + table_str + "\n")

    has_failures = any(r.status == "FAIL" for r in results)
    if has_failures:
        logger.error("Dataset verification FAILED. Please review the table above.")
        sys.exit(1)
    else:
        logger.success("All dataset verification checks PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
