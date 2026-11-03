"""
scripts/verify_splits.py
========================
Independent re-check of all splits, printing a PASS/FAIL table.

Checks:
  1. No id or group_id appears in more than one classifier split.
  2. No exact text duplicate across classifier splits.
  3. Fresh TF-IDF fit: count cross-split resume pairs with cosine >= 0.90 (must be 0).
  4. Per-category counts per split, and the minimum per split.
  5. No (title, description[:200]) key shared across JD splits.
  6. jrf_eval resume IDs and ner_gold IDs are subsets of classifier test IDs.
  7. The CSVs in data/splits/ contain no text columns.

Privacy: never prints resume or JD body text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

# -- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_RAW = ROOT / "data" / "raw"

TFIDF_THRESHOLD = 0.90
TFIDF_MAX_FEATURES = 10_000

# Long columns that should NOT appear in the split CSVs
TEXT_COLUMN_NAMES: Set[str] = {
    "text", "text_masked", "resume_str", "resume_html", "description",
    "body", "job_text", "resume_text",
}


# -- Helpers ----------------------------------------------------------------

def _fmt(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _check(label: str, condition: bool, detail: str = "") -> Tuple[str, bool, str]:
    status = _fmt(condition)
    return (label, condition, detail)


def check_no_id_leakage(cls_split: pd.DataFrame) -> Tuple[bool, str]:
    """Check that no id or group_id appears in more than one split."""
    for col in ("id", "group_id"):
        cross = (
            cls_split.groupby(col)["split"].nunique()
        )
        bad = cross[cross > 1]
        if len(bad):
            return False, f"{col}: {len(bad)} values appear in >1 split"
    return True, ""


def check_no_exact_text_duplicate(splits: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    """Check that no exact text appears in more than one split."""
    sets: Dict[str, Set[str]] = {
        sp: set(df["text"].tolist()) for sp, df in splits.items()
    }
    names = list(sets.keys())
    violations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                violations.append(f"{names[i]}/{names[j]}: {len(overlap)} overlapping texts")
    if violations:
        return False, "; ".join(violations)
    return True, ""


def check_tfidf_cross_split(splits: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    """Fresh TF-IDF fit: count cross-split pairs with cosine >= 0.90."""
    split_names = list(splits.keys())
    # Build labelled corpus
    texts: List[str] = []
    labels: List[str] = []
    for sp in split_names:
        texts.extend(splits[sp]["text"].tolist())
        labels.extend([sp] * len(splits[sp]))

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    norms = np.asarray(X.power(2).sum(axis=1)).ravel() ** 0.5
    norms[norms == 0] = 1.0

    n = len(texts)
    BLOCK = 200
    cross_pairs = 0
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        block = X[start:end]
        dots = (block @ X.T).toarray()
        cos = dots / (norms[start:end, None] * norms[None, :])
        for i_local, i_global in enumerate(range(start, end)):
            for j_global in range(i_global + 1, n):
                if labels[i_global] != labels[j_global]:
                    if cos[i_local, j_global] >= TFIDF_THRESHOLD:
                        cross_pairs += 1

    passed = cross_pairs == 0
    detail = f"cross-split pairs >= {TFIDF_THRESHOLD}: {cross_pairs}"
    return passed, detail


def check_per_category_counts(
    cls_split: pd.DataFrame,
) -> Tuple[bool, str]:
    """Print per-category counts per split and report minimum."""
    counts = cls_split.groupby(["category", "split"]).size().unstack(fill_value=0)
    logger.info("Per-category split counts:\n" + counts.to_string())
    min_per_split = {sp: counts[sp].min() for sp in counts.columns}
    logger.info("Minimum per split: " + str(min_per_split))
    # No hard failure condition; just report
    return True, f"min counts: {min_per_split}"


def check_no_jd_key_overlap(jd_split: pd.DataFrame, li_raw: pd.DataFrame) -> Tuple[bool, str]:
    """Check no (title, description[:200]) key shared across JD splits."""
    # Rebuild dedup key from raw linkedin data merged with jd_split
    li_raw = li_raw.copy()
    li_raw["jd_id"] = range(len(li_raw))
    li_raw["_dedup_key"] = (
        li_raw["title"].str.strip() + "|||" + li_raw["description"].str[:200].fillna("")
    )
    merged = jd_split.merge(li_raw[["jd_id", "_dedup_key"]], on="jd_id", how="left")

    splits = merged["split"].unique()
    violations = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            k1 = set(merged.loc[merged["split"] == s1, "_dedup_key"].tolist())
            k2 = set(merged.loc[merged["split"] == s2, "_dedup_key"].tolist())
            overlap = k1 & k2
            if overlap:
                violations.append(f"{s1}/{s2}: {len(overlap)} shared keys")

    if violations:
        return False, "; ".join(violations)
    return True, ""


def check_id_subsets(
    cls_split: pd.DataFrame,
    jrf_eval: pd.DataFrame,
    ner_gold: pd.DataFrame,
) -> Tuple[bool, str]:
    """Check jrf_eval and ner_gold IDs are subsets of classifier test IDs."""
    test_ids: Set[int] = set(cls_split.loc[cls_split["split"] == "test", "id"].tolist())
    errors: List[str] = []

    jrf_ids: Set[int] = set(jrf_eval["resume_id"].tolist())
    non_test_jrf = jrf_ids - test_ids
    if non_test_jrf:
        errors.append(f"jrf_eval has {len(non_test_jrf)} IDs not in test split")

    ner_ids: Set[int] = set(ner_gold["id"].tolist())
    non_test_ner = ner_ids - test_ids
    if non_test_ner:
        errors.append(f"ner_gold has {len(non_test_ner)} IDs not in test split")

    if errors:
        return False, "; ".join(errors)
    return True, f"jrf_eval={len(jrf_ids)} ner_gold={len(ner_ids)} both subsets of test ({len(test_ids)})"


def check_csv_no_text_columns(splits_dir: Path) -> Tuple[bool, str]:
    """Check that CSVs in data/splits/ contain no text columns."""
    errors: List[str] = []
    for csv_path in splits_dir.glob("*.csv"):
        df = pd.read_csv(csv_path, nrows=1)
        bad_cols = [c for c in df.columns if c.lower() in TEXT_COLUMN_NAMES]
        if bad_cols:
            errors.append(f"{csv_path.name}: found text columns {bad_cols}")
    if errors:
        return False, "; ".join(errors)
    return True, f"all CSVs in {splits_dir.name}/ are text-free"


# -- Main -------------------------------------------------------------------

def main() -> None:
    """Run all verification checks and print a PASS/FAIL table."""
    logger.info("=== verify_splits.py start ===")

    # Load artefacts
    cls_split_path = DATA_SPLITS / "classifier_split.csv"
    jd_split_path = DATA_SPLITS / "jd_split.csv"
    ner_gold_path = DATA_SPLITS / "ner_gold_ids.csv"
    jrf_eval_path = DATA_PROCESSED / "jrf_eval.parquet"
    li_path = DATA_RAW / "linkedin_sample" / "postings_5k.parquet"

    for p in [cls_split_path, jd_split_path, ner_gold_path, jrf_eval_path, li_path]:
        if not p.exists():
            logger.error(f"Required file missing: {p}")
            sys.exit(1)

    cls_split = pd.read_csv(cls_split_path)
    jd_split = pd.read_csv(jd_split_path)
    ner_gold = pd.read_csv(ner_gold_path)
    jrf_eval = pd.read_parquet(jrf_eval_path)
    li_raw = pd.read_parquet(li_path)

    # Dedupe li_raw to match build_jd_pool.py
    li_raw["_dedup_key"] = (
        li_raw["title"].str.strip() + "|||" + li_raw["description"].str[:200].fillna("")
    )
    li_raw = li_raw.drop_duplicates(subset=["_dedup_key"], keep="first").reset_index(drop=True)
    li_raw["jd_id"] = range(len(li_raw))

    # Load classifier parquets
    cls_parquets: Dict[str, pd.DataFrame] = {}
    for sp in ("train", "val", "test"):
        p = DATA_PROCESSED / f"classifier_{sp}.parquet"
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)
        cls_parquets[sp] = pd.read_parquet(p)

    results: List[Tuple[str, bool, str]] = []

    # Check 1: No id/group_id leakage
    ok, detail = check_no_id_leakage(cls_split)
    results.append(("1. No id/group_id in >1 classifier split", ok, detail))

    # Check 2: No exact text duplicate across splits
    ok, detail = check_no_exact_text_duplicate(cls_parquets)
    results.append(("2. No exact text duplicate across splits", ok, detail))

    # Check 3: TF-IDF cross-split cosine
    logger.info("Running fresh TF-IDF check (this may take ~30s) …")
    ok, detail = check_tfidf_cross_split(cls_parquets)
    results.append(("3. Fresh TF-IDF: 0 cross-split pairs >= 0.90", ok, detail))

    # Check 4: Per-category counts
    ok, detail = check_per_category_counts(cls_split)
    results.append(("4. Per-category counts reported", ok, detail))

    # Check 5: No JD key overlap
    ok, detail = check_no_jd_key_overlap(jd_split, li_raw)
    results.append(("5. No (title, desc[:200]) key across JD splits", ok, detail))

    # Check 6: ID subsets
    ok, detail = check_id_subsets(cls_split, jrf_eval, ner_gold)
    results.append(("6. jrf_eval & ner_gold IDs subset of test", ok, detail))

    # Check 7: CSVs have no text columns
    ok, detail = check_csv_no_text_columns(DATA_SPLITS)
    results.append(("7. data/splits/ CSVs contain no text columns", ok, detail))

    # Print table
    logger.info("\n" + "=" * 75)
    logger.info(f"{'CHECK':<52} {'STATUS':<6}  DETAIL")
    logger.info("=" * 75)
    all_passed = True
    for label, passed, detail in results:
        status = _fmt(passed)
        logger.info(f"{label:<52} {status:<6}  {detail}")
        if not passed:
            all_passed = False
    logger.info("=" * 75)
    if all_passed:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.error("SOME CHECKS FAILED — review above")
        sys.exit(1)

    logger.info("=== verify_splits.py done ===")


if __name__ == "__main__":
    main()
