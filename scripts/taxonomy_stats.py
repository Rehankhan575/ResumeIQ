"""Compute taxonomy document frequency (DF) stats across train splits.

Uses fast n-gram (1-5 word) lookup with alias mapping instead of hundreds
of individual regex searches. Evaluates against:
- Classifier train split (unmasked `text` column)
- JD train split (`text` column)

Generates:
- data/taxonomies/skills_stats.csv

Reports summary metrics for documentation in docs/taxonomy.md.
Privacy: prints only aggregate counts and skill strings, never raw body text.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_TAX = ROOT / "data" / "taxonomies"
DATA_PROCESSED = ROOT / "data" / "processed"

SKILLS_CSV = DATA_TAX / "skills.csv"
AMBIGUOUS_CSV = DATA_TAX / "ambiguous_terms.csv"
STATS_CSV = DATA_TAX / "skills_stats.csv"

CLASSIFIER_TRAIN = DATA_PROCESSED / "classifier_train.parquet"
JD_TRAIN = DATA_PROCESSED / "jd_train.parquet"

# Token pattern: matches words, dotted acronyms, c++/c#, .net, &, etc.
TOKEN_PATTERN = re.compile(
    r"\.net\b|[a-z0-9]+(?:#|\+\+)?(?:[/\'\&.-][a-z0-9]+)*(?:#|\+\+)?|\&"
)


def normalize_text(text: str) -> str:
    """Normalize unicode characters and lowercase."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return text.lower()


def build_taxonomy_lookup(
    skills_df: pd.DataFrame,
) -> Tuple[Dict[str, str], int]:
    """Map canonical skill names and aliases to canonical skill name.

    Returns:
        (term_to_skill, max_ngram_len)
    """
    term_to_skill: Dict[str, str] = {}
    max_len = 1

    for _, row in skills_df.iterrows():
        canonical = row["skill"]
        # Add canonical skill
        tokens = TOKEN_PATTERN.findall(normalize_text(canonical))
        if tokens:
            key = " ".join(tokens)
            term_to_skill[key] = canonical
            max_len = max(max_len, len(tokens))

        # Add aliases
        aliases = str(row.get("aliases", "")).split("|")
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            a_tokens = TOKEN_PATTERN.findall(normalize_text(alias))
            if a_tokens:
                a_key = " ".join(a_tokens)
                term_to_skill[a_key] = canonical
                max_len = max(max_len, len(a_tokens))

    return term_to_skill, max_len


def count_doc_skills(
    texts: pd.Series,
    term_to_skill: Dict[str, str],
    max_n: int = 5,
) -> Tuple[Dict[str, int], List[int]]:
    """Scan texts using sliding window n-grams and count document frequency.

    Returns:
        (df_per_skill, list_of_distinct_skills_per_document)
    """
    df_counts: Dict[str, int] = {skill: 0 for skill in set(term_to_skill.values())}
    per_doc_counts: List[int] = []

    for text in texts:
        tokens = TOKEN_PATTERN.findall(normalize_text(text))
        n_tokens = len(tokens)
        matched_skills: Set[str] = set()

        for n in range(1, max_n + 1):
            for i in range(n_tokens - n + 1):
                ngram = " ".join(tokens[i : i + n])
                if ngram in term_to_skill:
                    matched_skills.add(term_to_skill[ngram])

        per_doc_counts.append(len(matched_skills))
        for skill in matched_skills:
            df_counts[skill] += 1

    return df_counts, per_doc_counts


def run_taxonomy_stats() -> Dict[str, Any]:
    """Calculate and export taxonomy document frequency statistics."""
    logger.info("Loading skills taxonomy and train splits...")
    skills_df = pd.read_csv(SKILLS_CSV, keep_default_na=False)
    ambig_df = pd.read_csv(AMBIGUOUS_CSV, keep_default_na=False)
    ambig_set = set(ambig_df["term"].str.lower())

    classifier_train = pd.read_parquet(CLASSIFIER_TRAIN)
    jd_train = pd.read_parquet(JD_TRAIN)

    n_resumes = len(classifier_train)
    n_jds = len(jd_train)
    logger.info(f"Loaded {len(skills_df)} skills, {n_resumes} resumes, {n_jds} JDs.")

    term_to_skill, max_n = build_taxonomy_lookup(skills_df)
    logger.info(f"Built lookup dictionary with {len(term_to_skill)} entries (max n-gram: {max_n}).")

    logger.info("Scanning classifier train resumes (unmasked 'text' column)...")
    res_df, res_per_doc = count_doc_skills(classifier_train["text"], term_to_skill, max_n)

    logger.info("Scanning JD train postings ('text' column)...")
    jd_df, _ = count_doc_skills(jd_train["text"], term_to_skill, max_n)

    # Attach resume counts to DataFrame for per-category analysis
    classifier_train["skill_hits"] = res_per_doc

    # Build stats DataFrame
    rows = []
    for _, row in skills_df.iterrows():
        s = row["skill"]
        d_res = res_df.get(s, 0)
        d_jd = jd_df.get(s, 0)
        pct_res = round((d_res / n_resumes) * 100, 2) if n_resumes else 0.0
        # Active if df > 0 in either, or source is curated
        is_active = bool(d_res > 0 or d_jd > 0 or row["source"] == "curated")
        rows.append({
            "skill": s,
            "category": row["category"],
            "source": row["source"],
            "df_resume": d_res,
            "df_jd": d_jd,
            "df_resume_pct": pct_res,
            "active": is_active,
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(STATS_CSV, index=False)
    logger.info(f"Wrote {STATS_CSV} ({len(stats_df)} rows).")

    # 1. Active vs inactive counts per source
    active_summary = stats_df.groupby(["source", "active"]).size().unstack(fill_value=0)
    logger.info("\n=== 1. Active vs Inactive Counts Per Source ===")
    logger.info(active_summary.to_string())

    # 2. Top 40 skills by resume DF and by JD DF
    top40_resume = stats_df.sort_values(by="df_resume", ascending=False).head(40)
    top40_jd = stats_df.sort_values(by="df_jd", ascending=False).head(40)
    logger.info("\n=== 2a. Top 40 Skills by Resume Document Frequency ===")
    logger.info(top40_resume[["skill", "category", "source", "df_resume", "df_resume_pct"]].to_string(index=False))

    logger.info("\n=== 2b. Top 40 Skills by JD Document Frequency ===")
    logger.info(top40_jd[["skill", "category", "source", "df_jd"]].to_string(index=False))

    # 3. Suspects: skills with resume DF over 10% that are NOT in ambiguous_terms.csv
    suspects = stats_df[
        (stats_df["df_resume_pct"] > 10.0) & (~stats_df["skill"].str.lower().isin(ambig_set))
    ].sort_values(by="df_resume_pct", ascending=False)
    logger.info("\n=== 3. Suspect Skills (Resume DF > 10% & not in ambiguous_terms.csv) ===")
    logger.info(suspects[["skill", "category", "source", "df_resume", "df_resume_pct"]].to_string(index=False))

    # 4. Distinct skills per resume (overall and key categories)
    def compute_distribution(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "p5": round(float(np.percentile(series, 5)), 2),
            "pct_under_5": round(float((series < 5).mean() * 100), 2),
        }

    cat_stats: Dict[str, Dict[str, float]] = {
        "OVERALL": compute_distribution(classifier_train["skill_hits"])
    }
    target_cats = [
        "INFORMATION-TECHNOLOGY",
        "HEALTHCARE",
        "ACCOUNTANT",
        "CHEF",
        "SALES",
    ]
    for cat in target_cats:
        sub = classifier_train[classifier_train["category"] == cat]["skill_hits"]
        if len(sub) > 0:
            cat_stats[cat] = compute_distribution(sub)

    cat_stats_df = pd.DataFrame(cat_stats).T
    logger.info("\n=== 4. Distinct Skills Found Per Resume ===")
    logger.info(cat_stats_df.to_string())

    # 5. 15 O*NET skills with highest DF and 15 with DF=0 (skill strings only)
    onet_df = stats_df[stats_df["source"] == "onet"]
    onet_high = onet_df.sort_values(by="df_resume", ascending=False).head(15)["skill"].tolist()
    onet_zero = onet_df[onet_df["df_resume"] == 0].sort_values(by="skill").head(15)["skill"].tolist()

    logger.info("\n=== 5a. 15 O*NET Skills with Highest Resume DF ===")
    for s in onet_high:
        logger.info(f"  - {s}")

    logger.info("\n=== 5b. 15 O*NET Skills with Resume DF = 0 ===")
    for s in onet_zero:
        logger.info(f"  - {s}")

    # 6. 30 random O*NET skills printed with their category (seed 42)
    onet_sample_30 = onet_df.sample(30, random_state=42)[["skill", "category"]].sort_values(by="skill")
    logger.info("\n=== 6. 30 Random O*NET Skills (User Review List) ===")
    logger.info(onet_sample_30.to_string(index=False))

    return {
        "stats_df": stats_df,
        "active_summary": active_summary,
        "suspects": suspects,
        "cat_stats_df": cat_stats_df,
        "onet_high": onet_high,
        "onet_zero": onet_zero,
        "onet_sample_30": onet_sample_30,
        "top40_resume": top40_resume,
        "top40_jd": top40_jd,
    }


if __name__ == "__main__":
    run_taxonomy_stats()
