"""
scripts/build_jd_pool.py
========================
Build the JD pool for the matcher from linkedin_sample.

Steps
-----
1. Load linkedin_sample; dedupe on (title, description[:200]), keep first.
2. Create data/taxonomies/title_category_map.csv (keyword, category).
   Titles matched case-insensitively on word boundaries.
   Ambiguous (matches >1 category) and unmapped titles are excluded.
3. Print mapped/ambiguous/unmapped counts; 40 most frequent unmapped titles;
   60 random mapped (title -> category) pairs.
4. Split mapped JDs 70/15/15 at (title, description[:200]) group level,
   stratified by category, seed 42.
   Categories with < 20 mapped JDs flagged "too_small_for_matcher".
5. Write data/splits/jd_split.csv (jd_id, category, split, title).
   Write data/processed/jd_{train,val,test}.parquet (jd_id, title, text, category).
6. Write data/processed/jrf_eval.parquet: rows of job_resume_fit whose
   resume ID is in the classifier TEST split.
   Columns: resume_id, category, job_text, ai_match_score.

Privacy: never prints JD body text; prints titles only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# -- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_TAXONOMIES = ROOT / "data" / "taxonomies"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_SPLITS.mkdir(parents=True, exist_ok=True)
DATA_TAXONOMIES.mkdir(parents=True, exist_ok=True)

SEED = 42
SMALL_CATEGORY_THRESHOLD = 20


# -- Title -> Category mapping rules ----------------------------------------
# Each entry: (keyword_regex_fragment, category)
# Keyword matched case-insensitively on word boundaries (\b...\b).
# Order does NOT matter; ambiguous = matched to >1 category = excluded.

RAW_RULES: List[Tuple[str, str]] = [
    # ACCOUNTANT
    ("accountant", "ACCOUNTANT"),
    ("accounting manager", "ACCOUNTANT"),
    ("staff accountant", "ACCOUNTANT"),
    ("senior accountant", "ACCOUNTANT"),
    ("cpa", "ACCOUNTANT"),
    ("bookkeeper", "ACCOUNTANT"),
    ("controller", "ACCOUNTANT"),
    ("accounts payable", "ACCOUNTANT"),
    ("accounts receivable", "ACCOUNTANT"),
    ("tax accountant", "ACCOUNTANT"),
    ("payroll", "ACCOUNTANT"),
    ("auditor", "ACCOUNTANT"),
    ("audit", "ACCOUNTANT"),
    # ADVOCATE
    ("attorney", "ADVOCATE"),
    ("lawyer", "ADVOCATE"),
    ("paralegal", "ADVOCATE"),
    ("legal counsel", "ADVOCATE"),
    ("litigation", "ADVOCATE"),
    ("legal assistant", "ADVOCATE"),
    ("associate attorney", "ADVOCATE"),
    ("public defender", "ADVOCATE"),
    ("counsel", "ADVOCATE"),
    ("legal secretary", "ADVOCATE"),
    # AGRICULTURE
    ("farm", "AGRICULTURE"),
    ("agriculture", "AGRICULTURE"),
    ("agronomist", "AGRICULTURE"),
    ("horticultur", "AGRICULTURE"),
    ("livestock", "AGRICULTURE"),
    ("crop", "AGRICULTURE"),
    ("rancher", "AGRICULTURE"),
    ("vineyard", "AGRICULTURE"),
    # APPAREL
    ("fashion", "APPAREL"),
    ("apparel", "APPAREL"),
    ("textile", "APPAREL"),
    ("buyer.*fashion", "APPAREL"),
    ("garment", "APPAREL"),
    ("clothing", "APPAREL"),
    # ARTS
    ("artist", "ARTS"),
    ("musician", "ARTS"),
    ("actor", "ARTS"),
    ("photographer", "ARTS"),
    ("illustrator", "ARTS"),
    ("curator", "ARTS"),
    ("art director", "ARTS"),
    ("animator", "ARTS"),
    ("fine art", "ARTS"),
    # AUTOMOBILE
    ("automotive", "AUTOMOBILE"),
    ("auto technician", "AUTOMOBILE"),
    ("car dealer", "AUTOMOBILE"),
    ("auto mechanic", "AUTOMOBILE"),
    ("vehicle", "AUTOMOBILE"),
    ("dealership", "AUTOMOBILE"),
    ("auto sales", "AUTOMOBILE"),
    # AVIATION
    ("pilot", "AVIATION"),
    ("aviation", "AVIATION"),
    ("flight", "AVIATION"),
    ("airline", "AVIATION"),
    ("aircraft", "AVIATION"),
    ("air traffic", "AVIATION"),
    ("avionics", "AVIATION"),
    # BANKING
    ("banker", "BANKING"),
    ("bank teller", "BANKING"),
    ("loan officer", "BANKING"),
    ("mortgage", "BANKING"),
    ("credit analyst", "BANKING"),
    ("branch manager.*bank", "BANKING"),
    ("investment banker", "BANKING"),
    ("underwriter", "BANKING"),
    ("financial analyst", "BANKING"),
    ("wealth management", "BANKING"),
    # BPO
    ("bpo", "BPO"),
    ("call center", "BPO"),
    ("customer service representative", "BPO"),
    ("customer support", "BPO"),
    ("contact center", "BPO"),
    ("help desk", "BPO"),
    ("service desk", "BPO"),
    # BUSINESS-DEVELOPMENT
    ("business development", "BUSINESS-DEVELOPMENT"),
    ("strategy manager", "BUSINESS-DEVELOPMENT"),
    ("partnerships", "BUSINESS-DEVELOPMENT"),
    ("corporate development", "BUSINESS-DEVELOPMENT"),
    ("growth manager", "BUSINESS-DEVELOPMENT"),
    ("market development", "BUSINESS-DEVELOPMENT"),
    # CHEF
    ("chef", "CHEF"),
    ("cook", "CHEF"),
    ("sous chef", "CHEF"),
    ("pastry", "CHEF"),
    ("culinary", "CHEF"),
    ("line cook", "CHEF"),
    ("head cook", "CHEF"),
    ("food service", "CHEF"),
    ("kitchen manager", "CHEF"),
    ("executive chef", "CHEF"),
    # CONSTRUCTION
    ("construction", "CONSTRUCTION"),
    ("civil engineer", "CONSTRUCTION"),
    ("project engineer.*construct", "CONSTRUCTION"),
    ("site manager", "CONSTRUCTION"),
    ("superintendent.*construct", "CONSTRUCTION"),
    ("carpenter", "CONSTRUCTION"),
    ("electrician", "CONSTRUCTION"),
    ("plumber", "CONSTRUCTION"),
    ("estimator", "CONSTRUCTION"),
    ("foreman", "CONSTRUCTION"),
    ("hvac", "CONSTRUCTION"),
    ("welder", "CONSTRUCTION"),
    # CONSULTANT
    ("consultant", "CONSULTANT"),
    ("management consultant", "CONSULTANT"),
    ("strategy consultant", "CONSULTANT"),
    ("advisory", "CONSULTANT"),
    ("principal consultant", "CONSULTANT"),
    # DESIGNER
    ("graphic design", "DESIGNER"),
    ("ux designer", "DESIGNER"),
    ("ui designer", "DESIGNER"),
    ("product designer", "DESIGNER"),
    ("visual designer", "DESIGNER"),
    ("web designer", "DESIGNER"),
    ("interior design", "DESIGNER"),
    ("industrial designer", "DESIGNER"),
    ("motion designer", "DESIGNER"),
    ("brand designer", "DESIGNER"),
    # DIGITAL-MEDIA
    ("digital media", "DIGITAL-MEDIA"),
    ("social media", "DIGITAL-MEDIA"),
    ("content creator", "DIGITAL-MEDIA"),
    ("seo", "DIGITAL-MEDIA"),
    ("digital marketing", "DIGITAL-MEDIA"),
    ("content strategist", "DIGITAL-MEDIA"),
    ("email marketing", "DIGITAL-MEDIA"),
    ("influencer", "DIGITAL-MEDIA"),
    ("video editor", "DIGITAL-MEDIA"),
    ("content manager", "DIGITAL-MEDIA"),
    ("copywriter", "DIGITAL-MEDIA"),
    # ENGINEERING
    ("mechanical engineer", "ENGINEERING"),
    ("electrical engineer", "ENGINEERING"),
    ("industrial engineer", "ENGINEERING"),
    ("systems engineer", "ENGINEERING"),
    ("manufacturing engineer", "ENGINEERING"),
    ("chemical engineer", "ENGINEERING"),
    ("quality engineer", "ENGINEERING"),
    ("process engineer", "ENGINEERING"),
    ("structural engineer", "ENGINEERING"),
    ("field engineer", "ENGINEERING"),
    # FINANCE
    ("finance manager", "FINANCE"),
    ("financial planner", "FINANCE"),
    ("cfo", "FINANCE"),
    ("treasurer", "FINANCE"),
    ("risk analyst", "FINANCE"),
    ("investment analyst", "FINANCE"),
    ("portfolio manager", "FINANCE"),
    ("equity analyst", "FINANCE"),
    ("fund manager", "FINANCE"),
    ("actuary", "FINANCE"),
    ("compliance officer", "FINANCE"),
    # FITNESS
    ("personal trainer", "FITNESS"),
    ("fitness", "FITNESS"),
    ("gym manager", "FITNESS"),
    ("strength coach", "FITNESS"),
    ("yoga", "FITNESS"),
    ("pilates", "FITNESS"),
    ("nutritionist", "FITNESS"),
    ("wellness", "FITNESS"),
    ("physical education", "FITNESS"),
    ("sports coach", "FITNESS"),
    ("athletic trainer", "FITNESS"),
    # HEALTHCARE
    ("nurse", "HEALTHCARE"),
    ("registered nurse", "HEALTHCARE"),
    ("physician", "HEALTHCARE"),
    ("doctor", "HEALTHCARE"),
    ("medical", "HEALTHCARE"),
    ("pharmacist", "HEALTHCARE"),
    ("physical therapist", "HEALTHCARE"),
    ("occupational therapist", "HEALTHCARE"),
    ("dentist", "HEALTHCARE"),
    ("radiologist", "HEALTHCARE"),
    ("surgical", "HEALTHCARE"),
    ("clinical", "HEALTHCARE"),
    ("patient care", "HEALTHCARE"),
    ("phlebotomist", "HEALTHCARE"),
    ("healthcare administrator", "HEALTHCARE"),
    ("certified nursing assistant", "HEALTHCARE"),
    # HR
    ("human resources", "HR"),
    ("hr manager", "HR"),
    ("hr specialist", "HR"),
    ("talent acquisition", "HR"),
    ("recruiter", "HR"),
    ("hr director", "HR"),
    ("hr business partner", "HR"),
    ("people operations", "HR"),
    ("compensation.*benefits", "HR"),
    ("hris", "HR"),
    # INFORMATION-TECHNOLOGY
    ("software engineer", "INFORMATION-TECHNOLOGY"),
    ("software developer", "INFORMATION-TECHNOLOGY"),
    ("data engineer", "INFORMATION-TECHNOLOGY"),
    ("data scientist", "INFORMATION-TECHNOLOGY"),
    ("data analyst", "INFORMATION-TECHNOLOGY"),
    ("machine learning", "INFORMATION-TECHNOLOGY"),
    ("devops", "INFORMATION-TECHNOLOGY"),
    ("cloud engineer", "INFORMATION-TECHNOLOGY"),
    ("network engineer", "INFORMATION-TECHNOLOGY"),
    ("it manager", "INFORMATION-TECHNOLOGY"),
    ("systems administrator", "INFORMATION-TECHNOLOGY"),
    ("cybersecurity", "INFORMATION-TECHNOLOGY"),
    ("web developer", "INFORMATION-TECHNOLOGY"),
    ("java developer", "INFORMATION-TECHNOLOGY"),
    ("python developer", "INFORMATION-TECHNOLOGY"),
    ("full.?stack", "INFORMATION-TECHNOLOGY"),
    ("frontend developer", "INFORMATION-TECHNOLOGY"),
    ("backend developer", "INFORMATION-TECHNOLOGY"),
    ("database administrator", "INFORMATION-TECHNOLOGY"),
    ("qa engineer", "INFORMATION-TECHNOLOGY"),
    ("qa analyst", "INFORMATION-TECHNOLOGY"),
    # PUBLIC-RELATIONS
    ("public relations", "PUBLIC-RELATIONS"),
    ("pr manager", "PUBLIC-RELATIONS"),
    ("communications manager", "PUBLIC-RELATIONS"),
    ("communications specialist", "PUBLIC-RELATIONS"),
    ("media relations", "PUBLIC-RELATIONS"),
    ("brand communications", "PUBLIC-RELATIONS"),
    ("corporate communications", "PUBLIC-RELATIONS"),
    ("press secretary", "PUBLIC-RELATIONS"),
    ("spokesperson", "PUBLIC-RELATIONS"),
    # SALES
    ("sales", "SALES"),
    ("sales manager", "SALES"),
    ("sales representative", "SALES"),
    ("account executive", "SALES"),
    ("account manager", "SALES"),
    ("sales executive", "SALES"),
    ("sales associate", "SALES"),
    ("retail sales", "SALES"),
    ("inside sales", "SALES"),
    ("outside sales", "SALES"),
    ("salesperson", "SALES"),
    ("sales specialist", "SALES"),
    ("store manager", "SALES"),
    ("retail manager", "SALES"),
    # TEACHER
    ("teacher", "TEACHER"),
    ("professor", "TEACHER"),
    ("instructor", "TEACHER"),
    ("educator", "TEACHER"),
    ("tutor", "TEACHER"),
    ("principal.*school", "TEACHER"),
    ("school counselor", "TEACHER"),
    ("librarian", "TEACHER"),
    ("curriculum", "TEACHER"),
    ("adjunct", "TEACHER"),
]


GENERIC_ROLE_WORDS: Set[str] = {
    "consultant", "specialist", "associate", "manager", "analyst",
}
SALES_WORD_PATTERN = re.compile(r"\bsales\b", re.IGNORECASE)


def build_title_map() -> List[Tuple[re.Pattern[str], str, str]]:
    """Compile RAW_RULES into a list of (pattern, category, keyword) tuples.

    Returns:
        List of (compiled_pattern, category, keyword_str).
    """
    rules: List[Tuple[re.Pattern[str], str, str]] = []
    for kw, cat in RAW_RULES:
        # Wrap in word-boundary anchors
        pattern = re.compile(r"\b" + kw + r"\b", re.IGNORECASE)
        rules.append((pattern, cat, kw))
    return rules


def map_title(title: str, rules: List[Tuple[re.Pattern[str], str, str]]) -> Optional[str]:
    """Map a job title to a category, or return None if ambiguous/unmapped.

    Precedence rules:
    1. If the title contains word-boundary 'sales', it maps to SALES (precedence
       over generic role words and other co-occurring terms).
    2. When multiple keywords match and one is a generic role word (consultant,
       specialist, associate, manager, analyst), the domain keyword wins.
    3. If multiple domain categories match, the title is AMBIGUOUS (returns None).
    4. If no keywords match, the title is UNMAPPED (returns None).

    Args:
        title: Job title string.
        rules: Compiled (pattern, category, keyword) tuples.

    Returns:
        Category string, or None.
    """
    # 1. Sales precedence
    if SALES_WORD_PATTERN.search(title):
        return "SALES"

    # 2. General rule matching
    matches: List[Tuple[str, str]] = []
    for pattern, cat, kw in rules:
        if pattern.search(title):
            matches.append((cat, kw))

    if not matches:
        return None

    cats = {c for c, _ in matches}
    if len(cats) == 1:
        return next(iter(cats))

    # Precedence: domain keyword wins over generic role words / CONSULTANT
    domain_cats: Set[str] = set()
    for cat, kw in matches:
        if cat == "CONSULTANT" or kw.lower() in GENERIC_ROLE_WORDS:
            continue
        domain_cats.add(cat)

    if len(domain_cats) == 1:
        return domain_cats.pop()

    return None  # 0 domain cats (all generic) or >1 domain cats (ambiguous)


# -- Main -------------------------------------------------------------------

def main() -> None:
    """Run the JD pool build pipeline."""
    logger.info("=== build_jd_pool.py start ===")

    # Step 1: Load and dedupe linkedin_sample
    li_path = DATA_RAW / "linkedin_sample" / "postings_5k.parquet"
    logger.info(f"Loading LinkedIn from {li_path}")
    li = pd.read_parquet(li_path)
    logger.info(f"  Loaded {len(li)} rows, columns: {li.columns.tolist()}")

    # Dedupe on (title, description[:200])
    li["_dedup_key"] = li["title"].str.strip() + "|||" + li["description"].str[:200].fillna("")
    before = len(li)
    li = li.drop_duplicates(subset=["_dedup_key"], keep="first").reset_index(drop=True)
    logger.info(f"  After dedup: {len(li)} rows ({before - len(li)} duplicates removed).")

    # Assign jd_id
    li["jd_id"] = range(len(li))

    # Step 2: Build title->category rules
    logger.info("Building title->category rules …")
    rules = build_title_map()

    # Map each title
    li["category"] = li["title"].map(lambda t: map_title(str(t), rules))

    mapped = li[li["category"].notna()].copy()
    unmapped_df = li[li["category"].isna()].copy()

    # Count ambiguous vs truly unmapped
    def classify_title(title: str) -> str:
        if map_title(str(title), rules) is not None:
            return "mapped"
        cats: Set[str] = set()
        for pattern, cat, _ in rules:
            if pattern.search(str(title)):
                cats.add(cat)
        if len(cats) > 1:
            return "ambiguous"
        return "unmapped"

    unmapped_df["_status"] = unmapped_df["title"].map(classify_title)
    ambiguous_df = unmapped_df[unmapped_df["_status"] == "ambiguous"]
    truly_unmapped = unmapped_df[unmapped_df["_status"] == "unmapped"]

    logger.info(f"  Mapped:    {len(mapped)}")
    logger.info(f"  Ambiguous: {len(ambiguous_df)}")
    logger.info(f"  Unmapped:  {len(truly_unmapped)}")

    # Per-category counts
    logger.info("Mapped count per category:")
    cat_counts = mapped["category"].value_counts().sort_index()
    for cat, cnt in cat_counts.items():
        logger.info(f"  {cat:<28} {cnt}")

    # 40 most frequent unmapped titles
    logger.info("40 most frequent UNMAPPED titles:")
    unmapped_counts = truly_unmapped["title"].value_counts().head(40)
    for title, cnt in unmapped_counts.items():
        logger.info(f"  [{cnt:3d}] {title}")

    # 60 random mapped (title -> category) pairs (random_state=1)
    logger.info("60 random mapped (title -> category) pairs:")
    sample = mapped[["title", "category"]].sample(
        n=min(60, len(mapped)), random_state=1
    )
    for _, row in sample.iterrows():
        logger.info(f"  {row['title']} -> {row['category']}")

    # Write title_category_map.csv
    map_rows = []
    seen_kw: Set[str] = set()
    for kw, cat in RAW_RULES:
        if kw not in seen_kw:
            map_rows.append({"keyword": kw, "category": cat})
            seen_kw.add(kw)
    map_df = pd.DataFrame(map_rows)
    map_path = DATA_TAXONOMIES / "title_category_map.csv"
    map_df.to_csv(map_path, index=False)
    logger.info(f"Wrote {map_path} ({len(map_df)} rules)")

    # Step 4: Split mapped JDs at (title, description[:200]) group level
    mapped = mapped.copy()
    mapped["_grp_key"] = (
        mapped["title"].str.strip() + "|||" + mapped["description"].str[:200].fillna("")
    )
    # Assign group IDs
    grp_map = {k: i for i, k in enumerate(mapped["_grp_key"].unique())}
    mapped["_grp_id"] = mapped["_grp_key"].map(grp_map)

    # Majority category per group (trivial here since deduped, but kept for correctness)
    grp_label = (
        mapped.groupby("_grp_id")["category"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    # Convert to plain numpy arrays (PyArrow-backed pandas series cause
    # TypeError in sklearn's _safe_indexing)
    groups_arr = np.array(grp_label["_grp_id"].tolist())
    labels_arr = np.array(grp_label["category"].tolist())

    # Flag small categories BEFORE splitting
    small_cats = cat_counts[cat_counts < SMALL_CATEGORY_THRESHOLD].index.tolist()
    if small_cats:
        logger.warning(f"Categories too small for matcher (<{SMALL_CATEGORY_THRESHOLD} JDs): {small_cats}")

    val_test_frac = 0.30
    try:
        train_groups, valtest_groups, _, valtest_labels = train_test_split(
            groups_arr, labels_arr,
            test_size=val_test_frac, stratify=labels_arr, random_state=SEED,
        )
    except ValueError:
        logger.warning("JD stratification failed; falling back to no stratify.")
        train_groups, valtest_groups, _, valtest_labels = train_test_split(
            groups_arr, labels_arr, test_size=val_test_frac, random_state=SEED,
        )

    try:
        val_groups, test_groups = train_test_split(
            valtest_groups, test_size=0.5, stratify=valtest_labels, random_state=SEED,
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            valtest_groups, test_size=0.5, random_state=SEED,
        )


    train_set: Set[int] = set(int(x) for x in train_groups)
    val_set: Set[int] = set(int(x) for x in val_groups)

    def assign(gid: int) -> str:
        if gid in train_set:
            return "train"
        if gid in val_set:
            return "val"
        return "test"

    mapped["split"] = mapped["_grp_id"].map(assign)

    logger.info("JD split counts:")
    for sp, cnt in mapped.groupby("split").size().items():
        logger.info(f"  {sp}: {cnt}")

    logger.info("JD per-category per-split:")
    cat_split = mapped.groupby(["category", "split"]).size().unstack(fill_value=0)
    logger.info("\n" + cat_split.to_string())

    # Write jd_split.csv (no text columns)
    jd_split_df = mapped[["jd_id", "category", "split", "title"]].copy()
    jd_split_path = DATA_SPLITS / "jd_split.csv"
    jd_split_df.to_csv(jd_split_path, index=False)
    logger.info(f"Wrote {jd_split_path} ({len(jd_split_df)} rows)")

    # Write jd_{train,val,test}.parquet
    for sp in ("train", "val", "test"):
        subset = mapped[mapped["split"] == sp][
            ["jd_id", "title", "description", "category"]
        ].rename(columns={"description": "text"}).copy()
        out_path = DATA_PROCESSED / f"jd_{sp}.parquet"
        subset.to_parquet(out_path, index=False)
        logger.info(f"Wrote {out_path} ({len(subset)} rows)")

    # Step 6: jrf_eval.parquet
    logger.info("Building jrf_eval.parquet …")
    jrf_path = DATA_RAW / "job_resume_fit" / "train.parquet"
    jrf = pd.read_parquet(jrf_path)
    logger.info(f"  JRF loaded: {len(jrf)} rows, columns: {jrf.columns.tolist()}")

    # Load classifier test split IDs
    classifier_split_path = DATA_SPLITS / "classifier_split.csv"
    if not classifier_split_path.exists():
        logger.error(f"classifier_split.csv not found at {classifier_split_path}. Run make_splits.py first.")
        return

    cls_split = pd.read_csv(classifier_split_path)
    test_ids: Set[int] = set(cls_split.loc[cls_split["split"] == "test", "id"].tolist())
    logger.info(f"  Classifier test IDs: {len(test_ids)}")

    jrf_eval = jrf[jrf["ID"].isin(test_ids)][
        ["ID", "category", "job_text", "ai_match_score"]
    ].rename(columns={"ID": "resume_id"}).copy()

    out_path = DATA_PROCESSED / "jrf_eval.parquet"
    jrf_eval.to_parquet(out_path, index=False)
    logger.info(f"Wrote {out_path} ({len(jrf_eval)} rows)")

    logger.info("jrf_eval per-category counts:")
    for cat, cnt in jrf_eval["category"].value_counts().items():
        logger.info(f"  {cat}: {cnt}")

    logger.info("=== build_jd_pool.py done ===")


if __name__ == "__main__":
    main()
