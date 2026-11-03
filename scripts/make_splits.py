"""
scripts/make_splits.py
======================
Build the shared resume split (classifier / matcher / NER gold).

Steps
-----
1. Load LiveCareer; drop rows with Resume_str shorter than 200 chars.
2. Build groups via union-find over exact-duplicate Resume_str AND
   TF-IDF cosine >= 0.90 pairs.
3. Split at GROUP level, stratified by category, 70 / 15 / 15
   (train / val / test), seed 42.
4. Write data/splits/classifier_split.csv (id, category, split, group_id).
5. Write data/processed/classifier_{train,val,test}.parquet
   (id, category, text, text_masked, group_id).
6. Mask the leading title header; measure label-leak before/after.
7. Select NER gold candidates (36 resumes from test split);
   write data/splits/ner_gold_ids.csv (id column only).

Privacy: never prints resume body text; only prints job titles and tag names.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_CHARS = 200
TFIDF_THRESHOLD = 0.90
TFIDF_MAX_FEATURES = 10_000

# 24 LiveCareer categories
CATEGORIES: List[str] = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]


# -- Union-Find -------------------------------------------------------------

class UnionFind:
    """Weighted union-find with path compression.

    Args:
        n: Number of elements.
    """

    def __init__(self, n: int) -> None:
        """Initialise with n elements."""
        self._parent: List[int] = list(range(n))
        self._rank: List[int] = [0] * n

    def find(self, x: int) -> int:
        """Find root with path halving."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        """Return mapping root -> list of member indices."""
        g: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            g[self.find(i)].append(i)
        return dict(g)


# -- HTML header extraction -------------------------------------------------

def extract_header_from_html(html: str) -> Optional[str]:
    """Extract the job-title header from a LiveCareer resume HTML string.

    LiveCareer stores the job title inside
    ``<div class="paragraph PARAGRAPH_NAME firstparagraph">``.
    Fallback: ``<div class="name">``.

    Returns:
        Stripped job-title text, or None if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("div", class_=re.compile(r"PARAGRAPH_NAME"))
    if tag:
        return tag.get_text(separator=" ", strip=True)
    tag = soup.find("div", class_="name")
    if tag:
        return tag.get_text(separator=" ", strip=True)
    return None


def mask_header(text: str, header: Optional[str]) -> str:
    """Remove the leading job-title header from Resume_str.

    Normalises whitespace in both header and text before comparing,
    then strips the header prefix from the normalised text.

    Args:
        text: Original resume text.
        header: Job-title string to remove.

    Returns:
        Resume text with header removed, stripped.
    """
    if not header:
        return text
    h_norm = " ".join(header.split())
    text_norm = " ".join(text.split())
    if text_norm.lower().startswith(h_norm.lower()):
        remaining = text_norm[len(h_norm):].lstrip()
        return remaining
    pattern = re.compile(r"^\s*" + re.escape(h_norm) + r"\s*", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


# -- Leak measurement -------------------------------------------------------

def _first_n_words(text: str, n: int = 10) -> str:
    """Return first n words joined by spaces."""
    return " ".join(str(text).split()[:n])


def measure_leak(df: pd.DataFrame, text_col: str) -> Dict[str, float]:
    """Measure per-category label-leak using word-boundary check on first 10 words.

    A resume is a 'leak' if its category label appears (word-boundary match)
    in the first 10 words of the resume text.

    Args:
        df: DataFrame with columns 'category' and text_col.
        text_col: Column name to check.

    Returns:
        Dict mapping category (plus 'overall') to leak fraction.
    """
    results: Dict[str, float] = {}
    all_flags: List[bool] = []

    for cat, grp in df.groupby("category"):
        flat_cat = cat.replace("-", " ")
        patterns = [
            re.compile(r"\b" + re.escape(cat) + r"\b", re.IGNORECASE),
            re.compile(r"\b" + re.escape(flat_cat) + r"\b", re.IGNORECASE),
        ]
        flags: List[bool] = []
        for text in grp[text_col]:
            snippet = _first_n_words(text, 10)
            flags.append(any(p.search(snippet) for p in patterns))
        results[str(cat)] = sum(flags) / max(len(flags), 1)
        all_flags.extend(flags)

    results["overall"] = sum(all_flags) / max(len(all_flags), 1)
    return results


def report_leak(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    label: str = "",
) -> None:
    """Print a before/after masking leak report.

    Args:
        df: Full DataFrame.
        before_col: Text column before masking.
        after_col: Text column after masking.
        label: Optional label for the report header.
    """
    before = measure_leak(df, before_col)
    after = measure_leak(df, after_col)

    logger.info("=" * 60)
    logger.info(f"LABEL-LEAK REPORT {label}")
    logger.info(f"  Overall before masking: {before['overall']:.1%}")
    logger.info(f"  Overall after  masking: {after['overall']:.1%}")

    cat_before = {k: v for k, v in before.items() if k != "overall"}
    worst5 = sorted(cat_before.items(), key=lambda x: -x[1])[:5]
    logger.info("  5 worst categories (before -> after):")
    for cat, lb in worst5:
        la = after.get(cat, 0.0)
        logger.info(f"    {cat:<28} {lb:.1%} -> {la:.1%}")

    chars_removed = df[before_col].str.len() - df[after_col].str.len()
    logger.info(f"  Median chars removed: {chars_removed.median():.0f}")
    logger.info("=" * 60)

    if after["overall"] >= 0.10:
        logger.warning(
            f"Residual leak after masking: {after['overall']:.1%} (target <10%). "
            "Reported as-is; not forced further."
        )


# -- Group building ---------------------------------------------------------

def build_groups(df: pd.DataFrame) -> pd.Series:
    """Assign a group_id to each row using union-find.

    Groups are formed by:
      1. Exact duplicate Resume_str.
      2. TF-IDF cosine >= TFIDF_THRESHOLD pairs.

    Args:
        df: DataFrame with columns 'text' and 'category'.

    Returns:
        pd.Series of integer group IDs aligned with df.index.
    """
    n = len(df)
    uf = UnionFind(n)
    texts = df["text"].tolist()

    # Pass 1: exact duplicates
    logger.info("Groups: exact-duplicate pass …")
    exact_map: Dict[str, int] = {}
    for i, t in enumerate(texts):
        if t in exact_map:
            uf.union(exact_map[t], i)
        else:
            exact_map[t] = i

    # Pass 2: TF-IDF cosine
    logger.info("Groups: TF-IDF cosine pass …")
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)  # sparse (n, vocab)
    norms = np.asarray(X.power(2).sum(axis=1)).ravel() ** 0.5
    norms[norms == 0] = 1.0

    BLOCK = 200
    pair_count = 0
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        block = X[start:end]
        dots = (block @ X.T).toarray()  # (block_size, n)
        cos = dots / (norms[start:end, None] * norms[None, :])
        for i_local, i_global in enumerate(range(start, end)):
            row = cos[i_local, i_global + 1 :]
            hits = np.where(row >= TFIDF_THRESHOLD)[0]
            for h in hits:
                uf.union(i_global, i_global + 1 + int(h))
                pair_count += 1

    logger.info(f"  TF-IDF pairs merged: {pair_count}")

    # Compact group IDs
    roots = [uf.find(i) for i in range(n)]
    root_to_gid: Dict[int, int] = {}
    gid_counter = 0
    gids: List[int] = []
    for r in roots:
        if r not in root_to_gid:
            root_to_gid[r] = gid_counter
            gid_counter += 1
        gids.append(root_to_gid[r])

    series = pd.Series(gids, index=df.index, name="group_id")

    # Report mixed-category groups
    tmp = df[["category"]].copy()
    tmp["group_id"] = series
    mixed = tmp.groupby("group_id")["category"].nunique()
    mixed = mixed[mixed > 1]
    if len(mixed):
        logger.warning(f"Mixed-category groups: {len(mixed)}")
        for gid in mixed.index:
            cats = tmp.loc[tmp["group_id"] == gid, "category"].unique().tolist()
            logger.warning(f"  group_id={gid}: {cats}")
    else:
        logger.info("No mixed-category groups found.")

    logger.info(f"Total groups: {series.nunique()} from {n} resumes.")
    return series


# -- Splitting --------------------------------------------------------------

def group_stratified_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    label_col: str = "category",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> pd.Series:
    """Split at group level, stratified by majority label per group.

    Args:
        df: DataFrame with group_col and label_col columns.
        group_col: Column holding group IDs.
        label_col: Column holding category labels.
        val_size: Fraction for validation split.
        test_size: Fraction for test split.
        seed: Random state seed.

    Returns:
        pd.Series (same index as df) with values 'train'/'val'/'test'.
    """
    # Majority label per group
    group_label = (
        df.groupby(group_col)[label_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    # Convert to plain numpy arrays (PyArrow-backed pandas series cause
    # TypeError in sklearn's _safe_indexing)
    groups_arr = np.array(group_label[group_col].tolist())
    labels_arr = np.array(group_label[label_col].tolist())

    val_test_frac = val_size + test_size

    try:
        train_groups, valtest_groups, _, valtest_labels = train_test_split(
            groups_arr, labels_arr,
            test_size=val_test_frac, stratify=labels_arr, random_state=seed,
        )
    except ValueError:
        logger.warning("Stratification failed; falling back to unstratified split.")
        train_groups, valtest_groups, _, valtest_labels = train_test_split(
            groups_arr, labels_arr, test_size=val_test_frac, random_state=seed,
        )

    relative_test = test_size / val_test_frac
    try:
        val_groups, test_groups = train_test_split(
            valtest_groups, test_size=relative_test,
            stratify=valtest_labels, random_state=seed,
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            valtest_groups, test_size=relative_test, random_state=seed,
        )

    train_set: Set[int] = set(int(x) for x in train_groups)
    val_set: Set[int] = set(int(x) for x in val_groups)

    def assign(gid: int) -> str:
        if gid in train_set:
            return "train"
        if gid in val_set:
            return "val"
        return "test"

    return df[group_col].map(assign)


# -- NER gold selection -----------------------------------------------------

def select_ner_gold(
    df_test: pd.DataFrame,
    n_total: int = 36,
    min_chars: int = 1500,
    seed: int = SEED,
) -> List[int]:
    """Pick NER gold candidate IDs from the test split.

    Strategy: one per category (up to 24), then fill remaining slots
    from the largest categories.

    Args:
        df_test: Test split DataFrame with columns 'id', 'category', 'text'.
        n_total: Total number of IDs to select.
        min_chars: Minimum text length for eligibility.
        seed: Random seed.

    Returns:
        Sorted list of selected LiveCareer integer IDs.
    """
    rng = np.random.default_rng(seed)
    eligible = df_test[df_test["text"].str.len() >= min_chars].copy()
    eligible = eligible.sort_values("id")

    selected: Set[int] = set()

    # One per category
    for cat in sorted(eligible["category"].unique()):
        pool = [
            int(x) for x in eligible.loc[eligible["category"] == cat, "id"].values
            if int(x) not in selected
        ]
        if pool:
            selected.add(int(rng.choice(pool)))

    # Fill remaining slots cycling through largest categories
    remaining = n_total - len(selected)
    if remaining > 0:
        cat_sizes = (
            eligible[~eligible["id"].isin(selected)]
            .groupby("category").size()
            .sort_values(ascending=False)
        )
        cat_cycle = list(cat_sizes.index)
        i = 0
        attempts = 0
        while remaining > 0 and attempts < len(cat_cycle) * 10:
            cat = cat_cycle[i % len(cat_cycle)]
            pool = [
                int(x) for x in eligible.loc[
                    (eligible["category"] == cat) & (~eligible["id"].isin(selected)),
                    "id"
                ].values
            ]
            if pool:
                selected.add(int(rng.choice(pool)))
                remaining -= 1
            i += 1
            attempts += 1

    result = sorted(selected)
    logger.info(f"NER gold: {len(result)} IDs selected from test split.")
    return result


# -- Main -------------------------------------------------------------------

def main() -> None:
    """Run the full split pipeline."""
    logger.info("=== make_splits.py start ===")

    # Step 1: Load and filter
    lc_path = DATA_RAW / "livecareer" / "train.parquet"
    logger.info(f"Loading LiveCareer from {lc_path}")
    lc = pd.read_parquet(lc_path)
    logger.info(f"  Loaded {len(lc)} rows, columns: {lc.columns.tolist()}")

    before = len(lc)
    lc = lc[lc["Resume_str"].str.len() >= MIN_CHARS].copy()
    logger.info(f"  After MIN_CHARS={MIN_CHARS} filter: {len(lc)} rows ({before - len(lc)} dropped).")

    lc = lc.rename(columns={"ID": "id", "Category": "category", "Resume_str": "text"})

    # Step 2: HTML structure inspection (3 resumes, tags + class names only)
    logger.info("=== HTML structure (3 resumes, tags+classes only) ===")
    for _, row in lc.head(3).iterrows():
        soup = BeautifulSoup(row["Resume_html"], "html.parser")
        tags: Set[str] = set()
        for tag in soup.find_all():
            cls = " ".join(tag.get("class", []))
            tags.add(f"<{tag.name} class=\"{cls}\">")
        logger.info(f"  ID {row['id']}: {sorted(tags)[:15]}")

    # Step 3: Extract headers
    logger.info("Extracting headers from HTML …")
    lc["header"] = lc["Resume_html"].apply(extract_header_from_html)
    logger.info(f"  Headers found: {lc['header'].notna().sum()}/{len(lc)}")

    logger.info("=== First 6 words of first 10 headers (job titles only) ===")
    shown = 0
    for _, row in lc.iterrows():
        if pd.notna(row["header"]) and shown < 10:
            words = str(row["header"]).split()[:6]
            logger.info(f"  ID {row['id']}: {' '.join(words)}")
            shown += 1

    # Step 4: Mask headers
    logger.info("Applying header mask …")
    lc["text_masked"] = lc.apply(
        lambda r: mask_header(
            r["text"],
            r["header"] if pd.notna(r["header"]) else None,
        ),
        axis=1,
    )

    # Step 5: Build groups
    lc = lc.reset_index(drop=True)
    lc["group_id"] = build_groups(lc)

    # Step 6: Split
    logger.info("Splitting at group level (stratified by category) …")
    lc["split"] = group_stratified_split(lc)

    logger.info("Split counts:")
    for sp, cnt in lc.groupby("split").size().items():
        logger.info(f"  {sp}: {cnt}")

    logger.info("Per-category per-split counts:")
    cat_split = lc.groupby(["category", "split"]).size().unstack(fill_value=0)
    logger.info("\n" + cat_split.to_string())

    # Step 7: Leak measurement
    report_leak(lc, "text", "text_masked", label="(all data)")

    # Step 8a: classifier_split.csv (no text columns)
    split_csv = lc[["id", "category", "split", "group_id"]].copy()
    out = DATA_SPLITS / "classifier_split.csv"
    split_csv.to_csv(out, index=False)
    logger.info(f"Wrote {out} ({len(split_csv)} rows)")

    # Step 8b: classifier_{train,val,test}.parquet
    for sp in ("train", "val", "test"):
        subset = lc[lc["split"] == sp][["id", "category", "text", "text_masked", "group_id"]].copy()
        out = DATA_PROCESSED / f"classifier_{sp}.parquet"
        subset.to_parquet(out, index=False)
        logger.info(f"Wrote {out} ({len(subset)} rows)")

    # Step 9: NER gold
    df_test = lc[lc["split"] == "test"].copy()
    ner_ids = select_ner_gold(df_test)
    out = DATA_SPLITS / "ner_gold_ids.csv"
    pd.DataFrame({"id": ner_ids}).to_csv(out, index=False)
    logger.info(f"Wrote {out} ({len(ner_ids)} ids)")

    logger.info("=== make_splits.py done ===")


if __name__ == "__main__":
    main()
