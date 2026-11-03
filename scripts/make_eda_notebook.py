"""Generate and execute the exploratory data analysis (EDA) notebook.

Constructs notebooks/01_eda.ipynb programmatically using nbformat,
executes it with nbclient using the virtual environment kernel,
and saves the executed notebook with all outputs and figures in artifacts/reports/.

Privacy: outputs only counts, tables, and aggregate metrics; never prints
resume or JD body text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_eda.ipynb"
REPORTS_DIR = ROOT / "artifacts" / "reports"


def create_eda_notebook() -> nbf.NotebookNode:
    """Build the EDA notebook cells."""
    nb = nbf.v4.new_notebook()

    # Cell 1: Title
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# 01 — Exploratory Data Analysis: Resumes, JDs, and Taxonomies

**Author:** Deep Learning Engineering Team  
**Date:** November 2026  
**Purpose:** Comprehensive exploratory audit of the LiveCareer resume dataset, LinkedIn JD pool, dataset splits, label leak measurements, and taxonomy coverage.

> **Privacy Notice:** In accordance with repository privacy standards, this notebook presents **only** statistical summaries, counts, and aggregate visualizations. No raw resume text, candidate contact information, or body snippets are rendered.
"""
        )
    )

    # Cell 2: Setup and Imports
    nb.cells.append(
        nbf.v4.new_code_cell(
            """import os
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths configuration
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "artifacts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Environment configured.")
print("Root directory:", ROOT)
print("Reports directory:", REPORTS_DIR)
"""
        )
    )

    # Cell 3: Class Balance Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 1. Class Balance Per Split (LiveCareer Resumes)

The LiveCareer dataset is split into **train (70%)**, **validation (15%)**, and **test (15%)** partitions using a group-aware stratified split to guarantee that near-duplicate and duplicate resumes remain in the same partition.
"""
        )
    )

    # Cell 4: Class Balance Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """# Load classifier splits
df_train = pd.read_parquet(PROCESSED_DIR / "classifier_train.parquet")
df_val = pd.read_parquet(PROCESSED_DIR / "classifier_val.parquet")
df_test = pd.read_parquet(PROCESSED_DIR / "classifier_test.parquet")

df_train["split"] = "train"
df_val["split"] = "val"
df_test["split"] = "test"

all_resumes = pd.concat([df_train, df_val, df_test], ignore_index=True)

# Compute split counts per category
split_counts = all_resumes.groupby(["category", "split"]).size().unstack(fill_value=0)
split_counts = split_counts[["train", "val", "test"]]
split_counts["total"] = split_counts.sum(axis=1)
split_counts = split_counts.sort_values(by="total", ascending=False)

print("Resume split distribution per category:")
print(split_counts.to_string())

# Plot class balance
fig, ax = plt.subplots(figsize=(10, 8))
split_counts[["train", "val", "test"]].plot(
    kind="barh",
    stacked=True,
    color=["#2b5c8f", "#4f8bc9", "#8bbbe8"],
    ax=ax,
    edgecolor="black",
    linewidth=0.5
)
ax.set_title("LiveCareer Resume Category Distribution Across Splits", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Resumes", fontsize=11)
ax.set_ylabel("Category", fontsize=11)
ax.legend(["Train (70%)", "Validation (15%)", "Test (15%)"], loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

balance_fig_path = REPORTS_DIR / "eda_class_balance.png"
fig.savefig(balance_fig_path, dpi=300)
plt.close(fig)
print(f"Saved figure: {balance_fig_path}")
"""
        )
    )

    # Cell 5: Resume Lengths Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 2. Resume Length Distribution (`text` vs `text_masked`)

LiveCareer resumes begin with a prominent job-title header (from HTML `div.PARAGRAPH_NAME`). To eliminate trivial label leakage during classifier training, the header is masked. Here we evaluate the character length distributions before and after masking.
"""
        )
    )

    # Cell 6: Resume Lengths Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """train_chars_raw = df_train["text"].str.len()
train_chars_masked = df_train["text_masked"].str.len()
chars_removed = train_chars_raw - train_chars_masked

train_words_raw = df_train["text"].str.split().str.len()
train_words_masked = df_train["text_masked"].str.split().str.len()

stats_table = pd.DataFrame({
    "Raw Text (Chars)": train_chars_raw.describe(),
    "Masked Text (Chars)": train_chars_masked.describe(),
    "Chars Removed": chars_removed.describe(),
    "Raw Text (Words)": train_words_raw.describe(),
    "Masked Text (Words)": train_words_masked.describe(),
}).round(1)

print("Resume Length Distribution Summary (Train Split):")
print(stats_table.to_string())

# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(train_chars_raw, bins=40, color="#2b5c8f", alpha=0.7, label="Raw Text", edgecolor="black")
ax1.hist(train_chars_masked, bins=40, color="#d95f02", alpha=0.6, label="Masked Text", edgecolor="black")
ax1.set_title("Character Count Distribution", fontsize=12, fontweight="bold")
ax1.set_xlabel("Characters", fontsize=10)
ax1.set_ylabel("Resume Count", fontsize=10)
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.5)

ax2.hist(chars_removed, bins=40, color="#7570b3", edgecolor="black")
ax2.set_title("Characters Removed by Title Masking", fontsize=12, fontweight="bold")
ax2.set_xlabel("Number of Characters Removed", fontsize=10)
ax2.set_ylabel("Resume Count", fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
lengths_fig_path = REPORTS_DIR / "eda_resume_lengths.png"
fig.savefig(lengths_fig_path, dpi=300)
plt.close(fig)
print(f"Saved figure: {lengths_fig_path}")
"""
        )
    )

    # Cell 7: Null Counts Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 3. Null and Missing Value Counts

Integrity check verifying absence of null values across all processed splits.
"""
        )
    )

    # Cell 8: Null Counts Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_jd_train = pd.read_parquet(PROCESSED_DIR / "jd_train.parquet")
df_jd_val = pd.read_parquet(PROCESSED_DIR / "jd_val.parquet")
df_jd_test = pd.read_parquet(PROCESSED_DIR / "jd_test.parquet")
df_jrf_eval = pd.read_parquet(PROCESSED_DIR / "jrf_eval.parquet")

datasets = {
    "classifier_train": df_train,
    "classifier_val": df_val,
    "classifier_test": df_test,
    "jd_train": df_jd_train,
    "jd_val": df_jd_val,
    "jd_test": df_jd_test,
    "jrf_eval": df_jrf_eval,
}

null_summary = []
for name, df in datasets.items():
    nulls = df.isnull().sum().to_dict()
    for col, count in nulls.items():
        null_summary.append({
            "Dataset": name,
            "Total Rows": len(df),
            "Column": col,
            "Null Count": count,
            "Null %": round((count / len(df)) * 100, 2) if len(df) else 0.0
        })

null_df = pd.DataFrame(null_summary)
print("Missing value audit across all processed tables:")
print(null_df.to_string(index=False))
"""
        )
    )

    # Cell 9: Header Leak Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 4. Header Leak Measurement (Before vs After Masking)

Label leakage occurs when the target category string appears explicitly in the initial words of the resume text. We measure label leakage before and after title masking using word-boundary matching over the first 10 words.
"""
        )
    )

    # Cell 10: Header Leak Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """def measure_leak(df: pd.DataFrame, text_col: str) -> dict:
    results = {}
    all_flags = []
    for cat, grp in df.groupby("category"):
        flat_cat = cat.replace("-", " ")
        patterns = [
            re.compile(r"\\b" + re.escape(cat) + r"\\b", re.IGNORECASE),
            re.compile(r"\\b" + re.escape(flat_cat) + r"\\b", re.IGNORECASE),
        ]
        flags = []
        for text in grp[text_col]:
            first_10 = " ".join(str(text).split()[:10])
            flags.append(any(p.search(first_10) for p in patterns))
        results[cat] = sum(flags) / max(len(flags), 1)
        all_flags.extend(flags)
    results["OVERALL"] = sum(all_flags) / max(len(all_flags), 1)
    return results

leak_raw = measure_leak(all_resumes, "text")
leak_masked = measure_leak(all_resumes, "text_masked")

leak_df = pd.DataFrame({
    "Category": list(leak_raw.keys()),
    "Raw Leak Rate (%)": [round(v * 100, 1) for v in leak_raw.values()],
    "Masked Leak Rate (%)": [round(leak_masked[k] * 100, 1) for k in leak_raw.keys()],
})
leak_df["Absolute Reduction (%)"] = leak_df["Raw Leak Rate (%)"] - leak_df["Masked Leak Rate (%)"]
leak_df = leak_df.sort_values(by="Raw Leak Rate (%)", ascending=False)

print("Label leak rates before and after masking (First 10 words):")
print(leak_df.to_string(index=False))

# Plot leak comparison
plot_cats = leak_df[leak_df["Category"] != "OVERALL"].copy()
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(plot_cats))
height = 0.35

ax.barh(y - height/2, plot_cats["Raw Leak Rate (%)"], height, label="Before Masking (Raw)", color="#d95f02")
ax.barh(y + height/2, plot_cats["Masked Leak Rate (%)"], height, label="After Masking (Masked)", color="#1b9e77")

ax.set_yticks(y)
ax.set_yticklabels(plot_cats["Category"])
ax.set_xlabel("Leak Rate (%)", fontsize=11)
ax.set_title("Label Leakage in First 10 Words (Before vs After Masking)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

leak_fig_path = REPORTS_DIR / "eda_header_leak.png"
fig.savefig(leak_fig_path, dpi=300)
plt.close(fig)
print(f"Saved figure: {leak_fig_path}")
"""
        )
    )

    # Cell 11: JD Pool Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 5. Job Description Pool Per Category Per Split

LinkedIn job postings sampled and mapped to LiveCareer categories via title rules (`data/taxonomies/title_category_map.csv`).
Only categories with **at least 20 JDs** qualify for downstream matcher training and fine-grained evaluation (marked below).
"""
        )
    )

    # Cell 12: JD Pool Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """jd_split_df = pd.read_csv(SPLITS_DIR / "jd_split.csv")
mapped_jds = jd_split_df[jd_split_df["category"] != "UNMAPPED"].copy()

jd_cat_split = mapped_jds.groupby(["category", "split"]).size().unstack(fill_value=0)
for col in ["train", "val", "test"]:
    if col not in jd_cat_split:
        jd_cat_split[col] = 0
jd_cat_split = jd_cat_split[["train", "val", "test"]]
jd_cat_split["Total JDs"] = jd_cat_split.sum(axis=1)
jd_cat_split["Meets >= 20 Threshold"] = jd_cat_split["Total JDs"] >= 20
jd_cat_split = jd_cat_split.sort_values(by="Total JDs", ascending=False)

print("Job Description Pool per Category:")
print(jd_cat_split.to_string())

qual_count = jd_cat_split["Meets >= 20 Threshold"].sum()
print(f"\\nCategories meeting >= 20 threshold: {qual_count} / {len(jd_cat_split)}")

# Plot JD pool
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#2b5c8f" if meets else "#c4c4c4" for meets in jd_cat_split["Meets >= 20 Threshold"]]
bars = ax.barh(jd_cat_split.index, jd_cat_split["Total JDs"], color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(20, color="#d95f02", linestyle="--", linewidth=1.5, label="Qualification Threshold (N=20)")
ax.set_title("LinkedIn Job Description Pool by Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Total JDs Available", fontsize=11)
ax.set_ylabel("Category", fontsize=11)
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

jd_fig_path = REPORTS_DIR / "eda_jd_pool.png"
fig.savefig(jd_fig_path, dpi=300)
plt.close(fig)
print(f"Saved figure: {jd_fig_path}")
"""
        )
    )

    # Cell 13: Manifest Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 6. Dataset Manifest and Licensing Audit

Summary of external datasets, provenance, and license governance as documented in `docs/data_manifest.json`.
"""
        )
    )

    # Cell 14: Manifest Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(DOCS_DIR / "data_manifest.json") as f:
    manifest = json.load(f)

records = []
for key, item in manifest.items():
    total_rows = sum(file_info.get("rows", 0) for file_info in item.get("files", []))
    records.append({
        "Key": key,
        "Name": item.get("name"),
        "License": item.get("license"),
        "Total Rows": total_rows,
        "Source URL": item.get("source_url")
    })

manifest_df = pd.DataFrame(records)
print("Data Manifest Governance Overview:")
print(manifest_df.to_string(index=False))
"""
        )
    )

    # Cell 15: PII Header
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 7. Aggregate PII Presence in Resumes

Audit of personal identifiable information patterns (email addresses and phone numbers) across the LiveCareer dataset.
To maintain absolute privacy, **no individual PII instances are output**. Only aggregate counts and detection rates are reported.
"""
        )
    )

    # Cell 16: PII Code
    nb.cells.append(
        nbf.v4.new_code_cell(
            """EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(\\+?\\d{1,3}[-.\\s]?)?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}")

n_total = len(all_resumes)
email_hits = sum(bool(EMAIL_REGEX.search(t)) for t in all_resumes["text"])
phone_hits = sum(bool(PHONE_REGEX.search(t)) for t in all_resumes["text"])
both_hits = sum(bool(EMAIL_REGEX.search(t) and PHONE_REGEX.search(t)) for t in all_resumes["text"])

pii_summary = pd.DataFrame([
    {
        "PII Pattern": "Email Address",
        "Matching Resumes": email_hits,
        "Corpus Rate (%)": round((email_hits / n_total) * 100, 2),
        "Privacy Safeguard": "Scrubbed in fixtures; strictly synthetic @example.com in tests"
    },
    {
        "PII Pattern": "Phone Number",
        "Matching Resumes": phone_hits,
        "Corpus Rate (%)": round((phone_hits / n_total) * 100, 2),
        "Privacy Safeguard": "Scrubbed in fixtures; strictly synthetic 555-01xx in tests"
    },
    {
        "PII Pattern": "Both Email & Phone",
        "Matching Resumes": both_hits,
        "Corpus Rate (%)": round((both_hits / n_total) * 100, 2),
        "Privacy Safeguard": "No real contact details persisted in test or artifact suites"
    }
])

print("LiveCareer PII Audit Summary (N = 2,484 resumes):")
print(pii_summary.to_string(index=False))
"""
        )
    )

    # Cell 17: NER Synthetic & Decision Notes
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """---
## 8. NER Architectural Decisions: `ner_synthetic` vs `ner_real_skills`

### Decision 10: Rejection of `ner_real_skills`
- **Dataset:** `Annotated_NER_PDF_Resumes` (`ner_real_skills`, 5,029 resumes).
- **Audit Findings:** The annotations were derived from a mechanical, dictionary-lookup process rather than contextual human tagging. It produced **12,695 distinct skill strings**, of which **only 1.5%** mapped to recognized O*NET software skills. Spans contained severe noise, uncurated company names, single letters, and sentence fragments.
- **Action:** Formally **REJECTED** as ground truth. Preserved on disk in `data/raw/` for transparency but excluded from model training, validation, and testing.

### Decision 11: Adoption of `ner_synthetic`
- **Dataset:** `AuraATS-Synthetic-Resume-NER-12K-v2` (`ner_synthetic`, 12,000 resumes).
- **Audit Findings:** Clean, template-disjoint train/validation/test partitions featuring **301 distinct canonical skill strings**. Annotation quality is consistent and high-precision across technical and non-technical sections.
- **Evaluation Protocol:** Models trained on `ner_synthetic` are evaluated honestly on a hand-corrected gold benchmark of real LiveCareer resumes (`data/splits/ner_gold_ids.csv`), ensuring zero synthetic evaluation bias.
"""
        )
    )

    return nb


def run_and_save_notebook() -> None:
    """Build, execute, and write 01_eda.ipynb."""
    logger.info("Building EDA notebook cells...")
    nb = create_eda_notebook()

    logger.info("Executing notebook with nbclient (kernel: dl_project_venv)...")
    client = NotebookClient(nb, timeout=600, kernel_name="dl_project_venv")
    client.execute()

    logger.info(f"Writing executed notebook to {NOTEBOOK_PATH}...")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    logger.info("EDA notebook generated and executed successfully.")


if __name__ == "__main__":
    run_and_save_notebook()
