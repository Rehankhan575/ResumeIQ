#!/usr/bin/env python3
"""download_data.py – Download and prepare approved raw datasets.

Approved datasets
-----------------
A.     HF opensporks/resumes
       -> data/raw/livecareer/
B.     HF sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2
       -> data/raw/ner_synthetic/ (preserves train/validation/test splits)
B-real HF Mehyaar/Annotated_NER_PDF_Resumes
       -> data/raw/ner_real_skills/
C1.    HF batuhanmtl/job_resume_fit
       -> data/raw/job_resume_fit/
C2.    HF datastax/linkedin_job_listings
       -> data/raw/linkedin_sample/postings_5k.parquet (5 000-row sample, seed 42)
D1.    O*NET 31.0 direct CSV files
       -> data/raw/onet/ (software_skills.csv, occupation_data.csv)
D2.    ESCO skills_en.csv from official portal
       -> data/raw/esco/ (manual instructions if export form required)

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --only livecareer
    python scripts/download_data.py --only linkedin_sample --force
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger
from tqdm import tqdm

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "docs"
MANIFEST_PATH = DOCS_DIR / "data_manifest.json"

# ── Pinned Revisions & URLs ───────────────────────────────────────────────────
PINNED_REVISIONS = {
    "livecareer": {
        "repo_id": "opensporks/resumes",
        "revision": "ed4cb5f3fd1ce7e0a0e74e1a09c1a3b702c2c2eb",
        "url": "https://huggingface.co/datasets/opensporks/resumes",
        "license": "cc0-1.0",
    },
    "ner_synthetic": {
        "repo_id": "sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2",
        "revision": "d878412716a4d6bda378e1e0f60a6e422ff0062b",
        "url": "https://huggingface.co/datasets/sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2",
        "license": "NOT STATED",
    },
    "ner_real_skills": {
        "repo_id": "Mehyaar/Annotated_NER_PDF_Resumes",
        "revision": "1f1be3de47defbd77c1ca65e40e3bb9c7d01e63f",
        "url": "https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes",
        "license": "mit",
    },
    "job_resume_fit": {
        "repo_id": "batuhanmtl/job_resume_fit",
        "revision": "fd536486a5e99db00716b6fbe58e6368d2d038ef",
        "url": "https://huggingface.co/datasets/batuhanmtl/job_resume_fit",
        "license": "other",
    },
    "linkedin": {
        "repo_id": "datastax/linkedin_job_listings",
        "revision": "cb54240684b6ae7c847330765886b6d48e753e6b",
        "url": "https://huggingface.co/datasets/datastax/linkedin_job_listings",
        "license": "NOT STATED",
    },
    "onet": {
        "version": "db_31_0_csv",
        "url": "https://www.onetcenter.org/dl_files/database/db_31_0_csv/",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    },
    "esco": {
        "version": "v1.2.1",
        "url": "https://esco.ec.europa.eu/en/use-esco/download",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    },
}

# LinkedIn sampling configuration
LINKEDIN_SAMPLE_N = 5_000
LINKEDIN_SAMPLE_SEED = 42
LINKEDIN_MIN_DESC_CHARS = 200
LINKEDIN_KEEP_COLS = [
    "title",
    "description",
    "skills_desc",
    "formatted_experience_level",
    "location",
]


# ── Utility Functions ─────────────────────────────────────────────────────────

def _rel_path(path: Path) -> str:
    """Return path relative to ROOT if possible, else absolute string."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)



def calculate_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute hex SHA-256 digest of the given file path."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    """Load existing manifest or return an empty dict."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning(f"Could not load manifest ({e}); starting fresh.")
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Atomically write manifest to docs/data_manifest.json."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    temp_path.replace(MANIFEST_PATH)
    logger.info(f"Updated data manifest at {_rel_path(MANIFEST_PATH)}")


def download_http_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file over HTTP with progress bar and custom headers."""
    if dest.exists() and not force:
        logger.info(f"File already exists (skipping download): {_rel_path(dest)}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {_rel_path(dest)}")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()
    total_bytes = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as fh, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))

    return dest


# ── Dataset Downloaders ───────────────────────────────────────────────────────

def download_livecareer(force: bool = False) -> dict[str, Any]:
    """A. HF opensporks/resumes -> data/raw/livecareer/"""
    meta = PINNED_REVISIONS["livecareer"]
    dest_dir = DATA_RAW / "livecareer"
    out_file = dest_dir / "train.parquet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info(f"[livecareer] Already downloaded at {_rel_path(out_file)}")
        df = pd.read_parquet(out_file)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
    else:
        logger.info(f"[livecareer] Loading {meta['repo_id']} at revision {meta['revision'][:8]}...")
        # Use data_files='Resume/Resume.csv' to load the clean 2,484 CSV dataset without downloading 2,400 raw PDFs
        ds = load_dataset(
            meta["repo_id"],
            data_files="Resume/Resume.csv",
            split="train",
            revision=meta["revision"],
        )
        df = ds.to_pandas()
        df.to_parquet(out_file, index=False)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
        logger.success(f"[livecareer] Saved {rows:,} rows, {len(cols)} cols -> {_rel_path(out_file)}")

    return {
        "name": "LiveCareer Resume Dataset",
        "source_url": meta["url"],
        "revision": meta["revision"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "files": [
            {
                "path": _rel_path(out_file),
                "rows": rows,
                "columns": cols,
                "sha256": sha,
            }
        ],
    }


def download_ner_synthetic(force: bool = False) -> dict[str, Any]:
    """B. HF sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2 -> data/raw/ner_synthetic/"""
    meta = PINNED_REVISIONS["ner_synthetic"]
    dest_dir = DATA_RAW / "ner_synthetic"
    dest_dir.mkdir(parents=True, exist_ok=True)

    expected_splits = ["train", "validation", "test"]
    all_exist = all((dest_dir / f"{s}.parquet").exists() for s in expected_splits)

    file_records = []
    if all_exist and not force:
        logger.info(f"[ner_synthetic] All splits exist in {_rel_path(dest_dir)}")
        for s in expected_splits:
            p = dest_dir / f"{s}.parquet"
            df = pd.read_parquet(p)
            file_records.append(
                {
                    "path": _rel_path(p),
                    "split": s,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sha256": calculate_sha256(p),
                }
            )
    else:
        logger.info(f"[ner_synthetic] Loading {meta['repo_id']} at revision {meta['revision'][:8]}...")
        ds_splits = load_dataset(meta["repo_id"], revision=meta["revision"])
        for s in expected_splits:
            p = dest_dir / f"{s}.parquet"
            df = ds_splits[s].to_pandas()
            df.to_parquet(p, index=False)
            file_records.append(
                {
                    "path": _rel_path(p),
                    "split": s,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sha256": calculate_sha256(p),
                }
            )
            logger.success(f"[ner_synthetic] Split '{s}': {len(df):,} rows -> {_rel_path(p)}")

    return {
        "name": "AuraATS Synthetic Resume NER 12K v2",
        "source_url": meta["url"],
        "revision": meta["revision"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "files": file_records,
    }


def download_ner_real_skills(force: bool = False) -> dict[str, Any]:
    """B-real. HF Mehyaar/Annotated_NER_PDF_Resumes -> data/raw/ner_real_skills/"""
    meta = PINNED_REVISIONS["ner_real_skills"]
    dest_dir = DATA_RAW / "ner_real_skills"
    out_file = dest_dir / "ner_real_skills.parquet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info(f"[ner_real_skills] Already downloaded at {_rel_path(out_file)}")
        df = pd.read_parquet(out_file)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
    else:
        logger.info(f"[ner_real_skills] Downloading ResumesJsonAnnotated.zip from {meta['repo_id']}...")
        zip_path = hf_hub_download(
            repo_id=meta["repo_id"],
            filename="ResumesJsonAnnotated.zip",
            repo_type="dataset",
            revision=meta["revision"],
        )

        logger.info(f"[ner_real_skills] Parsing JSON annotations from {zip_path}...")
        records = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".json") and not name.startswith("__MACOSX"):
                    raw = zf.read(name).decode("utf-8", errors="replace")
                    data = json.loads(raw)
                    text = data.get("text", "")
                    if isinstance(text, str):
                        # Sanitize lone surrogate characters from PDF conversions
                        text = text.encode("utf-8", "replace").decode("utf-8", "replace")
                    annotations = data.get("annotations", [])
                    records.append(
                        {
                            "id": name,
                            "text": text,
                            "annotations": json.dumps(annotations),
                        }
                    )

        df = pd.DataFrame(records)
        df.to_parquet(out_file, index=False)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
        logger.success(f"[ner_real_skills] Saved {rows:,} rows -> {_rel_path(out_file)}")

    return {
        "name": "Annotated NER PDF Resumes (IT Skills)",
        "source_url": meta["url"],
        "revision": meta["revision"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "files": [
            {
                "path": _rel_path(out_file),
                "rows": rows,
                "columns": cols,
                "sha256": sha,
            }
        ],
    }


def download_job_resume_fit(force: bool = False) -> dict[str, Any]:
    """C1. HF batuhanmtl/job_resume_fit -> data/raw/job_resume_fit/"""
    meta = PINNED_REVISIONS["job_resume_fit"]
    dest_dir = DATA_RAW / "job_resume_fit"
    out_file = dest_dir / "train.parquet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info(f"[job_resume_fit] Already exists at {_rel_path(out_file)}")
        df = pd.read_parquet(out_file)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
    else:
        logger.info(f"[job_resume_fit] Loading {meta['repo_id']} at revision {meta['revision'][:8]}...")
        ds = load_dataset(meta["repo_id"], split="train", revision=meta["revision"])
        df = ds.to_pandas()
        df.to_parquet(out_file, index=False)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
        logger.success(f"[job_resume_fit] Saved {rows:,} rows, {len(cols)} cols -> {_rel_path(out_file)}")

    return {
        "name": "Resume-Job Fit Dataset",
        "source_url": meta["url"],
        "revision": meta["revision"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "files": [
            {
                "path": _rel_path(out_file),
                "rows": rows,
                "columns": cols,
                "sha256": sha,
            }
        ],
    }


def download_linkedin_sample(force: bool = False) -> dict[str, Any]:
    """C2. HF datastax/linkedin_job_listings -> 5,000 sample -> data/raw/linkedin_sample/"""
    meta = PINNED_REVISIONS["linkedin"]
    dest_dir = DATA_RAW / "linkedin_sample"
    out_file = dest_dir / "postings_5k.parquet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info(f"[linkedin_sample] Already exists at {_rel_path(out_file)}")
        df = pd.read_parquet(out_file)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(out_file)
    else:
        logger.info(
            f"[linkedin_sample] Downloading ~517 MB postings.csv from {meta['repo_id']} (rev: {meta['revision'][:8]})..."
        )
        csv_path = hf_hub_download(
            repo_id=meta["repo_id"],
            filename="postings.csv",
            repo_type="dataset",
            revision=meta["revision"],
        )

        logger.info("[linkedin_sample] Reading full postings dataset...")
        df_full = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"  Total raw postings: {len(df_full):,}")

        # Filter: non-empty title + description >= 200 chars
        mask = (
            df_full["title"].notna()
            & df_full["title"].astype(str).str.strip().ne("")
            & df_full["description"].notna()
            & df_full["description"].astype(str).str.len().ge(LINKEDIN_MIN_DESC_CHARS)
        )
        df_filtered = df_full.loc[mask].copy()
        logger.info(f"  Postings satisfying criteria (desc >= 200 chars): {len(df_filtered):,}")

        # Sample 5,000 rows with seed 42
        sample_n = min(LINKEDIN_SAMPLE_N, len(df_filtered))
        df_sample = df_filtered.sample(n=sample_n, random_state=LINKEDIN_SAMPLE_SEED).reset_index(drop=True)

        # Keep approved columns
        keep_cols = [c for c in LINKEDIN_KEEP_COLS if c in df_sample.columns]
        df_sample = df_sample[keep_cols]

        df_sample.to_parquet(out_file, index=False)
        rows, cols = len(df_sample), list(df_sample.columns)
        sha = calculate_sha256(out_file)
        logger.success(f"[linkedin_sample] Saved {rows:,} sample rows -> {_rel_path(out_file)}")

    return {
        "name": "LinkedIn Job Listings (5K Postings Sample)",
        "source_url": meta["url"],
        "revision": meta["revision"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "sampling": {
            "seed": LINKEDIN_SAMPLE_SEED,
            "sample_size": rows,
            "min_description_chars": LINKEDIN_MIN_DESC_CHARS,
        },
        "files": [
            {
                "path": _rel_path(out_file),
                "rows": rows,
                "columns": cols,
                "sha256": sha,
            }
        ],
    }


def download_onet(force: bool = False) -> dict[str, Any]:
    """D1. O*NET direct files software_skills.csv & occupation_data.csv -> data/raw/onet/"""
    meta = PINNED_REVISIONS["onet"]
    dest_dir = DATA_RAW / "onet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_get = ["software_skills.csv", "occupation_data.csv"]
    file_records = []

    for fname in files_to_get:
        out_path = dest_dir / fname
        file_url = f"{meta['url'].rstrip('/')}/{fname}"
        download_http_file(file_url, out_path, force=force)

        df = pd.read_csv(out_path)
        sha = calculate_sha256(out_path)
        file_records.append(
            {
                "path": _rel_path(out_path),
                "rows": len(df),
                "columns": list(df.columns),
                "sha256": sha,
            }
        )
        logger.success(f"[onet] {fname}: {len(df):,} rows -> {_rel_path(out_path)}")

    return {
        "name": "O*NET 31.0 Database (Software Skills & Occupation Data)",
        "source_url": meta["url"],
        "revision": meta["version"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "files": file_records,
    }


def download_esco(force: bool = False) -> dict[str, Any]:
    """D2. ESCO skills_en.csv -> data/raw/esco/ (official portal export instructions)."""
    meta = PINNED_REVISIONS["esco"]
    dest_dir = DATA_RAW / "esco"
    dest_dir.mkdir(parents=True, exist_ok=True)
    skills_csv = dest_dir / "skills_en.csv"

    if skills_csv.exists() and not force:
        logger.info(f"[esco] Found official export at {_rel_path(skills_csv)}")
        df = pd.read_csv(skills_csv)
        rows, cols = len(df), list(df.columns)
        sha = calculate_sha256(skills_csv)
        return {
            "name": "ESCO Skills Taxonomy",
            "source_url": meta["url"],
            "revision": meta["version"],
            "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "license": meta["license"],
            "files": [
                {
                    "path": _rel_path(skills_csv),
                    "rows": rows,
                    "columns": cols,
                    "sha256": sha,
                }
            ],
        }

    logger.warning("================================================================================")
    logger.warning("[esco] Manual export required from official European Commission ESCO portal:")
    logger.warning("  1. Navigate to: https://esco.ec.europa.eu/en/use-esco/download")
    logger.warning("  2. In the 'Your ESCO dataset' form, select:")
    logger.warning("       - Version:   ESCO dataset - v1.2.1")
    logger.warning("       - Content:   Skills")
    logger.warning("       - File type: CSV")
    logger.warning("       - Language:  English (en)")
    logger.warning("  3. Click 'Add to your package' then 'Export your dataset'.")
    logger.warning("  4. Extract 'skills_en.csv' into: data/raw/esco/skills_en.csv")
    logger.warning("================================================================================")

    return {
        "name": "ESCO Skills Taxonomy",
        "source_url": meta["url"],
        "revision": meta["version"],
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": meta["license"],
        "status": "manual_download_required",
        "instructions": (
            "Visit https://esco.ec.europa.eu/en/use-esco/download, select v1.2.1 Skills CSV (en), "
            "export package, and extract skills_en.csv into data/raw/esco/"
        ),
        "files": [],
    }


# ── CLI & Orchestrator ────────────────────────────────────────────────────────

DATASET_HANDLERS = {
    "livecareer": download_livecareer,
    "ner_synthetic": download_ner_synthetic,
    "ner_real_skills": download_ner_real_skills,
    "job_resume_fit": download_job_resume_fit,
    "linkedin_sample": download_linkedin_sample,
    "onet": download_onet,
    "esco": download_esco,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify approved raw datasets for AI Resume Analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=list(DATASET_HANDLERS.keys()),
        default=None,
        help="Download only a specific dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if target files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Starting data download pipeline (only={args.only}, force={args.force})")

    manifest = load_manifest()
    targets = [args.only] if args.only else list(DATASET_HANDLERS.keys())

    for target in targets:
        handler = DATASET_HANDLERS[target]
        logger.info(f"Processing dataset: {target}...")
        try:
            record = handler(force=args.force)
            manifest[target] = record
            save_manifest(manifest)
        except Exception as e:
            logger.exception(f"FATAL: Failed to download/process dataset '{target}': {e}")
            sys.exit(1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    main()
