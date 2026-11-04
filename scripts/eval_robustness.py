"""
scripts/eval_robustness.py
==========================
Robustness evaluation script for parsing and extraction on resume documents (.pdf, .docx).

Usage:
    python scripts/eval_robustness.py [FOLDER]

Default folder:
    data/real_samples/ (gitignored)

Privacy Guarantee:
    Prints per file ONLY: filename, pages, parse method, seconds, sections found,
    skills count, years experience, and warnings count.
    Never prints or logs candidate text, names, emails, phones, or contact details.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Segfault mitigation & headless execution
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.extraction.extractor import extract_entities
from app.parsing.parser import parse_resume


def evaluate_folder(folder_path: Path) -> List[Dict[str, Any]]:
    """Scan folder and evaluate parsing + entity extraction for every PDF/DOCX."""
    if not folder_path.exists():
        print(f"\n[INFO] Folder '{folder_path}' does not exist.")
        print("Hint: Place resume files in data/real_samples/ (gitignored) or run:")
        print("      python scripts/eval_robustness.py tests/fixtures/\n")
        return []

    # Gather PDF and DOCX files
    valid_exts = {".pdf", ".docx"}
    files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    if not files:
        print(f"\n[INFO] No .pdf or .docx files found in '{folder_path}'.")
        print("Hint: Place test files in the directory or run:")
        print("      python scripts/eval_robustness.py tests/fixtures/\n")
        return []

    # Use relative path if within project root to prevent leaking system user paths
    try:
        display_path = folder_path.resolve().relative_to(ROOT_DIR.resolve())
    except ValueError:
        display_path = folder_path.name

    print("\n" + "=" * 96)
    print(f"ROBUSTNESS EVALUATION: {display_path} ({len(files)} files)")
    print("=" * 96)
    header = f"{'Filename':<28} | {'Pages':<5} | {'Parse Method':<14} | {'Seconds':<8} | {'Sections':<8} | {'Skills':<6} | {'Years':<6} | {'Warnings':<8}"
    print(header)
    print("-" * 96)

    records: List[Dict[str, Any]] = []

    for fpath in files:
        fname = fpath.name
        t0 = time.perf_counter()
        try:
            raw_bytes = fpath.read_bytes()
            parsed = parse_resume(raw_bytes, fname)
            parsed = extract_entities(parsed)
            elapsed = time.perf_counter() - t0

            # Collect metadata only — strictly privacy safe
            pages = parsed.layout.page_count if parsed.layout else 1
            is_ocr = any("ocr fallback applied" in w.lower() for w in parsed.warnings)
            if is_ocr:
                method = "pdf_ocr"
            elif fpath.suffix.lower() == ".docx":
                method = "docx_native"
            else:
                method = "pdf_native"

            sections_found = len([k for k, v in parsed.sections.items() if v and str(v).strip()])
            skills_count = len(parsed.skills)
            years = parsed.total_years_experience
            warnings_count = len(parsed.warnings)

            rec = {
                "filename": fname,
                "pages": pages,
                "method": method,
                "seconds": round(elapsed, 3),
                "sections": sections_found,
                "skills": skills_count,
                "years": round(years, 1),
                "warnings": warnings_count,
                "status": "success",
            }
            records.append(rec)

            print(
                f"{fname[:28]:<28} | {pages:<5} | {method:<14} | {elapsed:<8.3f} | "
                f"{sections_found:<8} | {skills_count:<6} | {years:<6.1f} | {warnings_count:<8}"
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            rec = {
                "filename": fname,
                "pages": 0,
                "method": "failed",
                "seconds": round(elapsed, 3),
                "sections": 0,
                "skills": 0,
                "years": 0.0,
                "warnings": 1,
                "status": "error",
            }
            records.append(rec)
            print(
                f"{fname[:28]:<28} | {'0':<5} | {'failed':<14} | {elapsed:<8.3f} | "
                f"{'0':<8} | {'0':<6} | {'0.0':<6} | {'1':<8}"
            )

    print("=" * 96)

    # Print summary counts
    n_total = len(records)
    n_success = sum(1 for r in records if r["status"] == "success")
    n_ocr = sum(1 for r in records if "ocr" in r["method"].lower())
    avg_sec = sum(r["seconds"] for r in records) / n_total if n_total > 0 else 0.0
    avg_sec_success = sum(r["seconds"] for r in records if r["status"] == "success") / n_success if n_success > 0 else 0.0
    avg_skills = sum(r["skills"] for r in records) / n_total if n_total > 0 else 0.0
    avg_years = sum(r["years"] for r in records) / n_total if n_total > 0 else 0.0
    total_warnings = sum(r["warnings"] for r in records)

    print("\n" + "=" * 50)
    print("ROBUSTNESS SUMMARY COUNTS")
    print("=" * 50)
    print(f"Total Files Scanned:       {n_total}")
    print(f"Successfully Processed:    {n_success} ({n_success/n_total*100:.1f}%)" if n_total > 0 else "Successfully Processed: 0")
    print(f"OCR Fallback Invocations:  {n_ocr}")
    print(f"Average Duration (All):    {avg_sec:.3f} s")
    print(f"Average Duration (Success):{avg_sec_success:.3f} s")
    print(f"Average Skills Extracted:  {avg_skills:.1f}")
    print(f"Average Years Experience:  {avg_years:.1f}")
    print(f"Total Warnings Recorded:   {total_warnings}")
    print("=" * 50 + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline parsing and extraction robustness on resume files.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="data/real_samples/",
        help="Directory containing PDF/DOCX files (default: data/real_samples/)",
    )
    args = parser.parse_args()

    target_dir = Path(args.folder)
    evaluate_folder(target_dir)


if __name__ == "__main__":
    main()
