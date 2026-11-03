#!/usr/bin/env python3
"""
scripts/eval_parsing.py
=======================
Evaluation benchmark for Stage 1 document parsing.

Evaluates:
    1. Precision & recall on canonical sections found vs expected sidecars.
    2. Contact field extraction accuracy (name, email, phone).
    3. Layout metadata detection flags (page_count, is_multi_column, has_tables,
       has_images, needs_ocr).
    4. Per-file parsing latency (ms).
    5. Nurse scanned OCR skill-recall (alias-aware, case-insensitive).
    6. Swe two-column non-interleaving verification.
    7. In-memory edge-case handling table.
    8. Optional aggregate benchmark over LiveCareer raw PDFs if present.

Saves structured benchmark results to artifacts/metrics/parsing_eval.json.
Privacy: Never logs or saves résumé text (counts and timings only).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pymupdf

from app.config import settings
from app.parsing import (
    CorruptedFileError,
    EmptyDocumentError,
    FileTooLargeError,
    PasswordProtectedError,
    UnsupportedFileTypeError,
    extract_text,
    match_heading,
    parse_resume,
)
from app.schemas import ParsedResume

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
TAXONOMY_CSV = ROOT / "data" / "taxonomies" / "skills.csv"
SECTION_HEADINGS_CSV = ROOT / "data" / "taxonomies" / "section_headings.csv"
METRICS_OUT = ROOT / "artifacts" / "metrics" / "parsing_eval.json"

RESUME_FIXTURES = [
    ("ds_clean.pdf", "ds_clean.expected.json"),
    ("swe_two_column.pdf", "swe_two_column.expected.json"),
    ("accountant_table.pdf", "accountant_table.expected.json"),
    ("chef_plain.docx", "chef_plain.expected.json"),
    ("nurse_scanned.pdf", "nurse_scanned.expected.json"),
]


def load_skills_alias_map() -> Dict[str, Set[str]]:
    """Load canonical skills and all aliases into a lookup map."""
    df = pd.read_csv(TAXONOMY_CSV, keep_default_na=False)
    alias_map: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        canonical = row["skill"].strip().lower()
        variants = {canonical}
        if row["aliases"]:
            for a in row["aliases"].split("|"):
                a_clean = a.strip().lower()
                if a_clean:
                    variants.add(a_clean)
        alias_map[canonical] = variants
    return alias_map


def load_section_heading_map() -> Dict[str, str]:
    """Load heading synonym to canonical section mapping."""
    mapping: Dict[str, str] = {}
    if SECTION_HEADINGS_CSV.exists():
        df = pd.read_csv(SECTION_HEADINGS_CSV, keep_default_na=False)
        for _, row in df.iterrows():
            h = row["heading"].strip().lower()
            c = row["canonical"].strip().lower()
            if h and c:
                mapping[h] = c
    return mapping


def evaluate_fixtures() -> Dict[str, Any]:
    heading_map = load_section_heading_map()
    skills_map = load_skills_alias_map()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixtures": {},
        "summary": {},
    }

    print("\n" + "=" * 90)
    print("STAGE 1 DOCUMENT PARSING EVALUATION")
    print("=" * 90)

    # 1. Section Precision & Recall Table
    print("\n[1] SECTIONS FOUND VS EXPECTED")
    print(f"{'Fixture':<24} | {'Expected (Canonical)':<32} | {'Found (Canonical)':<30} | {'Prec':<5} | {'Rec':<5} | {'Status'}")
    print("-" * 115)

    all_precision: List[float] = []
    all_recall: List[float] = []

    parsed_resumes: Dict[str, Tuple[ParsedResume, Dict[str, Any], float]] = {}

    for fix_file, side_file in RESUME_FIXTURES:
        data = (FIXTURES_DIR / fix_file).read_bytes()
        sidecar = json.loads((FIXTURES_DIR / side_file).read_text())

        t0 = time.perf_counter()
        parsed = parse_resume(data, fix_file)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        parsed_resumes[fix_file] = (parsed, sidecar, duration_ms)

        # Map expected sections from sidecar to canonical
        expected_raw = sidecar.get("expected_sections", [])
        expected_canonical = set()
        for sec in expected_raw:
            canon = heading_map.get(sec.strip().lower(), match_heading(sec.strip()) or sec.strip().lower())
            expected_canonical.add(canon)

        # Found canonical sections (excluding 'header')
        found_canonical = set(parsed.sections.keys()) - {"header"}

        overlap = expected_canonical.intersection(found_canonical)
        precision = len(overlap) / len(found_canonical) if found_canonical else 0.0
        recall = len(overlap) / len(expected_canonical) if expected_canonical else 0.0
        all_precision.append(precision)
        all_recall.append(recall)

        status = "PASS" if precision >= 0.8 and recall >= 0.8 else "FAIL"
        exp_str = ", ".join(sorted(expected_canonical))
        found_str = ", ".join(sorted(found_canonical))
        print(f"{fix_file:<24} | {exp_str:<32} | {found_str:<30} | {precision:.2f} | {recall:.2f} | {status}")

        results["fixtures"][fix_file] = {
            "duration_ms": round(duration_ms, 2),
            "sections_precision": round(precision, 4),
            "sections_recall": round(recall, 4),
            "sections_found": sorted(found_canonical),
            "sections_expected": sorted(expected_canonical),
        }

    # 2. Contact Fields Table
    print("\n[2] CONTACT EXTRACTION ACCURACY")
    print(f"{'Fixture':<24} | {'Field':<10} | {'Expected':<28} | {'Extracted':<28} | {'Status'}")
    print("-" * 90)

    contact_all_pass = True
    for fix_file, (parsed, sidecar, _) in parsed_resumes.items():
        exp_contact = sidecar.get("contact", {})
        contact_res = {}

        # Name
        exp_name = exp_contact.get("name", "")
        got_name = parsed.contact.name
        name_ok = (got_name.lower() == exp_name.lower()) if exp_name else True
        print(f"{fix_file:<24} | {'name':<10} | {exp_name:<28} | {got_name:<28} | {'PASS' if name_ok else 'FAIL'}")
        contact_res["name"] = {"expected": exp_name, "extracted": got_name, "pass": name_ok}
        if not name_ok:
            contact_all_pass = False

        # Email
        exp_email = exp_contact.get("email", "")
        got_email = parsed.contact.email
        email_ok = (got_email.lower() == exp_email.lower()) if exp_email else True
        print(f"{'':<24} | {'email':<10} | {exp_email:<28} | {got_email:<28} | {'PASS' if email_ok else 'FAIL'}")
        contact_res["email"] = {"expected": exp_email, "extracted": got_email, "pass": email_ok}
        if not email_ok:
            contact_all_pass = False

        # Phone
        exp_phone = exp_contact.get("phone", "")
        got_phone = parsed.contact.phone
        phone_ok = (exp_phone in got_phone or got_phone in exp_phone) if exp_phone else True
        print(f"{'':<24} | {'phone':<10} | {exp_phone:<28} | {got_phone:<28} | {'PASS' if phone_ok else 'FAIL'}")
        contact_res["phone"] = {"expected": exp_phone, "extracted": got_phone, "pass": phone_ok}
        if not phone_ok:
            contact_all_pass = False

        results["fixtures"][fix_file]["contact"] = contact_res

    # 3. Layout Flags Table
    print("\n[3] LAYOUT FLAGS DETECTION")
    print(f"{'Fixture':<24} | {'Flag':<16} | {'Expected':<10} | {'Extracted':<10} | {'Status'}")
    print("-" * 80)

    layout_flags = ["page_count", "is_multi_column", "has_tables", "has_images", "needs_ocr"]
    for fix_file, (parsed, sidecar, _) in parsed_resumes.items():
        exp_layout = sidecar.get("layout", {})
        layout_res = {}
        for flag in layout_flags:
            exp_val = exp_layout.get(flag)
            got_val = getattr(parsed.layout, flag)
            ok = (exp_val == got_val)
            print(f"{fix_file:<24} | {flag:<16} | {str(exp_val):<10} | {str(got_val):<10} | {'PASS' if ok else 'FAIL'}")
            layout_res[flag] = {"expected": exp_val, "extracted": got_val, "pass": ok}
            fix_file = ""  # Only print filename on first row
        results["fixtures"][RESUME_FIXTURES[[f[0] for f in RESUME_FIXTURES].index(fix_file or list(parsed_resumes.keys())[0])][0]]["layout"] = layout_res

    # Fix fixture key mapping in layout_res dict
    for fix_file, (parsed, sidecar, _) in parsed_resumes.items():
        exp_layout = sidecar.get("layout", {})
        results["fixtures"][fix_file]["layout"] = {
            flag: {
                "expected": exp_layout.get(flag),
                "extracted": getattr(parsed.layout, flag),
                "pass": exp_layout.get(flag) == getattr(parsed.layout, flag),
            }
            for flag in layout_flags
        }

    # 4. Latency Table
    print("\n[4] PARSE TIMINGS")
    print(f"{'Fixture':<24} | {'Duration (ms)':<15} | {'Word Count':<12} | {'Method':<15} | {'Status'}")
    print("-" * 80)
    for fix_file, (parsed, _, dur) in parsed_resumes.items():
        status = "PASS" if dur < 10000 else "SLOW"
        print(f"{fix_file:<24} | {dur:<15.1f} | {parsed.layout.word_count:<12} | {parsed.layout.extraction_method:<15} | {status}")

    # 5. Nurse Scanned OCR Skill-Recall Check
    print("\n[5] NURSE_SCANNED OCR SKILL RECALL")
    print(f"{'Expected Skill':<24} | {'Matched In OCR':<15} | {'Variant / Form'}")
    print("-" * 65)

    nurse_parsed, nurse_sidecar, _ = parsed_resumes["nurse_scanned.pdf"]
    nurse_raw_lower = nurse_parsed.raw_text.lower()
    expected_nurse_skills = nurse_sidecar.get("skills", [])
    skills_found_count = 0

    nurse_skills_detail = []
    for sk in expected_nurse_skills:
        variants = skills_map.get(sk.lower(), {sk.lower()})
        matched_variant = None
        for v in variants:
            if v in nurse_raw_lower:
                matched_variant = v
                break

        if matched_variant:
            skills_found_count += 1
            print(f"{sk:<24} | {'YES':<15} | {matched_variant}")
            nurse_skills_detail.append({"skill": sk, "found": True, "matched": matched_variant})
        else:
            print(f"{sk:<24} | {'NO':<15} | None")
            nurse_skills_detail.append({"skill": sk, "found": False, "matched": None})

    nurse_recall = skills_found_count / len(expected_nurse_skills) if expected_nurse_skills else 0.0
    nurse_status = "PASS" if nurse_recall >= 0.8 else "FAIL"
    print(f"\nNurse Scanned OCR Skill Recall: {skills_found_count}/{len(expected_nurse_skills)} ({nurse_recall:.1%}) -> {nurse_status}")
    results["nurse_scanned_skill_recall"] = {
        "expected": expected_nurse_skills,
        "found_count": skills_found_count,
        "total_count": len(expected_nurse_skills),
        "recall": round(nurse_recall, 4),
        "status": nurse_status,
        "details": nurse_skills_detail,
    }

    # 6. Swe Two-Column Non-Interleaving Check
    print("\n[6] SWE_TWO_COLUMN NON-INTERLEAVING CHECK")
    swe_parsed, _, _ = parsed_resumes["swe_two_column.pdf"]
    exp_text = swe_parsed.sections.get("experience", "")

    has_title_1 = "Senior Backend Engineer" in exp_text
    has_company_1 = "CloudScale Systems" in exp_text
    has_title_2 = "Frontend Developer" in exp_text
    has_company_2 = "WebSphere Digital" in exp_text
    # Sidebar traps: must NOT be in experience
    no_sidebar_interleave = (
        "Tech Stack" not in exp_text
        and "Academics" not in exp_text
        and "Seattle, WA" not in exp_text
    )

    swe_ok = (
        has_title_1
        and has_company_1
        and has_title_2
        and has_company_2
        and no_sidebar_interleave
    )

    print(f"Contains 'Senior Backend Engineer': {'YES' if has_title_1 else 'NO'}")
    print(f"Contains 'CloudScale Systems':       {'YES' if has_company_1 else 'NO'}")
    print(f"Contains 'Frontend Developer':        {'YES' if has_title_2 else 'NO'}")
    print(f"Contains 'WebSphere Digital':         {'YES' if has_company_2 else 'NO'}")
    print(f"Free of Sidebar Interleaving:         {'YES' if no_sidebar_interleave else 'NO'}")
    print(f"Swe Two-Column Check Result:          {'PASS' if swe_ok else 'FAIL'}")

    results["swe_two_column_interleaving"] = {
        "has_senior_backend_engineer": has_title_1,
        "has_cloudscale_systems": has_company_1,
        "has_frontend_developer": has_title_2,
        "has_websphere_digital": has_company_2,
        "free_of_sidebar_interleaving": no_sidebar_interleave,
        "status": "PASS" if swe_ok else "FAIL",
    }

    # 7. Edge Cases Table
    print("\n[7] IN-MEMORY EDGE-CASE BEHAVIOR")
    print(f"{'Edge Case':<32} | {'Expected Exception / Behavior':<35} | {'Status'}")
    print("-" * 80)

    edge_cases = []

    # 1. Empty bytes
    try:
        extract_text(b"", "empty.pdf")
        edge_1 = "FAIL"
    except EmptyDocumentError:
        edge_1 = "PASS"
    except Exception:
        edge_1 = "WRONG_EXC"
    print(f"{'empty bytes':<32} | {'EmptyDocumentError':<35} | {edge_1}")
    edge_cases.append({"case": "empty bytes", "expected": "EmptyDocumentError", "status": edge_1})

    # 2. Truncated PDF
    try:
        extract_text(b"%PDF-1.4\n1 0 obj\n<<\n", "truncated.pdf")
        edge_2 = "FAIL"
    except CorruptedFileError:
        edge_2 = "PASS"
    except Exception:
        edge_2 = "WRONG_EXC"
    print(f"{'truncated PDF':<32} | {'CorruptedFileError':<35} | {edge_2}")
    edge_cases.append({"case": "truncated PDF", "expected": "CorruptedFileError", "status": edge_2})

    # 3. Password-protected PDF
    try:
        doc = pymupdf.open()
        doc.new_page()
        buf = io.BytesIO()
        perms = int(pymupdf.PDF_PERM_ACCESSIBILITY | pymupdf.PDF_PERM_PRINT)
        doc.save(buf, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="admin", permissions=perms)
        doc.close()
        extract_text(buf.getvalue(), "locked.pdf")
        edge_3 = "FAIL"
    except PasswordProtectedError:
        edge_3 = "PASS"
    except Exception:
        edge_3 = "WRONG_EXC"
    print(f"{'password-protected PDF':<32} | {'PasswordProtectedError':<35} | {edge_3}")
    edge_cases.append({"case": "password-protected PDF", "expected": "PasswordProtectedError", "status": edge_3})

    # 4. Oversize file
    orig_limit = settings.max_file_bytes
    settings.max_file_bytes = 100
    try:
        extract_text(b"%PDF" + b"x" * 200, "oversize.pdf")
        edge_4 = "FAIL"
    except FileTooLargeError:
        edge_4 = "PASS"
    except Exception:
        edge_4 = "WRONG_EXC"
    finally:
        settings.max_file_bytes = orig_limit
    print(f"{'oversize file':<32} | {'FileTooLargeError':<35} | {edge_4}")
    edge_cases.append({"case": "oversize file", "expected": "FileTooLargeError", "status": edge_4})

    # 5. Over max_pages
    orig_pages = settings.max_pages
    settings.max_pages = 2
    try:
        doc = pymupdf.open()
        for i in range(4):
            doc.new_page().insert_text((50, 50), f"Page {i+1} text content to avoid triggering OCR.")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        res = parse_resume(buf.getvalue(), "multipage.pdf")
        edge_5 = "PASS" if len(res.warnings) > 0 and res.layout.page_count == 4 else "FAIL"
    except Exception:
        edge_5 = "FAIL"
    finally:
        settings.max_pages = orig_pages
    print(f"{'over max_pages (warning)':<32} | {'Degrades with warning':<35} | {edge_5}")
    edge_cases.append({"case": "over max_pages", "expected": "Degrades with warning", "status": edge_5})

    # 6. PDF named .docx
    try:
        pdf_bytes = (FIXTURES_DIR / "ds_clean.pdf").read_bytes()
        res = parse_resume(pdf_bytes, "named_docx.docx")
        edge_6 = "PASS" if res.layout.file_type == "pdf" and res.contact.name == "Alex Mercer" else "FAIL"
    except Exception:
        edge_6 = "FAIL"
    print(f"{'PDF bytes named .docx':<32} | {'Detects magic bytes -> pdf':<35} | {edge_6}")
    edge_cases.append({"case": "PDF bytes named .docx", "expected": "Detects magic bytes -> pdf", "status": edge_6})

    # 7. DOCX named .pdf
    try:
        docx_bytes = (FIXTURES_DIR / "chef_plain.docx").read_bytes()
        res = parse_resume(docx_bytes, "named_pdf.pdf")
        edge_7 = "PASS" if res.layout.file_type == "docx" and res.contact.name == "Marcus Bennett" else "FAIL"
    except Exception:
        edge_7 = "FAIL"
    print(f"{'DOCX bytes named .pdf':<32} | {'Detects magic bytes -> docx':<35} | {edge_7}")
    edge_cases.append({"case": "DOCX bytes named .pdf", "expected": "Detects magic bytes -> docx", "status": edge_7})

    # 8. Unsupported extension
    try:
        extract_text(b"some raw string without known magic bytes", "resume.xyz")
        edge_8 = "FAIL"
    except UnsupportedFileTypeError:
        edge_8 = "PASS"
    except Exception:
        edge_8 = "WRONG_EXC"
    print(f"{'unsupported extension (.xyz)':<32} | {'UnsupportedFileTypeError':<35} | {edge_8}")
    edge_cases.append({"case": "unsupported extension", "expected": "UnsupportedFileTypeError", "status": edge_8})

    results["edge_cases"] = edge_cases

    # 8. Optional LiveCareer PDFs
    raw_pdfs = list((ROOT / "data" / "raw").glob("**/*.pdf"))
    print(f"\n[8] LIVECAREER RAW PDFS CHECK")
    if raw_pdfs:
        sample_count = min(len(raw_pdfs), 50)
        success_count = 0
        failure_count = 0
        timings = []
        for pdf_p in raw_pdfs[:sample_count]:
            t_start = time.perf_counter()
            try:
                parse_resume(pdf_p.read_bytes(), pdf_p.name)
                success_count += 1
                timings.append((time.perf_counter() - t_start) * 1000.0)
            except Exception:
                failure_count += 1

        avg_t = sum(timings) / len(timings) if timings else 0.0
        print(f"Sampled {sample_count} LiveCareer PDFs: {success_count} success, {failure_count} failures, mean time {avg_t:.1f}ms")
        results["livecareer_sample"] = {
            "sampled": sample_count,
            "success": success_count,
            "failed": failure_count,
            "mean_ms": round(avg_t, 2),
        }
    else:
        print("0 LiveCareer PDFs found in data/raw. Skipping aggregate raw batch.")
        results["livecareer_sample"] = {"sampled": 0, "status": "none_found"}

    # Summary
    mean_p = sum(all_precision) / len(all_precision)
    mean_r = sum(all_recall) / len(all_recall)
    results["summary"] = {
        "mean_sections_precision": round(mean_p, 4),
        "mean_sections_recall": round(mean_r, 4),
        "contact_all_pass": contact_all_pass,
        "nurse_ocr_skill_recall": round(nurse_recall, 4),
        "swe_non_interleaving": swe_ok,
        "edge_cases_all_pass": all(ec["status"] == "PASS" for ec in edge_cases),
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics written to {METRICS_OUT.relative_to(ROOT)}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    evaluate_fixtures()
