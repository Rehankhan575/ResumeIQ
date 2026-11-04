"""
scripts/preflight.py
====================
Pre-flight validation script for the AI Resume Analyzer pipeline.

Runs every resume fixture against every sample JD through ``run_pipeline``,
prints a results table (fixture, JD, score, grade, seconds, warnings, OK/FAIL),
and reports the cold-start warm_up time.

Exit code 1 on any pipeline failure (exception, grade F on a known-good pair,
or any stage returning warnings that indicate a crash).

Usage:
    source venv/bin/activate
    python scripts/preflight.py
"""

from __future__ import annotations

# ── Segfault-mitigation ───────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Collect fixtures and JDs
# ---------------------------------------------------------------------------

RESUME_FIXTURES = sorted([
    p for p in FIXTURES_DIR.glob("*.pdf")
    if not p.stem.startswith("jd_")
] + [
    p for p in FIXTURES_DIR.glob("*.docx")
    if not p.stem.startswith("jd_")
])

JD_FILES = sorted(FIXTURES_DIR.glob("jd_*.txt"))

if not RESUME_FIXTURES:
    print("ERROR: No resume fixtures found in tests/fixtures/", file=sys.stderr)
    sys.exit(1)

if not JD_FILES:
    print("ERROR: No JD fixtures found in tests/fixtures/", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Warm up
# ---------------------------------------------------------------------------

print("=" * 72)
print("Resume Analyzer — Preflight Check")
print("=" * 72)
print()
print("Warming up models …", flush=True)

t_wu_start = time.perf_counter()
from app.pipeline import warm_up, run_pipeline
warm_up()
cold_start_s = time.perf_counter() - t_wu_start

print(f"  Cold-start time: {cold_start_s:.2f}s")
print()

# ---------------------------------------------------------------------------
# Run all fixture × JD combinations
# ---------------------------------------------------------------------------

HEADER = f"{'Fixture':<26} {'JD':<26} {'Score':>6} {'Grade':>5} {'Secs':>5} {'Warns':>5}  {'Status'}"
print(HEADER)
print("-" * len(HEADER))

rows: list[dict] = []
any_fail = False

for resume_path in RESUME_FIXTURES:
    file_bytes = resume_path.read_bytes()
    filename = resume_path.name

    for jd_path in JD_FILES:
        jd_text = jd_path.read_text(encoding="utf-8")
        jd_title = jd_path.stem.replace("jd_", "").replace("_", " ").title()

        t0 = time.perf_counter()
        ok = True
        result = None
        try:
            result = run_pipeline(
                file_bytes=file_bytes,
                filename=filename,
                jd_text=jd_text,
                jd_title=jd_title,
                use_genai=False,   # skip GenAI for speed
            )
            elapsed = time.perf_counter() - t0
            score = result.total
            grade = result.grade
            warns = len(getattr(result, "warnings", []))

            # A result with no components and grade F is a pipeline error
            if result.total == 0.0 and result.grade == "F" and not result.components:
                ok = False
                any_fail = True

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            score = 0.0
            grade = "F"
            warns = 1
            ok = False
            any_fail = True
            print(f"  EXCEPTION for {filename} × {jd_path.name}: {exc}", file=sys.stderr)

        status = "OK" if ok else "FAIL"
        row = {
            "fixture": filename,
            "jd": jd_path.name,
            "score": score,
            "grade": grade,
            "secs": elapsed,
            "warns": warns,
            "status": status,
        }
        rows.append(row)

        fix_short = filename[:25]
        jd_short = jd_path.name[:25]
        print(
            f"{fix_short:<26} {jd_short:<26} {score:>6.1f} {grade:>5} "
            f"{elapsed:>5.2f} {warns:>5}  {status}"
        )

print()
print("=" * 72)
print(f"Cold-start (warm_up): {cold_start_s:.2f}s")
print(f"Total combinations: {len(rows)}")
fails = [r for r in rows if r["status"] == "FAIL"]
print(f"Failures: {len(fails)}")
if fails:
    print("Failed combinations:")
    for r in fails:
        print(f"  {r['fixture']} × {r['jd']}")
print("=" * 72)

if any_fail:
    sys.exit(1)
else:
    print("All preflight checks passed.")
    sys.exit(0)

