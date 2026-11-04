"""
app/pipeline.py
===============
Top-level pipeline orchestrator for the AI Resume Analyzer.

``run_pipeline`` chains all 7 stages in order and returns a :class:`ScoreResult`.
All résumé text is held in memory only; it is never written to disk or logged.

Segfault mitigation (item 0c): TOKENIZERS_PARALLELISM, OMP_NUM_THREADS, and
PYTORCH_ENABLE_MPS_FALLBACK are set before any torch import to prevent MPS / HF
tokenizer fork-related crashes that produced exit code 139 intermittently.
"""

from __future__ import annotations

# ── Segfault-mitigation env vars (must be set before torch/tokenizer imports) ──
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import logging
import time
from pathlib import Path
from typing import Optional

from app.config import settings
from app.extraction.extractor import extract_entities
from app.extras.ranking import _populate_jd_skills
from app.genai.generator import generate_suggestions
from app.models.classifier import classify_resume
from app.models.matcher import match_resume_jd
from app.parsing.errors import ParseError
from app.parsing.parser import parse_document
from app.schemas import JobDescription, MatchResult, ParsedResume, ScoreResult
from app.scoring.ats import compute_ats_score
from app.scoring.explain import add_highlights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-type / size constants
# ---------------------------------------------------------------------------

MAX_FILE_BYTES: int = 5 * 1024 * 1024   # 5 MB hard limit for run_pipeline
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Warm-up singleton
# ---------------------------------------------------------------------------

_WARMED_UP: bool = False


def warm_up() -> None:
    """Pre-load the classifier and matcher models once (lazy singleton).

    Safe to call multiple times; subsequent calls are no-ops.
    Models are loaded into module-level caches inside their respective modules.
    """
    global _WARMED_UP
    if _WARMED_UP:
        return

    logger.info("warm_up: loading classifier …")
    try:
        from app.models.classifier import classify
        classify("warm up text", top_k=1)
    except Exception as exc:
        logger.warning("warm_up: classifier load failed: %s", exc)

    logger.info("warm_up: loading matcher …")
    try:
        from app.models.matcher import load_sentence_model
        load_sentence_model()
    except Exception as exc:
        logger.warning("warm_up: matcher load failed: %s", exc)

    _WARMED_UP = True
    logger.info("warm_up: complete")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _error_result(message: str, timings: dict[str, float]) -> ScoreResult:
    """Return a ScoreResult that surfaces an error without a stack trace."""
    from app.schemas import Issue
    return ScoreResult(
        total=0.0,
        grade="F",
        components={},
        missing_skills=[],
        issues=[Issue(
            component="pipeline",
            message=message,
            severity="high",
            estimated_gain=0.0,
        )],
        warnings=[message],
        timings_ms=timings,
    )


def _safe_stage(
    stage_name: str,
    fn,
    *args,
    timings: dict[str, float],
    **kwargs,
):
    """Run *fn* with *args* / *kwargs*, record timing, catch and log any error.

    Returns ``(result, error_message)`` where *error_message* is ``None`` on
    success.  On failure *result* is ``None`` and the error is logged without
    exposing résumé text.
    """
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        timings[f"{stage_name}_ms"] = (time.perf_counter() - t0) * 1000
        return result, None
    except Exception as exc:
        timings[f"{stage_name}_ms"] = (time.perf_counter() - t0) * 1000
        # Log only the exception type and message, never the resume text
        logger.warning("Stage %s failed: %s: %s", stage_name, type(exc).__name__, exc)
        return None, f"Stage '{stage_name}' failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    file_bytes: bytes,
    filename: str,
    jd_text: str = "",
    jd_title: str = "",
    use_genai: bool = True,
) -> ScoreResult:
    """Run the full AI Resume Analyzer pipeline.

    Stages (in order):
        1. Parsing       — extract raw text + layout metadata
        2. Extraction    — NER, skills, education, experience, projects
        3. Classification — resume category prediction
        4. JD Matching   — semantic + keyword match score
        5. ATS Scoring   — composite weighted score + issues
        6. Explainability — character-level highlights
        7. Generative    — improvement suggestions

    Privacy: résumé text is only held in memory during this call.
    It is never written to disk, logged, or persisted anywhere.

    Args:
        file_bytes:  Raw bytes of the uploaded résumé file.
        filename:    Original filename (e.g. ``"resume.pdf"``).
        jd_text:     Optional raw job-description text; empty string if not provided.
        jd_title:    Optional job title for the job description.
        use_genai:   If ``False``, skip Stage 7 generative suggestions.

    Returns:
        A fully populated :class:`ScoreResult`, or a minimal error result
        (grade ``"F"``, ``total=0.0``, ``issues`` with the error message)
        if the file is invalid.  Never raises an exception.
    """
    timings: dict[str, float] = {}
    warnings: list[str] = []

    logger.info("Pipeline start | file=%s bytes=%d device=%s", filename, len(file_bytes), settings.device)

    # ── Pre-flight: file size check ────────────────────────────────────────
    if len(file_bytes) > MAX_FILE_BYTES:
        msg = (
            f"File '{filename}' is {len(file_bytes) / 1024 / 1024:.1f} MB "
            f"which exceeds the 5 MB limit. Please upload a smaller file."
        )
        logger.warning("Pipeline rejected oversized file: %s bytes", len(file_bytes))
        return _error_result(msg, timings)

    # ── Pre-flight: file type check ────────────────────────────────────────
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        msg = (
            f"File type '{ext}' is not supported. "
            f"Please upload a PDF, DOCX, PNG, JPG, or JPEG file."
        )
        logger.warning("Pipeline rejected unsupported file type: %s", ext)
        return _error_result(msg, timings)

    # ── Stage 1: Parsing ───────────────────────────────────────────────────
    parsed, parse_err = _safe_stage("parsing", parse_document, file_bytes, filename, timings=timings)
    if parse_err or parsed is None:
        err_msg = parse_err or "Parsing produced no output."
        warnings.append(err_msg)
        # Return immediately on parse failure — nothing else can proceed
        result = _error_result(
            f"Could not parse '{filename}'. The file may be corrupted, password-protected, or empty.",
            timings,
        )
        result.warnings = warnings
        return result

    # ── Stage 2: Extraction ────────────────────────────────────────────────
    extracted, ext_err = _safe_stage("extraction", extract_entities, parsed, timings=timings)
    if ext_err or extracted is None:
        warnings.append(ext_err or "Extraction stage failed.")
        extracted = parsed   # use partially parsed resume

    # ── Stage 3: Classification ────────────────────────────────────────────
    categories, cls_err = _safe_stage("classification", classify_resume, extracted, timings=timings)
    if cls_err or categories is None:
        warnings.append(cls_err or "Classification stage failed.")
        categories = []

    # ── Stage 4: JD Matching ───────────────────────────────────────────────
    jd: Optional[JobDescription] = None
    if jd_text.strip():
        jd = JobDescription(text=jd_text, title=jd_title or "Job Description")
        # Fix (item 0): ensure required_skills populated from JD text
        jd = _populate_jd_skills(jd)

    match_result, match_err = _safe_stage("matching", match_resume_jd, extracted, jd, timings=timings)
    if match_err or match_result is None:
        warnings.append(match_err or "Matching stage failed.")
        match_result = MatchResult()

    # ── Stage 5: ATS Scoring ───────────────────────────────────────────────
    score_result, score_err = _safe_stage(
        "scoring", compute_ats_score, extracted, match_result, jd, timings=timings
    )
    if score_err or score_result is None:
        warnings.append(score_err or "Scoring stage failed.")
        score_result = ScoreResult(total=0.0, grade="F", timings_ms=timings, warnings=warnings)
        score_result.predicted_categories = categories
        return score_result

    score_result.predicted_categories = categories

    # ── Stage 6: Explainability ────────────────────────────────────────────
    explained, explain_err = _safe_stage("explain", add_highlights, score_result, extracted, timings=timings)
    if explain_err or explained is None:
        warnings.append(explain_err or "Explainability stage failed.")
        # Keep unenhanced score_result
    else:
        score_result = explained

    # ── Stage 7: Generative Suggestions ────────────────────────────────────
    if use_genai:
        suggested, genai_err = _safe_stage(
            "genai", generate_suggestions, score_result, extracted, jd_text, timings=timings
        )
        if genai_err or suggested is None:
            warnings.append(genai_err or "GenAI stage failed.")
            timings.setdefault("genai_ms", 0.0)
        else:
            score_result = suggested
    else:
        timings["genai_ms"] = 0.0

    score_result.timings_ms = timings
    if warnings:
        score_result.warnings = getattr(score_result, "warnings", []) + warnings

    logger.info(
        "Pipeline complete | total=%.2f | grade=%s | warnings=%d",
        score_result.total,
        score_result.grade,
        len(warnings),
    )
    return score_result
