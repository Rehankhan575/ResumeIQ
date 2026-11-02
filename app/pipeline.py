"""
app/pipeline.py
===============
Top-level pipeline orchestrator for the AI Resume Analyzer.

``run_pipeline`` chains all 7 stages in order and returns a :class:`ScoreResult`.
All résumé text is held in memory only; it is never written to disk or logged.
"""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.extraction.extractor import extract_entities
from app.genai.generator import generate_suggestions
from app.models.classifier import classify_resume
from app.models.matcher import match_resume_jd
from app.parsing.parser import parse_document
from app.schemas import JobDescription, ScoreResult
from app.scoring.ats import compute_ats_score
from app.scoring.explain import add_highlights

logger = logging.getLogger(__name__)


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    jd_text: str = "",
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
        file_bytes: Raw bytes of the uploaded résumé file.
        filename:   Original filename (e.g. ``"resume.pdf"``).
        jd_text:    Optional raw job-description text; empty string if not provided.

    Returns:
        A fully populated :class:`ScoreResult`.
    """
    timings: dict[str, float] = {}
    logger.info("Pipeline start | file=%s | device=%s", filename, settings.device)

    # ── Stage 1: Parsing ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    parsed = parse_document(file_bytes, filename)
    timings["parsing_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 2: Extraction ────────────────────────────────────────────────
    t0 = time.perf_counter()
    parsed = extract_entities(parsed)
    timings["extraction_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 3: Classification ────────────────────────────────────────────
    t0 = time.perf_counter()
    categories = classify_resume(parsed)
    timings["classification_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 4: JD Matching ───────────────────────────────────────────────
    t0 = time.perf_counter()
    jd: JobDescription | None = None
    if jd_text.strip():
        jd = JobDescription(text=jd_text, title="Job Description")
    match_result = match_resume_jd(parsed, jd)
    timings["matching_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 5: ATS Scoring ───────────────────────────────────────────────
    t0 = time.perf_counter()
    score_result = compute_ats_score(parsed, match_result)
    score_result.predicted_categories = categories
    timings["scoring_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 6: Explainability ────────────────────────────────────────────
    t0 = time.perf_counter()
    score_result = add_highlights(score_result, parsed)
    timings["explain_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 7: Generative Suggestions ────────────────────────────────────
    t0 = time.perf_counter()
    score_result = generate_suggestions(score_result, parsed, jd_text)
    timings["genai_ms"] = (time.perf_counter() - t0) * 1000

    score_result.timings_ms = timings
    logger.info("Pipeline complete | total=%.2f | grade=%s", score_result.total, score_result.grade)

    return score_result
