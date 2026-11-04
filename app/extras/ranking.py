"""
app/extras/ranking.py
=====================
Multi-job description ranking for candidate résumés.

Ranks a candidate résumé across multiple job descriptions using ATS composite scores
(or match scores if ATS scoring is unavailable).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from app.extraction.extractor import extract_entities
from app.parsing.parser import parse_document
from app.schemas import JobDescription, ParsedResume, ScoreResult

logger = logging.getLogger(__name__)


def _ensure_parsed_resume(resume: Union[ParsedResume, bytes, str], filename: str = "resume.pdf") -> ParsedResume:
    """Ensure input is an enriched ParsedResume object."""
    if isinstance(resume, ParsedResume):
        return resume
    if isinstance(resume, bytes):
        parsed = parse_document(resume, filename)
        return extract_entities(parsed)
    if isinstance(resume, str):
        # Treat as raw text
        parsed = ParsedResume(raw_text=resume)
        return extract_entities(parsed)
    raise ValueError(f"Unsupported resume input type: {type(resume)}")


def _ensure_jd(jd: Union[JobDescription, Dict[str, Any], str], default_title: str = "Target Job") -> JobDescription:
    """Ensure input is a JobDescription object."""
    if isinstance(jd, JobDescription):
        return jd
    if isinstance(jd, dict):
        return JobDescription(
            text=jd.get("text", ""),
            title=jd.get("title", default_title),
            required_skills=jd.get("required_skills", []),
            preferred_skills=jd.get("preferred_skills", []),
            min_years=jd.get("min_years", 0.0),
        )
    if isinstance(jd, str):
        return JobDescription(text=jd, title=default_title)
    raise ValueError(f"Unsupported JD input type: {type(jd)}")


def rank_jds(
    resume: Union[ParsedResume, bytes, str],
    jds: List[Union[JobDescription, Dict[str, Any], str]],
) -> List[Dict[str, Any]]:
    """Rank multiple job descriptions against a single candidate résumé.

    Sorted descending by ATS composite total score (or match score if ATS scoring fails).

    Args:
        resume: :class:`ParsedResume`, raw bytes, or text string.
        jds: List of :class:`JobDescription` objects, dicts, or raw texts.

    Returns:
        List of dicts: ``{"jd_title", "score", "grade", "missing_skills", "rank"}``
        sorted descending by score.
    """
    if not jds:
        return []

    parsed_resume = _ensure_parsed_resume(resume)
    ranked_results: List[Dict[str, Any]] = []

    # Import ATS scoring and Matcher public functions
    try:
        from app.scoring.ats import score_resume
    except ImportError:
        score_resume = None

    try:
        from app.models.matcher import match
    except ImportError:
        match = None

    for idx, raw_jd in enumerate(jds):
        jd_obj = _ensure_jd(raw_jd, default_title=f"Job {idx + 1}")
        title = jd_obj.title or f"Job {idx + 1}"

        score = 0.0
        grade = "N/A"
        missing_skills: List[str] = []

        # 1. Try ATS composite scoring
        scored = False
        if score_resume is not None:
            try:
                res: ScoreResult = score_resume(parsed_resume, jd_obj)
                score = res.total
                grade = res.grade
                missing_skills = list(res.missing_skills)
                scored = True
            except Exception as exc:
                logger.debug("score_resume failed during ranking: %s", exc)

        # 2. Fall back to matcher if ATS scoring failed
        if not scored and match is not None:
            try:
                m_res = match(parsed_resume.raw_text, jd_obj.text)
                if isinstance(m_res, dict):
                    score = round(float(m_res.get("score", 0.0)) * 100.0, 2)
                    missing_skills = list(m_res.get("missing_skills", []))
                    grade = "A" if score >= 80 else ("B" if score >= 65 else "C")
                    scored = True
            except Exception as exc:
                logger.debug("matcher.match failed during ranking: %s", exc)

        # 3. Final Fallback: Simple TF-IDF cosine
        if not scored:
            msg = "Falling back to simple TF-IDF cosine for JD ranking."
            logger.info(msg)
            print(msg)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(stop_words="english")
            X = vec.fit_transform([parsed_resume.raw_text, jd_obj.text])
            sim = float(cosine_similarity(X[0:1], X[1:2])[0][0])
            score = round(min(100.0, max(0.0, sim * 150.0)), 2)
            grade = "Pass" if score >= 50 else "Review"

        ranked_results.append({
            "jd_title": title,
            "score": score,
            "grade": grade,
            "missing_skills": missing_skills,
        })

    # Sort descending by score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # Assign 1-based rank
    for rank_idx, item in enumerate(ranked_results, 1):
        item["rank"] = rank_idx

    return ranked_results
