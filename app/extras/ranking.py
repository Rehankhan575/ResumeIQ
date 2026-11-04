"""
app/extras/ranking.py
=====================
Multi-job description ranking for candidate résumés.

Ranks a candidate résumé across multiple job descriptions using ATS composite scores
(or match scores if ATS scoring is unavailable).

Bug fix (item 0): JDs with empty required_skills previously received keyword coverage
100%, inflating every score by +20 and masking missing skills.  ``rank_jds`` now calls
``extract_jd_skills`` on the raw JD text whenever ``required_skills`` is empty so that
real required-skill lists are always populated before scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from app.extraction.extractor import extract_entities, get_skill_extractor
from app.parsing.parser import parse_document
from app.schemas import JobDescription, ParsedResume, ScoreResult

logger = logging.getLogger(__name__)


def _populate_jd_skills(jd: JobDescription) -> JobDescription:
    """Ensure a JobDescription has required_skills populated from its text.

    If ``jd.required_skills`` is empty and ``jd.text`` is non-empty, calls
    :func:`~app.extraction.skills.TaxonomySkillExtractor.extract_jd_skills` on
    the raw JD text and fills in ``required_skills`` and ``preferred_skills``.

    Args:
        jd: A :class:`JobDescription` whose ``required_skills`` may be empty.

    Returns:
        The same object, with ``required_skills`` populated when previously empty.
    """
    if not jd.required_skills and jd.text.strip():
        try:
            extractor = get_skill_extractor()
            extracted = extractor.extract_jd_skills(jd.text)
            # Use model_copy to avoid mutating shared objects
            return jd.model_copy(update={
                "required_skills": extracted.get("required_skills", []),
                "preferred_skills": extracted.get("preferred_skills", []) or jd.preferred_skills,
                "min_years": jd.min_years if jd.min_years is not None else extracted.get("min_years"),
            })
        except Exception as exc:
            logger.debug("extract_jd_skills failed during ranking populate: %s", exc)
    return jd


# Fix (item 0): Ensure every path that scores a JD runs extract_jd_skills on the JD text
# whenever required_skills is empty. This wraps app.scoring.ats.score_resume transparently
# so that standalone scoring paths outside rank_jds also populate missing required_skills.
try:
    import app.scoring.ats as _ats_mod
    if hasattr(_ats_mod, "score_resume") and not getattr(_ats_mod, "_jd_skills_patched", False):
        _orig_ats_score_resume = _ats_mod.score_resume
        _orig_ats_compute_ats_score = getattr(_ats_mod, "compute_ats_score", None)

        def _patched_score_resume(
            parsed_resume: Any,
            jd: Optional[JobDescription] = None,
            match: Any = None,
        ) -> Any:
            if isinstance(jd, JobDescription) and not jd.required_skills and jd.text.strip():
                jd = _populate_jd_skills(jd)
            return _orig_ats_score_resume(parsed_resume, jd=jd, match=match)

        def _patched_compute_ats_score(
            parsed: Any,
            match: Any = None,
            jd: Optional[JobDescription] = None,
        ) -> Any:
            if isinstance(jd, JobDescription) and not jd.required_skills and jd.text.strip():
                jd = _populate_jd_skills(jd)
            if callable(_orig_ats_compute_ats_score):
                return _orig_ats_compute_ats_score(parsed, match=match, jd=jd)
            return _orig_ats_score_resume(parsed, jd=jd, match=match)

        _ats_mod.score_resume = _patched_score_resume
        if hasattr(_ats_mod, "compute_ats_score"):
            _ats_mod.compute_ats_score = _patched_compute_ats_score
        setattr(_ats_mod, "_jd_skills_patched", True)
except Exception as _patch_err:
    logger.debug("Could not wrap ats.score_resume: %s", _patch_err)


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


def _ensure_jd(jd: Union[JobDescription, Dict[str, Any], str], default_title: str = "Target Job") -> JobDescription:  # noqa: E501
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
        # Fix (item 0): populate required_skills from JD text when empty so that
        # keyword coverage is never inflated to 100% for skill-free JD objects.
        jd_obj = _populate_jd_skills(jd_obj)
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

