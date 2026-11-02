"""
app/models/matcher.py
=====================
Stage 4 — Résumé–JD Semantic Matching.

Uses a Siamese Sentence-BERT (all-MiniLM-L6-v2) fine-tuned with triplet/contrastive loss,
combined with keyword overlap and skill-gap analysis.
Evaluation: MRR, nDCG@10 on résumé–JD pairs (ranx).
"""

from __future__ import annotations

import logging

from app.schemas import JobDescription, MatchResult, ParsedResume

logger = logging.getLogger(__name__)


def match_resume_jd(
    parsed: ParsedResume,
    jd: JobDescription | None,
) -> MatchResult:
    """Compute a semantic and keyword-based match score between résumé and JD.

    This is currently a **stub** returning dummy match data.
    Real implementation: Siamese SBERT cosine similarity + TF-IDF baseline
    (for comparison), plus skill-gap computation against ESCO taxonomy.

    Args:
        parsed: Enriched :class:`ParsedResume`.
        jd:     Optional :class:`JobDescription`; returns a neutral result if None.

    Returns:
        A :class:`MatchResult` with overall score, per-section scores,
        matched skills, and missing skills.
    """
    logger.debug("match_resume_jd called (jd=%s)", jd.title if jd else "None")

    if jd is None:
        return MatchResult(
            score=0.0,
            section_scores={},
            matched_skills=[],
            missing_skills=[],
            method="none",
        )

    return MatchResult(
        score=0.73,
        section_scores={
            "skills": 0.80,
            "experience": 0.70,
            "education": 0.65,
        },
        matched_skills=["Python", "Machine Learning"],
        missing_skills=["Kubernetes", "Spark"],
        method="sbert_cosine",
    )
