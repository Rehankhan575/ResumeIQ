"""
app/scoring/ats.py
==================
Stage 5 — ATS Composite Scorer.

Computes a transparent weighted score (0–100) from:
  - semantic_match        (weight: config.weight_semantic_match)
  - keyword_coverage      (weight: config.weight_keyword_coverage)
  - section_completeness  (weight: config.weight_section_completeness)
  - formatting            (weight: config.weight_formatting)
  - quantified_achievements (weight: config.weight_quantified_achievements)
  - action_verbs          (weight: config.weight_action_verbs)
  - readability           (weight: config.weight_readability)
  - length                (weight: config.weight_length)

No ground-truth "ATS score" exists in any dataset; this is a designed,
explainable composite. Each component is validated separately.
"""

from __future__ import annotations

import logging

from app.schemas import Issue, MatchResult, ParsedResume, ScoreResult

logger = logging.getLogger(__name__)


def compute_ats_score(
    parsed: ParsedResume,
    match: MatchResult | None,
) -> ScoreResult:
    """Compute the composite ATS score for a résumé.

    This is currently a **stub** returning dummy score data.
    Real implementation: each sub-component computed and weighted per
    ``app/config.py`` weights; issues detected via formatting heuristics,
    action-verb lists, and readability metrics (textstat).

    Args:
        parsed: Enriched :class:`ParsedResume`.
        match:  Optional :class:`MatchResult` from the matcher stage.

    Returns:
        A partial :class:`ScoreResult` with ``total``, ``grade``,
        ``components``, ``missing_skills``, and ``issues`` populated.
        ``highlights`` and ``suggestions`` are filled by later stages.
    """
    logger.debug("compute_ats_score called")

    components = {
        "semantic_match": 75.0,
        "keyword_coverage": 65.0,
        "section_completeness": 80.0,
        "formatting": 90.0,
        "quantified_achievements": 60.0,
        "action_verbs": 70.0,
        "readability": 78.0,
        "length": 85.0,
    }

    # Placeholder weighted average (weights will come from config in real impl)
    total = sum(components.values()) / len(components)

    issues = [
        Issue(
            component="quantified_achievements",
            message="Only 1 of 2 bullet points contain quantified results.",
            severity="medium",
            estimated_gain=5.0,
        ),
        Issue(
            component="keyword_coverage",
            message="Missing keywords: Kubernetes, Spark.",
            severity="high",
            estimated_gain=8.0,
        ),
    ]

    return ScoreResult(
        total=round(total, 2),
        components=components,
        missing_skills=match.missing_skills if match else [],
        issues=issues,
        match=match,
    )
