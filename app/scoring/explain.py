"""
app/scoring/explain.py
======================
Stage 6 — Explainability.

Generates character-level :class:`Highlight` spans showing which words/phrases
raised or lowered the ATS score. Backends: SHAP, LIME, attention heatmaps (captum).
"""

from __future__ import annotations

import logging

from app.schemas import Highlight, ParsedResume, ScoreResult

logger = logging.getLogger(__name__)


def add_highlights(
    score_result: ScoreResult,
    parsed: ParsedResume,
) -> ScoreResult:
    """Attach explainability highlights to a :class:`ScoreResult`.

    This is currently a **stub** returning dummy highlights.
    Real implementation: SHAP values over TF-IDF features for keyword
    component; attention rollout from DistilBERT for semantic component;
    LIME perturbation for individual sentence contributions.

    Args:
        score_result: Partial :class:`ScoreResult` from the ATS scorer.
        parsed:       Enriched :class:`ParsedResume` (raw_text used for offsets).

    Returns:
        The :class:`ScoreResult` with ``highlights`` list populated.
    """
    logger.debug("add_highlights called")

    score_result.highlights = [
        Highlight(
            start=0,
            end=6,
            text="Python",
            weight=0.15,
            type="keyword_match",
        ),
        Highlight(
            start=8,
            end=15,
            text="PyTorch",
            weight=0.12,
            type="positive",
        ),
        Highlight(
            start=16,
            end=19,
            text="SQL",
            weight=0.05,
            type="keyword_match",
        ),
        Highlight(
            start=50,
            end=60,
            text="Kubernetes",
            weight=-0.10,
            type="keyword_missing",
        ),
    ]

    return score_result
