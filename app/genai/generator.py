"""
app/genai/generator.py
======================
Stage 7 — Generative Feedback.

Rewrites weak bullet points, generates cover-letter drafts, and suggests
missing skills as improvement plans. Backends:
  - Local: fine-tuned Flan-T5 (google/flan-t5-base)
  - API: OpenAI / Google Gemini (controlled by ``config.use_llm_api``)
  - Fallback: rule-based templates (always available)
"""

from __future__ import annotations

import logging

from app.schemas import ParsedResume, ScoreResult, Suggestion

logger = logging.getLogger(__name__)


def generate_suggestions(
    score_result: ScoreResult,
    parsed: ParsedResume,
    jd_text: str = "",
) -> ScoreResult:
    """Add generative improvement suggestions to a :class:`ScoreResult`.

    This is currently a **stub** returning dummy suggestions.
    Real implementation: prompt Flan-T5 (or LLM API) with the weak bullet,
    JD context, and a structured prompt template; fall back to rule-based
    rewrites if the model is unavailable.

    Args:
        score_result: :class:`ScoreResult` from the explain stage.
        parsed:       Enriched :class:`ParsedResume`.
        jd_text:      Raw JD text for context (empty string if no JD provided).

    Returns:
        The :class:`ScoreResult` with ``suggestions`` list populated.
    """
    logger.debug("generate_suggestions called (jd_text length=%d)", len(jd_text))

    score_result.suggestions = [
        Suggestion(
            kind="bullet_rewrite",
            original="Worked on ML pipeline.",
            suggested=(
                "Engineered an end-to-end ML inference pipeline using PyTorch and FastAPI, "
                "reducing p99 latency by 40% and cutting cloud compute costs by $12k/month."
            ),
            notes="Added quantified impact and action verb.",
        ),
        Suggestion(
            kind="improvement_plan",
            original="",
            suggested=(
                "Add Kubernetes and Apache Spark to your skills section and include a project "
                "demonstrating distributed data processing to close the skill gap for this role."
            ),
            notes="Based on missing_skills: ['Kubernetes', 'Spark'].",
        ),
    ]

    return score_result
