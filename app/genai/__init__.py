"""
app/genai
=========
Stage 7 — Generative AI feedback, bullet rewriting, cover letter drafting,
and grounding verification.
"""

from app.genai.generator import (
    generate_cover_letter,
    generate_suggestions,
    improvement_plan,
    rewrite_bullets,
)
from app.genai.grounding import GroundingResult, check_grounding
from app.genai.provider import call_llm, get_active_provider

__all__ = [
    "rewrite_bullets",
    "generate_cover_letter",
    "improvement_plan",
    "check_grounding",
    "generate_suggestions",
    "call_llm",
    "get_active_provider",
    "GroundingResult",
]
