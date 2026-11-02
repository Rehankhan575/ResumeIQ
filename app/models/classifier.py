"""
app/models/classifier.py
========================
Stage 3 — Resume Category Classifier.

Baseline: TF-IDF + MLP.
Improved: BiLSTM.
Best: Fine-tuned BERT (distilbert-base-uncased).

Returns a ranked list of :class:`CategoryPrediction` objects.
"""

from __future__ import annotations

import logging

from app.schemas import CategoryPrediction, ParsedResume

logger = logging.getLogger(__name__)


def classify_resume(parsed: ParsedResume) -> list[CategoryPrediction]:
    """Predict the résumé's job category.

    This is currently a **stub** returning dummy predictions.
    Real implementation: fine-tuned DistilBERT classifier trained on the
    Kaggle LiveCareer resume dataset (~2,400 samples, 25 categories).

    Args:
        parsed: Enriched :class:`ParsedResume` from extraction stage.

    Returns:
        List of :class:`CategoryPrediction` sorted by probability descending.
    """
    logger.debug("classify_resume called")

    return [
        CategoryPrediction(category="Data Science", probability=0.72),
        CategoryPrediction(category="Software Engineering", probability=0.20),
        CategoryPrediction(category="Machine Learning", probability=0.08),
    ]
