"""
app/scoring/explain.py
======================
Stage 6 — Explainability & Attribution.

Generates character-level :class:`Highlight` spans showing which words, phrases,
and sentences raised or lowered the ATS and match scores.

Methods used:
  1. Sentence contributions: Leave-One-Sentence-Out (LOSO) against the semantic match score.
     Capped at 40 sentences per resume and batched for sub-second CPU latency.
  2. Classifier word importance: TF-IDF + Logistic Regression linear coefficient attribution.
     (Explicitly chosen over SHAP/LIME for deterministic sub-10ms CPU execution).
  3. Keyword matching: Character spans for matched JD skills and zero-span entries for missing skills.
  4. HTML Rendering: Safe HTML rendering with escaping of raw text and HTML tags.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.classifier import classify, load_classifier
from app.parsing.cleaner import clean_text
from app.schemas import Highlight, JobDescription, ParsedResume, ScoreResult

logger = logging.getLogger(__name__)

# Explainability method declaration
CLASSIFIER_XAI_METHOD: str = "TF-IDF + Logistic Regression linear coefficient attribution"

LR_MODEL_PATH: Path = Path("artifacts/models/classifier/tfidf_lr.joblib")
VEC_MODEL_PATH: Path = Path("artifacts/models/classifier/tfidf_vectorizer.joblib")
CLASSES_PATH: Path = Path("artifacts/models/classifier/classes.json")
_LR_CACHE: Dict[str, Any] = {}


def _get_lr_classifier() -> Optional[Dict[str, Any]]:
    """Load cached TF-IDF + Logistic Regression classifier for feature attribution."""
    if "info" in _LR_CACHE:
        return _LR_CACHE["info"]

    if LR_MODEL_PATH.exists() and VEC_MODEL_PATH.exists() and CLASSES_PATH.exists():
        try:
            import json
            import joblib
            clf = joblib.load(LR_MODEL_PATH)
            vec = joblib.load(VEC_MODEL_PATH)
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                classes = json.load(f)
            info = {"model": clf, "vectorizer": vec, "classes": classes}
            _LR_CACHE["info"] = info
            return info
        except Exception as e:
            logger.warning("Failed to load LR model for XAI: %s", e)

    # Fall back to general classifier info
    gen_info = load_classifier()
    if gen_info and hasattr(gen_info.get("model"), "coef_"):
        _LR_CACHE["info"] = gen_info
        return gen_info

    return None


def _split_into_sentences(text: str, max_sentences: int = 40) -> List[Tuple[int, int, str]]:
    """Split resume text into non-empty sentences with precise character offsets.

    Args:
        text: Raw resume text.
        max_sentences: Maximum number of sentence spans to extract (default: 40).

    Returns:
        List of (start_char, end_char, sentence_text) tuples where
        ``text[start_char:end_char] == sentence_text``.
    """
    if not text:
        return []

    spans: List[Tuple[int, int, str]] = []
    # Match sentences ending with punctuation or double-newlines
    pattern = re.compile(r"[^\n.!?]+(?:[.!?]+|\n+|$)", re.UNICODE)

    for match in pattern.finditer(text):
        raw_slice = match.group()
        s_strip = raw_slice.strip()
        # Skip very short fragments
        if len(s_strip) < 8 or len(s_strip.split()) < 2:
            continue

        l_trim = len(raw_slice) - len(raw_slice.lstrip())
        r_trim = len(raw_slice) - len(raw_slice.rstrip())
        start = match.start() + l_trim
        end = match.end() - r_trim

        if start < end:
            spans.append((start, end, text[start:end]))
            if len(spans) >= max_sentences:
                break

    return spans


def _snap_to_word_boundary(text: str, start: int, end: int) -> Tuple[int, int]:
    """Snap character offsets to clean whole-word boundaries."""
    if not text:
        return start, end
    s = max(0, min(start, len(text)))
    e = max(s, min(end, len(text)))
    if s == e:
        return s, e

    # Expand left while preceding char is alphanumeric or underscore/hyphen
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] in "_-"):
        s -= 1

    # Expand right while current char is alphanumeric or underscore/hyphen
    while e < len(text) and (text[e].isalnum() or text[e] in "_-"):
        e += 1

    # Strip leading/trailing hyphens or underscores
    while s < e and text[s] in "-_":
        s += 1
    while e > s and text[e - 1] in "-_":
        e -= 1

    return s, e


def compute_sentence_contributions(
    text: str,
    jd_text: str,
    max_sentences: int = 40,
) -> List[Highlight]:
    """Compute sentence importance using batched Leave-One-Sentence-Out (LOSO).

    Args:
        text: Raw resume text.
        jd_text: Job description text.
        max_sentences: Cap on sentences evaluated (default: 40).

    Returns:
        List of :class:`Highlight` objects with type 'positive' or 'negative'.
    """
    if not text.strip() or not jd_text.strip():
        return []

    sentence_spans = _split_into_sentences(text, max_sentences=max_sentences)
    if not sentence_spans:
        return []

    # Fit TF-IDF on full resume and JD
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        vec.fit([text, jd_text])
        jd_vec = vec.transform([jd_text])
        full_vec = vec.transform([text])
        base_sim = float(cosine_similarity(full_vec, jd_vec)[0][0])
    except Exception as e:
        logger.debug("LOSO TF-IDF fit failed: %s", e)
        return []

    # Construct all leave-one-out texts for a single batch transformation
    loo_texts = []
    for start, end, _ in sentence_spans:
        loo_text = text[:start] + " " + text[end:]
        loo_texts.append(loo_text)

    try:
        loo_mat = vec.transform(loo_texts)
        loo_sims = cosine_similarity(loo_mat, jd_vec).flatten()
    except Exception as e:
        logger.debug("LOSO batch scoring failed: %s", e)
        return []

    highlights: List[Highlight] = []
    for (start, end, s_text), loo_sim in zip(sentence_spans, loo_sims):
        delta = float(base_sim - loo_sim)
        # Positive delta means score dropped when sentence was removed (sentence helped score)
        if abs(delta) >= 0.005:
            h_type = "positive" if delta > 0 else "negative"
            highlights.append(
                Highlight(
                    start=start,
                    end=end,
                    text=s_text,
                    weight=round(delta, 4),
                    type=h_type,
                )
            )

    return highlights


def compute_classifier_word_importance(
    text: str,
    top_n_words: int = 15,
) -> List[Highlight]:
    """Compute word attribution using TF-IDF + Logistic Regression coefficients.

    Method:
      TF-IDF + Logistic Regression linear coefficient attribution.
      Fast, deterministic, sub-10ms execution on CPU (SHAP/LIME omitted per performance budget).

    Args:
        text: Resume raw text.
        top_n_words: Number of top influential words to extract.

    Returns:
        List of :class:`Highlight` spans with positive or negative weights.
    """
    if not text.strip():
        return []

    try:
        info = _get_lr_classifier()
        if not info:
            return []
        vec = info.get("vectorizer")
        model = info.get("model")
        classes = info.get("classes", [])

        if not vec or not model or not hasattr(model, "coef_"):
            return []

        # Predict class
        preds = classify(text, top_k=1)
        if not preds:
            return []
        top_cat = preds[0].category
        if top_cat not in classes:
            return []
        cat_idx = classes.index(top_cat)

        feat = vec.transform([text])
        coefs = model.coef_[cat_idx]
        feature_names = vec.get_feature_names_out()

        # Non-zero words in document
        nonzeros = feat.nonzero()[1]
        word_scores: List[Tuple[str, float]] = []
        for idx in nonzeros:
            word = feature_names[idx]
            val = float(feat[0, idx] * coefs[idx])
            if abs(val) > 0.01:
                word_scores.append((word, val))

        # Sort by absolute contribution
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_words = word_scores[:top_n_words]

        highlights: List[Highlight] = []
        for word, weight in top_words:
            # Locate all occurrences of this word in text
            for match in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                start, end = match.start(), match.end()
                start, end = _snap_to_word_boundary(text, start, end)
                matched_str = text[start:end]
                highlights.append(
                    Highlight(
                        start=start,
                        end=end,
                        text=matched_str,
                        weight=round(weight, 4),
                        type="positive" if weight > 0 else "negative",
                    )
                )

        return highlights
    except Exception as e:
        logger.debug("Classifier word attribution failed: %s", e)
        return []


def compute_keyword_highlights(
    text: str,
    jd: JobDescription | None,
    matched_skills: List[str] | None = None,
    missing_skills: List[str] | None = None,
) -> List[Highlight]:
    """Find character spans for matched keywords and record missing required keywords.

    Args:
        text: Resume raw text.
        jd: Optional JobDescription.
        matched_skills: Explicit list of matched skills.
        missing_skills: Explicit list of missing skills.

    Returns:
        List of :class:`Highlight` spans.
    """
    highlights: List[Highlight] = []
    if jd is None and not matched_skills:
        return highlights

    matched_set = set(matched_skills or [])
    missing_set = set(missing_skills or [])

    if jd is not None and not matched_set and not missing_set:
        text_lower = text.lower()
        for s in jd.required_skills:
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower):
                matched_set.add(s)
            else:
                missing_set.add(s)

    # 1. Matched skills in resume text
    for skill in matched_set:
        for match in re.finditer(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
            s_snap, e_snap = _snap_to_word_boundary(text, match.start(), match.end())
            highlights.append(
                Highlight(
                    start=s_snap,
                    end=e_snap,
                    text=text[s_snap:e_snap],
                    weight=1.0,
                    type="keyword_match",
                )
            )

    # 2. Missing skills (character offset 0-0)
    for skill in missing_set:
        highlights.append(
            Highlight(
                start=0,
                end=0,
                text=skill,
                weight=-1.0,
                type="keyword_missing",
            )
        )

    return highlights


def render_html(highlights: List[Highlight], text: str) -> str:
    """Render resume text with explainability highlight spans escaped for safe HTML display.

    All original text is HTML-escaped. Non-overlapping spans are highlighted with CSS classes.

    Args:
        highlights: List of :class:`Highlight` spans.
        text: Original raw resume text.

    Returns:
        HTML string with <mark> tags and escaped text.
    """
    if not text:
        return ""

    # Filter out zero-span highlights (such as keyword_missing) and invalid offsets
    valid_spans: List[Tuple[int, int, str, float, str]] = []
    for h in highlights:
        if h.type == "keyword_missing":
            continue
        s, e = _snap_to_word_boundary(text, h.start, h.end)
        if 0 <= s < e <= len(text):
            valid_spans.append((s, e, text[s:e], h.weight, h.type))

    if not valid_spans:
        return html.escape(text)

    # Priority ranking for overlapping spans: keyword_match > positive > negative
    type_priority = {"keyword_match": 3, "positive": 2, "negative": 1}
    valid_spans.sort(
        key=lambda x: (x[0], -type_priority.get(x[4], 0), -(x[1] - x[0]))
    )

    # De-overlap spans (greedy interval selection)
    non_overlapping: List[Tuple[int, int, str, float, str]] = []
    current_end = 0
    for span in valid_spans:
        start, end, s_text, weight, s_type = span
        if start >= current_end:
            non_overlapping.append(span)
            current_end = end

    # Assemble HTML string
    parts: List[str] = []
    last_idx = 0
    for start, end, s_text, weight, s_type in non_overlapping:
        if start > last_idx:
            parts.append(html.escape(text[last_idx:start]))

        css_class = f"ats-highlight ats-highlight-{s_type}"
        escaped_content = html.escape(s_text)
        parts.append(
            f'<mark class="{css_class}" data-type="{s_type}" data-weight="{weight:.4f}">'
            f"{escaped_content}</mark>"
        )
        last_idx = end

    if last_idx < len(text):
        parts.append(html.escape(text[last_idx:]))

    return "".join(parts)


def add_highlights(
    score_result: ScoreResult,
    parsed: ParsedResume,
    jd: JobDescription | None = None,
) -> ScoreResult:
    """Attach explainability highlights to a :class:`ScoreResult`.

    Combines:
      - Sentence-level LOSO contributions
      - Classifier linear attribution words
      - Keyword matches and missing skills

    Args:
        score_result: :class:`ScoreResult` from ATS scorer.
        parsed: Enriched :class:`ParsedResume`.
        jd: Optional :class:`JobDescription`.

    Returns:
        The updated :class:`ScoreResult` with populated highlights.
    """
    logger.debug("add_highlights called")
    text = parsed.raw_text

    all_highlights: List[Highlight] = []

    # 1. Keyword highlights
    matched = score_result.match.matched_skills if score_result.match else []
    missing = score_result.missing_skills
    kw_highlights = compute_keyword_highlights(text, jd, matched_skills=matched, missing_skills=missing)
    all_highlights.extend(kw_highlights)

    # 2. Sentence contributions (LOSO)
    jd_text = jd.text if jd else ""
    if jd_text:
        sent_highlights = compute_sentence_contributions(text, jd_text, max_sentences=40)
        all_highlights.extend(sent_highlights)

    # 3. Classifier word importance
    clf_highlights = compute_classifier_word_importance(text, top_n_words=10)
    all_highlights.extend(clf_highlights)

    # 4. Fallback to extracted skills if no highlights generated yet
    if not all_highlights and parsed.skills:
        for skill in parsed.skills:
            for match in re.finditer(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
                all_highlights.append(
                    Highlight(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        weight=1.0,
                        type="keyword_match",
                    )
                )

    score_result.highlights = all_highlights
    return score_result
