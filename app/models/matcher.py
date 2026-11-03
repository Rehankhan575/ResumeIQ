"""
app/models/matcher.py
=====================
Stage 4 — Résumé–JD Semantic Matching.

Implements semantic matching between résumés and job descriptions using:
1. Fine-tuned Siamese Sentence-BERT (all-MiniLM-L6-v2) bi-encoder.
2. Zero-shot Sentence-BERT fallback.
3. TF-IDF cosine similarity fallback when neural models are unavailable.

Provides section-wise similarity:
- Skills similarity: Overlap between candidate skills and JD required skills (via app.extraction).
- Experience similarity: Semantic similarity between candidate experience section and JD text.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.config import settings
from app.extraction.skills import TaxonomySkillExtractor
from app.parsing.cleaner import clean_text
from app.parsing.sections import split_sections
from app.schemas import JobDescription, MatchResult, ParsedResume

logger = logging.getLogger(__name__)

# Default model directory
MATCHER_DIR = settings.models_dir / "matcher"
FINE_TUNED_MODEL_PATH = MATCHER_DIR / "model"
TFIDF_MATCHER_PATH = MATCHER_DIR / "tfidf_matcher.pkl"

# Global singletons for cached models
_SENTENCE_MODEL: Any = None
_TFIDF_MODEL: Optional[Dict[str, Any]] = None
_SKILL_EXTRACTOR: Optional[TaxonomySkillExtractor] = None


def get_skill_extractor() -> TaxonomySkillExtractor:
    """Return cached TaxonomySkillExtractor instance."""
    global _SKILL_EXTRACTOR
    if _SKILL_EXTRACTOR is None:
        _SKILL_EXTRACTOR = TaxonomySkillExtractor()
    return _SKILL_EXTRACTOR


def load_sentence_model(model_path: Optional[Path] = None) -> Tuple[Any, str]:
    """Load the best available SentenceTransformer model.

    Tries:
        1. Fine-tuned model at specified path or default artifacts/models/matcher/model/
        2. Zero-shot all-MiniLM-L6-v2 base model
        3. None (triggers TF-IDF fallback)

    Returns:
        Tuple of (model_instance, method_name).
    """
    global _SENTENCE_MODEL
    target_path = model_path or FINE_TUNED_MODEL_PATH

    # Check if fine-tuned checkpoint exists
    if target_path.exists() and (target_path / "modules.json").exists():
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading fine-tuned matcher model from %s", target_path)
            model = SentenceTransformer(str(target_path), device=settings.device)
            model.max_seq_length = 256
            return model, "sbert_fine_tuned"
        except Exception as exc:
            logger.warning("Failed to load fine-tuned matcher model (%s); falling back", exc)

    # Fallback to zero-shot base model
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading zero-shot base model: %s", settings.sbert_model_name)
        model = SentenceTransformer(settings.sbert_model_name, device=settings.device)
        model.max_seq_length = 256
        return model, "sbert_zero_shot"
    except Exception as exc:
        logger.warning(
            "Failed to load SentenceTransformer base model (%s); falling back to TF-IDF",
            exc,
        )
        return None, "tfidf_cosine"


def load_tfidf_model() -> Optional[Dict[str, Any]]:
    """Load cached TF-IDF vectorizer if available on disk."""
    global _TFIDF_MODEL
    if _TFIDF_MODEL is not None:
        return _TFIDF_MODEL

    if TFIDF_MATCHER_PATH.exists():
        try:
            with open(TFIDF_MATCHER_PATH, "rb") as f:
                _TFIDF_MODEL = pickle.load(f)
                return _TFIDF_MODEL
        except Exception as exc:
            logger.warning("Failed to load TF-IDF matcher cache: %s", exc)

    return None


def _compute_cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D vectors in range [0, 1]."""
    v1 = np.asarray(vec1).flatten()
    v2 = np.asarray(vec2).flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos = float(np.dot(v1, v2) / (norm1 * norm2))
    # Bound to [0.0, 1.0]
    return max(0.0, min(1.0, (cos + 1.0) / 2.0 if cos < 0 else cos))


def _extract_experience_text(resume_text: str) -> str:
    """Extract experience section text from a résumé string."""
    cleaned = clean_text(resume_text)
    sections = split_sections(cleaned)
    if "experience" in sections and sections["experience"].strip():
        return sections["experience"]
    # Fallback: look for common experience keywords
    m = re.search(r"(?:experience|work\s*history|employment)[\s:]+(.*?)(?=\n[A-Z][a-z]+|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return cleaned


def match(
    resume_text: str,
    jd_text: str,
    model_path: Optional[Path] = None,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """Compute match score and section-wise similarities between résumé and JD.

    Compares:
        - Overall semantic similarity
        - Section similarity:
            * skills: Overlap between résumé skills and JD required skills
            * experience: Semantic similarity between candidate experience and JD text

    Args:
        resume_text: Full raw or parsed text of the résumé.
        jd_text: Raw text of the job description.
        model_path: Optional path to fine-tuned SentenceTransformer model.
        force_fallback: If True, forces TF-IDF fallback for testing.

    Returns:
        Dictionary with keys:
            - score (float 0.0-1.0)
            - section_similarity (dict with 'skills' and 'experience' floats)
            - matched_skills (list of matched skill strings)
            - missing_skills (list of missing required skill strings)
            - method (str)
    """
    if not resume_text.strip() or not jd_text.strip():
        return {
            "score": 0.0,
            "section_similarity": {"skills": 0.0, "experience": 0.0},
            "matched_skills": [],
            "missing_skills": [],
            "method": "none",
        }

    # 1. Skills Extraction & Comparison (via app.extraction)
    extractor = get_skill_extractor()
    resume_skills = set(extractor.extract_canonical_skills(resume_text, section_name="skills"))

    jd_info = extractor.extract_jd_skills(jd_text)
    jd_required = set(jd_info.get("required_skills", []))
    jd_preferred = set(jd_info.get("preferred_skills", []))

    all_target_skills = jd_required if jd_required else jd_preferred
    if not all_target_skills:
        # Fall back to extracting all skills from JD text
        all_target_skills = set(extractor.extract_canonical_skills(jd_text, section_name="skills"))

    matched_skills = sorted(resume_skills.intersection(all_target_skills))
    missing_skills = sorted(all_target_skills - resume_skills)

    if all_target_skills:
        skills_sim = float(len(matched_skills) / len(all_target_skills))
    else:
        # Neutral skill overlap if JD mentions zero taxonomy skills
        skills_sim = 0.5

    # 2. Experience Section Semantic Comparison
    exp_text = _extract_experience_text(resume_text)

    # 3. Overall Semantic Similarity
    model = None
    method = "tfidf_cosine"

    if not force_fallback:
        model, method = load_sentence_model(model_path)

    if model is not None and method.startswith("sbert"):
        try:
            # Encode texts using SBERT
            emb_res = model.encode(resume_text[:2000], convert_to_numpy=True, show_progress_bar=False)
            emb_jd = model.encode(jd_text[:2000], convert_to_numpy=True, show_progress_bar=False)
            overall_score = _compute_cosine_sim(emb_res, emb_jd)

            # Experience similarity
            emb_exp = model.encode(exp_text[:2000], convert_to_numpy=True, show_progress_bar=False)
            exp_sim = _compute_cosine_sim(emb_exp, emb_jd)
        except Exception as exc:
            logger.warning("Neural encoding failed (%s); falling back to TF-IDF", exc)
            model = None
            method = "tfidf_cosine"

    if model is None or method == "tfidf_cosine":
        # TF-IDF Fallback
        method = "tfidf_cosine"
        tfidf_cache = load_tfidf_model()
        if tfidf_cache is not None and "vectorizer" in tfidf_cache:
            vec = tfidf_cache["vectorizer"]
            v_res = vec.transform([resume_text])
            v_jd = vec.transform([jd_text])
            v_exp = vec.transform([exp_text])
            overall_score = _compute_cosine_sim(v_res.toarray(), v_jd.toarray())
            exp_sim = _compute_cosine_sim(v_exp.toarray(), v_jd.toarray())
        else:
            # Dynamic TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(stop_words="english", max_features=1000)
            X = vec.fit_transform([resume_text, jd_text, exp_text])
            overall_score = _compute_cosine_sim(X[0].toarray(), X[1].toarray())
            exp_sim = _compute_cosine_sim(X[2].toarray(), X[1].toarray())

    # Bound overall score to [0.0, 1.0]
    final_score = round(float(np.clip(overall_score, 0.0, 1.0)), 4)
    skills_sim = round(float(np.clip(skills_sim, 0.0, 1.0)), 4)
    exp_sim = round(float(np.clip(exp_sim, 0.0, 1.0)), 4)

    return {
        "score": final_score,
        "section_similarity": {
            "skills": skills_sim,
            "experience": exp_sim,
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "method": method,
    }


def match_resume_jd(
    parsed: ParsedResume,
    jd: Optional[JobDescription],
) -> MatchResult:
    """Public pipeline entry point called by app/pipeline.py.

    Args:
        parsed: Enriched :class:`ParsedResume`.
        jd: Optional :class:`JobDescription`.

    Returns:
        Structured :class:`MatchResult` populated with score, section scores,
        matched skills, missing skills, and method.
    """
    if jd is None or not jd.text.strip():
        return MatchResult(
            score=0.0,
            section_scores={},
            matched_skills=[],
            missing_skills=[],
            method="none",
        )

    result = match(resume_text=parsed.raw_text, jd_text=jd.text)

    return MatchResult(
        score=result["score"],
        section_scores=result["section_similarity"],
        matched_skills=result["matched_skills"],
        missing_skills=result["missing_skills"],
        method=result["method"],
    )
