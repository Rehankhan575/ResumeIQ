"""
app/scoring/ats.py
==================
Stage 5 — ATS Composite Scorer.

Computes a transparent weighted score (0–100) from:
  - semantic_match          (weight: config.SCORE_WEIGHTS["semantic_match"])
  - keyword_coverage        (weight: config.SCORE_WEIGHTS["keyword_coverage"])
  - section_completeness    (weight: config.SCORE_WEIGHTS["section_completeness"])
  - formatting              (weight: config.SCORE_WEIGHTS["formatting"])
  - experience_fit          (weight: config.SCORE_WEIGHTS["experience_fit"])
  - quantified_achievements (weight: config.SCORE_WEIGHTS["quantified_achievements"])
  - action_verbs            (weight: config.SCORE_WEIGHTS["action_verbs"])
  - readability             (weight: config.SCORE_WEIGHTS["readability"])
  - contact_completeness    (weight: config.SCORE_WEIGHTS["contact_completeness"])

No ground-truth "ATS score" exists in any dataset; this is a designed,
explainable composite. Each component is validated separately.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import ACTION_VERBS, SCORE_WEIGHTS, settings
from app.models.classifier import classify
import app.models.matcher as matcher_module
from app.parsing.cleaner import clean_text
from app.schemas import Issue, JobDescription, MatchResult, ParsedResume, ScoreResult
from app.scoring.formatting import score_formatting

logger = logging.getLogger(__name__)

# Regex for numbers, percentages, and currencies in bullets
QUANTIFIED_RE = re.compile(
    r"(?:\b\d+(?:,\d{3})*(?:\.\d+)?%?|\$\s*\d+|\b\d+\s*(?:percent|\+|k|m|b|x)\b|[€£¥]\s*\d+)",
    re.IGNORECASE,
)

CORE_SECTIONS: Dict[str, float] = {
    "experience": 0.35,
    "education": 0.25,
    "skills": 0.25,
    "contact": 0.10,
    "other": 0.05,
}


def _compute_tfidf_cosine(resume_text: str, jd_text: str) -> float:
    """Compute normalized cosine similarity between resume and JD texts via TF-IDF."""
    clean_resume = clean_text(resume_text).strip()
    clean_jd = clean_text(jd_text).strip()
    if not clean_resume or not clean_jd:
        return 0.0

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vec.fit_transform([clean_resume, clean_jd])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        # Uncalibrated TF-IDF cosine across resumes/JDs typically peaks around ~0.50.
        # Scale by 2.0 (clamp at 1.0) for an intuitive 0.0-1.0 component metric.
        scaled_sim = min(1.0, max(0.0, sim * 2.0))
        return round(scaled_sim, 4)
    except Exception as e:
        logger.warning("TF-IDF cosine computation failed: %s", e)
        return 0.0


# In-memory caching for sentence transformer to prevent reloading per inference
_orig_load_sentence_model = getattr(matcher_module, "load_sentence_model", None)
_CACHED_SBERT_MODEL: Any = None


def _cached_load_sentence_model(model_path: Any = None) -> Tuple[Any, str]:
    global _CACHED_SBERT_MODEL
    if _CACHED_SBERT_MODEL is not None:
        return _CACHED_SBERT_MODEL
    if callable(_orig_load_sentence_model):
        _CACHED_SBERT_MODEL = _orig_load_sentence_model(model_path)
        return _CACHED_SBERT_MODEL
    return None, "tfidf_cosine"


if callable(_orig_load_sentence_model):
    matcher_module.load_sentence_model = _cached_load_sentence_model


def _get_matcher_result(parsed: ParsedResume, jd: JobDescription | None) -> MatchResult:
    """Call matcher or fall back to TF-IDF cosine if matcher is a stub or missing."""
    if jd is None:
        return MatchResult(
            score=0.0,
            section_scores={},
            matched_skills=[],
            missing_skills=[],
            method="none",
        )

    # Check public function in app.models.matcher
    is_stub = True
    result: Optional[MatchResult] = None

    try:
        if hasattr(matcher_module, "match") and callable(matcher_module.match):
            raw_match = matcher_module.match(parsed.raw_text, jd.text)
            if isinstance(raw_match, dict):
                score = float(raw_match.get("score", 0.0))
                matched_skills = list(raw_match.get("matched_skills", []))
                missing_skills = list(raw_match.get("missing_skills", []))
                method = raw_match.get("method", "matcher")
                # Detect dummy stub output: static 0.73 score with hardcoded skills or method == "stub"
                if method != "stub" and not (
                    score == 0.73
                    and set(matched_skills) == {"Python", "Machine Learning"}
                ):
                    is_stub = False
                    result = MatchResult(
                        score=score,
                        section_scores=raw_match.get("section_similarity", {}),
                        matched_skills=matched_skills,
                        missing_skills=missing_skills,
                        method=method,
                    )
        elif hasattr(matcher_module, "match_resume_jd") and callable(matcher_module.match_resume_jd):
            res = matcher_module.match_resume_jd(parsed, jd)
            if res and (
                res.score != 0.73
                or set(res.matched_skills) != {"Python", "Machine Learning"}
            ):
                is_stub = False
                result = res
    except Exception as e:
        logger.warning("Matcher invocation raised error: %s", e)

    if is_stub or result is None:
        msg = "app.models.matcher is a stub or missing; falling back to TF-IDF cosine inside ats.py."
        logger.info(msg)
        print(msg)
        sim_score = _compute_tfidf_cosine(parsed.raw_text, jd.text)

        # Keyword overlap for matched and missing skills
        resume_skills_lower = {s.lower() for s in parsed.skills}
        resume_text_lower = parsed.raw_text.lower()

        matched: List[str] = []
        missing: List[str] = []
        for req_skill in jd.required_skills:
            req_lower = req_skill.lower()
            if req_lower in resume_skills_lower or re.search(r"\b" + re.escape(req_lower) + r"\b", resume_text_lower):
                matched.append(req_skill)
            else:
                missing.append(req_skill)

        result = MatchResult(
            score=sim_score,
            section_scores={"full_text": sim_score},
            matched_skills=matched,
            missing_skills=missing,
            method="tfidf_cosine (matcher fallback: model stub/missing)",
        )

    return result


def _score_section_completeness(parsed: ParsedResume) -> Tuple[float, List[str]]:
    """Score resume for presence of standard core sections."""
    if not parsed.raw_text.strip():
        return 0.0, ["Missing all sections; resume is empty."]

    found_sections: Set[str] = set()
    missing_msgs: List[str] = []

    # Check experience
    has_exp = len(parsed.experience) > 0 or any(
        k in parsed.sections for k in ["experience", "work_experience", "employment", "professional_experience"]
    )
    if has_exp:
        found_sections.add("experience")
    else:
        missing_msgs.append("Missing Work Experience section.")

    # Check education
    has_edu = len(parsed.education) > 0 or any(
        k in parsed.sections for k in ["education", "academic", "qualifications"]
    )
    if has_edu:
        found_sections.add("education")
    else:
        missing_msgs.append("Missing Education section.")

    # Check skills
    has_skills = len(parsed.skills) > 0 or any(
        k in parsed.sections for k in ["skills", "technical_skills", "core_competencies", "areas_of_expertise"]
    )
    if has_skills:
        found_sections.add("skills")
    else:
        missing_msgs.append("Missing Skills section.")

    # Check contact
    has_contact = bool(parsed.contact.email or parsed.contact.phone or parsed.contact.name)
    if has_contact:
        found_sections.add("contact")
    else:
        missing_msgs.append("Missing Contact Information.")

    # Check other (projects, summary, certifications)
    has_other = (
        len(parsed.projects) > 0
        or len(parsed.certifications) > 0
        or any(k in parsed.sections for k in ["projects", "certifications", "summary", "profile", "objective"])
    )
    if has_other:
        found_sections.add("other")

    score = sum(CORE_SECTIONS[s] for s in found_sections)
    return round(min(1.0, score), 4), missing_msgs


def _score_quantified_achievements(parsed: ParsedResume) -> Tuple[float, str]:
    """Score the proportion of experience bullets with numbers, percentages, or currencies."""
    bullets: List[str] = []
    for exp in parsed.experience:
        for b in exp.bullets:
            b_clean = b.strip()
            if b_clean:
                bullets.append(b_clean)

    if not bullets:
        # Fallback to scanning raw text lines starting with bullets
        for line in parsed.raw_text.splitlines():
            m = re.match(r"^\s*[-*•\d+.]\s+(.*)", line)
            if m and len(m.group(1).split()) >= 4:
                bullets.append(m.group(1).strip())

    if not bullets:
        return 0.0, "No experience bullet points found to measure quantified impact."

    quant_count = sum(1 for b in bullets if QUANTIFIED_RE.search(b))
    ratio = quant_count / len(bullets)
    msg = f"{quant_count} of {len(bullets)} bullets ({ratio:.0%}) contain quantified metrics."
    return round(ratio, 4), msg


def _score_action_verbs(parsed: ParsedResume) -> Tuple[float, str]:
    """Score the proportion of experience bullets starting with strong action verbs."""
    bullets: List[str] = []
    for exp in parsed.experience:
        for b in exp.bullets:
            b_clean = b.strip()
            if b_clean:
                bullets.append(b_clean)

    if not bullets:
        for line in parsed.raw_text.splitlines():
            m = re.match(r"^\s*[-*•\d+.]\s+(.*)", line)
            if m and len(m.group(1).split()) >= 4:
                bullets.append(m.group(1).strip())

    if not bullets:
        return 0.0, "No experience bullet points found to verify action verbs."

    action_count = 0
    for b in bullets:
        words = re.findall(r"\b[a-zA-Z]+\b", b)
        if words and words[0].lower() in ACTION_VERBS:
            action_count += 1
        elif len(words) >= 2 and words[1].lower() in ACTION_VERBS:
            # Handles adverbs e.g. "Successfully deployed"
            action_count += 1

    ratio = action_count / len(bullets)
    msg = f"{action_count} of {len(bullets)} bullets ({ratio:.0%}) begin with strong action verbs."
    return round(ratio, 4), msg


def _score_readability(parsed: ParsedResume) -> Tuple[float, str]:
    """Score resume readability based on sentence length and bullet density."""
    text = parsed.raw_text.strip()
    if not text:
        return 0.0, "Resume has no text for readability analysis."

    # Sentence length
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip().split()) >= 3]
    if sentences:
        words_per_sent = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        words_per_sent = 0.0

    # Ideal words per sentence: 10 - 22
    if 10.0 <= words_per_sent <= 22.0:
        sent_score = 1.0
    elif words_per_sent < 10.0:
        sent_score = max(0.4, words_per_sent / 10.0)
    else:
        sent_score = max(0.2, 1.0 - (words_per_sent - 22.0) * 0.04)

    # Bullet density
    bullets: List[str] = []
    for exp in parsed.experience:
        bullets.extend([b for b in exp.bullets if b.strip()])

    n_exp = max(1, len(parsed.experience))
    density = len(bullets) / n_exp

    # Ideal bullet density: 3 to 6 bullets per position
    if 3.0 <= density <= 6.0:
        density_score = 1.0
    elif density < 3.0:
        density_score = max(0.3, density / 3.0)
    else:
        density_score = max(0.5, 1.0 - (density - 6.0) * 0.05)

    readability_score = 0.5 * sent_score + 0.5 * density_score
    msg = f"Average sentence length: {words_per_sent:.1f} words; bullet density: {density:.1f} bullets/role."
    return round(readability_score, 4), msg


def _score_contact_completeness(parsed: ParsedResume) -> Tuple[float, List[str]]:
    """Score contact details completeness (name, email, phone, links)."""
    c = parsed.contact
    score = 0.0
    missing: List[str] = []

    if c.name.strip():
        score += 0.25
    else:
        missing.append("Candidate name is missing.")

    if c.email.strip():
        score += 0.35
    else:
        missing.append("Email address is missing.")

    if c.phone.strip():
        score += 0.25
    else:
        missing.append("Phone number is missing.")

    if c.linkedin.strip() or c.github.strip() or c.other_links:
        score += 0.15
    else:
        missing.append("No professional links (LinkedIn/GitHub/portfolio) provided.")

    return round(min(1.0, score), 4), missing


def _score_experience_fit(parsed: ParsedResume, jd: JobDescription | None) -> Tuple[float, str]:
    """Score candidate experience fit against JD minimum years requirement."""
    cand_years = parsed.total_years_experience
    if cand_years == 0.0 and parsed.experience:
        months = sum(e.duration_months for e in parsed.experience)
        cand_years = months / 12.0

    if jd is None:
        # Without JD, evaluate career progression depth
        if cand_years >= 2.0:
            return 1.0, f"Strong experience record ({cand_years:.1f} years)."
        elif cand_years > 0.0:
            return 0.85, f"Moderate experience record ({cand_years:.1f} years)."
        return 0.60, "Entry level or unspecified experience duration."

    min_years = jd.min_years if jd.min_years is not None else jd.min_years_experience
    if min_years is None or min_years <= 0.0:
        return 1.0, f"No minimum years specified in JD (candidate has {cand_years:.1f} years)."

    if cand_years >= min_years:
        return 1.0, f"Meets experience requirement ({cand_years:.1f} years vs {min_years:.1f} years required)."

    fit = cand_years / min_years
    return round(max(0.1, min(1.0, fit)), 4), (
        f"Candidate has {cand_years:.1f} years; job requires {min_years:.1f} years."
    )


def _compute_grade(total: float) -> str:
    """Return standard letter grade (A-F) for a 0-100 total score."""
    if total >= 90.0:
        return "A"
    if total >= 80.0:
        return "B"
    if total >= 70.0:
        return "C"
    if total >= 60.0:
        return "D"
    return "F"


def score_resume(
    parsed_resume: ParsedResume,
    jd: JobDescription | None = None,
    match: MatchResult | None = None,
) -> ScoreResult:
    """Score a parsed resume against a job description across 9 ATS components.

    Components evaluated:
      - semantic_match (TF-IDF cosine or SBERT)
      - keyword_coverage (JD required skills coverage)
      - section_completeness (core sections present)
      - formatting (layout_meta: columns, tables, images, pages, words)
      - experience_fit (candidate years vs JD required years)
      - quantified_achievements (proportion of quantified bullets)
      - action_verbs (proportion of strong verb-initiated bullets)
      - readability (sentence length & bullet density)
      - contact_completeness (name, email, phone, links)

    Args:
        parsed_resume: Enriched :class:`ParsedResume` object.
        jd: Optional :class:`JobDescription` for targeted ATS scoring.
        match: Optional pre-computed :class:`MatchResult`.

    Returns:
        :class:`ScoreResult` with total 0-100, letter grade (A-F), components,
        missing_skills, and issues sorted by estimated point gain.
    """
    weights = getattr(settings, "score_weights", SCORE_WEIGHTS)

    # 1. Handle empty or corrupted resume gracefully
    if not parsed_resume or not parsed_resume.raw_text.strip():
        components = {k: 0.0 for k in weights}
        missing_skills = jd.required_skills if jd else []
        issues = [
            Issue(
                component="general",
                message="Resume is empty or unreadable; no content could be parsed.",
                severity="high",
                estimated_gain=100.0,
            )
        ]
        return ScoreResult(
            total=0.0,
            grade="F",
            components=components,
            missing_skills=missing_skills,
            issues=issues,
            match=match,
        )

    # 2. MatchResult & Semantic match
    match_result = match if match is not None else _get_matcher_result(parsed_resume, jd)
    semantic_score = match_result.score if jd is not None else 0.80

    # 3. Keyword coverage & missing skills
    missing_skills: List[str] = []
    if jd is not None and jd.required_skills:
        resume_skills_lower = {s.lower() for s in parsed_resume.skills}
        resume_text_lower = parsed_resume.raw_text.lower()
        matched_req = []
        for req in jd.required_skills:
            r_low = req.lower()
            if r_low in resume_skills_lower or re.search(r"\b" + re.escape(r_low) + r"\b", resume_text_lower):
                matched_req.append(req)
            else:
                missing_skills.append(req)
        coverage_score = len(matched_req) / len(jd.required_skills)
    else:
        coverage_score = 1.0 if jd is not None else 0.85

    # 4. Section completeness
    sec_score, sec_msgs = _score_section_completeness(parsed_resume)

    # 5. Formatting
    fmt_score, fmt_issues = score_formatting(parsed_resume.layout)

    # 6. Experience fit
    exp_fit_score, exp_fit_msg = _score_experience_fit(parsed_resume, jd)

    # 7. Quantified achievements
    quant_score, quant_msg = _score_quantified_achievements(parsed_resume)

    # 8. Action verbs
    action_score, action_msg = _score_action_verbs(parsed_resume)

    # 9. Readability
    read_score, read_msg = _score_readability(parsed_resume)

    # 10. Contact completeness
    contact_score, contact_msgs = _score_contact_completeness(parsed_resume)

    components: Dict[str, float] = {
        "semantic_match": round(semantic_score, 4),
        "keyword_coverage": round(coverage_score, 4),
        "section_completeness": round(sec_score, 4),
        "formatting": round(fmt_score, 4),
        "experience_fit": round(exp_fit_score, 4),
        "quantified_achievements": round(quant_score, 4),
        "action_verbs": round(action_score, 4),
        "readability": round(read_score, 4),
        "contact_completeness": round(contact_score, 4),
    }

    # Total composite score: sum(weight * component_score) * 100
    raw_total = sum(weights[k] * components[k] for k in weights) * 100.0
    total = round(max(0.0, min(100.0, raw_total)), 2)
    grade = _compute_grade(total)

    # 11. Generate issues sorted by estimated point gain
    # gain = weight * (1 - component_score) * 100
    issues: List[Issue] = []

    def _calc_gain(comp_name: str) -> float:
        w = weights.get(comp_name, 0.0)
        c_val = components.get(comp_name, 1.0)
        return round(w * (1.0 - c_val) * 100.0, 2)

    # Keyword coverage issue
    gain_kw = _calc_gain("keyword_coverage")
    if missing_skills:
        issues.append(
            Issue(
                component="keyword_coverage",
                message=f"Missing required skills: {', '.join(missing_skills[:5])}.",
                severity="high" if gain_kw >= 5.0 else "medium",
                estimated_gain=gain_kw,
            )
        )

    # Semantic match issue
    gain_sem = _calc_gain("semantic_match")
    if semantic_score < 0.60:
        issues.append(
            Issue(
                component="semantic_match",
                message="Low semantic alignment with job description; align summary and role titles.",
                severity="high" if gain_sem >= 5.0 else "medium",
                estimated_gain=gain_sem,
            )
        )

    # Section completeness issue
    gain_sec = _calc_gain("section_completeness")
    if sec_msgs:
        issues.append(
            Issue(
                component="section_completeness",
                message=" ".join(sec_msgs),
                severity="high" if gain_sec >= 4.0 else "medium",
                estimated_gain=gain_sec,
            )
        )

    # Formatting issues
    gain_fmt = _calc_gain("formatting")
    if fmt_issues:
        for fi in fmt_issues:
            fi.estimated_gain = round(gain_fmt / len(fmt_issues), 2)
            issues.append(fi)

    # Experience fit issue
    gain_exp = _calc_gain("experience_fit")
    if exp_fit_score < 0.80:
        issues.append(
            Issue(
                component="experience_fit",
                message=exp_fit_msg,
                severity="medium" if gain_exp >= 2.0 else "low",
                estimated_gain=gain_exp,
            )
        )

    # Quantified achievements issue
    gain_quant = _calc_gain("quantified_achievements")
    if quant_score < 0.50:
        issues.append(
            Issue(
                component="quantified_achievements",
                message=f"Only {quant_msg} Add metrics, %, or revenue figures to demonstrate impact.",
                severity="medium" if gain_quant >= 2.0 else "low",
                estimated_gain=gain_quant,
            )
        )

    # Action verbs issue
    gain_act = _calc_gain("action_verbs")
    if action_score < 0.50:
        issues.append(
            Issue(
                component="action_verbs",
                message=f"Only {action_msg} Begin bullets with active power verbs.",
                severity="low",
                estimated_gain=gain_act,
            )
        )

    # Readability issue
    gain_read = _calc_gain("readability")
    if read_score < 0.70:
        issues.append(
            Issue(
                component="readability",
                message=f"Readability could be improved: {read_msg}",
                severity="low",
                estimated_gain=gain_read,
            )
        )

    # Contact completeness issue
    gain_con = _calc_gain("contact_completeness")
    if contact_msgs:
        issues.append(
            Issue(
                component="contact_completeness",
                message=" ".join(contact_msgs),
                severity="medium" if gain_con >= 2.0 else "low",
                estimated_gain=gain_con,
            )
        )

    # Sort issues strictly by estimated point gain descending
    issues.sort(key=lambda x: x.estimated_gain, reverse=True)

    # Category predictions from classifier
    cats = []
    try:
        cats = classify(parsed_resume.raw_text, top_k=3)
    except Exception as e:
        logger.debug("Classifier failed: %s", e)

    return ScoreResult(
        total=total,
        grade=grade,
        components=components,
        missing_skills=missing_skills,
        issues=issues,
        match=match_result,
        predicted_categories=cats,
    )


def compute_ats_score(
    parsed: ParsedResume,
    match: MatchResult | None = None,
    jd: JobDescription | None = None,
) -> ScoreResult:
    """Backward-compatible wrapper for pipeline Stage 5."""
    return score_resume(parsed, jd=jd, match=match)
