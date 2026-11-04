"""
app/extras/bias.py
==================
Unconscious Bias and Demographic Signal Auditing.

Detects potential demographic cues in candidate résumés:
  - Gendered wording & third-person pronouns
  - Age cues (graduation years older than 15 years, age-correlated adjectives)
  - Sensitive personal details (DOB, marital status, nationality, religion, photographs)

Privacy Guarantee:
  Outputs abstract category types and evidence labels ONLY.
  Never outputs or logs matched personal data.
  Frames all findings as constructive suggestions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Union

from app.schemas import ParsedResume

logger = logging.getLogger(__name__)

# Personal detail detection patterns
DOB_RE = re.compile(
    r"\b(?:date\s*of\s*birth|d\.?o\.?b\.?|birth\s*date|born\s*(?:on|in)?)\b",
    re.IGNORECASE,
)
MARITAL_RE = re.compile(
    r"\b(?:marital\s*status|married|single|divorced|unmarried|widowed|spouse)\b",
    re.IGNORECASE,
)
NATIONALITY_RE = re.compile(
    r"\b(?:nationality|citizenship|citizen\s*of|visa\s*status|work\s*permit\s*holder|passport\s*holder)\b",
    re.IGNORECASE,
)
RELIGION_RE = re.compile(
    r"\b(?:religion|religious\s*affiliation|christian|catholic|muslim|jewish|hindu|buddhist|parish|church\s*member)\b",
    re.IGNORECASE,
)

# Age cue word patterns
AGE_WORDS_RE = re.compile(
    r"\b(?:young|youthful|energetic|digital\s*native|recent\s*college\s*grad|mature|seasoned\s*veteran)\b",
    re.IGNORECASE,
)

# Third-person gender pronouns
GENDER_PRONOUNS_RE = re.compile(
    r"\b(?:he|him|his|she|her|hers)\b",
    re.IGNORECASE,
)

# Strongly gender-coded trait terms (gauged against standard industrial I/O psychology rubrics)
GENDER_TRAITS_MASCULINE = {
    "aggressive", "dominant", "competitive", "outspoken", "assertive",
    "forceful", "authoritative", "boastful", "fearless", "reckless",
}
GENDER_TRAITS_FEMININE = {
    "nurturing", "supportive", "compassionate", "affectionate", "interpersonal",
    "sensitive", "caring", "understanding", "empathetic",
}


def bias_check(
    resume: Union[ParsedResume, str],
) -> List[Dict[str, str]]:
    """Scan resume text and metadata for potential bias signals.

    Args:
        resume: :class:`ParsedResume` or raw resume text.

    Returns:
        List of dicts: ``{"type", "evidence_label", "suggestion"}``.
        Empty list if no bias signals are detected.
    """
    if not resume:
        return []

    if isinstance(resume, ParsedResume):
        parsed = resume
        text = parsed.raw_text
        layout = parsed.layout
    else:
        parsed = None
        text = str(resume)
        layout = None

    if not text.strip():
        return []

    findings: List[Dict[str, str]] = []

    # 1. Personal Details: Date of Birth
    if DOB_RE.search(text):
        findings.append({
            "type": "personal_info",
            "evidence_label": "Date of birth disclosed",
            "suggestion": "Remove date of birth to protect against potential age discrimination during initial screening.",
        })

    # 2. Personal Details: Marital Status
    if MARITAL_RE.search(text):
        findings.append({
            "type": "personal_info",
            "evidence_label": "Marital status disclosed",
            "suggestion": "Marital status is non-job-relevant personal information and should be excluded.",
        })

    # 3. Personal Details: Nationality / Citizenship
    if NATIONALITY_RE.search(text):
        findings.append({
            "type": "personal_info",
            "evidence_label": "Nationality or citizenship disclosed",
            "suggestion": "Omit nationality or citizenship declarations unless required for specific government security clearances.",
        })

    # 4. Personal Details: Religion
    if RELIGION_RE.search(text):
        findings.append({
            "type": "personal_info",
            "evidence_label": "Religious affiliation reference detected",
            "suggestion": "Remove faith-based or religious references to eliminate potential demographic bias.",
        })

    # 5. Personal Details: Photograph / Headshot
    has_photo = False
    if layout and layout.has_images:
        has_photo = True
    elif re.search(r"\b(?:photo|headshot|picture\s*attached)\b", text, re.IGNORECASE):
        has_photo = True

    if has_photo:
        findings.append({
            "type": "personal_info",
            "evidence_label": "Photograph or graphic headshot detected",
            "suggestion": "Corporate screening standards in North America and Western Europe strongly advise removing headshots to minimize unconscious appearance bias.",
        })

    # 6. Age Cues: Very Old Graduation Year
    # Standard threshold: Graduation years older than 15 years (e.g. <= 2010)
    old_grad_found = False
    if parsed and parsed.education:
        for edu in parsed.education:
            year = edu.graduation_year
            if year and year <= 2010:
                old_grad_found = True
                break
    if not old_grad_found:
        # Fall back to regex scan in education sections or full text
        edu_sec = parsed.sections.get("education", text) if parsed else text
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20[0-1][0-9])\b", edu_sec)]
        if any(y <= 2010 for y in years):
            old_grad_found = True

    if old_grad_found:
        findings.append({
            "type": "age_bias",
            "evidence_label": "Graduation year older than 15 years",
            "suggestion": "Consider omitting college graduation years older than 10-15 years to prevent age-related filtering.",
        })

    # 7. Age Cues: Age-correlated descriptors
    if AGE_WORDS_RE.search(text):
        findings.append({
            "type": "age_bias",
            "evidence_label": "Age-correlated descriptor detected",
            "suggestion": "Replace age-suggestive descriptors (such as 'young', 'energetic', or 'seasoned veteran') with quantifiable career accomplishments.",
        })

    # 8. Gendered Wording: Third-person pronouns & gendered traits
    pronoun_match = GENDER_PRONOUNS_RE.search(text)
    lower_words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    has_masc_traits = bool(lower_words.intersection(GENDER_TRAITS_MASCULINE))
    has_fem_traits = bool(lower_words.intersection(GENDER_TRAITS_FEMININE))

    if pronoun_match or has_masc_traits or has_fem_traits:
        evidence_parts = []
        if pronoun_match:
            evidence_parts.append("third-person gender pronouns")
        if has_masc_traits or has_fem_traits:
            evidence_parts.append("gender-coded descriptor language")
        findings.append({
            "type": "gender_bias",
            "evidence_label": f"Gendered wording ({', '.join(evidence_parts)}) detected",
            "suggestion": "Adopt first-person or implied-first-person syntax with neutral technical phrasing rather than gender-indicative terminology.",
        })

    return findings
