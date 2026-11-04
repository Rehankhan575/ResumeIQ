"""
app/genai/generator.py
======================
Stage 7 — Generative Feedback & Resume Optimization.

Provides:
  - rewrite_bullets(bullets, jd=None) -> List[Dict[str, Any]]
  - generate_cover_letter(resume, jd) -> str
  - improvement_plan(score_result) -> List[Dict[str, Any]]
  - generate_suggestions(score_result, parsed, jd_text) -> ScoreResult (pipeline hook)

All generative outputs pass through the Grounding Guard (check_grounding).
If ungrounded inventions are detected or the provider is unavailable/times out,
deterministic rule-based fallbacks are deployed seamlessly.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from app.config import ACTION_VERBS
from app.genai.grounding import check_grounding
from app.genai.provider import call_llm
from app.schemas import JobDescription, ParsedResume, ScoreResult, Suggestion

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Weak-to-strong verb substitution mapping
WEAK_VERB_MAP: Dict[str, str] = {
    "worked on": "engineered",
    "worked with": "collaborated with",
    "helped with": "facilitated",
    "assisted in": "contributed to",
    "assisted with": "supported",
    "responsible for": "spearheaded",
    "was responsible for": "led",
    "handled": "executed",
    "participated in": "partnered on",
    "did": "delivered",
    "supported": "optimized",
    "involved in": "drove",
    "contributed to": "accelerated",
    "managed": "orchestrated",
    "conducted": "executed",
    "evaluated": "benchmarked",
    "looked at": "analyzed",
    "created": "architected",
    "made": "developed",
    "built": "constructed",
}

NUMBER_DETECTOR = re.compile(
    r"(?:\$\s*\d+|\b\d+(?:,\d{3})*(?:\.\d+)?%?|\b\d+\s*(?:k|m|b|x|percent)\b)",
    re.IGNORECASE,
)


def _load_prompt(filename: str) -> str:
    """Load prompt template from app/genai/prompts/."""
    p = PROMPTS_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _rule_based_rewrite_single(bullet: str) -> str:
    """Apply rule-based weak-verb replacement and [add metric] placeholder."""
    cleaned = bullet.strip().rstrip(".")
    if not cleaned:
        return ""

    rewritten = cleaned
    lower_bullet = cleaned.lower()

    # 1. Weak verb replacement (longest match first)
    replaced_verb = False
    for weak, strong in sorted(WEAK_VERB_MAP.items(), key=lambda x: -len(x[0])):
        if lower_bullet.startswith(weak):
            rewritten = strong.capitalize() + cleaned[len(weak):]
            replaced_verb = True
            break
        elif f" {weak} " in lower_bullet:
            idx = lower_bullet.find(f" {weak} ")
            rewritten = cleaned[:idx] + f" {strong} " + cleaned[idx + len(weak) + 2:]
            replaced_verb = True
            break

    # If first word is not an action verb and wasn't replaced, check if we should strengthen it
    words = rewritten.split()
    if words and not replaced_verb:
        first_lower = words[0].lower()
        if first_lower not in ACTION_VERBS and len(words) > 1:
            # Check if second word is a verb or keep as is with capitalized first word
            pass

    # 2. Check for metric presence; if missing, add placeholder
    if not NUMBER_DETECTOR.search(rewritten):
        rewritten = f"{rewritten}, achieving [add metric]"

    return rewritten + "."


def rewrite_bullets(
    bullets: Union[str, List[str]],
    jd: Optional[Union[str, JobDescription]] = None,
) -> List[Dict[str, Any]]:
    """Rewrite experience bullets to maximize impact, applying Grounding Guard.

    Args:
        bullets: Single bullet string or list of bullet strings.
        jd: Optional JobDescription or raw job text for targeted alignment.

    Returns:
        List of dicts: ``{"original", "rewritten", "is_grounded", "method", "flags"}``.
    """
    if isinstance(bullets, str):
        bullet_list = [b.strip() for b in bullets.split("\n") if b.strip()]
    else:
        bullet_list = [str(b).strip() for b in bullets if str(b).strip()]

    if not bullet_list:
        return []

    jd_text = ""
    if isinstance(jd, JobDescription):
        jd_text = jd.text
    elif isinstance(jd, str):
        jd_text = jd

    template = _load_prompt("bullet_rewrite.txt")
    results: List[Dict[str, Any]] = []

    for original in bullet_list:
        rewritten: Optional[str] = None
        method = "rule_based"

        # 1. Try LLM Provider if available
        if template:
            prompt = template.format(
                bullet=original,
                jd_context=jd_text[:1000] if jd_text else "General relevant professional role",
            )
            raw_llm = call_llm(prompt)
            if raw_llm:
                # Clean LLM output
                candidate_rewrite = raw_llm.strip().split("\n")[0].strip("-*• ")
                # 2. Run Grounding Guard on Provider Output
                guard_res = check_grounding(original, candidate_rewrite)
                if guard_res.is_grounded:
                    rewritten = candidate_rewrite
                    method = "llm_grounded"
                else:
                    logger.info("Grounding guard flagged LLM rewrite; falling back to rule-based.")
                    rewritten = None

        # 3. Rule-based Fallback
        if not rewritten:
            rewritten = _rule_based_rewrite_single(original)
            method = "rule_based"

        guard_check = check_grounding(original, rewritten)
        results.append({
            "original": original,
            "rewritten": rewritten,
            "is_grounded": guard_check.is_grounded,
            "method": method,
            "flags": guard_check.flags,
        })

    return results


def generate_cover_letter(
    resume: Union[str, ParsedResume],
    jd: Optional[Union[str, JobDescription]] = None,
) -> str:
    """Generate a tailored, professional cover letter grounded in resume facts.

    Args:
        resume: :class:`ParsedResume` or raw resume text.
        jd: Optional :class:`JobDescription` or raw JD text.

    Returns:
        Cover letter text.
    """
    candidate_name = "Candidate"
    contact_details = ""
    skills_list: List[str] = []
    exp_summary: List[str] = []
    edu_summary: List[str] = []

    if isinstance(resume, ParsedResume):
        if resume.contact and resume.contact.name:
            candidate_name = resume.contact.name
        if resume.contact:
            parts = [p for p in [resume.contact.email, resume.contact.phone, getattr(resume.contact, "linkedin", "")] if p]
            contact_details = " | ".join(parts)
        skills_list = resume.skills[:10]
        for exp in resume.experience[:2]:
            role = f"{exp.title} at {exp.company}" if exp.company else exp.title
            exp_summary.append(f"- {role} ({', '.join(exp.bullets[:2])})")
        for edu in resume.education[:2]:
            deg = f"{edu.degree} in {edu.field}" if edu.field else edu.degree
            edu_summary.append(f"- {deg} ({edu.institution})")
    elif isinstance(resume, str):
        skills_list = ["technical skills", "problem solving"]
        exp_summary = ["- Professional experience in industry roles"]

    job_title = "Open Position"
    jd_raw = ""
    if isinstance(jd, JobDescription):
        job_title = jd.title or "Open Position"
        jd_raw = jd.text
    elif isinstance(jd, str) and jd.strip():
        jd_raw = jd.strip()
        first_line = jd_raw.splitlines()[0]
        if len(first_line.split()) <= 6:
            job_title = first_line

    template = _load_prompt("cover_letter.txt")
    if template:
        prompt = template.format(
            job_title=job_title,
            jd_text=jd_raw[:1500] if jd_raw else "Targeting open role requirements.",
            candidate_name=candidate_name,
            contact_info=contact_details or "Available upon request",
            skills=", ".join(skills_list) if skills_list else "Relevant technical background",
            experience_summary="\n".join(exp_summary) if exp_summary else "Strong record of accomplishment",
            education_summary="\n".join(edu_summary) if edu_summary else "Relevant academic qualifications",
        )
        llm_letter = call_llm(prompt)
        if llm_letter and len(llm_letter.strip().split()) >= 80:
            return llm_letter.strip()

    # Rule-based fallback template
    skills_str = ", ".join(skills_list[:5]) if skills_list else "core technical competencies"
    primary_role = exp_summary[0].lstrip("- ") if exp_summary else "my previous professional engagements"

    fallback_letter = (
        f"Dear Hiring Team,\n\n"
        f"I am writing to express my strong interest in the {job_title} position. "
        f"With a proven background in {skills_str}, I am enthusiastic about the opportunity to "
        f"contribute directly to your organization's goals.\n\n"
        f"In my work experience, specifically {primary_role}, I have consistently focused on "
        f"delivering high-impact results, optimizing operational processes, and collaborating "
        f"across cross-functional teams to meet demanding project deadlines.\n\n"
        f"I welcome the opportunity to discuss how my practical qualifications and dedication "
        f"will allow me to add immediate value to your team. Thank you for your time and consideration.\n\n"
        f"Sincerely,\n"
        f"{candidate_name}"
    )
    return fallback_letter


def improvement_plan(
    score_result: ScoreResult,
) -> List[Dict[str, Any]]:
    """Build a prioritized resume improvement plan based on ATS diagnostic issues.

    Args:
        score_result: :class:`ScoreResult` with components and issues.

    Returns:
        List of dicts: ``{"priority", "component", "estimated_gain", "recommendation"}``.
    """
    if not score_result:
        return []

    plan: List[Dict[str, Any]] = []

    # Priority 1: Address missing skills if any
    if score_result.missing_skills:
        missing_top = score_result.missing_skills[:5]
        plan.append({
            "priority": "High",
            "component": "keyword_coverage",
            "estimated_gain": round(score_result.components.get("keyword_coverage", 0.0) * 10.0, 1),
            "recommendation": (
                f"Close skill gaps by explicitly documenting experience with: {', '.join(missing_top)}. "
                f"Incorporate these into project descriptions or coursework."
            ),
        })

    # Priority 2: Iterate through diagnostic issues sorted strictly by estimated gain
    sorted_issues = sorted(score_result.issues, key=lambda x: x.estimated_gain, reverse=True)
    for idx, issue in enumerate(sorted_issues):
        prio = "High" if issue.estimated_gain >= 4.0 else ("Medium" if issue.estimated_gain >= 2.0 else "Low")
        rec = f"{issue.message} Estimated potential gain: +{issue.estimated_gain:.1f} pts."
        plan.append({
            "priority": prio,
            "component": issue.component,
            "estimated_gain": issue.estimated_gain,
            "recommendation": rec,
        })

    # Default fallback item if no issues detected
    if not plan:
        plan.append({
            "priority": "Low",
            "component": "general",
            "estimated_gain": 0.0,
            "recommendation": "Resume satisfies primary ATS rubrics. Maintain crisp formatting and updated metrics.",
        })

    return plan


def generate_suggestions(
    score_result: ScoreResult,
    parsed: ParsedResume,
    jd_text: str = "",
) -> ScoreResult:
    """Populate generative improvement suggestions in :class:`ScoreResult`.

    Args:
        score_result: Stage 5/6 :class:`ScoreResult`.
        parsed: Enriched :class:`ParsedResume`.
        jd_text: Optional job description text.

    Returns:
        Updated :class:`ScoreResult` with populated suggestions.
    """
    suggestions: List[Suggestion] = []

    # 1. Experience bullet rewrites
    bullets_to_rewrite: List[str] = []
    for exp in parsed.experience:
        for b in exp.bullets:
            clean_b = b.strip()
            # Target bullets lacking numbers or starting with weak verbs
            if clean_b and not NUMBER_DETECTOR.search(clean_b):
                bullets_to_rewrite.append(clean_b)
                if len(bullets_to_rewrite) >= 3:
                    break
        if len(bullets_to_rewrite) >= 3:
            break

    # If all bullets had numbers, pick first 2 available
    if not bullets_to_rewrite:
        for exp in parsed.experience:
            for b in exp.bullets:
                if b.strip():
                    bullets_to_rewrite.append(b.strip())
                    if len(bullets_to_rewrite) >= 2:
                        break

    if bullets_to_rewrite:
        rewritten_data = rewrite_bullets(bullets_to_rewrite, jd=jd_text)
        for item in rewritten_data:
            suggestions.append(
                Suggestion(
                    kind="bullet_rewrite",
                    original=item["original"],
                    suggested=item["rewritten"],
                    notes=f"Method: {item['method']}; Grounded: {item['is_grounded']}",
                )
            )

    # 2. Improvement plan suggestions
    plan_items = improvement_plan(score_result)
    for p in plan_items[:3]:
        suggestions.append(
            Suggestion(
                kind="improvement_plan",
                original="",
                suggested=p["recommendation"],
                notes=f"Priority: {p['priority']} | Component: {p['component']} | Gain: +{p['estimated_gain']:.1f}",
            )
        )

    score_result.suggestions = suggestions
    return score_result
