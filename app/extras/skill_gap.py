"""
app/extras/skill_gap.py
=======================
Skill Gap Analysis and Course Recommendation.

Maps missing required skills against curated learning resources from data/courses.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_COURSES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "courses.json"
_COURSES_CACHE: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _load_courses(courses_path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load curated courses from JSON file."""
    global _COURSES_CACHE
    target_path = courses_path or DEFAULT_COURSES_PATH

    if target_path == DEFAULT_COURSES_PATH and _COURSES_CACHE is not None:
        return _COURSES_CACHE

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if target_path == DEFAULT_COURSES_PATH:
                    _COURSES_CACHE = data
                return data
        except Exception as exc:
            logger.warning("Failed loading courses catalog from %s: %s", target_path, exc)

    return {}


def skill_gap(
    resume_skills: Iterable[str],
    jd_required: Iterable[str],
    courses_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Map candidate missing skills to curated learning recommendations.

    Args:
        resume_skills: Skills possessed by the candidate.
        jd_required: Skills required by the target job description.
        courses_path: Optional custom path to courses.json catalog.

    Returns:
        Dictionary mapping each missing skill to a list of learning resources:
        ``{skill: [{"title", "provider", "level", "estimated_hours"}]}``.
    """
    res_lower: Set[str] = {str(s).strip().lower() for s in resume_skills if str(s).strip()}
    catalog = _load_courses(courses_path)

    gap_map: Dict[str, List[Dict[str, Any]]] = {}

    for req in jd_required:
        req_clean = str(req).strip()
        req_lower = req_clean.lower()
        if not req_clean:
            continue

        # Check if candidate possesses the required skill
        if req_lower in res_lower:
            continue

        # Find matching course in catalog (exact or substring)
        courses: List[Dict[str, Any]] = []
        if req_lower in catalog:
            courses = catalog[req_lower]
        else:
            # Substring / partial match
            for cat_skill, items in catalog.items():
                if cat_skill in req_lower or req_lower in cat_skill:
                    courses = items
                    break

        # Fallback guidance if skill is not yet in curated courses
        if not courses:
            courses = [
                {
                    "title": f"Applied Professional Training: {req_clean}",
                    "provider": "Self-Directed / Industry Courseware",
                    "level": "Intermediate",
                    "estimated_hours": 20,
                }
            ]

        gap_map[req_clean] = courses

    return gap_map
