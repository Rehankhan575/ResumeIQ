"""
app/parsing/contact.py
======================
Regex-based extraction of structured contact information from résumé text.

Extracts:
    - Email address
    - Phone number
    - LinkedIn profile URL
    - GitHub profile URL
    - Best-effort candidate name (from header text, 2-4 capitalized words, no digits)
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas import ContactInfo

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
)
# Simpler robust phone regex for standard formats (e.g. 555-0142, (555) 014-2345, +1-555-0142)
PHONE_SIMPLE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?", re.IGNORECASE
)
GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+/?", re.IGNORECASE
)

# Common words/phrases in header blocks that should not be mistaken for names
NON_NAME_STOP_WORDS = {
    "curriculum vitae",
    "resume",
    "résumé",
    "contact",
    "contact info",
    "contact information",
    "personal information",
    "personal profile",
    "personal details",
    "summary",
    "overview",
    "experience",
    "work experience",
    "work history",
    "academics",
    "education",
    "skills",
    "tech stack",
    "technical skills",
    "certifications",
    "projects",
    "awards",
    "publications",
    "languages",
    "interests",
    "references",
}


def extract_candidate_name(header_text: str, full_text: str = "") -> str:
    """Extract candidate name using best-effort heuristic.

    Looks for the first line of 2-4 capitalized words containing no digits.
    Checks header lines first, then falls back to early document lines.

    Args:
        header_text: Content of the "header" section (before first heading).
        full_text: Full document text (used as fallback).

    Returns:
        Best-effort candidate name string or empty string.
    """
    candidates = []
    if header_text:
        candidates.extend(header_text.splitlines())
    if full_text:
        candidates.extend(full_text.splitlines()[:15])

    for raw_line in candidates:
        line = raw_line.strip()
        if not line or "@" in line or "http" in line:
            continue

        # Strip pipe or dash delimiters (e.g. "Alex Mercer | alex@example.com")
        for delim in ["|", "•", "·", "—", "/"]:
            if delim in line:
                line = line.split(delim)[0].strip()

        # Strip degrees/credentials after commas (e.g. "Elena Rostova, BSN, RN")
        if "," in line:
            line = line.split(",")[0].strip()

        if line.lower() in NON_NAME_STOP_WORDS:
            continue

        # Must not contain digits
        if any(ch.isdigit() for ch in line):
            continue

        words = line.split()
        if 2 <= len(words) <= 4:
            # All words must be alphabetic and capitalized
            if all(w.isalpha() and w[0].isupper() for w in words):
                return line

    return ""


def extract_contact_info(
    header_text: str = "",
    full_text: str = "",
) -> ContactInfo:
    """Extract structured contact details from résumé text.

    Args:
        header_text: Text preceding the first canonical heading.
        full_text: Full document text.

    Returns:
        Populated :class:`ContactInfo` model.
    """
    search_space = f"{header_text}\n{full_text}".strip()

    # Email
    email_match = EMAIL_RE.search(search_space)
    email = email_match.group(0).strip() if email_match else ""

    # Phone
    phone_match = PHONE_SIMPLE_RE.search(search_space)
    phone = phone_match.group(0).strip() if phone_match else ""

    # LinkedIn
    li_match = LINKEDIN_RE.search(search_space)
    linkedin = li_match.group(0).strip() if li_match else ""

    # GitHub
    gh_match = GITHUB_RE.search(search_space)
    github = gh_match.group(0).strip() if gh_match else ""

    # Name
    name = extract_candidate_name(header_text, full_text)

    return ContactInfo(
        name=name,
        email=email,
        phone=phone,
        linkedin=linkedin,
        github=github,
        other_links=[],
    )
