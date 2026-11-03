"""
app/extraction/entries.py
=========================
Education and Experience extraction for Stage 2.

Features:
- Normalized education parsing (degree, field, institution, graduation year).
- Experience entry extraction anchored on robust date ranges.
- Title and company parsing from standard delimiters and adjacent lines.
- Union-of-intervals total_years_experience calculation with month granularity
  and deterministic as_of parameter.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.schemas import EducationEntry, ExperienceEntry

logger = logging.getLogger(__name__)

# Standard month abbreviation/name mapping
MONTH_MAP = {
    "jan": 1, "jan.": 1, "january": 1,
    "feb": 2, "feb.": 2, "february": 2,
    "mar": 3, "mar.": 3, "march": 3,
    "apr": 4, "apr.": 4, "april": 4,
    "may": 5,
    "jun": 6, "jun.": 6, "june": 6,
    "jul": 7, "jul.": 7, "july": 7,
    "aug": 8, "aug.": 8, "august": 8,
    "sep": 9, "sep.": 9, "sept": 9, "sept.": 9, "september": 9,
    "oct": 10, "oct.": 10, "october": 10,
    "nov": 11, "nov.": 11, "november": 11,
    "dec": 12, "dec.": 12, "december": 12,
}

MONTH_REGEX_STR = (
    r"(?:jan(?:uary|\.)?|feb(?:ruary|\.)?|mar(?:ch|\.)?|apr(?:il|\.)?|may|"
    r"jun(?:e|\.)?|jul(?:y|\.)?|aug(?:ust|\.)?|sep(?:tember|\.|\b)|sept(?:\.|\b)|"
    r"oct(?:ober|\.)?|nov(?:ember|\.)?|dec(?:ember|\.)?)"
)

CURRENT_REGEX_STR = r"(?:present|current|ongoing|till\s*date|now)"

# Date range separator: -, –, —, to, /
SEP_REGEX_STR = r"(?:\s*(?:[-–—]|to)\s*)"

# Pattern matching a complete date range
DATE_RANGE_PATTERN = re.compile(
    rf"(?i)\b("
    # Start date
    rf"(?:{MONTH_REGEX_STR}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}-\d{{2}}|\d{{4}})"
    rf")"
    rf"{SEP_REGEX_STR}"
    rf"("
    # End date
    rf"{CURRENT_REGEX_STR}|{MONTH_REGEX_STR}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}-\d{{2}}|\d{{4}}"
    rf")\b"
)

# Known degrees and normalizations
DEGREE_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"(?i)\b(?:bachelor\s+of\s+science\s+in\s+nursing|bsn)\b"),
        "Bachelor of Science in Nursing",
        "Nursing",
    ),
    (
        re.compile(r"(?i)\b(?:associate\s+of\s+applied\s+science|aas)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Associate of Applied Science",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:associate\s+of\s+science|as)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Associate of Science",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:associate\s+of\s+arts|aa)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Associate of Arts",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:associate\s+degree|associate)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Associate Degree",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:master\s+of\s+science|ms|m\.s\.|msc|m\.sc)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Master of Science",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:master\s+of\s+business\s+administration|mba)\b"),
        "Master of Business Administration",
        "Business Administration",
    ),
    (
        re.compile(r"(?i)\b(?:master\s+of\s+arts|ma|m\.a\.)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Master of Arts",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:doctor\s+of\s+philosophy|phd|ph\.d\.)\s*(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Doctor of Philosophy",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:bachelor\s+of\s+technology|b\.tech|btech)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Bachelor of Technology",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:bachelor\s+of\s+engineering|b\.e\.|be)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Bachelor of Engineering",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:bachelor\s+of\s+science|bs|b\.s\.|bsc|b\.sc)\s+(?:in\s+|,?\s*)([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Bachelor of Science",
        r"\1",
    ),
    (
        re.compile(r"(?i)\b(?:bachelor\s+of\s+arts|ba|b\.a\.)\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Bachelor of Arts",
        r"\1",
    ),
    (
        re.compile(r"(?i)\bdiploma\s+(?:in\s+)?([A-Za-z\s]+?)(?=[—\-,|\n]|$)"),
        "Diploma",
        r"\1",
    ),
]


def parse_date_token(token: str, is_end: bool = False, as_of: str = "2026-11-01") -> Tuple[int, int]:
    """Parse a date token into (year, month).

    Supports:
    - 'Present', 'Current', etc. -> as_of date
    - 'YYYY-MM'
    - 'MM/YYYY'
    - 'Month YYYY'
    - 'YYYY' -> (YYYY, 1) for start, (YYYY, 12) for end
    """
    token = token.strip()
    token_lower = token.lower()

    if re.match(rf"^{CURRENT_REGEX_STR}$", token_lower):
        parts = as_of.split("-")
        return int(parts[0]), int(parts[1])

    # YYYY-MM
    m_iso = re.match(r"^(\d{4})-(\d{1,2})$", token)
    if m_iso:
        return int(m_iso.group(1)), int(m_iso.group(2))

    # MM/YYYY
    m_slash = re.match(r"^(\d{1,2})/(\d{4})$", token)
    if m_slash:
        return int(m_slash.group(2)), int(m_slash.group(1))

    # Month YYYY
    m_month = re.match(rf"^({MONTH_REGEX_STR})\s+(\d{{4}})$", token, re.IGNORECASE)
    if m_month:
        m_str = m_month.group(1).lower()
        year = int(m_month.group(2))
        month = MONTH_MAP.get(m_str, 1)
        return year, month

    # Just 4-digit YYYY
    m_year = re.match(r"^(\d{4})$", token)
    if m_year:
        year = int(m_year.group(1))
        month = 12 if is_end else 1
        return year, month

    return 2020, 1


def format_date_output(year: int, month: int, is_current: bool = False) -> str:
    """Format date to standard YYYY-MM or 'Present'."""
    if is_current:
        return "Present"
    return f"{year:04d}-{month:02d}"


def extract_education(text: str) -> List[EducationEntry]:
    """Extract structured education entries from resume text."""
    if not text or not text.strip():
        return []

    entries: List[EducationEntry] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Patterns for institutions
    inst_pattern = re.compile(
        r"(?:university|college|institute|school|academy|polytechnic|campus|uc\s+[A-Za-z]+|stanford|mit|harvard)",
        re.IGNORECASE,
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for degree match in line
        matched_degree: Optional[str] = None
        matched_field: str = ""
        for pattern, deg_name, field_tmpl in DEGREE_PATTERNS:
            m = pattern.search(line)
            if m:
                matched_degree = deg_name
                if "\\" in field_tmpl:
                    matched_field = m.expand(field_tmpl).strip()
                else:
                    matched_field = field_tmpl.strip()
                break

        if not matched_degree:
            i += 1
            continue

        # Look for institution and year in current line and next 2 lines
        search_block = [line]
        if i + 1 < len(lines):
            search_block.append(lines[i + 1])
        if i + 2 < len(lines):
            search_block.append(lines[i + 2])
        block_text = " — ".join(search_block)

        # Extract institution
        institution = ""
        # Check standard delimiters on current line first: '—', '-', '|', 'at'
        parts = re.split(r"\s*(?:—|–| - |\|| at )\s*", line)
        for p in parts[1:]:
            p_clean = re.sub(r",?\s*\b(?:19\d{2}|20\d{2})\b.*", "", p).strip()
            if inst_pattern.search(p_clean) or len(p_clean) > 3:
                institution = p_clean
                break

        if not institution and i + 1 < len(lines):
            next_line = lines[i + 1]
            if inst_pattern.search(next_line):
                institution = re.sub(r",?\s*\b(?:19\d{2}|20\d{2})\b.*", "", next_line).strip()
                i += 1

        # Extract graduation year
        grad_year: Optional[int] = None
        year_m = re.search(r"\b(?:graduated:?\s*|expected\s+)?(19\d{2}|20\d{2})\b", block_text, re.IGNORECASE)
        if year_m:
            # Check if this year was part of a trap like Grade C honor
            if "grade" not in line.lower() or year_m.group(1) in line:
                grad_year = int(year_m.group(1))

        # Clean field: strip trailing punctuation or institution words
        matched_field = re.sub(r"^[,\s–—]+|[,\s–—]+$", "", matched_field).strip()
        matched_field = re.sub(r"\s*(?:university|college|institute|school).*", "", matched_field, flags=re.IGNORECASE).strip()

        # Clean institution: strip trailing location or year
        institution = re.sub(r"\s*,\s*(?:[A-Z]{2}|\d{5}|[A-Za-z\s]+)$", lambda m: "" if re.search(r"\b[A-Z]{2}\b", m.group()) else m.group(), institution).strip()

        entries.append(
            EducationEntry(
                degree=matched_degree,
                field=matched_field,
                institution=institution,
                graduation_year=grad_year,
                end_year=grad_year,
            )
        )
        i += 1

    return entries


def clean_title_company(raw_text: str) -> Tuple[str, str]:
    """Parse title and company from a single line with standard delimiters."""
    # Delimiters: '—', '–', '|', ' at ', ' - '
    # If multiple delimiters, e.g.:
    # "Senior Specialist — Core Platform — Example Corporation, Chicago, IL"
    delims = [m.start() for m in re.finditer(r"\s*(?:—|–|\||\bat\b| - )\s*", raw_text, re.IGNORECASE)]
    if not delims:
        return raw_text.strip(), ""

    # Split into parts
    parts = re.split(r"\s*(?:—|–|\||\bat\b| - )\s*", raw_text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 1:
        return parts[0], ""
    elif len(parts) == 2:
        title = parts[0]
        company_raw = parts[1]
    else:
        # e.g. ["Senior Specialist", "Core Platform", "Example Corporation, Chicago, IL"]
        # Combine leading parts into Title if they look like department/role
        title = f"{parts[0]} – {parts[1]}"
        company_raw = parts[2]

    # Clean location (city, state) from company
    # E.g. "Example Corporation, Chicago, IL" -> "Example Corporation"
    company = re.sub(r",\s*[A-Za-z\s]+,\s*[A-Z]{2}\b", "", company_raw).strip()
    company = re.sub(r",\s*[A-Z]{2}\b", "", company).strip()

    return title, company


def extract_experience(
    text: str,
    as_of: str = "2026-11-01",
) -> List[ExperienceEntry]:
    """Extract structured experience records from text."""
    if not text or not text.strip():
        return []

    entries: List[ExperienceEntry] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Find lines containing date ranges
    i = 0
    while i < len(lines):
        line = lines[i]
        date_match = DATE_RANGE_PATTERN.search(line)

        if not date_match:
            i += 1
            continue

        raw_start = date_match.group(1).strip()
        raw_end = date_match.group(2).strip()

        is_current = bool(re.match(rf"^{CURRENT_REGEX_STR}$", raw_end, re.IGNORECASE))
        sy, sm = parse_date_token(raw_start, is_end=False, as_of=as_of)
        ey, em = parse_date_token(raw_end, is_end=True, as_of=as_of)

        start_str = format_date_output(sy, sm, is_current=False)
        end_str = format_date_output(ey, em, is_current=is_current)

        # Compute duration in months
        end_inc = 0 if is_current else 1
        duration_months = max(1, (ey - sy) * 12 + (em - sm) + end_inc)

        # Find title and company from adjacent lines
        title = ""
        company = ""

        # Case 1: Title & company are on the same line before the date range
        date_line_prefix = line[: date_match.start()].strip()
        if date_line_prefix and len(date_line_prefix) > 3:
            title, company = clean_title_company(date_line_prefix)
        # Case 2: Title & company are on the previous line(s)
        elif i > 0:
            prev_line = lines[i - 1]
            title, company = clean_title_company(prev_line)
            # If company missing and line before prev exists:
            if not company and i > 1:
                title_line = lines[i - 2]
                title = title_line
                company = prev_line

        # Extract bullet points following this entry until next date range
        bullets: List[str] = []
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            if DATE_RANGE_PATTERN.search(next_line):
                break
            # Bullet point indicator or descriptive line
            if re.match(r"^[-*•+]\s*", next_line):
                cleaned_bullet = re.sub(r"^[-*•+]\s*", "", next_line).strip()
                if cleaned_bullet:
                    bullets.append(cleaned_bullet)
            j += 1

        entries.append(
            ExperienceEntry(
                title=title,
                company=company,
                start=start_str,
                end=end_str,
                start_date=start_str,
                end_date=end_str,
                is_current=is_current,
                duration_months=duration_months,
                bullets=bullets,
            )
        )
        i = j if j > i + 1 else i + 1

    return entries


def compute_total_years_experience(
    entries: List[ExperienceEntry],
    as_of: str = "2026-11-01",
) -> float:
    """Compute total experience years as the union of monthly intervals."""
    if not entries:
        return 0.0

    all_months: Set[int] = set()

    for entry in entries:
        s_date = entry.start or entry.start_date
        e_date = entry.end or entry.end_date
        if not s_date:
            continue

        sy, sm = parse_date_token(s_date, is_end=False, as_of=as_of)
        ey, em = parse_date_token(e_date or "Present", is_end=True, as_of=as_of)

        # Exclusive of end month for ongoing roles, inclusive (+1) for past completed roles
        end_inc = 0 if entry.is_current else 1
        for m in range(sy * 12 + sm, ey * 12 + em + end_inc):
            all_months.add(m)

    total_years = len(all_months) / 12.0
    return round(total_years, 1)

