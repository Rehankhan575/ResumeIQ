"""
app/extraction/skills.py
========================
Taxonomy-based skill extractor for Stage 2 extraction.

Features:
- Loads active taxonomy skills (active=True in data/taxonomies/skills_stats.csv),
  canonical skills, aliases, and ambiguous term disambiguation rules.
- 1-5 word n-gram lookup with longest-match-first non-overlapping span resolution.
- Context and case validation for ambiguous terms according to ambiguous_terms.csv.
- Resume skill extraction with graceful fallback to full text.
- JD skill extraction splitting required vs preferred skills and parsing min_years.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.config import settings
from app.extraction.text_utils import (
    find_raw_token_spans,
    normalize_text,
    tokenize,
)

logger = logging.getLogger(__name__)

# Canonical skills section names
SKILL_SECTIONS = {
    "skills",
    "technical_skills",
    "core_competencies",
    "areas_of_expertise",
    "technologies",
    "tools",
    "key_skills",
    "proficiencies",
    "technical_proficiencies",
    "skills_and_expertise",
}

# Negative prefix patterns indicating non-skill usage
INVALID_PREFIX_TERMS = {
    "c": {"grade", "class", "level", "vitamin", "type", "schedule", "section", "category", "room"},
    "r": {"grade", "class", "level", "vitamin", "phase", "r&d", "r & d"},
    "excel": {"to", "can", "will", "excelled"},
    "word": {"a", "one", "the", "every", "in", "key", "written", "spoken", "his", "her", "my"},
    "access": {"to", "have", "gain", "grant", "give", "provide", "unauthorized", "direct"},
    "rest": {"the", "a", "of"},
}

HEALTHCARE_CONTEXT_TERMS = {
    "patient", "vital", "signs", "nursing", "nurse", "emergency", "clinical", "bls",
    "wound", "care", "hospital", "cardiac", "trauma", "physician", "icu", "medical",
    "health", "healthcare", "acuity", "resuscitation", "intravenous", "bedside",
}

CULINARY_CONTEXT_TERMS = {
    "chef", "kitchen", "culinary", "cook", "cooking", "food", "recipe", "menu",
    "scratch", "banquet", "pastry", "catering", "dining", "restaurant",
}

DEVOPS_CONTEXT_TERMS = {
    "devops", "cloud", "docker", "kubernetes", "ansible", "puppet", "terraform",
    "ci/cd", "pipeline", "infrastructure", "linux", "automation", "sysadmin",
}


class TaxonomySkillExtractor:
    """Extracts canonical skills from resumes and job descriptions."""

    def __init__(
        self,
        skills_csv_path: Optional[Path] = None,
        skills_stats_csv_path: Optional[Path] = None,
        ambiguous_terms_csv_path: Optional[Path] = None,
    ) -> None:
        self.skills_csv_path = skills_csv_path or settings.skills_csv_path
        self.skills_stats_csv_path = skills_stats_csv_path or settings.skills_stats_csv_path
        self.ambiguous_terms_csv_path = ambiguous_terms_csv_path or settings.ambiguous_terms_csv_path

        self.term_to_canonical: Dict[str, str] = {}
        self.canonical_category: Dict[str, str] = {}
        self.active_skills: Set[str] = set()
        self.ambiguous_rules: Dict[str, str] = {}
        self.max_ngram_len: int = 1

        self._load_taxonomies()

    def _load_taxonomies(self) -> None:
        """Load active taxonomy skills, aliases, and ambiguous term rules."""
        logger.debug("Loading skills taxonomy and statistics...")
        stats_df = pd.read_csv(self.skills_stats_csv_path, keep_default_na=False)
        active_mask = stats_df["active"].astype(str).str.lower() == "true"
        self.active_skills = set(stats_df[active_mask]["skill"].str.lower())

        ambig_df = pd.read_csv(self.ambiguous_terms_csv_path, keep_default_na=False)
        for _, row in ambig_df.iterrows():
            t = str(row["term"]).strip().lower()
            r = str(row["rule"]).strip().lower()
            if t:
                self.ambiguous_rules[t] = r

        skills_df = pd.read_csv(self.skills_csv_path, keep_default_na=False)
        for _, row in skills_df.iterrows():
            canonical = str(row["skill"]).strip()
            canonical_lower = canonical.lower()

            # Strictly ignore inactive skills
            if canonical_lower not in self.active_skills:
                continue

            category = str(row.get("category", "")).strip()
            self.canonical_category[canonical] = category

            # Add canonical name tokens
            c_tokens = tokenize(canonical)
            if c_tokens:
                key = " ".join(c_tokens)
                self.term_to_canonical[key] = canonical
                self.max_ngram_len = max(self.max_ngram_len, len(c_tokens))

            # Add aliases
            aliases_str = str(row.get("aliases", "")).strip()
            if aliases_str:
                for alias in aliases_str.split("|"):
                    alias = alias.strip()
                    if not alias:
                        continue
                    a_tokens = tokenize(alias)
                    if a_tokens:
                        a_key = " ".join(a_tokens)
                        self.term_to_canonical[a_key] = canonical
                        self.max_ngram_len = max(self.max_ngram_len, len(a_tokens))

        self.max_ngram_len = min(self.max_ngram_len, 5)
        logger.debug(
            f"TaxonomySkillExtractor loaded: {len(self.active_skills)} active skills, "
            f"{len(self.term_to_canonical)} lookup keys, max ngram {self.max_ngram_len}."
        )

    def _check_ambiguity(
        self,
        canonical: str,
        raw_slice: str,
        prev_raw_token: str,
        next_raw_token: str,
        section_name: str,
        line_context: str,
        full_text: str,
    ) -> bool:
        """Validate an ambiguous term match according to rules and context."""
        canonical_lower = canonical.lower()
        rule = self.ambiguous_rules.get(canonical_lower)
        if not rule:
            return True

        is_skill_sec = section_name.lower().strip() in SKILL_SECTIONS

        # Check list context: bullet points, commas, pipes, slashes, semicolons
        is_list = bool(
            re.search(r"^\s*[-*•\d+.]", line_context)
            or re.search(r"[,|/•;]\s*", line_context)
        )

        # 1. Negative prefixes and exact traps
        prev_lower = prev_raw_token.lower()
        next_lower = next_raw_token.lower()

        if canonical_lower in INVALID_PREFIX_TERMS:
            if prev_lower in INVALID_PREFIX_TERMS[canonical_lower]:
                return False

        # Trap: R&D (or R & D)
        if canonical_lower == "r":
            if next_lower in {"&", "d"} or prev_lower in {"&", "r"}:
                return False
            if "r&d" in raw_slice.lower() or "r & d" in line_context.lower():
                # Verify raw slice itself is not part of R&D
                if re.search(r"\bR\s*&\s*D\b", line_context):
                    return False

        # Trap: go-to-market
        if canonical_lower == "go":
            if "go-to-market" in line_context.lower():
                return False

        # Trap: excelled at
        if canonical_lower == "excel":
            if "excelled" in line_context.lower() or "excel at" in line_context.lower():
                return False

        # Trap: chef (DevOps tool vs culinary profession)
        if canonical_lower == "chef":
            # If in culinary text / title, reject
            norm_full = full_text.lower()
            culinary_hits = sum(1 for w in CULINARY_CONTEXT_TERMS if w in norm_full)
            devops_hits = sum(1 for w in DEVOPS_CONTEXT_TERMS if w in norm_full)
            if culinary_hits > 2 and devops_hits == 0:
                return False
            if not is_skill_sec and not any(w in line_context.lower() for w in DEVOPS_CONTEXT_TERMS):
                return False

        # 2. Case sensitivity / exact-case checks
        if rule == "case_sensitive" or rule == "require_context":
            # Short single/two letter tokens need strict casing
            if canonical_lower in {"c", "r"}:
                if raw_slice != canonical.upper():  # Must be 'C' or 'R'
                    return False
            elif canonical_lower == "go":
                if raw_slice not in {"Go", "GO", "Golang"}:
                    return False
            elif canonical_lower in {"java", "jira"}:
                if raw_slice not in {canonical.capitalize(), canonical.upper()}:
                    return False
            elif canonical_lower in {"react", "swift", "rust", "spark", "puppet", "spring", "bash", "git", "dart", "sass", "yarn"}:
                if raw_slice.islower() and not is_skill_sec and not is_list:
                    return False
            elif canonical_lower in {"excel", "word", "access"}:
                if raw_slice.islower() and not is_skill_sec:
                    return False

        # 3. Domain context checks for triage, baking, grilling, roasting, welding
        if canonical_lower == "triage":
            norm_ctx = (line_context + " " + full_text).lower()
            has_healthcare = any(term in norm_ctx for term in HEALTHCARE_CONTEXT_TERMS)
            if not has_healthcare and not is_skill_sec:
                return False

        if canonical_lower in {"baking", "grilling", "roasting"}:
            norm_ctx = (line_context + " " + full_text).lower()
            has_culinary = any(term in norm_ctx for term in CULINARY_CONTEXT_TERMS)
            if not has_culinary and not is_skill_sec:
                return False

        # 4. Context rules (require_context vs exclude_standalone)
        if rule == "exclude_standalone":
            # Soft skills / generic terms: management, project, sales, communication, etc.
            # Only valid in a dedicated skills section or structured list
            if not is_skill_sec and not is_list:
                return False

        if rule == "require_context":
            # Must be in a skills section, a list context, or near another skill
            if not is_skill_sec and not is_list:
                # If neither, check if it's triage in healthcare
                if canonical_lower == "triage":
                    norm_line = line_context.lower()
                    if any(term in norm_line for term in HEALTHCARE_CONTEXT_TERMS):
                        return True
                return False

        return True

    def extract_from_text(
        self,
        text: str,
        section_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract canonical skills from text using longest-match n-grams."""
        if not text or not text.strip():
            return []

        token_spans = find_raw_token_spans(text)
        n_tokens = len(token_spans)
        if n_tokens == 0:
            return []

        # Find candidate matches
        candidates: List[Tuple[int, int, int, str, str, str, int, int]] = []
        for length in range(self.max_ngram_len, 0, -1):
            for i in range(n_tokens - length + 1):
                ngram_tokens = [token_spans[j][1] for j in range(i, i + length)]
                ngram_key = " ".join(ngram_tokens)
                if ngram_key in self.term_to_canonical:
                    canonical = self.term_to_canonical[ngram_key]
                    start_char = token_spans[i][2]
                    end_char = token_spans[i + length - 1][3]
                    raw_slice = text[start_char:end_char]
                    candidates.append((
                        length,
                        i,
                        i + length,
                        ngram_key,
                        canonical,
                        raw_slice,
                        start_char,
                        end_char,
                    ))

        # Longest match first: length DESC, start index ASC
        candidates.sort(key=lambda x: (-x[0], x[1]))

        matched_token_indices: Set[int] = set()
        matched_results: List[Dict[str, Any]] = []

        for length, start_idx, end_idx, ngram_key, canonical, raw_slice, start_char, end_char in candidates:
            # Check overlap
            if any(idx in matched_token_indices for idx in range(start_idx, end_idx)):
                continue

            # Context extraction for ambiguity check
            prev_token = token_spans[start_idx - 1][0] if start_idx > 0 else ""
            next_token = token_spans[end_idx][0] if end_idx < n_tokens else ""

            # Line context surrounding span
            line_start = text.rfind("\n", 0, start_char)
            line_start = 0 if line_start == -1 else line_start + 1
            line_end = text.find("\n", end_char)
            line_end = len(text) if line_end == -1 else line_end
            line_context = text[line_start:line_end]

            if not self._check_ambiguity(
                canonical=canonical,
                raw_slice=raw_slice,
                prev_raw_token=prev_token,
                next_raw_token=next_token,
                section_name=section_name,
                line_context=line_context,
                full_text=text,
            ):
                continue

            # Mark tokens as consumed
            for idx in range(start_idx, end_idx):
                matched_token_indices.add(idx)

            category = self.canonical_category.get(canonical, "")
            matched_results.append({
                "skill": canonical,
                "category": category,
                "section": section_name,
                "start": start_char,
                "end": end_char,
            })

        return matched_results

    def extract_canonical_skills(
        self,
        text: str,
        section_name: str = "",
    ) -> List[str]:
        """Return deduplicated list of canonical skill names preserved in order."""
        matches = self.extract_from_text(text, section_name=section_name)
        seen: Set[str] = set()
        deduped: List[str] = []
        for m in matches:
            s = m["skill"].lower()
            if s not in seen:
                seen.add(s)
                deduped.append(m["skill"])
        return deduped

    def extract_skills_from_resume_sections(
        self,
        sections: Dict[str, str],
        raw_text: str = "",
    ) -> List[str]:
        """Extract skills from structured resume sections, falling back to full text."""
        # 1. Search dedicated skill sections
        skill_section_texts = []
        for sec_name, sec_text in sections.items():
            if sec_name.lower().strip() in SKILL_SECTIONS:
                skill_section_texts.append((sec_name, sec_text))

        if skill_section_texts:
            all_skills: List[str] = []
            seen: Set[str] = set()
            for sec_name, sec_text in skill_section_texts:
                extracted = self.extract_canonical_skills(sec_text, section_name=sec_name)
                for s in extracted:
                    if s.lower() not in seen:
                        seen.add(s.lower())
                        all_skills.append(s)
            return all_skills

        # 2. Fall back to scanning all sections or raw text
        target_text = raw_text if raw_text else "\n\n".join(sections.values())
        return self.extract_canonical_skills(target_text, section_name="")

    def extract_jd_skills(self, text: str) -> Dict[str, Any]:
        """Extract required skills, preferred skills, and min_years from JD text."""
        if not text:
            return {"required_skills": [], "preferred_skills": [], "min_years": None}

        # Parse min_years
        min_years: Optional[float] = None
        # Match patterns like:
        # "Minimum 4+ years", "4+ years of experience", "Minimum 5 years", "3-5 years"
        m_exp = re.search(
            r"(?:minimum|at\s*least)?\s*(\d+(?:\.\d+)?)\+?\s*(?:to|-)?\s*(?:\d+\+?)?\s*years?\b",
            text,
            re.IGNORECASE,
        )
        if m_exp:
            try:
                val = float(m_exp.group(1))
                min_years = int(val) if val.is_integer() else val
            except ValueError:
                pass

        # Split into required vs preferred sections
        req_pattern = re.compile(
            r"(?:required\s*(?:skills|qualifications|requirements)?|qualifications|requirements|must\s*have):?",
            re.IGNORECASE,
        )
        pref_pattern = re.compile(
            r"(?:preferred\s*(?:skills|qualifications|requirements)?|nice\s*to\s*have|bonus|desired):?",
            re.IGNORECASE,
        )

        req_match = req_pattern.search(text)
        pref_match = pref_pattern.search(text)

        required_skills: List[str] = []
        preferred_skills: List[str] = []

        if req_match and pref_match:
            if req_match.start() < pref_match.start():
                req_text = text[req_match.end() : pref_match.start()]
                pref_text = text[pref_match.end() :]
            else:
                pref_text = text[pref_match.end() : req_match.start()]
                req_text = text[req_match.end() :]

            required_skills = self.extract_canonical_skills(req_text, section_name="skills")
            preferred_skills = self.extract_canonical_skills(pref_text, section_name="skills")
        elif req_match:
            req_text = text[req_match.end() :]
            required_skills = self.extract_canonical_skills(req_text, section_name="skills")
        elif pref_match:
            pref_text = text[pref_match.end() :]
            preferred_skills = self.extract_canonical_skills(pref_text, section_name="skills")
        else:
            required_skills = self.extract_canonical_skills(text, section_name="skills")

        return {
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "min_years": min_years,
        }
