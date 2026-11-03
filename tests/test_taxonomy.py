"""
tests/test_taxonomy.py
======================
Unit tests for the skills taxonomy and ambiguity rules.

Validates:
  1. No empty skill names.
  2. No duplicate keys (unique canonical skills).
  3. Every alias maps to exactly one skill; no alias collides with primary skills.
  4. Ambiguous terms all exist in skills.csv with valid rules.
  5. Minimum row count thresholds met (>=1,500 total, >=400 curated).
  6. Single-character entries strictly conform to allowlist ('c', 'r').
  7. Valid sources and non-empty categories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_TAXONOMIES = ROOT / "data" / "taxonomies"
SKILLS_PATH = DATA_TAXONOMIES / "skills.csv"
CURATED_PATH = DATA_TAXONOMIES / "skills_curated.csv"
AMBIGUOUS_PATH = DATA_TAXONOMIES / "ambiguous_terms.csv"

ALLOWLIST_SINGLE_CHAR: Set[str] = {"c", "r", "go"}
VALID_AMBIGUOUS_RULES: Set[str] = {
    "case_sensitive",
    "require_context",
    "exclude_standalone",
}
VALID_SOURCES: Set[str] = {"curated", "onet", "ner_synthetic"}


@pytest.fixture(scope="module")
def skills_df() -> pd.DataFrame:
    """Fixture to load skills.csv."""
    assert SKILLS_PATH.exists(), f"Missing {SKILLS_PATH}"
    df = pd.read_csv(SKILLS_PATH, keep_default_na=False)
    return df


@pytest.fixture(scope="module")
def curated_df() -> pd.DataFrame:
    """Fixture to load skills_curated.csv."""
    assert CURATED_PATH.exists(), f"Missing {CURATED_PATH}"
    df = pd.read_csv(CURATED_PATH, keep_default_na=False)
    return df


@pytest.fixture(scope="module")
def ambiguous_df() -> pd.DataFrame:
    """Fixture to load ambiguous_terms.csv."""
    assert AMBIGUOUS_PATH.exists(), f"Missing {AMBIGUOUS_PATH}"
    df = pd.read_csv(AMBIGUOUS_PATH, keep_default_na=False)
    return df


class TestTaxonomyIntegrity:
    """Tests for skills.csv integrity."""

    def test_schema_columns(self, skills_df: pd.DataFrame) -> None:
        """skills.csv must contain required columns."""
        expected = ["skill", "aliases", "category", "source"]
        assert list(skills_df.columns) == expected

    def test_no_empty_skill_names(self, skills_df: pd.DataFrame) -> None:
        """Skill names must not be empty, null, or whitespace-only."""
        assert (skills_df["skill"].str.strip() == "").sum() == 0

    def test_no_duplicate_skill_keys(self, skills_df: pd.DataFrame) -> None:
        """Every canonical skill name must be unique."""
        dups = skills_df[skills_df["skill"].duplicated(keep=False)]
        assert len(dups) == 0, f"Duplicate skill keys found: {dups['skill'].tolist()}"

    def test_minimum_row_count(self, skills_df: pd.DataFrame) -> None:
        """skills.csv must meet target minimum of 1,500 rows."""
        assert len(skills_df) >= 1500, f"skills.csv has only {len(skills_df)} rows (< 1500)"

    def test_sources_valid(self, skills_df: pd.DataFrame) -> None:
        """Sources must be one of curated, onet, or ner_synthetic."""
        sources = set(skills_df["source"].unique())
        assert sources.issubset(VALID_SOURCES), f"Invalid sources: {sources - VALID_SOURCES}"

    def test_categories_not_empty(self, skills_df: pd.DataFrame) -> None:
        """Every skill must have a non-empty category."""
        assert (skills_df["category"].str.strip() == "").sum() == 0

    def test_single_character_allowlist(self, skills_df: pd.DataFrame) -> None:
        """Single-character skills must be in the approved allowlist."""
        single_chars = skills_df[skills_df["skill"].str.len() == 1]["skill"].tolist()
        for ch in single_chars:
            assert ch in ALLOWLIST_SINGLE_CHAR, f"Unauthorized single-character skill: '{ch}'"

    def test_every_alias_maps_to_one_skill(self, skills_df: pd.DataFrame) -> None:
        """Every alias must map to exactly one skill; no cross-skill collisions."""
        all_skills: Set[str] = set(skills_df["skill"].tolist())
        alias_to_skill: Dict[str, str] = {}
        collisions: List[str] = []

        for _, row in skills_df.iterrows():
            skill = row["skill"]
            aliases_str = row["aliases"]
            if not aliases_str:
                continue

            for alias in aliases_str.split("|"):
                alias_clean = alias.strip().lower()
                assert alias_clean != "", f"Empty alias found in skill '{skill}'"
                assert alias_clean != skill, f"Alias '{alias_clean}' identical to skill '{skill}'"

                # Alias cannot conflict with any canonical skill name
                if alias_clean in all_skills:
                    collisions.append(
                        f"Alias '{alias_clean}' of skill '{skill}' conflicts with canonical skill '{alias_clean}'"
                    )

                # Alias cannot belong to multiple skills
                if alias_clean in alias_to_skill:
                    collisions.append(
                        f"Alias '{alias_clean}' mapped to multiple skills: '{alias_to_skill[alias_clean]}' and '{skill}'"
                    )
                alias_to_skill[alias_clean] = skill

        assert not collisions, f"Alias collisions detected:\n" + "\n".join(collisions[:10])


class TestCuratedTaxonomy:
    """Tests for skills_curated.csv."""

    def test_schema_columns(self, curated_df: pd.DataFrame) -> None:
        """skills_curated.csv must contain required columns."""
        expected = ["skill", "aliases", "category"]
        assert list(curated_df.columns) == expected

    def test_curated_minimum_count(self, curated_df: pd.DataFrame) -> None:
        """skills_curated.csv must contain at least 400 skills."""
        assert len(curated_df) >= 400, f"Curated skills count {len(curated_df)} < 400"

    def test_no_empty_curated_names(self, curated_df: pd.DataFrame) -> None:
        """Curated skills must not have empty names or categories."""
        assert (curated_df["skill"].str.strip() == "").sum() == 0
        assert (curated_df["category"].str.strip() == "").sum() == 0

    def test_no_duplicate_curated_keys(self, curated_df: pd.DataFrame) -> None:
        """Curated skills must be unique."""
        dups = curated_df[curated_df["skill"].duplicated(keep=False)]
        assert len(dups) == 0, f"Duplicate curated skills: {dups['skill'].tolist()}"


class TestAmbiguousTerms:
    """Tests for ambiguous_terms.csv."""

    def test_schema_columns(self, ambiguous_df: pd.DataFrame) -> None:
        """ambiguous_terms.csv must contain term and rule."""
        expected = ["term", "rule"]
        assert list(ambiguous_df.columns) == expected

    def test_all_ambiguous_terms_exist_in_skills(
        self, ambiguous_df: pd.DataFrame, skills_df: pd.DataFrame
    ) -> None:
        """All ambiguous terms must exist in skills.csv (as canonical skill or alias)."""
        all_skills: Set[str] = set(skills_df["skill"].tolist())
        for aliases_str in skills_df["aliases"]:
            if aliases_str:
                for a in aliases_str.split("|"):
                    all_skills.add(a.strip().lower())

        missing: List[str] = []
        for term in ambiguous_df["term"]:
            t_clean = term.strip().lower()
            if t_clean not in all_skills:
                missing.append(t_clean)

        assert not missing, f"Ambiguous terms missing from skills.csv: {missing}"

    def test_ambiguous_rules_valid(self, ambiguous_df: pd.DataFrame) -> None:
        """Every rule in ambiguous_terms.csv must be an approved rule type."""
        for _, row in ambiguous_df.iterrows():
            term = row["term"]
            rule = row["rule"]
            assert rule in VALID_AMBIGUOUS_RULES, (
                f"Invalid rule '{rule}' for term '{term}'. Must be in {VALID_AMBIGUOUS_RULES}"
            )

    def test_required_terms_present(self, ambiguous_df: pd.DataFrame) -> None:
        """Prompt-specified collision terms must all be present."""
        prompt_terms = {
            "go", "r", "c", "java", "swift", "rust", "spark", "excel",
            "word", "access", "sales", "project", "management",
        }
        actual_terms = set(ambiguous_df["term"].str.strip().str.lower().tolist())
        missing = prompt_terms - actual_terms
        assert not missing, f"Prompt-mandated collision terms missing from ambiguous_terms.csv: {missing}"

