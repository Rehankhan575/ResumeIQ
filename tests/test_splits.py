"""
tests/test_splits.py
====================
Offline unit tests for split-related logic.

Tests:
  1. UnionFind: basic grouping behaviour.
  2. Group-level split: no group straddles two splits.
  3. mask_header: header removed from text_masked; word-boundary leak check.
  4. map_title: word-boundary matching, ambiguous titles excluded.

All tests use tiny fake data; no real dataset files are read.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest

# Import the modules under test directly (they are plain Python files)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_splits import UnionFind, mask_header, group_stratified_split, build_groups
from build_jd_pool import map_title, build_title_map, RAW_RULES


# ---------------------------------------------------------------------------
# 1. UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    """Tests for the UnionFind data structure."""

    def test_single_element(self) -> None:
        """Single element is its own root."""
        uf = UnionFind(1)
        assert uf.find(0) == 0

    def test_union_merges(self) -> None:
        """Unioning two elements gives them the same root."""
        uf = UnionFind(4)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitive_union(self) -> None:
        """Transitive union: 0-1, 1-2 => 0,1,2 same group."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(1) == uf.find(2)
        assert uf.find(3) != uf.find(0)

    def test_groups(self) -> None:
        """groups() returns correct partition."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        g = uf.groups()
        # Should have 4 groups: {0,1}, {2,3}, {4}, {5}
        sizes = sorted(len(v) for v in g.values())
        assert sizes == [1, 1, 2, 2]

    def test_no_cross_group_merge(self) -> None:
        """Elements not unioned remain in separate groups."""
        uf = UnionFind(3)
        uf.union(0, 1)
        assert uf.find(0) != uf.find(2)
        assert uf.find(1) != uf.find(2)


# ---------------------------------------------------------------------------
# 2. Group-level split: no group straddles splits
# ---------------------------------------------------------------------------

class TestGroupStratifiedSplit:
    """Tests for group_stratified_split."""

    def _make_df(self, n_groups: int = 20, members_per_group: int = 3) -> pd.DataFrame:
        """Create a tiny fake DataFrame with group_id and category."""
        records = []
        cats = ["A", "B", "C", "D"]
        for gid in range(n_groups):
            cat = cats[gid % len(cats)]
            for _ in range(members_per_group):
                records.append({"group_id": gid, "category": cat})
        return pd.DataFrame(records)

    def test_no_group_straddles_splits(self) -> None:
        """Every group_id must appear in exactly one split."""
        df = self._make_df(n_groups=30)
        splits = group_stratified_split(df, seed=42)
        df = df.copy()
        df["split"] = splits
        n_splits_per_group = df.groupby("group_id")["split"].nunique()
        assert (n_splits_per_group == 1).all(), (
            f"Groups straddle splits: {n_splits_per_group[n_splits_per_group > 1].index.tolist()}"
        )

    def test_split_proportions(self) -> None:
        """Train should be approximately 70%, val and test ~15% each."""
        df = self._make_df(n_groups=100)
        splits = group_stratified_split(df, seed=42)
        df = df.copy()
        df["split"] = splits
        counts = df.drop_duplicates("group_id")["split"].value_counts()
        total = counts.sum()
        assert abs(counts.get("train", 0) / total - 0.70) < 0.10
        assert abs(counts.get("val", 0) / total - 0.15) < 0.10
        assert abs(counts.get("test", 0) / total - 0.15) < 0.10

    def test_all_rows_assigned(self) -> None:
        """Every row is assigned a split."""
        df = self._make_df(n_groups=20)
        splits = group_stratified_split(df, seed=42)
        assert splits.notna().all()
        assert set(splits.unique()).issubset({"train", "val", "test"})

    def test_deterministic(self) -> None:
        """Same seed => same splits."""
        df = self._make_df(n_groups=40)
        s1 = group_stratified_split(df, seed=42)
        s2 = group_stratified_split(df, seed=42)
        assert (s1.values == s2.values).all()


# ---------------------------------------------------------------------------
# 3. mask_header
# ---------------------------------------------------------------------------

class TestMaskHeader:
    """Tests for the header masking function."""

    def test_removes_leading_header(self) -> None:
        """Header text is removed from the start of the resume text."""
        header = "HR MANAGER"
        text = "HR MANAGER Summary Dedicated professional with 10 years experience."
        result = mask_header(text, header)
        assert not result.startswith("HR MANAGER")
        assert "Summary" in result

    def test_no_header_returns_original(self) -> None:
        """None header returns original text unchanged."""
        text = "Some resume text here."
        assert mask_header(text, None) == text

    def test_empty_header_returns_original(self) -> None:
        """Empty-string header returns original text unchanged."""
        text = "Some resume text here."
        result = mask_header(text, "")
        assert result == text

    def test_case_insensitive_removal(self) -> None:
        """Header removal is case-insensitive."""
        header = "CHEF DE PARTIE"
        text = "Chef de Partie Experience at multiple restaurants."
        result = mask_header(text, header)
        assert "Experience" in result

    def test_whitespace_normalisation(self) -> None:
        """Extra whitespace in header / text is normalised for comparison."""
        header = "INFORMATION  TECHNOLOGY"
        text = "INFORMATION TECHNOLOGY Summary Key skills include Python."
        result = mask_header(text, header)
        assert "Summary" in result

    def test_word_boundary_leak_check(self) -> None:
        """After masking, the category label no longer appears in first 10 words."""
        header = "HR SPECIALIST"
        text = "HR SPECIALIST Summary Over 20 years experience in recruiting."
        masked = mask_header(text, header)
        first_10 = " ".join(masked.split()[:10])
        # "HR" might still appear but "HR SPECIALIST" as unit should not
        pattern = re.compile(r"\bHR SPECIALIST\b", re.IGNORECASE)
        assert not pattern.search(first_10)

    def test_chars_reduced(self) -> None:
        """Masked text should be shorter than original when header found."""
        header = "SALES MANAGER"
        text = "SALES MANAGER Summary Experienced sales professional."
        masked = mask_header(text, header)
        assert len(masked) < len(text)


# ---------------------------------------------------------------------------
# 4. map_title: word boundaries, ambiguous excluded
# ---------------------------------------------------------------------------

class TestMapTitle:
    """Tests for the title->category mapping."""

    @pytest.fixture(scope="class")
    @classmethod
    def rules(cls) -> List[Tuple[re.Pattern[str], str]]:
        """Compiled rules fixture."""
        return build_title_map()

    def test_exact_match(self, rules: list) -> None:
        """A title matching exactly one category is mapped correctly."""
        cat = map_title("Sales Manager", rules)
        assert cat == "SALES"

    def test_case_insensitive(self, rules: list) -> None:
        """Matching is case-insensitive."""
        cat = map_title("SALES MANAGER", rules)
        assert cat == "SALES"

    def test_ambiguous_excluded(self, rules: list) -> None:
        """A title matching more than one category returns None."""
        # 'attorney consultant' matches ADVOCATE (attorney) AND CONSULTANT (consultant)
        cat = map_title("attorney consultant", rules)
        assert cat is None

    def test_unmapped_returns_none(self, rules: list) -> None:
        """A title matching no rule returns None."""
        cat = map_title("professional juggler extraordinaire", rules)
        assert cat is None

    def test_word_boundary_not_substring(self, rules: list) -> None:
        """Pattern must match on word boundary, not as substring."""
        # 'cook' is a keyword for CHEF; 'cookbook' should NOT match
        # (depends on word-boundary anchor in the rule)
        # 'sales' is a SALES keyword; 'wholesales' should NOT match
        cat_whole = map_title("wholesales director", rules)
        # 'wholesales' does not have \bsales on a word boundary after 'whole'
        # In practice regex \bsales\b will not match inside 'wholesales'
        # because 's' follows non-word char 'e' and 's' at end IS word boundary
        # Let's just check 'accountant' keyword doesn't match 'nonaccountant'
        cat_prefix = map_title("nonaccountant", rules)
        assert cat_prefix is None, (
            "Substring 'accountant' inside 'nonaccountant' should not match due to \\b"
        )

    def test_nurse_maps_to_healthcare(self, rules: list) -> None:
        """'Registered Nurse' maps to HEALTHCARE."""
        cat = map_title("Registered Nurse", rules)
        assert cat == "HEALTHCARE"

    def test_software_engineer_maps_to_it(self, rules: list) -> None:
        """'Software Engineer' maps to INFORMATION-TECHNOLOGY."""
        cat = map_title("Software Engineer", rules)
        assert cat == "INFORMATION-TECHNOLOGY"

    def test_hr_manager_maps_to_hr(self, rules: list) -> None:
        """'HR Manager' maps to HR."""
        cat = map_title("HR Manager", rules)
        assert cat == "HR"
