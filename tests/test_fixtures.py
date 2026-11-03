"""Unit tests for synthetic test fixtures and expected sidecars.

Verifies:
1. All fixtures open cleanly via PyMuPDF (PDFs) or python-docx (DOCX).
2. nurse_scanned.pdf is an image-only PDF with <20 extractable characters and >=1 image.
3. Resume sidecars contain all required schema keys and layout flags.
4. All expected skills in sidecars exist in data/taxonomies/skills.csv (canonical or alias).
5. JD fixtures exist, meet word count thresholds (250-400 words), and have valid sidecars.
6. The only email domain in any fixture or in notebooks/01_eda.ipynb is 'example.com'.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set

import fitz  # PyMuPDF
import pandas as pd
import pytest
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
TAXONOMY_CSV = ROOT / "data" / "taxonomies" / "skills.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "01_eda.ipynb"

RESUME_FIXTURES = [
    ("ds_clean.pdf", "ds_clean.expected.json"),
    ("swe_two_column.pdf", "swe_two_column.expected.json"),
    ("accountant_table.pdf", "accountant_table.expected.json"),
    ("chef_plain.docx", "chef_plain.expected.json"),
    ("nurse_scanned.pdf", "nurse_scanned.expected.json"),
]

JD_FIXTURES = [
    ("jd_data_scientist.txt", "jd_data_scientist.expected.json"),
    ("jd_software_engineer.txt", "jd_software_engineer.expected.json"),
    ("jd_chef.txt", "jd_chef.expected.json"),
]

EMAIL_DOMAIN_REGEX = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")


@pytest.fixture(scope="module")
def valid_skills() -> Set[str]:
    """Load canonical skills and aliases from taxonomy."""
    df = pd.read_csv(TAXONOMY_CSV, keep_default_na=False)
    skills: Set[str] = set(df["skill"].str.lower())
    for aliases in df["aliases"]:
        for alias in aliases.split("|"):
            alias = alias.strip().lower()
            if alias:
                skills.add(alias)
    return skills


def test_fixtures_exist():
    """Verify all fixture files and sidecars exist on disk."""
    for fixture_file, sidecar_file in RESUME_FIXTURES:
        assert (FIXTURES_DIR / fixture_file).exists(), f"Missing {fixture_file}"
        assert (FIXTURES_DIR / sidecar_file).exists(), f"Missing {sidecar_file}"

    for jd_file, sidecar_file in JD_FIXTURES:
        assert (FIXTURES_DIR / jd_file).exists(), f"Missing {jd_file}"
        assert (FIXTURES_DIR / sidecar_file).exists(), f"Missing {sidecar_file}"


def test_resume_fixtures_open():
    """Verify all resume fixture documents open without error."""
    for fixture_file, _ in RESUME_FIXTURES:
        path = FIXTURES_DIR / fixture_file
        if path.suffix == ".pdf":
            doc = fitz.open(path)
            assert len(doc) >= 1
            doc.close()
        elif path.suffix == ".docx":
            doc = Document(path)
            assert len(doc.paragraphs) > 0


def test_scanned_pdf_has_no_text_layer():
    """Verify nurse_scanned.pdf is an image-only scan (<20 text chars, >=1 image)."""
    path = FIXTURES_DIR / "nurse_scanned.pdf"
    doc = fitz.open(path)
    assert len(doc) == 1

    page = doc[0]
    extracted_text = page.get_text().strip()
    image_list = page.get_images()

    assert len(extracted_text) < 20, f"Expected <20 chars, got {len(extracted_text)}: '{extracted_text}'"
    assert len(image_list) >= 1, "Expected at least 1 raster image in scanned PDF"
    doc.close()


def test_resume_sidecar_schema():
    """Verify all resume sidecars conform to expected structure."""
    required_keys = {
        "as_of",
        "contact",
        "expected_sections",
        "skills",
        "education",
        "experience",
        "total_years_experience",
        "layout",
        "traps",
    }
    required_contact_keys = {"name", "email", "phone", "location"}
    required_layout_keys = {
        "page_count",
        "is_multi_column",
        "has_tables",
        "has_images",
        "needs_ocr",
    }

    for _, sidecar_file in RESUME_FIXTURES:
        path = FIXTURES_DIR / sidecar_file
        with open(path) as f:
            data = json.load(f)

        assert required_keys.issubset(data.keys()), f"Missing keys in {sidecar_file}: {required_keys - set(data.keys())}"
        assert data["as_of"] == "2026-11-01"
        assert required_contact_keys.issubset(data["contact"].keys())
        assert required_layout_keys.issubset(data["layout"].keys())
        assert isinstance(data["expected_sections"], list)
        assert isinstance(data["skills"], list)
        assert len(data["skills"]) >= 3
        assert isinstance(data["experience"], list)
        for exp in data["experience"]:
            assert {"title", "company", "start", "end", "is_current"}.issubset(exp.keys())
        assert isinstance(data["traps"], list)
        assert len(data["traps"]) >= 1


def test_expected_skills_exist_in_taxonomy(valid_skills: Set[str]):
    """Verify all skills declared in resume and JD sidecars exist in skills.csv."""
    all_sidecars = [s for _, s in RESUME_FIXTURES] + [s for _, s in JD_FIXTURES]
    for sidecar_file in all_sidecars:
        path = FIXTURES_DIR / sidecar_file
        with open(path) as f:
            data = json.load(f)

        skills = data.get("skills", [])
        skills += data.get("required_skills", [])
        skills += data.get("preferred_skills", [])

        for skill in skills:
            canonical_check = skill.strip().lower()
            assert canonical_check in valid_skills, (
                f"Skill '{skill}' from {sidecar_file} not found in taxonomy"
            )


def test_jd_fixtures_and_sidecars():
    """Verify JD fixtures meet word-count limits (250-400 words) and have valid sidecars."""
    required_jd_keys = {"as_of", "title", "required_skills", "preferred_skills", "min_years"}

    for jd_file, sidecar_file in JD_FIXTURES:
        text_path = FIXTURES_DIR / jd_file
        with open(text_path) as f:
            text = f.read()

        words = text.split()
        assert 250 <= len(words) <= 400, (
            f"{jd_file} word count {len(words)} outside [250, 400] range"
        )
        assert "years" in text.lower(), f"{jd_file} missing minimum-years specification"
        assert "required" in text.lower() or "requirements" in text.lower()
        assert "preferred" in text.lower()

        sidecar_path = FIXTURES_DIR / sidecar_file
        with open(sidecar_path) as f:
            data = json.load(f)

        assert required_jd_keys.issubset(data.keys()), f"Missing keys in {sidecar_file}"
        assert data["as_of"] == "2026-11-01"
        assert len(data["required_skills"]) >= 2
        assert len(data["preferred_skills"]) >= 1
        assert isinstance(data["min_years"], (int, float))


def test_email_domain_isolation():
    """Verify the only email domain in any fixture, sidecar, or the EDA notebook is example.com."""
    # Check fixtures and sidecars
    for path in FIXTURES_DIR.iterdir():
        if path.is_file():
            content = path.read_bytes().decode("latin1")
            matches = EMAIL_DOMAIN_REGEX.findall(content)
            for domain in matches:
                # Strip trailing punctuation if matched
                domain_clean = domain.rstrip(".-/)>'\"")
                assert domain_clean == "example.com", (
                    f"Forbidden email domain '{domain_clean}' found in {path.name}"
                )

    # Check EDA notebook
    assert NOTEBOOK_PATH.exists(), "EDA notebook not found"
    nb_content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    nb_domains = EMAIL_DOMAIN_REGEX.findall(nb_content)
    for domain in nb_domains:
        domain_clean = domain.rstrip(".-/)>'\"")
        assert domain_clean == "example.com", (
            f"Forbidden email domain '{domain_clean}' found in notebook"
        )
