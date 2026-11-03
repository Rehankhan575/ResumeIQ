"""
tests/test_parsing.py
=====================
Comprehensive test suite for Stage 1 document parsing.

Tests:
    1. Text cleaning and normalization (clean_text).
    2. Heading matching and section splitting (split_sections).
    3. Contact extraction (email, phone, LinkedIn, GitHub, name).
    4. In-memory edge cases and typed errors:
        - Empty bytes -> EmptyDocumentError
        - Truncated PDF -> CorruptedFileError
        - Encrypted PDF -> PasswordProtectedError
        - Oversize file -> FileTooLargeError
        - Document over max_pages -> Degrades with warning, doesn't crash
        - PDF bytes named .docx -> Detected by magic bytes and parses cleanly
        - DOCX bytes named .pdf -> Detected by magic bytes and parses cleanly
        - Unsupported extension -> UnsupportedFileTypeError
    5. Real fixture parsing validation across all 5 test files.
"""

from __future__ import annotations

import io
from pathlib import Path

import docx
import pymupdf
import pytest

from app.config import settings
from app.parsing import (
    CorruptedFileError,
    EmptyDocumentError,
    ExtractedDocument,
    FileTooLargeError,
    ParseError,
    PasswordProtectedError,
    UnsupportedFileTypeError,
    clean_text,
    detect_file_type,
    extract_candidate_name,
    extract_contact_info,
    extract_text,
    match_heading,
    parse_document,
    parse_resume,
    split_sections,
)
from app.schemas import ParsedResume

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# 1. Text Cleaning Tests
# ---------------------------------------------------------------------------

def test_clean_text_bullets() -> None:
    raw = "• Item one\n▪ Item two\n◦ Item three\n● Item four\n‣ Item five"
    cleaned = clean_text(raw)
    for line in cleaned.splitlines():
        assert line.startswith("- ")


def test_clean_text_zero_width_and_nfkc() -> None:
    # Contains zero-width space \u200b and fullwidth characters
    raw = "Hello\u200b \uFEFFWorld\u200D! Ｔｅｓｔ"
    cleaned = clean_text(raw)
    assert cleaned == "Hello World! Test"


def test_clean_text_dehyphenation() -> None:
    # Lowercase-to-lowercase across newline must de-hyphenate
    raw = "This is a compu-\ntational method.\nPre-\nPost\n2021-\n2022\ngo-to-market"
    cleaned = clean_text(raw)
    lines = cleaned.splitlines()
    assert "computational method." in lines[0]
    # Uppercase-uppercase or non-alpha must NOT de-hyphenate
    assert "Pre-" in lines[1]
    assert "Post" in lines[2]
    assert "go-to-market" in cleaned


def test_clean_text_blank_lines_limit() -> None:
    # At most 1 blank line (max 2 consecutive newlines)
    raw = "Paragraph 1\n\n\n\n\n\nParagraph 2"
    cleaned = clean_text(raw)
    assert cleaned == "Paragraph 1\n\nParagraph 2"


# ---------------------------------------------------------------------------
# 2. Section Splitting Tests
# ---------------------------------------------------------------------------

def test_match_heading_exact_and_synonyms() -> None:
    assert match_heading("Work History") == "experience"
    assert match_heading("Academics") == "education"
    assert match_heading("Tech Stack") == "skills"
    assert match_heading("Work Experience:") == "experience"
    assert match_heading("CERTIFICATIONS") == "certifications"
    assert match_heading("Core Competencies") == "skills"
    assert match_heading("Education & Credentials") == "education"


def test_match_heading_rejections() -> None:
    # Line too long (>4 words)
    assert match_heading("This is five words long here") is None
    # Trailing period
    assert match_heading("Experience.") is None
    # Empty
    assert match_heading("") is None


def test_split_sections_header_and_duplicate_concatenation() -> None:
    text = (
        "John Doe\nSoftware Engineer\n\n"
        "Experience\nCompany A, 2020\n\n"
        "Education\nMIT, 2019\n\n"
        "Work History\nCompany B, 2022"
    )
    sections = split_sections(text)
    assert "header" in sections
    assert "John Doe" in sections["header"]
    assert "education" in sections
    assert "experience" in sections
    # Experience and Work History both map to 'experience' and are concatenated
    assert "Company A, 2020" in sections["experience"]
    assert "Company B, 2022" in sections["experience"]


# ---------------------------------------------------------------------------
# 3. Contact Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_contact_info() -> None:
    header = "Jane Smith\njane.smith@example.com | 555-0199 | Austin, TX\nhttps://linkedin.com/in/janesmith github.com/janesmith"
    info = extract_contact_info(header_text=header, full_text=header)
    assert info.name == "Jane Smith"
    assert info.email == "jane.smith@example.com"
    assert info.phone == "555-0199"
    assert "linkedin.com/in/janesmith" in info.linkedin
    assert "github.com/janesmith" in info.github


def test_extract_candidate_name_filtering() -> None:
    # Heading phrases and non-name lines must be rejected
    assert extract_candidate_name("CURRICULUM VITAE\nTech Stack\nAlex Mercer") == "Alex Mercer"
    assert extract_candidate_name("Elena Rostova, BSN, RN") == "Elena Rostova"


# ---------------------------------------------------------------------------
# 4. In-Memory Edge Cases & Typed Errors
# ---------------------------------------------------------------------------

def test_empty_bytes_raises_empty_document_error() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text(b"", "empty.pdf")


def test_truncated_pdf_raises_corrupted_file_error() -> None:
    truncated_pdf = b"%PDF-1.5\n%truncated junk bytes that do not form a valid pdf xref\n"
    with pytest.raises(CorruptedFileError):
        extract_text(truncated_pdf, "corrupted.pdf")


def test_password_protected_pdf_raises_password_error() -> None:
    # Create encrypted PDF in memory with PyMuPDF
    doc = pymupdf.open()
    doc.new_page()
    buf = io.BytesIO()
    perms = int(pymupdf.PDF_PERM_ACCESSIBILITY | pymupdf.PDF_PERM_PRINT)
    doc.save(
        buf,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="master_secret",
        permissions=perms,
    )
    doc.close()
    encrypted_bytes = buf.getvalue()

    with pytest.raises(PasswordProtectedError):
        extract_text(encrypted_bytes, "encrypted.pdf")


def test_file_too_large_raises_file_too_large_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_file_bytes", 1024)
    large_bytes = b"%PDF" + b"0" * 2048

    with pytest.raises(FileTooLargeError):
        extract_text(large_bytes, "large.pdf")


def test_more_than_max_pages_degrades_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_pages", 3)
    # Generate 5-page PDF in memory
    doc = pymupdf.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((50, 50), f"Page {i+1} Content with sufficient length to avoid OCR.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    multi_page_pdf = buf.getvalue()

    res = parse_resume(multi_page_pdf, "multipage.pdf")
    assert res.layout.page_count == 5
    assert any("exceeds the limit" in w for w in res.warnings)


def test_pdf_bytes_named_docx_detected_by_magic_bytes() -> None:
    # Valid PDF bytes given a .docx filename
    pdf_path = FIXTURES_DIR / "ds_clean.pdf"
    pdf_bytes = pdf_path.read_bytes()

    res = parse_resume(pdf_bytes, "tricky_name.docx")
    assert res.layout.file_type == "pdf"
    assert res.contact.name == "Alex Mercer"
    assert "summary" in res.sections


def test_docx_bytes_named_pdf_detected_by_magic_bytes() -> None:
    # Valid DOCX bytes given a .pdf filename
    docx_path = FIXTURES_DIR / "chef_plain.docx"
    docx_bytes = docx_path.read_bytes()

    res = parse_resume(docx_bytes, "tricky_name.pdf")
    assert res.layout.file_type == "docx"
    assert res.contact.name == "Marcus Bennett"
    assert "experience" in res.sections


def test_unsupported_file_extension_raises_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(b"some plain text that is not a supported format", "resume.xyz")


# ---------------------------------------------------------------------------
# 5. Real Fixture Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected_name,expected_sections,is_multi_col,has_tbl,has_img,needs_ocr",
    [
        (
            "ds_clean.pdf",
            "Alex Mercer",
            {"summary", "experience", "education", "skills"},
            False,
            False,
            False,
            False,
        ),
        (
            "swe_two_column.pdf",
            "Jordan Lee",
            {"education", "skills", "experience"},
            True,
            True,
            True,
            False,
        ),
        (
            "accountant_table.pdf",
            "Morgan Vance",
            {"summary", "experience", "skills", "education"},
            False,
            True,
            False,
            False,
        ),
        (
            "chef_plain.docx",
            "Marcus Bennett",
            {"summary", "experience", "certifications", "education"},
            False,
            False,
            False,
            False,
        ),
        (
            "nurse_scanned.pdf",
            "Elena Rostova",
            {"summary", "experience", "education"},
            False,
            False,
            True,
            True,
        ),
    ],
)
def test_fixture_parsing(
    filename: str,
    expected_name: str,
    expected_sections: set[str],
    is_multi_col: bool,
    has_tbl: bool,
    has_img: bool,
    needs_ocr: bool,
) -> None:
    path = FIXTURES_DIR / filename
    data = path.read_bytes()
    res = parse_resume(data, filename)

    assert isinstance(res, ParsedResume)
    assert res.contact.name == expected_name
    assert expected_sections.issubset(set(res.sections.keys()))
    assert res.layout.is_multi_column is is_multi_col
    assert res.layout.has_tables is has_tbl
    assert res.layout.has_images is has_img
    assert res.layout.needs_ocr is needs_ocr


def test_parse_document_alias() -> None:
    """Verify parse_document is a backward-compatible alias of parse_resume."""
    path = FIXTURES_DIR / "ds_clean.pdf"
    res = parse_document(path.read_bytes(), "ds_clean.pdf")
    assert isinstance(res, ParsedResume)
    assert res.contact.name == "Alex Mercer"
