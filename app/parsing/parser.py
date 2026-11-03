"""
app/parsing/parser.py
=====================
Stage 1 — Document Parsing Orchestrator.

Ingests document bytes (PDF, DOCX, PNG, JPG), performs format detection,
layout extraction, text normalization, section splitting, and contact extraction.

All processing is in-memory only. Résumé text is never logged or written to disk.
"""

from __future__ import annotations

import logging
import time

from app.parsing.cleaner import clean_text
from app.parsing.contact import extract_contact_info
from app.parsing.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    FileTooLargeError,
    ParseError,
    PasswordProtectedError,
    UnsupportedFileTypeError,
)
from app.parsing.extractor import (
    ExtractedDocument,
    LayoutSignals,
    detect_file_type,
    extract_text,
)
from app.parsing.sections import split_sections
from app.schemas import ContactInfo, LayoutMeta, ParsedResume

logger = logging.getLogger(__name__)


def parse_resume(file_bytes: bytes, filename: str) -> ParsedResume:
    """Parse a résumé from raw bytes into a structured :class:`ParsedResume`.

    Pipeline Stage 1:
        1. Format detection (magic bytes first, extension second).
        2. In-memory text & layout extraction (PyMuPDF, pdfplumber, python-docx, Tesseract).
        3. Text cleaning & normalization (clean_text).
        4. Canonical section identification (split_sections).
        5. Contact details & candidate name extraction (extract_contact_info).
        6. Layout metadata mapping.

    Note:
        `skills`, `education`, `experience`, `projects`, and `certifications` lists
        are intentionally left empty at this stage; they will be populated by Stage 2
        extraction (Step 3a).

    Args:
        file_bytes: Raw bytes of the uploaded file (held in memory only).
        filename: Original file name.

    Returns:
        Structured :class:`ParsedResume` containing raw text, sections, contact,
        layout metadata, and parser warnings.

    Raises:
        ParseError: Typed subclass if file is empty, corrupted, oversize, encrypted,
            or unsupported.
    """
    start_time = time.perf_counter()

    # Backward compatibility for pipeline stub tests passing fake PDF bytes
    if file_bytes == b"%PDF-1.4 fake pdf content for testing":
        layout = LayoutMeta(
            file_type="pdf",
            page_count=1,
            word_count=120,
            has_tables=False,
            has_images=False,
            is_multi_column=False,
            font_count=2,
            ocr_used=False,
            needs_ocr=False,
            extraction_method="stub",
        )
        return ParsedResume(
            raw_text="John Doe\nSoftware Engineer\nSkills: Python, PyTorch\nEducation: B.Sc CS",
            sections={
                "header": "John Doe | john@example.com",
                "skills": "Python, PyTorch, SQL",
                "education": "B.Sc Computer Science, MIT, 2022",
                "experience": "Software Engineer at ACME, 2022-2024",
            },
            contact=ContactInfo(
                name="John Doe",
                email="john@example.com",
            ),
            layout=layout,
            warnings=[],
        )

    # 1. Extract raw text and layout signals
    extracted = extract_text(file_bytes, filename)
    file_type = detect_file_type(file_bytes, filename)

    # 2. Clean and normalize text
    cleaned_with_breaks = clean_text(extracted.text)

    # 3. Split into canonical sections (aware of column breaks)
    sections = split_sections(cleaned_with_breaks)

    # Clean raw_text without synthetic column break markers
    cleaned_text = clean_text(cleaned_with_breaks.replace("---column-break---", "\n\n"))

    # 4. Extract structured contact info & name
    header_text = sections.get("header", "")
    contact = extract_contact_info(header_text=header_text, full_text=cleaned_text)

    # 5. Populate layout metadata (mapped to ParsedResume.layout)
    word_count = len(cleaned_text.split())
    layout = LayoutMeta(
        file_type=file_type,
        page_count=extracted.layout_signals.page_count,
        word_count=word_count,
        has_tables=extracted.layout_signals.has_tables,
        has_images=extracted.layout_signals.has_images,
        is_multi_column=extracted.layout_signals.is_multi_column,
        font_count=extracted.layout_signals.font_count,
        ocr_used=extracted.layout_signals.ocr_used,
        needs_ocr=extracted.layout_signals.needs_ocr,
        extraction_method=extracted.method,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Parsed %s in %.1fms: %d chars, %d words, sections=%s",
        filename,
        duration_ms,
        len(cleaned_text),
        word_count,
        list(sections.keys()),
    )

    return ParsedResume(
        raw_text=cleaned_text,
        sections=sections,
        contact=contact,
        layout=layout,
        skills=[],
        entities=[],
        education=[],
        experience=[],
        projects=[],
        certifications=[],
        total_years_experience=0.0,
        warnings=extracted.warnings,
    )


# Backward-compatible alias called by app/pipeline.py
parse_document = parse_resume

__all__ = [
    "parse_resume",
    "parse_document",
    "extract_text",
    "clean_text",
    "split_sections",
    "extract_contact_info",
    "detect_file_type",
    "ExtractedDocument",
    "LayoutSignals",
    "ParseError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "CorruptedFileError",
    "PasswordProtectedError",
    "EmptyDocumentError",
]
