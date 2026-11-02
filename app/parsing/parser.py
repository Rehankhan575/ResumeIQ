"""
app/parsing/parser.py
=====================
Stage 1 — Document Parsing.

Extracts raw text and layout metadata from PDF, DOCX, or image files.
Backends: PyMuPDF (primary), pdfplumber (layout), python-docx, Tesseract OCR (fallback).

NOTE: ``import pymupdf`` is used — NOT ``import fitz`` (deprecated in PyMuPDF 2.0).
NOTE: Tesseract binary must be installed via ``brew install tesseract`` before OCR is active.
"""

from __future__ import annotations

import logging

from app.schemas import LayoutMeta, ParsedResume

logger = logging.getLogger(__name__)


def parse_document(file_bytes: bytes, filename: str) -> ParsedResume:
    """Parse a résumé file and return a structured :class:`ParsedResume`.

    This is currently a **stub** that returns valid dummy data.
    Real implementation will use PyMuPDF, pdfplumber, python-docx, and Tesseract.

    Args:
        file_bytes: Raw bytes of the uploaded résumé file.
        filename:   Original filename (used to infer file type).

    Returns:
        A :class:`ParsedResume` with ``raw_text``, ``sections``, and ``layout`` populated.
    """
    logger.debug("parse_document called for %s (%d bytes)", filename, len(file_bytes))

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"

    layout = LayoutMeta(
        file_type=ext,
        page_count=1,
        word_count=120,
        has_tables=False,
        has_images=False,
        is_multi_column=False,
        font_count=2,
        ocr_used=False,
    )

    return ParsedResume(
        raw_text="John Doe\nSoftware Engineer\nSkills: Python, PyTorch\nEducation: B.Sc CS",
        sections={
            "contact": "John Doe | john@example.com",
            "skills": "Python, PyTorch, SQL",
            "education": "B.Sc Computer Science, MIT, 2022",
            "experience": "Software Engineer at ACME, 2022-2024",
        },
        layout=layout,
        warnings=[],
    )
