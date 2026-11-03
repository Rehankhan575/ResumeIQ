"""
app/parsing/extractor.py
========================
Core document text and layout extraction engine.

Handles format detection (magic bytes first, extension second), PyMuPDF
extraction with column-aware layout reconstruction, pdfplumber fallback,
Tesseract OCR fallback for scanned documents, and python-docx document-order
traversal.
"""

from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import docx
import pdfplumber
import pymupdf
import pytesseract
from PIL import Image, ImageOps

from app.config import settings
from app.parsing.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    FileTooLargeError,
    ParseError,
    PasswordProtectedError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

# Magic byte signatures
PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPG_MAGIC = b"\xff\xd8\xff"


@dataclass
class LayoutSignals:
    """Internal signals describing physical document layout and extraction state."""

    page_count: int = 0
    has_tables: bool = False
    has_images: bool = False
    is_multi_column: bool = False
    font_count: int = 0
    needs_ocr: bool = False
    ocr_used: bool = False
    extraction_method: str = ""


@dataclass
class ExtractedDocument:
    """Internal container for raw extracted text, signals, and warnings."""

    text: str
    page_count: int
    method: str
    warnings: List[str] = field(default_factory=list)
    layout_signals: LayoutSignals = field(default_factory=LayoutSignals)


def detect_file_type(data: bytes, filename: str) -> str:
    """Detect file format using magic bytes first, falling back to extension.

    Supported types: "pdf", "docx", "png", "jpg".

    Args:
        data: Raw file bytes.
        filename: Original file name.

    Returns:
        One of "pdf", "docx", "png", "jpg".

    Raises:
        UnsupportedFileTypeError: If format is unrecognized or unsupported.
    """
    # 1. Magic bytes check
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(DOCX_MAGIC):
        return "docx"
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPG_MAGIC):
        return "jpg"

    # 2. Extension fallback
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext == ".png":
        return "png"
    if ext in (".jpg", ".jpeg"):
        return "jpg"

    raise UnsupportedFileTypeError(
        f"Unsupported file format for '{filename}'. Allowed: .pdf, .docx, .png, .jpg, .jpeg"
    )


# ---------------------------------------------------------------------------
# PDF Layout & Column Ordering Helpers
# ---------------------------------------------------------------------------

def _order_page_blocks(
    page: pymupdf.Page,
) -> Tuple[List[str], bool, bool, bool, Set[str]]:
    """Extract and order text blocks on a PyMuPDF page, handling columns and tables.

    Returns:
        Tuple of:
            - text_lines: list of extracted text chunks in logical reading order
            - is_multi_column: whether 2 columns with vertical overlap were detected
            - page_has_tables: whether tables were found on this page
            - page_has_images: whether raster images exist on this page
            - fonts: set of font names found
    """
    page_has_images = len(page.get_images()) > 0

    # Collect fonts
    fonts: Set[str] = set()
    try:
        page_dict = page.get_text("dict")
        for blk in page_dict.get("blocks", []):
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if font_name := span.get("font"):
                        fonts.add(font_name)
    except Exception:
        pass

    # Find tables
    tables = []
    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            tables = tabs.tables
    except Exception:
        pass

    page_has_tables = len(tables) > 0

    # Extract raw text blocks (block_type 0 = text)
    raw_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

    # If tables exist, substitute overlapping text blocks with formatted table rows
    table_bboxes = [t.bbox for t in tables]
    processed_blocks = []

    # Map tables to virtual blocks so they participate in sorting
    for t in tables:
        rows = t.extract()
        if rows:
            lines = [
                " | ".join(str(cell or "").replace("\n", " ").strip() for cell in row)
                for row in rows
            ]
            table_str = "\n".join(lines).strip()
            if table_str:
                # Use table bbox (x0, y0, x1, y1)
                bx0, by0, bx1, by1 = t.bbox
                processed_blocks.append((bx0, by0, bx1, by1, table_str, 0, 0))

    # Keep non-table text blocks
    for b in raw_blocks:
        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        in_table = any(
            tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for (tx0, ty0, tx1, ty1) in table_bboxes
        )
        if not in_table:
            processed_blocks.append(b)

    if not processed_blocks:
        return [], False, page_has_tables, page_has_images, fonts

    if len(processed_blocks) < 2:
        return [b[4] for b in processed_blocks], False, page_has_tables, page_has_images, fonts

    # Multi-column detection: cluster by x0
    x0_values = [b[0] for b in processed_blocks]
    min_x0, max_x0 = min(x0_values), max(x0_values)

    is_multi_column = False
    ordered_texts: List[str] = []

    if max_x0 - min_x0 >= 80:
        # Candidate 2-column layout
        mid_x = (min_x0 + max_x0) / 2.0
        left_cand = [b for b in processed_blocks if b[0] < mid_x]
        right_cand = [b for b in processed_blocks if b[0] >= mid_x]

        if left_cand and right_cand:
            max_left_x1 = max(b[2] for b in left_cand)
            min_right_x0 = min(b[0] for b in right_cand)

            if max_left_x1 < min_right_x0:
                split_x = (max_left_x1 + min_right_x0) / 2.0
                left_blocks = [b for b in processed_blocks if b[2] <= split_x]
                right_blocks = [b for b in processed_blocks if b[0] >= split_x]
                spanning = [
                    b for b in processed_blocks if b not in left_blocks and b not in right_blocks
                ]

                if left_blocks and right_blocks:
                    left_y0 = min(b[1] for b in left_blocks)
                    left_y1 = max(b[3] for b in left_blocks)
                    right_y0 = min(b[1] for b in right_blocks)
                    right_y1 = max(b[3] for b in right_blocks)

                    overlap = min(left_y1, right_y1) - max(left_y0, right_y0)
                    min_height = min(left_y1 - left_y0, right_y1 - right_y0)

                    if min_height > 0 and (overlap / min_height) >= 0.2:
                        is_multi_column = True
                        left_blocks.sort(key=lambda b: (b[1], b[0]))
                        right_blocks.sort(key=lambda b: (b[1], b[0]))

                        # Top / bottom spanning blocks
                        top_span = [b for b in spanning if b[3] <= min(left_y0, right_y0) + 10]
                        bottom_span = [b for b in spanning if b[1] >= max(left_y1, right_y1) - 10]
                        mid_span = [
                            b
                            for b in spanning
                            if b not in top_span and b not in bottom_span
                        ]

                        column_pieces: List[str] = []
                        if top_span:
                            column_pieces.extend(b[4] for b in top_span)
                        column_pieces.extend(b[4] for b in left_blocks)
                        if mid_span:
                            column_pieces.extend(b[4] for b in mid_span)
                        column_pieces.append("---column-break---")
                        column_pieces.extend(b[4] for b in right_blocks)
                        if bottom_span:
                            column_pieces.extend(b[4] for b in bottom_span)
                        ordered_texts = column_pieces

    # Check vector drawings for vertical line separators in multi-column layouts
    if is_multi_column:
        drawings = page.get_drawings()
        for d in drawings:
            rect = d.get("rect")
            if rect and rect.height > 50 and rect.width <= 2:
                page_has_tables = True
                break

    if not is_multi_column:
        # Fall back to standard PyMuPDF reading order sorted by top-to-bottom
        processed_blocks.sort(key=lambda b: (b[1], b[0]))
        ordered_texts = [b[4] for b in processed_blocks]

    return ordered_texts, is_multi_column, page_has_tables, page_has_images, fonts


# ---------------------------------------------------------------------------
# OCR Helper
# ---------------------------------------------------------------------------

def _run_ocr_on_page(
    page: pymupdf.Page, dpi: int
) -> Tuple[str, float]:
    """Render a PyMuPDF page at given DPI and run Tesseract OCR.

    Returns:
        Tuple of (extracted_text, mean_confidence).
    """
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)

    text = pytesseract.image_to_string(img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data.get("conf", []) if float(c) >= 0]
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0

    return text, mean_conf


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def _extract_pdf(data: bytes) -> ExtractedDocument:
    """Extract text and layout from PDF bytes using PyMuPDF and fallback mechanisms."""
    warnings: List[str] = []
    tesseract_available = shutil.which("tesseract") is not None

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptedFileError(f"Corrupted or invalid PDF document: {exc}") from exc

    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise PasswordProtectedError("PDF document is encrypted or password-protected.")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise EmptyDocumentError("PDF document has 0 pages.")

    pages_to_process = min(total_pages, settings.max_pages)
    if total_pages > settings.max_pages:
        warnings.append(
            f"Document has {total_pages} pages, which exceeds the limit of {settings.max_pages}. "
            "Extra pages were ignored."
        )

    all_page_texts: List[str] = []
    all_fonts: Set[str] = set()
    has_tables = False
    has_images = False
    is_multi_column = False
    method = "pymupdf"

    try:
        for p_idx in range(pages_to_process):
            page = doc[p_idx]
            page_lines, p_multi, p_tables, p_images, p_fonts = _order_page_blocks(page)
            all_page_texts.append("\n\n".join(page_lines))
            all_fonts.update(p_fonts)
            if p_multi:
                is_multi_column = True
            if p_tables:
                has_tables = True
            if p_images:
                has_images = True
    except Exception as exc:
        logger.warning("PyMuPDF block extraction failed (%s); falling back to pdfplumber", exc)
        doc.close()
        return _extract_pdfplumber_fallback(data, warnings)

    combined_text = "\n\n".join(all_page_texts).strip()
    avg_chars_per_page = (len(combined_text) / pages_to_process) if pages_to_process else 0
    needs_ocr = avg_chars_per_page < settings.min_chars_per_page
    ocr_used = False

    if needs_ocr:
        if not tesseract_available:
            warnings.append(
                "Document appears to be scanned (<50 chars/page), but Tesseract OCR binary "
                "was not found on PATH. Returning empty text layer."
            )
            combined_text = ""
            method = "pymupdf_no_ocr"
        else:
            method = "tesseract"
            ocr_used = True
            ocr_texts: List[str] = []
            confidences: List[float] = []

            for p_idx in range(pages_to_process):
                page = doc[p_idx]
                p_text, p_conf = _run_ocr_on_page(page, settings.ocr_dpi)
                ocr_texts.append(p_text)
                if p_conf > 0:
                    confidences.append(p_conf)

            combined_text = "\n\n".join(ocr_texts).strip()
            mean_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
            warnings.append(
                f"OCR fallback applied on {pages_to_process} page(s) (mean confidence: {mean_conf:.1f}%)."
            )

    doc.close()

    signals = LayoutSignals(
        page_count=total_pages,
        has_tables=has_tables,
        has_images=has_images,
        is_multi_column=is_multi_column,
        font_count=len(all_fonts) if all_fonts else 1,
        needs_ocr=needs_ocr,
        ocr_used=ocr_used,
        extraction_method=method,
    )

    return ExtractedDocument(
        text=combined_text,
        page_count=total_pages,
        method=method,
        warnings=warnings,
        layout_signals=signals,
    )


def _extract_pdfplumber_fallback(data: bytes, existing_warnings: List[str]) -> ExtractedDocument:
    """Fallback extraction using pdfplumber when PyMuPDF encounters errors."""
    warnings = list(existing_warnings)
    warnings.append("Extracted using pdfplumber fallback.")

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise EmptyDocumentError("PDF has 0 pages in pdfplumber.")

            pages_to_process = min(total_pages, settings.max_pages)
            texts = []
            has_tables = False
            has_images = False

            for p in pdf.pages[:pages_to_process]:
                if p.images:
                    has_images = True
                if p.extract_tables():
                    has_tables = True
                t = p.extract_text() or ""
                texts.append(t)

            combined = "\n\n".join(texts).strip()
            signals = LayoutSignals(
                page_count=total_pages,
                has_tables=has_tables,
                has_images=has_images,
                is_multi_column=False,
                font_count=1,
                needs_ocr=False,
                ocr_used=False,
                extraction_method="pdfplumber",
            )
            return ExtractedDocument(
                text=combined,
                page_count=total_pages,
                method="pdfplumber",
                warnings=warnings,
                layout_signals=signals,
            )
    except Exception as exc:
        raise CorruptedFileError(f"Both PyMuPDF and pdfplumber failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DOCX Extraction
# ---------------------------------------------------------------------------

def _extract_docx(data: bytes) -> ExtractedDocument:
    """Extract text from DOCX in document body order (paragraphs and tables)."""
    try:
        doc = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise CorruptedFileError(f"Corrupted or invalid DOCX document: {exc}") from exc

    lines: List[str] = []
    fonts: Set[str] = set()
    has_tables = len(doc.tables) > 0
    has_images = False

    for child in doc.element.body:
        tag = child.tag
        if tag.endswith("p"):
            para = docx.text.paragraph.Paragraph(child, doc)
            text = para.text.strip()
            if text:
                lines.append(text)
            for run in para.runs:
                if run.font and run.font.name:
                    fonts.add(run.font.name)
                # Check for images in run
                if run._element.xpath(".//a:blip|.//w:drawing"):
                    has_images = True
        elif tag.endswith("tbl"):
            tbl = docx.table.Table(child, doc)
            for row in tbl.rows:
                row_texts = [cell.text.strip() for cell in row.cells]
                if any(row_texts):
                    lines.append(" | ".join(row_texts))
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for run in p.runs:
                            if run.font and run.font.name:
                                fonts.add(run.font.name)

    combined_text = "\n".join(lines).strip()
    signals = LayoutSignals(
        page_count=1,
        has_tables=has_tables,
        has_images=has_images,
        is_multi_column=False,
        font_count=len(fonts) if fonts else 1,
        needs_ocr=False,
        ocr_used=False,
        extraction_method="python-docx",
    )

    return ExtractedDocument(
        text=combined_text,
        page_count=1,
        method="python-docx",
        warnings=[],
        layout_signals=signals,
    )


# ---------------------------------------------------------------------------
# Image Extraction (PNG / JPG)
# ---------------------------------------------------------------------------

def _extract_image(data: bytes, file_type: str) -> ExtractedDocument:
    """Extract text from PNG / JPG using Tesseract OCR."""
    warnings: List[str] = []
    tesseract_available = shutil.which("tesseract") is not None

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise CorruptedFileError(f"Corrupted or invalid image: {exc}") from exc

    if not tesseract_available:
        warnings.append("Image received, but Tesseract OCR binary not found on PATH.")
        signals = LayoutSignals(
            page_count=1,
            has_tables=False,
            has_images=True,
            is_multi_column=False,
            font_count=1,
            needs_ocr=True,
            ocr_used=False,
            extraction_method="no_ocr",
        )
        return ExtractedDocument(
            text="",
            page_count=1,
            method="no_ocr",
            warnings=warnings,
            layout_signals=signals,
        )

    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    text = pytesseract.image_to_string(img)
    data_dict = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data_dict.get("conf", []) if float(c) >= 0]
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0

    warnings.append(f"Image parsed via Tesseract OCR (mean confidence: {mean_conf:.1f}%).")
    signals = LayoutSignals(
        page_count=1,
        has_tables=False,
        has_images=True,
        is_multi_column=False,
        font_count=1,
        needs_ocr=True,
        ocr_used=True,
        extraction_method="tesseract",
    )

    return ExtractedDocument(
        text=text.strip(),
        page_count=1,
        method="tesseract",
        warnings=warnings,
        layout_signals=signals,
    )


# ---------------------------------------------------------------------------
# Public extract_text function
# ---------------------------------------------------------------------------

def extract_text(data: bytes, filename: str) -> ExtractedDocument:
    """Extract raw text and layout signals from document bytes.

    Detects file format by magic bytes first, extension second. Supported formats:
    PDF, DOCX, PNG, JPG/JPEG.

    Args:
        data: Raw file bytes (held in memory only).
        filename: Original file name.

    Returns:
        :class:`ExtractedDocument` with raw text, layout signals, and warnings.

    Raises:
        EmptyDocumentError: If data is empty.
        FileTooLargeError: If data size exceeds config.max_file_bytes.
        UnsupportedFileTypeError: If format is not supported.
        PasswordProtectedError: If document is password protected.
        CorruptedFileError: If file is truncated or unreadable.
        ParseError: For any other parser failure.
    """
    if not data or len(data) == 0:
        raise EmptyDocumentError(f"Document '{filename}' is empty (0 bytes).")

    if len(data) > settings.max_file_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds maximum allowed "
            f"limit of {settings.max_file_bytes} bytes ({settings.max_file_bytes // (1024 * 1024)} MB)."
        )

    file_type = detect_file_type(data, filename)

    try:
        if file_type == "pdf":
            return _extract_pdf(data)
        if file_type == "docx":
            return _extract_docx(data)
        if file_type in ("png", "jpg"):
            return _extract_image(data, file_type)
        raise UnsupportedFileTypeError(f"Unsupported file format: {file_type}")
    except (
        EmptyDocumentError,
        FileTooLargeError,
        UnsupportedFileTypeError,
        PasswordProtectedError,
        CorruptedFileError,
    ):
        raise
    except Exception as exc:
        logger.error("Unhandled parsing failure on '%s': %s", filename, exc)
        raise CorruptedFileError(f"Failed to parse document '{filename}': {exc}") from exc
