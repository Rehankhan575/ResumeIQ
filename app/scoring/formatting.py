"""
app/scoring/formatting.py
=========================
ATS formatting scorer evaluating layout compatibility.

Scores resume layout metadata (tables, multi-column, images, page count, word count, OCR)
for applicant tracking system (ATS) friendliness.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from app.schemas import Issue, LayoutMeta

logger = logging.getLogger(__name__)


def score_formatting(layout: LayoutMeta | None) -> Tuple[float, List[Issue]]:
    """Score resume layout metadata for ATS friendliness on a 0.0 - 1.0 scale.

    Evaluates:
      - Multi-column layouts (ATS parsers often read across columns)
      - Tables (ATS parsers frequently misalign cell order)
      - Images (Graphics and icons are non-parsable by text extractors)
      - Page count (1-2 pages optimal; 0 is corrupt; >3 is penalized)
      - Word count (300-1000 words optimal; <150 is too brief; >1500 is verbose)
      - Scanned / OCR usage (searchable text preferred over scanned images)

    Args:
        layout: Extracted :class:`LayoutMeta` from the parsing stage.

    Returns:
        Tuple of (score in [0.0, 1.0], list of detected :class:`Issue` objects).
    """
    if layout is None:
        issue = Issue(
            component="formatting",
            message="No layout metadata available; format could not be verified.",
            severity="medium",
            estimated_gain=0.0,
        )
        return 0.5, [issue]

    deductions: float = 0.0
    issues: List[Issue] = []

    # 1. Multi-column layout
    if layout.is_multi_column:
        deductions += 0.20
        issues.append(
            Issue(
                component="formatting",
                message="Multi-column layout detected; ATS parsers may misorder or scramble text.",
                severity="medium",
            )
        )

    # 2. Tables
    if layout.has_tables:
        deductions += 0.15
        issues.append(
            Issue(
                component="formatting",
                message="Tables detected; many ATS engines fail to extract table cells in logical order.",
                severity="medium",
            )
        )

    # 3. Images
    if layout.has_images:
        deductions += 0.10
        issues.append(
            Issue(
                component="formatting",
                message="Images or graphic elements detected; embedded text cannot be indexed by ATS.",
                severity="low",
            )
        )

    # 4. Page count
    if layout.page_count <= 0 and layout.word_count <= 0:
        deductions += 0.80
        issues.append(
            Issue(
                component="formatting",
                message="Empty or unreadable document (0 pages and 0 words detected).",
                severity="high",
            )
        )
    elif layout.page_count == 0:
        deductions += 0.20
        issues.append(
            Issue(
                component="formatting",
                message="Unspecified page count in layout metadata.",
                severity="low",
            )
        )
    elif layout.page_count > 3:
        deductions += 0.25
        issues.append(
            Issue(
                component="formatting",
                message=f"Resume is {layout.page_count} pages long; 1-2 pages is standard for ATS screening.",
                severity="medium",
            )
        )
    elif layout.page_count == 3:
        deductions += 0.10
        issues.append(
            Issue(
                component="formatting",
                message="Resume is 3 pages long; condensing to 1-2 pages is recommended.",
                severity="low",
            )
        )

    # 5. Word count
    if layout.word_count == 0:
        if layout.page_count > 0:
            deductions += 0.70
            issues.append(
                Issue(
                    component="formatting",
                    message="Document contains no readable text (possible scanned/image-only PDF).",
                    severity="high",
                )
            )
    elif layout.word_count < 150:
        deductions += 0.30
        issues.append(
            Issue(
                component="formatting",
                message=f"Resume text is very short ({layout.word_count} words); ensure all sections are present.",
                severity="high",
            )
        )
    elif layout.word_count < 300:
        deductions += 0.10
        issues.append(
            Issue(
                component="formatting",
                message=f"Resume is relatively brief ({layout.word_count} words); recommended range is 350-900 words.",
                severity="low",
            )
        )
    elif layout.word_count > 1500:
        deductions += 0.15
        issues.append(
            Issue(
                component="formatting",
                message=f"Resume exceeds 1,500 words ({layout.word_count} words); consider editing for brevity.",
                severity="low",
            )
        )

    # 6. OCR fallback
    if layout.ocr_used or layout.needs_ocr:
        deductions += 0.15
        issues.append(
            Issue(
                component="formatting",
                message="Document required OCR image extraction; text-based vector PDFs parse far more reliably in ATS.",
                severity="medium",
            )
        )

    score = max(0.0, min(1.0, 1.0 - deductions))
    return round(score, 4), issues
