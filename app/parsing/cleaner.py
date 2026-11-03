"""
app/parsing/cleaner.py
======================
Text cleaning and normalization rules for extracted résumé text.

Preserves structural newlines while standardising bullets, whitespace,
hyphenation breaks, and Unicode quirks.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width spaces, joiners, directional formatting, BOM
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F]")

# Bullet characters to standardize to "- "
BULLETS_RE = re.compile(r"[•▪◦●‣][ \t]*")

# De-hyphenate lowercase-lowercase line-breaks only
# e.g., "imple-\nmentation" -> "implementation"
# Preserves "Pre-\nPost" or "go-to-market" or "2021-\n2022"
LINE_BREAK_HYPHEN_RE = re.compile(r"([a-z])- *\n *([a-z])")

# Collapse multiple horizontal whitespace (spaces, tabs)
HORIZONTAL_WS_RE = re.compile(r"[ \t]+")

# At most 1 blank line (max 2 consecutive newlines)
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize and clean raw extracted document text.

    Rules applied:
        1. Unicode NFKC normalization.
        2. Strip zero-width and invisible directional markers.
        3. Normalize bullet characters (•, ▪, ◦, ●, ‣) to "- ".
        4. De-hyphenate lowercase-to-lowercase line breaks.
        5. Collapse horizontal spaces within lines.
        6. Enforce at most 1 blank line between text blocks while preserving newlines.

    Args:
        text: Raw text string from parser or OCR.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # 1. NFKC normalization
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Strip zero-width / directional / BOM markers
    cleaned = ZERO_WIDTH_RE.sub("", cleaned)

    # 3. Normalize bullets to standard "- "
    cleaned = BULLETS_RE.sub("- ", cleaned)

    # 4. De-hyphenate lowercase-lowercase line-break splits
    cleaned = LINE_BREAK_HYPHEN_RE.sub(r"\1\2", cleaned)

    # 5. Collapse horizontal whitespace per line and strip line edges
    lines = [HORIZONTAL_WS_RE.sub(" ", line).strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)

    # 6. At most 1 blank line (collapse 3+ newlines to 2)
    cleaned = EXCESS_NEWLINES_RE.sub("\n\n", cleaned)

    return cleaned.strip()
