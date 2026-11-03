"""
app/parsing
===========
Stage 1: Document parsing and text/layout extraction.
"""

from app.parsing.cleaner import clean_text
from app.parsing.contact import extract_candidate_name, extract_contact_info
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
from app.parsing.parser import parse_document, parse_resume
from app.parsing.sections import match_heading, split_sections

__all__ = [
    "parse_resume",
    "parse_document",
    "extract_text",
    "clean_text",
    "split_sections",
    "match_heading",
    "extract_contact_info",
    "extract_candidate_name",
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
