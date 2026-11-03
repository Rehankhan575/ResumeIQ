"""
app/parsing/errors.py
=====================
Typed exception hierarchy for Stage 1 document parsing.

All parsing exceptions inherit from :class:`ParseError`.
Callers catch :class:`ParseError` to handle any parser failure without bare exceptions.
"""

from __future__ import annotations


class ParseError(Exception):
    """Base exception for all document parsing errors."""


class UnsupportedFileTypeError(ParseError):
    """Raised when the document file format is unsupported or unrecognized."""


class FileTooLargeError(ParseError):
    """Raised when document file size exceeds the configured max_file_bytes."""


class CorruptedFileError(ParseError):
    """Raised when file content is truncated, invalid, or cannot be parsed."""


class PasswordProtectedError(ParseError):
    """Raised when document is encrypted or requires a password to open."""


class EmptyDocumentError(ParseError):
    """Raised when file bytes are empty (0 bytes) or contain zero readable content."""
