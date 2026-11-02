"""
tests/conftest.py
=================
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def dummy_pdf_bytes() -> bytes:
    """Minimal fake PDF bytes for pipeline tests."""
    return b"%PDF-1.4 fake pdf content for testing"


@pytest.fixture(scope="session")
def dummy_filename() -> str:
    """Sample filename for pipeline tests."""
    return "test_resume.pdf"


@pytest.fixture(scope="session")
def dummy_jd_text() -> str:
    """Sample job description text for matching tests."""
    return (
        "We are looking for a Senior Python Developer with experience in "
        "PyTorch, SQL, and distributed systems. Kubernetes and Spark experience preferred."
    )
