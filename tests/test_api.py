"""
tests/test_api.py
=================
Integration tests for the FastAPI REST backend (app/api/main.py).
Tests health, sample fixtures, analyze endpoint, file size/type boundaries, and multi-JD ranking.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_health_check() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "app": "ResumeIQ"}


def test_list_samples() -> None:
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "resumes" in data
    assert "jds" in data
    assert len(data["resumes"]) > 0
    assert any(r["filename"] == "ds_clean.pdf" for r in data["resumes"])


def test_analyze_with_sample_fixtures() -> None:
    res = client.post(
        "/api/analyze",
        data={
            "resume_choice": "ds_clean.pdf",
            "jd_choice": "jd_data_scientist.txt",
            "use_genai": "false",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "grade" in data
    assert 0.0 <= data["total"] <= 100.0
    assert data["grade"] == "B"
    assert "semantic_match" in data["components"]
    assert "issues" in data
    assert len(data["issues"]) > 0


def test_analyze_rejects_unsupported_extension() -> None:
    res = client.post(
        "/api/analyze",
        files={"file": ("resume.exe", b"binarycontent", "application/octet-stream")},
        data={"jd_choice": "jd_data_scientist.txt"},
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_analyze_rejects_empty_file() -> None:
    res = client.post(
        "/api/analyze",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"jd_choice": "jd_data_scientist.txt"},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_analyze_rejects_oversized_file() -> None:
    oversized = b"a" * (5 * 1024 * 1024 + 1024)  # > 5 MB
    res = client.post(
        "/api/analyze",
        files={"file": ("large.pdf", oversized, "application/pdf")},
        data={"jd_choice": "jd_data_scientist.txt"},
    )
    assert res.status_code == 413
    assert "exceeds maximum allowed size" in res.json()["detail"]


def test_rank_multi_jds() -> None:
    res = client.post(
        "/api/rank",
        data={
            "resume_choice": "ds_clean.pdf",
            "jd_filenames": ["jd_data_scientist.txt", "jd_chef.txt"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "ranked" in data
    assert len(data["ranked"]) == 2
    assert data["ranked"][0]["jd_title"] == "Data Scientist"
