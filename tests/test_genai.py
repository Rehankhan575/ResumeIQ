"""
tests/test_genai.py
===================
Offline test suite for Stage 7 GenAI Provider Abstraction, Prompts, and Grounding Guard.

Verifies:
  1. Fallback works with no env vars set.
  2. Provider failure and timeout fall back to deterministic rule-based output.
  3. Grounding guard catches invented numbers, companies/organisations, and skills.
  4. Grounding guard drops ungrounded provider output and preserves safe fallback.
  5. API keys and resume text never appear in application logs.
  6. Empty/corrupt input is handled gracefully everywhere.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest
import requests

from app.genai.generator import (
    generate_cover_letter,
    generate_suggestions,
    improvement_plan,
    rewrite_bullets,
)
from app.genai.grounding import check_grounding
from app.genai.provider import _IN_MEMORY_CACHE, call_llm, get_active_provider
from app.schemas import Issue, ParsedResume, ScoreResult


@pytest.fixture(autouse=True)
def clean_env_and_cache(monkeypatch):
    """Ensure clean environment and in-memory cache for each test."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    _IN_MEMORY_CACHE.clear()


def test_fallback_works_with_no_env_vars() -> None:
    """With no environment variables set, provider is none and fallbacks execute cleanly."""
    provider, auth, model = get_active_provider()
    assert provider == "none"
    assert call_llm("Test prompt") is None

    # Bullet rewrite fallback
    bullets = ["Worked on internal database queries and report generation."]
    rewrites = rewrite_bullets(bullets)
    assert len(rewrites) == 1
    assert rewrites[0]["method"] == "rule_based"
    assert rewrites[0]["is_grounded"] is True
    assert "[add metric]" in rewrites[0]["rewritten"]

    # Cover letter fallback
    letter = generate_cover_letter("Software Engineer with Python skills", "Senior Developer")
    assert "Dear Hiring Team" in letter
    assert len(letter.split()) >= 60

    # Improvement plan fallback
    score_res = ScoreResult(
        total=65.0,
        grade="C",
        components={"keyword_coverage": 0.5, "quantified_achievements": 0.3},
        missing_skills=["Docker", "Kubernetes"],
        issues=[
            Issue(component="keyword_coverage", message="Missing Docker and Kubernetes.", estimated_gain=5.0),
            Issue(component="quantified_achievements", message="Add numbers to bullets.", estimated_gain=3.5),
        ],
    )
    plan = improvement_plan(score_res)
    assert len(plan) >= 2
    assert any("Docker" in p["recommendation"] for p in plan)
    assert plan[0]["estimated_gain"] >= plan[1]["estimated_gain"]


def test_provider_failure_and_timeout_fall_back(monkeypatch) -> None:
    """Network failure or 8s timeout triggers retries and falls back gracefully."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-test-key")

    provider, auth, model = get_active_provider()
    assert provider == "gemini"

    # Mock timeout
    with patch("requests.post", side_effect=requests.Timeout("Connection timed out after 8.0s")):
        res = call_llm("Generate something")
        assert res is None

    # Mock 500 error
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch("requests.post", return_value=mock_resp):
        res = call_llm("Generate something")
        assert res is None

    # End-to-end bullet rewrite should still succeed via fallback
    with patch("requests.post", side_effect=requests.RequestException("Offline network")):
        rewrites = rewrite_bullets(["Managed data pipelines"])
        assert len(rewrites) == 1
        assert rewrites[0]["method"] == "rule_based"
        assert "[add metric]" in rewrites[0]["rewritten"]


def test_mocked_provider_success_and_caching(monkeypatch) -> None:
    """Mocked LLM provider succeeds and caches in-memory without duplicate network calls."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Engineered scalable data architectures."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as post_mock:
        res1 = call_llm("Write a strong bullet")
        assert res1 == "Engineered scalable data architectures."
        assert post_mock.call_count == 1

        # Second call with identical prompt should hit in-memory cache
        res2 = call_llm("Write a strong bullet")
        assert res2 == "Engineered scalable data architectures."
        assert post_mock.call_count == 1


def test_grounding_guard_catches_invented_number_company_and_skill() -> None:
    """check_grounding flags any invented number, organisation, or taxonomy skill."""
    original = "Worked on database queries and improved response times."

    # Invented number
    res_num = check_grounding(original, "Engineered database queries, improving response times by 45%.")
    assert not res_num.is_grounded
    assert any("invented_number" in f and "45%" in f for f in res_num.flags)

    # Invented company
    res_org = check_grounding(original, "Engineered database queries at Google, improving response times.")
    assert not res_org.is_grounded
    assert any("invented_org" in f and "Google" in f for f in res_org.flags)

    # Invented taxonomy skill
    res_skill = check_grounding(original, "Engineered database queries using Kubernetes and PyTorch.")
    assert not res_skill.is_grounded
    assert any("invented_skill" in f and "kubernetes" in f for f in res_skill.flags)

    # Legitimate rewrite with placeholder passes cleanly
    res_clean = check_grounding(
        original,
        "Engineered relational database queries, improving response times by [add metric].",
    )
    assert res_clean.is_grounded
    assert len(res_clean.flags) == 0


def test_grounding_guard_drops_unclean_provider_output(monkeypatch) -> None:
    """If provider invents ungrounded facts, rewrite_bullets drops it and uses rule-based fallback."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Mock provider returning hallucinated bullet
    hallucinated_text = "Spearheaded cloud migration at Amazon saving $100k using Terraform."
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": hallucinated_text}]}}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        original_bullet = "Assisted in database maintenance and SQL backups."
        rewrites = rewrite_bullets([original_bullet])
        assert len(rewrites) == 1
        # Provider output must be dropped because Amazon, $100k, and Terraform are ungrounded
        assert rewrites[0]["method"] == "rule_based"
        assert "Amazon" not in rewrites[0]["rewritten"]
        assert "$100k" not in rewrites[0]["rewritten"]
        assert rewrites[0]["is_grounded"] is True


def test_keys_and_resume_never_appear_in_logs(monkeypatch, caplog) -> None:
    """Verify that credentials and resume text never leak into application logs."""
    secret_key = "super-secret-api-key-xyz-987"
    sensitive_resume_snippet = "John Doe Secret Project X-12345"

    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    with caplog.at_level(logging.DEBUG):
        # Trigger provider call and failure
        with patch("requests.post", side_effect=requests.RequestException("Mock error")):
            _ = call_llm(f"Please review: {sensitive_resume_snippet}")
            _ = rewrite_bullets([f"Worked on {sensitive_resume_snippet}"])

    all_logs = caplog.text
    assert secret_key not in all_logs, "API key leaked in log messages!"
    assert sensitive_resume_snippet not in all_logs, "Resume snippet leaked in log messages!"


def test_empty_input_handled_in_genai() -> None:
    """Empty inputs to GenAI functions must return safe defaults without throwing exceptions."""
    assert rewrite_bullets([]) == []
    assert rewrite_bullets("") == []

    letter = generate_cover_letter("", "")
    assert isinstance(letter, str)
    assert "Dear Hiring Team" in letter

    plan = improvement_plan(None)
    assert plan == []

    plan_empty = improvement_plan(ScoreResult(total=100.0, grade="A", components={}))
    assert len(plan_empty) >= 1

    res = generate_suggestions(
        ScoreResult(total=70.0, grade="B"),
        ParsedResume(raw_text=""),
    )
    assert isinstance(res, ScoreResult)
