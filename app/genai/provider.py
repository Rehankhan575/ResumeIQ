"""
app/genai/provider.py
=====================
LLM Provider Abstraction supporting Gemini, Groq, and Ollama with rule-based fallback.

Chosen strictly via environment variables (from .env or os.environ):
  - GEMINI_API_KEY (+ GEMINI_MODEL)
  - GROQ_API_KEY   (+ GROQ_MODEL)
  - OLLAMA_HOST    (+ OLLAMA_MODEL)

Guarantees:
  - 8-second timeout per request.
  - At most 2 retries (3 total attempts).
  - In-memory only cache (SHA-256 keyed, never persisted to disk).
  - Zero logging of API keys, resume text, or candidate information.
  - Returns None on missing credentials, network failure, or timeout,
    signaling the caller to use deterministic rule-based fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: float = 8.0
MAX_RETRIES: int = 2

# In-memory transient cache
_IN_MEMORY_CACHE: Dict[str, str] = {}


def _read_env_file() -> Dict[str, str]:
    """Read key-value pairs from .env file at repo root if present."""
    env_vars: Dict[str, str] = {}
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        env_vars[k] = v
        except Exception as exc:
            logger.debug("Could not parse .env file: %s", exc)
    return env_vars


def get_genai_env(var_name: str, default: str = "") -> str:
    """Retrieve an environment variable from .env or os.environ.

    Never logs the retrieved value.
    """
    # 1. os.environ takes precedence (e.g. pytest monkeypatch)
    if var_name in os.environ and os.environ[var_name]:
        return os.environ[var_name]

    # 2. Check .env file directly
    dot_env = _read_env_file()
    if var_name in dot_env and dot_env[var_name]:
        return dot_env[var_name]

    return default


def get_active_provider() -> Tuple[str, str, str]:
    """Detect configured provider from env vars.

    Returns:
        Tuple of (provider_name, api_key_or_host, model_name).
        If no provider is configured, returns ("none", "", "").
    """
    gemini_key = get_genai_env("GEMINI_API_KEY")
    if gemini_key:
        model = get_genai_env("GEMINI_MODEL", "gemini-1.5-flash")
        return "gemini", gemini_key, model

    groq_key = get_genai_env("GROQ_API_KEY")
    if groq_key:
        model = get_genai_env("GROQ_MODEL", "llama-3.1-8b-instant")
        return "groq", groq_key, model

    ollama_host = get_genai_env("OLLAMA_HOST")
    if ollama_host:
        model = get_genai_env("OLLAMA_MODEL", "llama3")
        return "ollama", ollama_host, model

    return "none", "", ""


def _compute_cache_key(provider: str, model: str, prompt: str) -> str:
    """Compute SHA-256 hash for in-memory response caching."""
    payload = f"{provider}:{model}:{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def call_llm(prompt: str) -> Optional[str]:
    """Execute a prompt against the configured LLM provider.

    Returns response text or None if fallback is required.
    Does not log prompt text or keys.
    """
    if not prompt or not prompt.strip():
        return None

    provider, auth_or_host, model = get_active_provider()
    if provider == "none":
        return None

    cache_key = _compute_cache_key(provider, model, prompt)
    if cache_key in _IN_MEMORY_CACHE:
        logger.info("GenAI cache hit | provider=%s", provider)
        return _IN_MEMORY_CACHE[cache_key]

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info("GenAI call attempt %d/%d | provider=%s", attempt + 1, MAX_RETRIES + 1, provider)

            if provider == "gemini":
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={auth_or_host}"
                )
                headers = {"Content-Type": "application/json"}
                body = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(url, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                _IN_MEMORY_CACHE[cache_key] = text.strip()
                return text.strip()

            elif provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {auth_or_host}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }
                resp = requests.post(url, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                _IN_MEMORY_CACHE[cache_key] = text.strip()
                return text.strip()

            elif provider == "ollama":
                host = auth_or_host.rstrip("/")
                url = f"{host}/api/generate"
                body = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                }
                resp = requests.post(url, json=body, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                _IN_MEMORY_CACHE[cache_key] = text.strip()
                return text.strip()

        except Exception as exc:
            logger.warning(
                "GenAI provider request failed on attempt %d/%d | provider=%s | error_type=%s",
                attempt + 1,
                MAX_RETRIES + 1,
                provider,
                exc.__class__.__name__,
            )

    logger.info("GenAI attempts exhausted; falling back to deterministic rule-based output.")
    return None
