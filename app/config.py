"""
app/config.py
=============
Central settings for the AI Resume Analyzer.

All paths, hyper-parameters, score weights, and device selection live here.
Import Settings from this module everywhere; never hardcode values elsewhere.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
TAXONOMIES_DIR: Path = DATA_DIR / "taxonomies"


# ---------------------------------------------------------------------------
# Device detection: CUDA → MPS → CPU
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return the best available PyTorch device string.

    Checks in order: CUDA, MPS (Apple Silicon), CPU.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Device: MPS (Apple Silicon)")
        return "mps"
    logger.info("Device: CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Settings (loaded from .env if present)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """All runtime configuration for the pipeline.

    Values can be overridden via environment variables or a ``.env`` file.
    All paths are absolute ``pathlib.Path`` objects derived from ``ROOT_DIR``.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Device ─────────────────────────────────────────────────────────────
    device: str = "cpu"  # overridden at runtime by detect_device()

    # ── File ingestion ─────────────────────────────────────────────────────
    max_file_bytes: int = 10 * 1024 * 1024   # 10 MB
    allowed_extensions: list[str] = [".pdf", ".docx", ".png", ".jpg", ".jpeg"]

    # ── Paths ───────────────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    metrics_dir: Path = METRICS_DIR
    reports_dir: Path = REPORTS_DIR
    taxonomies_dir: Path = TAXONOMIES_DIR

    # ── NER model ───────────────────────────────────────────────────────────
    ner_model_name: str = "dslim/distilbert-NER"
    ner_batch_size: int = 16

    # ── Classifier ──────────────────────────────────────────────────────────
    classifier_model_name: str = "distilbert-base-uncased"
    classifier_max_length: int = 512
    classifier_batch_size: int = 8
    num_resume_categories: int = 25

    # ── Matcher (Siamese SBERT) ─────────────────────────────────────────────
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    matcher_top_k: int = 5

    # ── ATS Score weights (must sum to 1.0) ─────────────────────────────────
    # Placeholder values — will be tuned after ablation study.
    weight_semantic_match: float = 0.30
    weight_keyword_coverage: float = 0.20
    weight_section_completeness: float = 0.15
    weight_formatting: float = 0.10
    weight_quantified_achievements: float = 0.10
    weight_action_verbs: float = 0.05
    weight_readability: float = 0.05
    weight_length: float = 0.05

    # ── LLM / Generative ────────────────────────────────────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""
    flan_t5_model_name: str = "google/flan-t5-base"
    use_llm_api: bool = False    # False → use local Flan-T5

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings = Settings()
settings.device = detect_device()
set_seed(settings.seed)
