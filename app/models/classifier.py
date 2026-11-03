"""
app/models/classifier.py
========================
Stage 3 — Resume Category Classifier.

Provides:
- classify(text) -> list[CategoryPrediction] (top-3 categories with probabilities)
- classify_resume(parsed) -> list[CategoryPrediction] (pipeline entrypoint)

Supports fine-tuned DistilBERT, BiLSTM, TF-IDF + MLP, and TF-IDF + Logistic Regression.
If no model checkpoint exists, gracefully falls back to an on-the-fly trained TF-IDF + LR baseline
or returns a clean error without crashing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

from app.config import detect_device
from app.schemas import CategoryPrediction, ParsedResume

logger = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/models/classifier")

# Global in-memory cache for loaded model and artifacts
_LOADED_MODEL_CACHE: Dict[str, Any] = {}


# ============================================================================
# PyTorch Architectures for Inference
# ============================================================================

class _TextMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 24, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 24, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        feat = torch.cat([h_forward, h_backward], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


class _WordVocab:
    def __init__(self, max_vocab: int = 20000):
        self.max_vocab = max_vocab
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.w2i: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}

    def encode(self, text: str, max_len: int = 400) -> List[int]:
        token_pattern = re.compile(r"\b\w+\b")
        tokens = token_pattern.findall(text.lower())[:max_len]
        unk_idx = self.w2i.get(self.unk_token, 1)
        encoded = [self.w2i.get(tok, unk_idx) for tok in tokens]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return encoded

    @classmethod
    def load(cls, path: Path) -> _WordVocab:
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.w2i = data["w2i"]
        vocab.max_vocab = data.get("max_vocab", len(vocab.w2i))
        return vocab


# ============================================================================
# Model Loading & Fallback
# ============================================================================

def _get_inference_device() -> torch.device:
    dev_str = detect_device()
    if dev_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_fallback_baseline() -> Tuple[Any, Any, List[str]]:
    """Train a quick TF-IDF + Logistic Regression model on the fly if no saved model exists."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data_path = Path("data/processed/classifier_train.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path} for fallback classifier.")

    logger.warning("No saved classifier model found; training on-the-fly TF-IDF+LR baseline fallback...")
    df = pd.read_parquet(data_path)
    classes = sorted(df["category"].unique().tolist())
    class2idx = {c: i for i, c in enumerate(classes)}

    vec = TfidfVectorizer(max_features=3000, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(df["text_masked"])
    y = [class2idx[c] for c in df["category"]]

    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X, y)
    return vec, clf, classes


def load_classifier() -> Dict[str, Any]:
    """Load the best trained classifier or create an in-memory fallback."""
    global _LOADED_MODEL_CACHE
    if "model_info" in _LOADED_MODEL_CACHE:
        return _LOADED_MODEL_CACHE["model_info"]

    meta_file = MODEL_DIR / "best_model_meta.json"
    classes_file = MODEL_DIR / "classes.json"

    if not meta_file.exists() or not classes_file.exists():
        try:
            vec, clf, classes = _train_fallback_baseline()
            info = {
                "type": "fallback_lr",
                "vectorizer": vec,
                "model": clf,
                "classes": classes,
            }
            _LOADED_MODEL_CACHE["model_info"] = info
            return info
        except Exception as e:
            logger.error("Failed to build fallback classifier: %s", e)
            info = {
                "type": "error",
                "classes": ["UNKNOWN"],
                "error": str(e),
            }
            _LOADED_MODEL_CACHE["model_info"] = info
            return info

    with open(classes_file) as f:
        classes = json.load(f)
    with open(meta_file) as f:
        meta = json.load(f)

    best_model_type = meta.get("best_model", "distilbert")
    device = _get_inference_device()

    try:
        if best_model_type == "distilbert" and (MODEL_DIR / "distilbert").exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            bert_dir = MODEL_DIR / "distilbert"
            tok = AutoTokenizer.from_pretrained(bert_dir)
            model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
            model.eval()
            info = {
                "type": "distilbert",
                "tokenizer": tok,
                "model": model,
                "device": device,
                "classes": classes,
            }
        elif best_model_type == "bilstm" and (MODEL_DIR / "bilstm.pt").exists():
            vocab = _WordVocab.load(MODEL_DIR / "bilstm_vocab.json")
            model = _BiLSTMClassifier(vocab_size=len(vocab.w2i), num_classes=len(classes)).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / "bilstm.pt", map_location=device))
            model.eval()
            info = {
                "type": "bilstm",
                "vocab": vocab,
                "model": model,
                "device": device,
                "classes": classes,
            }
        elif best_model_type == "tfidf_mlp" and (MODEL_DIR / "tfidf_mlp.pt").exists():
            vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            in_dim = len(vec.get_feature_names_out())
            model = _TextMLP(in_features=in_dim, num_classes=len(classes)).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / "tfidf_mlp.pt", map_location=device))
            model.eval()
            info = {
                "type": "tfidf_mlp",
                "vectorizer": vec,
                "model": model,
                "device": device,
                "classes": classes,
            }
        elif (MODEL_DIR / "tfidf_lr.joblib").exists():
            vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            clf = joblib.load(MODEL_DIR / "tfidf_lr.joblib")
            info = {
                "type": "tfidf_lr",
                "vectorizer": vec,
                "model": clf,
                "classes": classes,
            }
        else:
            vec, clf, classes = _train_fallback_baseline()
            info = {
                "type": "fallback_lr",
                "vectorizer": vec,
                "model": clf,
                "classes": classes,
            }
    except Exception as e:
        logger.warning("Error loading model %s: %s. Falling back to on-the-fly baseline.", best_model_type, e)
        vec, clf, classes = _train_fallback_baseline()
        info = {
            "type": "fallback_lr",
            "vectorizer": vec,
            "model": clf,
            "classes": classes,
        }

    _LOADED_MODEL_CACHE["model_info"] = info
    return info


def clear_classifier_cache() -> None:
    """Clear in-memory cached classifier."""
    global _LOADED_MODEL_CACHE
    _LOADED_MODEL_CACHE.clear()


# ============================================================================
# Classification Entrypoints
# ============================================================================

def classify(text: str, top_k: int = 3, normalize: bool = True) -> List[CategoryPrediction]:
    """Predict the job category for arbitrary text, returning top-k predictions with probabilities.

    Args:
        text: Resume or JD text string.
        top_k: Number of highest-probability classes to return (default: 3).
        normalize: If True, normalizes top-k probabilities so they sum to 1.0 (default: True).

    Returns:
        List of :class:`CategoryPrediction` sorted by probability descending.
    """
    if not text or not text.strip():
        # Handle empty text gracefully
        return [
            CategoryPrediction(category="INFORMATION-TECHNOLOGY", probability=0.34),
            CategoryPrediction(category="BUSINESS-DEVELOPMENT", probability=0.33),
            CategoryPrediction(category="ENGINEERING", probability=0.33),
        ][:top_k]

    info = load_classifier()
    if info.get("type") == "error":
        return [CategoryPrediction(category="UNKNOWN", probability=1.0)]

    classes: List[str] = info["classes"]
    m_type = info["type"]

    try:
        if m_type == "distilbert":
            tokenizer = info["tokenizer"]
            model = info["model"]
            device = info["device"]
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elif m_type == "bilstm":
            vocab: _WordVocab = info["vocab"]
            model = info["model"]
            device = info["device"]
            tokens = vocab.encode(text, max_len=400)
            x_t = torch.tensor([tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elif m_type == "tfidf_mlp":
            vec = info["vectorizer"]
            model = info["model"]
            device = info["device"]
            feat = vec.transform([text]).toarray()
            x_t = torch.tensor(feat, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elif m_type in ("tfidf_lr", "fallback_lr"):
            vec = info["vectorizer"]
            clf = info["model"]
            feat = vec.transform([text])
            probs = clf.predict_proba(feat)[0]

        else:
            return [CategoryPrediction(category="UNKNOWN", probability=1.0)]

    except Exception as e:
        logger.error("Inference failed: %s", e)
        return [CategoryPrediction(category="UNKNOWN", probability=1.0)]

    # Extract top_k
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]

    if normalize:
        s = float(np.sum(top_probs))
        if s > 0:
            top_probs = top_probs / s
        else:
            top_probs = np.ones(top_k) / float(top_k)

    predictions = []
    for idx, p in zip(top_indices, top_probs):
        category = classes[idx] if idx < len(classes) else f"CLASS_{idx}"
        predictions.append(CategoryPrediction(category=category, probability=float(round(p, 4))))

    # Guarantee floating point sum equals 1.0 when normalized
    if normalize and len(predictions) > 0:
        diff = 1.0 - sum(p.probability for p in predictions)
        if abs(diff) > 1e-6:
            predictions[0] = CategoryPrediction(
                category=predictions[0].category,
                probability=float(round(predictions[0].probability + diff, 4)),
            )

    return predictions


def classify_resume(parsed: ParsedResume) -> List[CategoryPrediction]:
    """Predict the resume's job category from a ParsedResume object.

    Args:
        parsed: Enriched :class:`ParsedResume` from extraction stage.

    Returns:
        List of :class:`CategoryPrediction` sorted by probability descending.
    """
    raw_text = parsed.raw_text if parsed and parsed.raw_text else ""
    return classify(raw_text)
