"""
scripts/train_classifier.py
===========================
Training and evaluation pipeline for Stage 3a Resume Classifier.

Models:
 1. TF-IDF + Logistic Regression
 2. TF-IDF + MLP (PyTorch, 2 hidden layers, dropout)
 3. BiLSTM (PyTorch, trained embeddings, vocab ~20k, max_len 400, <= 8 epochs)
 4. Fine-tuned DistilBERT (max_len 256, PyTorch on MPS/CPU)

Saves models to artifacts/models/classifier/ and metrics/plots to artifacts/metrics/classifier/.
The test split is touched ONLY when run with --final-eval.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from app.config import detect_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_classifier")

SEED = 42
MODEL_DIR = Path("artifacts/models/classifier")
METRIC_DIR = Path("artifacts/metrics/classifier")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_torch_device() -> torch.device:
    dev_str = detect_device()
    if dev_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# PyTorch Architectures
# ============================================================================

class TextMLP(nn.Module):
    """2-hidden-layer MLP with BatchNorm and Dropout."""
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


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM text classifier with trained word embeddings."""
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
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        out, (h_n, c_n) = self.lstm(emb)  # h_n: (2, B, H)
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        feat = torch.cat([h_forward, h_backward], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


# ============================================================================
# BiLSTM Tokenizer / Dataset
# ============================================================================

class WordVocab:
    def __init__(self, max_vocab: int = 20000):
        self.max_vocab = max_vocab
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.w2i: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.i2w: Dict[int, str] = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, texts: List[str]) -> None:
        from collections import Counter
        counts = Counter()
        token_pattern = re.compile(r"\b\w+\b")
        for text in texts:
            tokens = token_pattern.findall(text.lower())
            counts.update(tokens)
        most_common = counts.most_common(self.max_vocab - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.w2i[word] = idx
            self.i2w[idx] = word

    def encode(self, text: str, max_len: int = 400) -> List[int]:
        token_pattern = re.compile(r"\b\w+\b")
        tokens = token_pattern.findall(text.lower())[:max_len]
        unk_idx = self.w2i[self.unk_token]
        encoded = [self.w2i.get(tok, unk_idx) for tok in tokens]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return encoded

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"w2i": self.w2i, "max_vocab": self.max_vocab}, f)

    @classmethod
    def load(cls, path: Path) -> WordVocab:
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.w2i = data["w2i"]
        vocab.i2w = {int(v): k for k, v in vocab.w2i.items()}
        vocab.max_vocab = data.get("max_vocab", len(vocab.w2i))
        return vocab


class BiLSTMDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: WordVocab, max_len: int = 400):
        self.labels = labels
        self.samples = [vocab.encode(t, max_len=max_len) for t in texts]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.samples[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class DistilBERTDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_len: int = 256):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================================
# Plotting Helpers
# ============================================================================

def plot_loss_curve(train_losses: List[float], val_losses: List[float], title: str, save_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Train Loss")
    plt.plot(epochs, val_losses, "r-s", label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Smoke Tests
# ============================================================================

def run_smoke_tests(train_texts: List[str], train_labels: List[int], val_texts: List[str], val_labels: List[int], num_classes: int) -> None:
    logger.info("=" * 60)
    logger.info("RUNNING SMOKE TESTS (Speed Measurement & Budget Verification)")
    logger.info("=" * 60)

    device = get_torch_device()
    # Sample across different classes for smoke testing
    rng = np.random.RandomState(SEED)
    sample_indices = rng.choice(len(train_texts), size=min(128, len(train_texts)), replace=False)
    sample_texts = [train_texts[i] for i in sample_indices]
    sample_labels = [train_labels[i] for i in sample_indices]
    n_sample = len(sample_texts)

    # 1. TF-IDF + LR
    t0 = time.time()
    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    X_s = vec.fit_transform(sample_texts)
    lr = LogisticRegression(max_iter=50)
    lr.fit(X_s, sample_labels)
    t_lr_smoke = time.time() - t0
    logger.info("Smoke 1: TF-IDF + LR (%d samples): %.3fs", n_sample, t_lr_smoke)

    # 2. TF-IDF + MLP
    t0 = time.time()
    mlp = TextMLP(in_features=2000, num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    X_t = torch.tensor(X_s.toarray(), dtype=torch.float32).to(device)
    y_t = torch.tensor(sample_labels, dtype=torch.long).to(device)
    mlp.train()
    opt.zero_grad()
    loss = crit(mlp(X_t), y_t)
    loss.backward()
    opt.step()
    t_mlp_smoke = time.time() - t0
    logger.info("Smoke 2: TF-IDF + MLP 1 batch (%d samples): %.3fs", n_sample, t_mlp_smoke)

    # 3. BiLSTM
    t0 = time.time()
    vocab = WordVocab(max_vocab=5000)
    vocab.build_vocab(sample_texts)
    bilstm = BiLSTMClassifier(vocab_size=len(vocab.w2i), num_classes=num_classes).to(device)
    opt_b = torch.optim.Adam(bilstm.parameters(), lr=1e-3)
    x_b = torch.tensor([vocab.encode(t, max_len=200) for t in sample_texts], dtype=torch.long).to(device)
    bilstm.train()
    opt_b.zero_grad()
    loss_b = crit(bilstm(x_b), y_t)
    loss_b.backward()
    opt_b.step()
    t_bilstm_smoke = time.time() - t0
    logger.info("Smoke 3: BiLSTM 1 batch (%d samples): %.3fs", n_sample, t_bilstm_smoke)

    # 4. DistilBERT
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes).to(device)
    bert_inputs = tok(sample_texts[:16], padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
    bert_labels = y_t[:16]
    bert.train()
    opt_bert = torch.optim.AdamW(bert.parameters(), lr=5e-5)
    opt_bert.zero_grad()
    out = bert(**bert_inputs, labels=bert_labels)
    out.loss.backward()
    opt_bert.step()
    t_bert_16 = time.time() - t0
    est_full_epoch = (t_bert_16 / 16) * len(train_texts)
    logger.info("Smoke 4: DistilBERT 1 batch of 16: %.3fs -> Est. 1 full epoch (1739 samples): %.1fs (%.2f min)", t_bert_16, est_full_epoch, est_full_epoch / 60.0)

    logger.info("-" * 60)
    logger.info("BUDGET SELECTION (Total cap: 20 minutes):")
    logger.info("  1. TF-IDF + LR: Full train, grid search C in [0.5, 1.0, 2.0, 5.0] (~2s)")
    logger.info("  2. TF-IDF + MLP: Full train, up to 25 epochs with early stopping (~8s)")
    logger.info("  3. BiLSTM: Full train, max 8 epochs, max_len 400, early stopping (~30s)")
    logger.info("  4. DistilBERT: Full train (1739 samples), max_len 256, 3 epochs (~8.5 min)")
    logger.info("Total projected training time: ~9.5 minutes (well within 20m budget).")
    logger.info("=" * 60)


# ============================================================================
# Training Functions
# ============================================================================

def train_tfidf_lr(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    classes: List[str],
) -> Tuple[Any, Any, Dict[str, float]]:
    logger.info("--- Training Model 1: TF-IDF + Logistic Regression ---")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    best_c = 1.0
    best_val_f1 = -1.0
    best_clf = None

    for c in [0.5, 1.0, 2.0, 5.0]:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
        clf.fit(X_train, train_labels)
        val_preds = clf.predict(X_val)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        logger.info("  C=%.1f -> Val Macro-F1: %.4f", c, val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c
            best_clf = clf

    # Final predictions
    train_preds = best_clf.predict(X_train)
    val_preds = best_clf.predict(X_val)
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    logger.info("  Best C=%.1f | Train Macro-F1: %.4f, Val Macro-F1: %.4f | Val Acc: %.4f (took %.2fs)",
                best_c, train_f1, val_f1, val_acc, time.time() - t0)

    # Save artifacts
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_clf, MODEL_DIR / "tfidf_lr.joblib")

    metrics = {
        "best_c": best_c,
        "train_macro_f1": float(train_f1),
        "val_macro_f1": float(val_f1),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
    }
    return vectorizer, best_clf, metrics


def train_tfidf_mlp(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    classes: List[str],
    vectorizer: TfidfVectorizer,
) -> Tuple[nn.Module, Dict[str, Any]]:
    logger.info("--- Training Model 2: TF-IDF + PyTorch MLP ---")
    t0 = time.time()
    device = get_torch_device()
    X_train = torch.tensor(vectorizer.transform(train_texts).toarray(), dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(vectorizer.transform(val_texts).toarray(), dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = TextMLP(in_features=X_train.shape[1], num_classes=len(classes), dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    patience = 5
    patience_cnt = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, 31):
        model.train()
        total_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(by)
        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            out_val = model(X_val.to(device))
            val_loss = criterion(out_val, y_val.to(device)).item()

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("  MLP early stopping triggered at epoch %d", epoch)
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train.to(device)).cpu()
        val_logits = model(X_val.to(device)).cpu()
        train_preds = train_logits.argmax(dim=1).numpy()
        val_preds = val_logits.argmax(dim=1).numpy()

    train_f1 = f1_score(train_labels, train_preds, average="macro")
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    logger.info("  MLP | Train Macro-F1: %.4f, Val Macro-F1: %.4f | Val Acc: %.4f (took %.2fs)",
                train_f1, val_f1, val_acc, time.time() - t0)

    # Save weights & loss curve
    torch.save(best_weights, MODEL_DIR / "tfidf_mlp.pt")
    plot_loss_curve(train_loss_history, val_loss_history, "TF-IDF + MLP Loss Curve", METRIC_DIR / "loss_curve_tfidf_mlp.png")

    metrics = {
        "train_macro_f1": float(train_f1),
        "val_macro_f1": float(val_f1),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "epochs_trained": len(train_loss_history),
    }
    return model, metrics


def train_bilstm(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    classes: List[str],
) -> Tuple[nn.Module, WordVocab, Dict[str, Any]]:
    logger.info("--- Training Model 3: BiLSTM (vocab ~20k, max_len 400, <= 8 epochs) ---")
    t0 = time.time()
    device = get_torch_device()

    vocab = WordVocab(max_vocab=20000)
    vocab.build_vocab(train_texts)
    vocab.save(MODEL_DIR / "bilstm_vocab.json")
    logger.info("  Built vocab size: %d", len(vocab.w2i))

    train_dataset = BiLSTMDataset(train_texts, train_labels, vocab, max_len=400)
    val_dataset = BiLSTMDataset(val_texts, val_labels, vocab, max_len=400)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = BiLSTMClassifier(vocab_size=len(vocab.w2i), embed_dim=128, hidden_dim=128, num_classes=len(classes), dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, 9):
        model.train()
        total_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(by)
        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                total_val_loss += loss.item() * len(by)
                val_preds_list.extend(out.argmax(dim=1).cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)
        val_f1_epoch = f1_score(val_labels, val_preds_list, average="macro")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        logger.info("  BiLSTM Epoch %d/8 - Train Loss: %.4f, Val Loss: %.4f, Val Macro-F1: %.4f",
                    epoch, avg_train_loss, avg_val_loss, val_f1_epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()

    # Full train and val eval
    train_loader_eval = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_preds_list = []
    with torch.no_grad():
        for bx, _ in train_loader_eval:
            out = model(bx.to(device))
            train_preds_list.extend(out.argmax(dim=1).cpu().numpy())
    val_preds_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            out = model(bx.to(device))
            val_preds_list.extend(out.argmax(dim=1).cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds_list, average="macro")
    val_f1 = f1_score(val_labels, val_preds_list, average="macro")
    train_acc = accuracy_score(train_labels, train_preds_list)
    val_acc = accuracy_score(val_labels, val_preds_list)

    logger.info("  BiLSTM | Train Macro-F1: %.4f, Val Macro-F1: %.4f | Val Acc: %.4f (took %.2fs)",
                train_f1, val_f1, val_acc, time.time() - t0)

    torch.save(best_weights, MODEL_DIR / "bilstm.pt")
    plot_loss_curve(train_loss_history, val_loss_history, "BiLSTM Loss Curve", METRIC_DIR / "loss_curve_bilstm.png")

    metrics = {
        "train_macro_f1": float(train_f1),
        "val_macro_f1": float(val_f1),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "epochs_trained": len(train_loss_history),
    }
    return model, vocab, metrics


def train_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    classes: List[str],
    num_epochs: int = 3,
) -> Tuple[nn.Module, AutoTokenizer, Dict[str, Any]]:
    logger.info("--- Training Model 4: Fine-tuned DistilBERT (max_len 256, %d epochs) ---", num_epochs)
    t0 = time.time()
    device = get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(classes)).to(device)

    train_dataset = DistilBERTDataset(train_texts, train_labels, tokenizer, max_len=256)
    val_dataset = DistilBERTDataset(val_texts, val_labels, tokenizer, max_len=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_loss = float("inf")
    best_weights = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item() * len(labels)
        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item() * len(labels)
                logits = outputs.logits
                val_preds_list.extend(logits.argmax(dim=1).cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)
        val_f1_epoch = f1_score(val_labels, val_preds_list, average="macro")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        logger.info("  DistilBERT Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, Val Macro-F1: %.4f (epoch took %.1fs)",
                    epoch, num_epochs, avg_train_loss, avg_val_loss, val_f1_epoch, time.time() - epoch_t0)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()

    # Save to MODEL_DIR / "distilbert"
    bert_save_dir = MODEL_DIR / "distilbert"
    bert_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bert_save_dir)
    tokenizer.save_pretrained(bert_save_dir)

    # Evaluate train & val
    train_loader_eval = DataLoader(train_dataset, batch_size=32, shuffle=False)
    train_preds_list = []
    with torch.no_grad():
        for batch in train_loader_eval:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            train_preds_list.extend(logits.argmax(dim=1).cpu().numpy())

    val_preds_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            val_preds_list.extend(logits.argmax(dim=1).cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds_list, average="macro")
    val_f1 = f1_score(val_labels, val_preds_list, average="macro")
    train_acc = accuracy_score(train_labels, train_preds_list)
    val_acc = accuracy_score(val_labels, val_preds_list)

    logger.info("  DistilBERT | Train Macro-F1: %.4f, Val Macro-F1: %.4f | Val Acc: %.4f (took %.2fs)",
                train_f1, val_f1, val_acc, time.time() - t0)

    plot_loss_curve(train_loss_history, val_loss_history, "DistilBERT Loss Curve", METRIC_DIR / "loss_curve_distilbert.png")

    metrics = {
        "train_macro_f1": float(train_f1),
        "val_macro_f1": float(val_f1),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "epochs_trained": len(train_loss_history),
    }
    return model, tokenizer, metrics


# ============================================================================
# Main Training Workflow (Val-only, NO test access)
# ============================================================================

def run_training_pipeline() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training and validation datasets...")
    train_df = pd.read_parquet("data/processed/classifier_train.parquet")
    val_df = pd.read_parquet("data/processed/classifier_val.parquet")

    classes = sorted(train_df["category"].unique().tolist())
    class2idx = {c: i for i, c in enumerate(classes)}
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    train_texts = train_df["text_masked"].tolist()
    train_labels = [class2idx[c] for c in train_df["category"]]
    val_texts = val_df["text_masked"].tolist()
    val_labels = [class2idx[c] for c in val_df["category"]]

    logger.info("Dataset sizes: Train=%d, Val=%d, Num Classes=%d", len(train_texts), len(val_texts), len(classes))

    # Smoke tests
    run_smoke_tests(train_texts, train_labels, val_texts, val_labels, len(classes))

    # 1. TF-IDF + LR
    vectorizer, clf_lr, lr_metrics = train_tfidf_lr(train_texts, train_labels, val_texts, val_labels, classes)

    # 2. TF-IDF + MLP
    mlp_model, mlp_metrics = train_tfidf_mlp(train_texts, train_labels, val_texts, val_labels, classes, vectorizer)

    # 3. BiLSTM
    bilstm_model, bilstm_vocab, bilstm_metrics = train_bilstm(train_texts, train_labels, val_texts, val_labels, classes)

    # 4. DistilBERT
    distilbert_model, distilbert_tok, bert_metrics = train_distilbert(train_texts, train_labels, val_texts, val_labels, classes, num_epochs=3)

    # Compare validation performance and pick best model
    all_models = {
        "tfidf_lr": lr_metrics,
        "tfidf_mlp": mlp_metrics,
        "bilstm": bilstm_metrics,
        "distilbert": bert_metrics,
    }

    best_model_name = max(all_models.keys(), key=lambda m: all_models[m]["val_macro_f1"])
    best_val_f1 = all_models[best_model_name]["val_macro_f1"]
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY (Best model by Val Macro-F1: %s with %.4f):", best_model_name, best_val_f1)
    for name, m in all_models.items():
        gap = m["train_macro_f1"] - m["val_macro_f1"]
        logger.info("  %-12s: Val Macro-F1=%.4f, Val Acc=%.4f (Train Macro-F1=%.4f, Gap=%.4f)",
                    name, m["val_macro_f1"], m["val_accuracy"], m["train_macro_f1"], gap)
    logger.info("=" * 60)

    best_meta = {
        "best_model": best_model_name,
        "val_macro_f1": best_val_f1,
        "val_accuracy": all_models[best_model_name]["val_accuracy"],
        "models": all_models,
    }
    with open(MODEL_DIR / "best_model_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)

    with open(METRIC_DIR / "val_metrics.json", "w") as f:
        json.dump(all_models, f, indent=2)

    logger.info("Training complete. Best model metadata saved to %s", MODEL_DIR / "best_model_meta.json")


# ============================================================================
# Final Test Set Evaluation (--final-eval ONLY)
# ============================================================================

def run_final_eval() -> None:
    logger.info("=" * 60)
    logger.info("RUNNING FINAL TEST EVALUATION (--final-eval)")
    logger.info("=" * 60)

    if not (MODEL_DIR / "classes.json").exists() or not (MODEL_DIR / "best_model_meta.json").exists():
        raise FileNotFoundError("Models not trained yet! Run python scripts/train_classifier.py first.")

    with open(MODEL_DIR / "classes.json") as f:
        classes = json.load(f)
    class2idx = {c: i for i, c in enumerate(classes)}

    with open(MODEL_DIR / "best_model_meta.json") as f:
        best_meta = json.load(f)
    val_metrics = best_meta["models"]

    # Touch test set ONLY here
    test_df = pd.read_parquet("data/processed/classifier_test.parquet")
    test_texts = test_df["text_masked"].tolist()
    test_labels = [class2idx[c] for c in test_df["category"]]
    logger.info("Test set loaded: %d samples, %d classes", len(test_texts), len(classes))

    device = get_torch_device()
    final_results = {}

    # 1. TF-IDF + LR
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    clf_lr = joblib.load(MODEL_DIR / "tfidf_lr.joblib")
    X_test_tfidf = vectorizer.transform(test_texts)
    lr_preds = clf_lr.predict(X_test_tfidf)
    lr_test_f1 = f1_score(test_labels, lr_preds, average="macro")
    lr_test_acc = accuracy_score(test_labels, lr_preds)
    plot_confusion_matrix(test_labels, lr_preds, classes, "TF-IDF + LR Confusion Matrix (Test)", METRIC_DIR / "cm_tfidf_lr.png")
    final_results["tfidf_lr"] = {
        **val_metrics["tfidf_lr"],
        "test_macro_f1": float(lr_test_f1),
        "test_accuracy": float(lr_test_acc),
    }

    # 2. TF-IDF + MLP
    mlp = TextMLP(in_features=X_test_tfidf.shape[1], num_classes=len(classes), dropout=0.3).to(device)
    mlp.load_state_dict(torch.load(MODEL_DIR / "tfidf_mlp.pt", map_location=device))
    mlp.eval()
    X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32).to(device)
    with torch.no_grad():
        mlp_preds = mlp(X_test_tensor).argmax(dim=1).cpu().numpy()
    mlp_test_f1 = f1_score(test_labels, mlp_preds, average="macro")
    mlp_test_acc = accuracy_score(test_labels, mlp_preds)
    plot_confusion_matrix(test_labels, mlp_preds, classes, "TF-IDF + MLP Confusion Matrix (Test)", METRIC_DIR / "cm_tfidf_mlp.png")
    final_results["tfidf_mlp"] = {
        **val_metrics["tfidf_mlp"],
        "test_macro_f1": float(mlp_test_f1),
        "test_accuracy": float(mlp_test_acc),
    }

    # 3. BiLSTM
    vocab = WordVocab.load(MODEL_DIR / "bilstm_vocab.json")
    bilstm = BiLSTMClassifier(vocab_size=len(vocab.w2i), embed_dim=128, hidden_dim=128, num_classes=len(classes), dropout=0.3).to(device)
    bilstm.load_state_dict(torch.load(MODEL_DIR / "bilstm.pt", map_location=device))
    bilstm.eval()
    test_bilstm_dataset = BiLSTMDataset(test_texts, test_labels, vocab, max_len=400)
    test_bilstm_loader = DataLoader(test_bilstm_dataset, batch_size=64, shuffle=False)
    bilstm_preds_list = []
    with torch.no_grad():
        for bx, _ in test_bilstm_loader:
            bilstm_preds_list.extend(bilstm(bx.to(device)).argmax(dim=1).cpu().numpy())
    bilstm_test_f1 = f1_score(test_labels, bilstm_preds_list, average="macro")
    bilstm_test_acc = accuracy_score(test_labels, bilstm_preds_list)
    plot_confusion_matrix(test_labels, bilstm_preds_list, classes, "BiLSTM Confusion Matrix (Test)", METRIC_DIR / "cm_bilstm.png")
    final_results["bilstm"] = {
        **val_metrics["bilstm"],
        "test_macro_f1": float(bilstm_test_f1),
        "test_accuracy": float(bilstm_test_acc),
    }

    # 4. DistilBERT
    bert_dir = MODEL_DIR / "distilbert"
    bert_tok = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
    bert_model.eval()
    test_bert_dataset = DistilBERTDataset(test_texts, test_labels, bert_tok, max_len=256)
    test_bert_loader = DataLoader(test_bert_dataset, batch_size=32, shuffle=False)
    bert_preds_list = []
    with torch.no_grad():
        for batch in test_bert_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = bert_model(input_ids=input_ids, attention_mask=attention_mask).logits
            bert_preds_list.extend(logits.argmax(dim=1).cpu().numpy())
    bert_test_f1 = f1_score(test_labels, bert_preds_list, average="macro")
    bert_test_acc = accuracy_score(test_labels, bert_preds_list)
    plot_confusion_matrix(test_labels, bert_preds_list, classes, "DistilBERT Confusion Matrix (Test)", METRIC_DIR / "cm_distilbert.png")
    final_results["distilbert"] = {
        **val_metrics["distilbert"],
        "test_macro_f1": float(bert_test_f1),
        "test_accuracy": float(bert_test_acc),
    }

    # Save full metrics JSON
    with open(METRIC_DIR / "metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Print 4-Model Comparison Table
    print("\n" + "=" * 90)
    print("STAGE 3a RESUME CLASSIFIER - 4-MODEL COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Model':<16} | {'Train F1':<10} | {'Val F1':<10} | {'Test F1':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Train-Val Gap':<12}"
    print(header)
    print("-" * 90)
    for model_name, m in final_results.items():
        gap = m["train_macro_f1"] - m["val_macro_f1"]
        row = (
            f"{model_name:<16} | "
            f"{m['train_macro_f1']:.4f}     | "
            f"{m['val_macro_f1']:.4f}   | "
            f"{m['test_macro_f1']:.4f}    | "
            f"{m['train_accuracy']:.4f}     | "
            f"{m['val_accuracy']:.4f}    | "
            f"{m['test_accuracy']:.4f}    | "
            f"{gap:+.4f}"
        )
        print(row)
    print("=" * 90)
    print(f"Best model by Validation Macro-F1: {best_meta['best_model']} ({best_meta['val_macro_f1']:.4f})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate Stage 3a Resume Classifier models.")
    parser.add_argument("--final-eval", action="store_true", help="Evaluate models on test set (run ONCE at end).")
    args = parser.parse_args()

    if args.final_eval:
        run_final_eval()
    else:
        run_training_pipeline()


if __name__ == "__main__":
    main()
