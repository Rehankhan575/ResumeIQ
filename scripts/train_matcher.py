#!/usr/bin/env python3
"""
scripts/train_matcher.py
========================
Trains and evaluates Stage 4 Résumé–JD Matching models:
1. TF-IDF cosine baseline
2. Zero-shot sentence-transformers (all-MiniLM-L6-v2)
3. Fine-tuned Siamese bi-encoder (all-MiniLM-L6-v2) trained with MultipleNegativesRankingLoss

Enforces:
- Strict split isolation: Training pairs come ONLY from classifier_train and jd_train.
- Category capping: JDs per category capped (default 20) to prevent HEALTHCARE/SALES/IT dominance.
- Top 15 categories with >= 20 JDs.
- Hard negatives sampled from top-similar categories based on TF-IDF centroid similarity.
- MRR, nDCG@10, Recall@5, and Recall@10 evaluation overall and per-category.
- Test split touched ONLY when --final-eval is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings

logger = logging.getLogger(__name__)

MATCHER_DIR = ROOT / "artifacts" / "models" / "matcher"
METRICS_DIR = ROOT / "artifacts" / "metrics" / "matcher"
FINE_TUNED_MODEL_PATH = MATCHER_DIR / "model"
TFIDF_MATCHER_PATH = MATCHER_DIR / "tfidf_matcher.pkl"

TOP_15_CATEGORIES: List[str] = [
    "HEALTHCARE",
    "SALES",
    "INFORMATION-TECHNOLOGY",
    "ACCOUNTANT",
    "ENGINEERING",
    "CONSTRUCTION",
    "ADVOCATE",
    "CONSULTANT",
    "HR",
    "TEACHER",
    "BANKING",
    "BPO",
    "CHEF",
    "BUSINESS-DEVELOPMENT",
    "DIGITAL-MEDIA",
]


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load and filter resume and JD datasets to TOP_15_CATEGORIES."""
    resumes: Dict[str, pd.DataFrame] = {}
    jds: Dict[str, pd.DataFrame] = {}

    for split in ["train", "val", "test"]:
        df_r = pd.read_parquet(ROOT / f"data/processed/classifier_{split}.parquet")
        df_r = df_r[df_r["category"].isin(TOP_15_CATEGORIES)].reset_index(drop=True)
        resumes[split] = df_r

        df_j = pd.read_parquet(ROOT / f"data/processed/jd_{split}.parquet")
        df_j = df_j[df_j["category"].isin(TOP_15_CATEGORIES)].reset_index(drop=True)
        jds[split] = df_j

    return resumes, jds


def compute_similar_categories_map(df_jd_train: pd.DataFrame) -> Dict[str, List[str]]:
    """Compute top 3 similar categories for each category using TF-IDF centroids."""
    vec = TfidfVectorizer(max_features=2500, stop_words="english")
    X = vec.fit_transform(df_jd_train["text"])

    centroids: Dict[str, np.ndarray] = {}
    for cat in TOP_15_CATEGORIES:
        mask = (df_jd_train["category"] == cat).values
        if mask.sum() > 0:
            centroids[cat] = np.asarray(X[mask].mean(axis=0)).flatten()
        else:
            centroids[cat] = np.zeros(X.shape[1])

    cat_matrix = np.vstack([centroids[c] for c in TOP_15_CATEGORIES])
    sim_matrix = cosine_similarity(cat_matrix)

    similar_map: Dict[str, List[str]] = {}
    for i, c in enumerate(TOP_15_CATEGORIES):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # exclude self
        top_indices = np.argsort(-sims)[:3]
        similar_map[c] = [TOP_15_CATEGORIES[idx] for idx in top_indices]

    return similar_map


def build_training_pairs(
    df_res_train: pd.DataFrame,
    df_jd_train: pd.DataFrame,
    jd_cap: int = 20,
    seed: int = 42,
) -> Tuple[List[Any], List[str], List[str]]:
    """Construct (resume, positive_jd, hard_negative_jd) triplets for fine-tuning.

    Caps JDs per category to ensure balanced representation across domains.
    Returns:
        triplets, used_resume_ids, used_jd_ids
    """
    from sentence_transformers import InputExample

    rng = random.Random(seed)
    similar_cats_map = compute_similar_categories_map(df_jd_train)

    # Cap JDs per category in train
    capped_jds_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat in TOP_15_CATEGORIES:
        cat_jds = df_jd_train[df_jd_train["category"] == cat].to_dict("records")
        if len(cat_jds) > jd_cap:
            cat_jds = rng.sample(cat_jds, jd_cap)
        capped_jds_by_cat[cat] = cat_jds

    # Group resumes by category
    resumes_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat in TOP_15_CATEGORIES:
        resumes_by_cat[cat] = df_res_train[df_res_train["category"] == cat].to_dict("records")

    triplets: List[InputExample] = []
    used_resume_ids: Set[str] = set()
    used_jd_ids: Set[str] = set()

    for cat in TOP_15_CATEGORIES:
        cat_res = resumes_by_cat.get(cat, [])
        cat_jds = capped_jds_by_cat.get(cat, [])
        if not cat_res or not cat_jds:
            continue

        sim_cats = similar_cats_map.get(cat, [])
        hard_neg_pool: List[Dict[str, Any]] = []
        for sc in sim_cats:
            hard_neg_pool.extend(capped_jds_by_cat.get(sc, []))

        # Other categories pool
        other_cats = [c for c in TOP_15_CATEGORIES if c != cat and c not in sim_cats]
        random_neg_pool: List[Dict[str, Any]] = []
        for oc in other_cats:
            random_neg_pool.extend(capped_jds_by_cat.get(oc, []))

        for res in cat_res:
            res_id = str(res["id"])
            res_text = str(res["text"])[:1000]

            # 1-2 positive JDs per resume
            pos_samples = rng.sample(cat_jds, min(2, len(cat_jds)))
            for pos in pos_samples:
                pos_id = str(pos["jd_id"])
                pos_text = str(pos["text"])[:1000]

                # Sample hard negative from similar categories if available, else random
                if hard_neg_pool and rng.random() < 0.7:
                    neg = rng.choice(hard_neg_pool)
                elif random_neg_pool:
                    neg = rng.choice(random_neg_pool)
                else:
                    neg = rng.choice(hard_neg_pool)

                neg_id = str(neg["jd_id"])
                neg_text = str(neg["text"])[:1000]

                triplets.append(InputExample(texts=[res_text, pos_text, neg_text]))
                used_resume_ids.add(res_id)
                used_jd_ids.add(pos_id)
                used_jd_ids.add(neg_id)

    rng.shuffle(triplets)
    return triplets, sorted(used_resume_ids), sorted(used_jd_ids)


def train_tfidf_baseline(
    df_res_train: pd.DataFrame,
    df_jd_train: pd.DataFrame,
) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on combined training corpus and save to disk."""
    corpus = list(df_res_train["text"].fillna("")) + list(df_jd_train["text"].fillna(""))
    vec = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    vec.fit(corpus)

    MATCHER_DIR.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_MATCHER_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec}, f)

    return vec


def compute_ranking_metrics_for_matrix(
    scores_matrix: np.ndarray,
    query_categories: List[str],
    doc_categories: List[str],
) -> Dict[str, Any]:
    """Compute MRR, nDCG@10, Recall@5, and Recall@10 overall and per category."""
    mrrs: List[float] = []
    ndcgs: List[float] = []
    r5s: List[float] = []
    r10s: List[float] = []

    per_cat_data: Dict[str, Dict[str, List[float]]] = {
        cat: {"mrr": [], "ndcg@10": [], "recall@5": [], "recall@10": []}
        for cat in TOP_15_CATEGORIES
    }

    doc_cats_arr = np.array(doc_categories)

    for i, q_cat in enumerate(query_categories):
        scores = scores_matrix[i]
        order = np.argsort(-scores)
        sorted_doc_cats = doc_cats_arr[order]
        rel = (sorted_doc_cats == q_cat).astype(int)
        total_rel = int(rel.sum())
        if total_rel == 0:
            continue

        # MRR
        first_rel = np.where(rel == 1)[0]
        mrr = 1.0 / (first_rel[0] + 1) if len(first_rel) > 0 else 0.0
        mrrs.append(mrr)

        # Recall@5 and Recall@10
        r5 = float(rel[:5].sum() / total_rel)
        r10 = float(rel[:10].sum() / total_rel)
        r5s.append(r5)
        r10s.append(r10)

        # nDCG@10
        k = 10
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg10 = float((rel[:k] * discounts).sum())
        ideal_rel = np.zeros(k)
        ideal_rel[: min(k, total_rel)] = 1
        idcg10 = float((ideal_rel * discounts).sum())
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
        ndcgs.append(ndcg10)

        if q_cat in per_cat_data:
            per_cat_data[q_cat]["mrr"].append(mrr)
            per_cat_data[q_cat]["ndcg@10"].append(ndcg10)
            per_cat_data[q_cat]["recall@5"].append(r5)
            per_cat_data[q_cat]["recall@10"].append(r10)

    overall = {
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "ndcg@10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "recall@5": round(float(np.mean(r5s)), 4) if r5s else 0.0,
        "recall@10": round(float(np.mean(r10s)), 4) if r10s else 0.0,
    }

    per_category: Dict[str, Dict[str, float]] = {}
    for cat in TOP_15_CATEGORIES:
        cd = per_cat_data[cat]
        per_category[cat] = {
            "mrr": round(float(np.mean(cd["mrr"])), 4) if cd["mrr"] else 0.0,
            "ndcg@10": round(float(np.mean(cd["ndcg@10"])), 4) if cd["ndcg@10"] else 0.0,
            "recall@5": round(float(np.mean(cd["recall@5"])), 4) if cd["recall@5"] else 0.0,
            "recall@10": round(float(np.mean(cd["recall@10"])), 4) if cd["recall@10"] else 0.0,
            "count": len(cd["mrr"]),
        }

    return {"overall": overall, "per_category": per_category}


def evaluate_split_models(
    tfidf_vec: TfidfVectorizer,
    zero_shot_model: Any,
    fine_tuned_model: Any,
    df_queries: pd.DataFrame,
    df_corpus: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate TF-IDF, Zero-shot, and Fine-tuned models on a given split."""
    query_texts = df_queries["text"].fillna("").tolist()
    query_cats = df_queries["category"].tolist()
    corpus_texts = df_corpus["text"].fillna("").tolist()
    corpus_cats = df_corpus["category"].tolist()

    # 1. TF-IDF
    q_tfidf = tfidf_vec.transform(query_texts)
    c_tfidf = tfidf_vec.transform(corpus_texts)
    sim_tfidf = cosine_similarity(q_tfidf, c_tfidf)
    res_tfidf = compute_ranking_metrics_for_matrix(sim_tfidf, query_cats, corpus_cats)

    # 2. Zero-Shot MiniLM
    q_zs = zero_shot_model.encode(
        query_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    c_zs = zero_shot_model.encode(
        corpus_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    sim_zs = np.dot(q_zs, c_zs.T)
    res_zs = compute_ranking_metrics_for_matrix(sim_zs, query_cats, corpus_cats)

    # 3. Fine-Tuned MiniLM
    q_ft = fine_tuned_model.encode(
        query_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    c_ft = fine_tuned_model.encode(
        corpus_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    sim_ft = np.dot(q_ft, c_ft.T)
    res_ft = compute_ranking_metrics_for_matrix(sim_ft, query_cats, corpus_cats)

    return {
        "tfidf": res_tfidf,
        "zero_shot": res_zs,
        "fine_tuned": res_ft,
    }


def print_comparison_table(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"{'Model Variant':<28} | {'MRR':<8} | {'nDCG@10':<8} | {'Recall@5':<10} | {'Recall@10':<10}")
    print("-" * 72)
    for key, name in [
        ("tfidf", "1. TF-IDF Cosine Baseline"),
        ("zero_shot", "2. Zero-Shot all-MiniLM-L6-v2"),
        ("fine_tuned", "3. Fine-Tuned Siamese Bi-Encoder"),
    ]:
        ov = metrics[key]["overall"]
        print(f"{name:<28} | {ov['mrr']:<8.4f} | {ov['ndcg@10']:<8.4f} | {ov['recall@5']:<10.4f} | {ov['recall@10']:<10.4f}")


def print_per_category_table(per_cat: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'PER-CATEGORY PERFORMANCE (Fine-Tuned Siamese Bi-Encoder)':^75}")
    print(f"{'Category':<24} | {'Count':<6} | {'MRR':<8} | {'nDCG@10':<8} | {'Recall@5':<10} | {'Recall@10':<10}")
    print("-" * 75)
    for cat in TOP_15_CATEGORIES:
        m = per_cat[cat]
        print(f"{cat:<24} | {m['count']:<6} | {m['mrr']:<8.4f} | {m['ndcg@10']:<8.4f} | {m['recall@5']:<10.4f} | {m['recall@10']:<10.4f}")


def run_training_pipeline(args: argparse.Namespace) -> None:
    set_all_seeds(settings.seed)
    device = settings.device
    print(f"Matcher Training Pipeline | Device: {device} | Seed: {settings.seed}")

    resumes, jds = load_datasets()
    print(f"Loaded datasets (15 categories):")
    print(f"  Resumes: train={len(resumes['train'])}, val={len(resumes['val'])}, test={len(resumes['test'])}")
    print(f"  JDs:     train={len(jds['train'])}, val={len(jds['val'])}, test={len(jds['test'])}")

    # Build training pairs
    t0 = time.time()
    train_triplets, train_res_ids, train_jd_ids = build_training_pairs(
        resumes["train"], jds["train"], jd_cap=args.jd_cap, seed=settings.seed
    )
    print(f"Generated {len(train_triplets)} triplets from {len(train_res_ids)} resumes and {len(train_jd_ids)} JDs in {time.time()-t0:.2f}s")

    # Fit and save TF-IDF baseline
    print("Fitting TF-IDF baseline...")
    tfidf_vec = train_tfidf_baseline(resumes["train"], jds["train"])

    from sentence_transformers import SentenceTransformer, losses

    # Zero-shot baseline model
    print(f"Loading base model: {settings.sbert_model_name}...")
    zero_shot_model = SentenceTransformer(settings.sbert_model_name, device=device)
    zero_shot_model.max_seq_length = 256

    # Fine-tuning
    print(f"Initializing fine-tuned model for training ({args.epochs} epochs, batch_size={args.batch_size})...")
    fine_tuned_model = SentenceTransformer(settings.sbert_model_name, device=device)
    fine_tuned_model.max_seq_length = 256

    # Speed smoke test on small subset
    smoke_triplets = train_triplets[:100]
    smoke_loader = DataLoader(smoke_triplets, batch_size=args.batch_size, shuffle=True)
    smoke_loss = losses.MultipleNegativesRankingLoss(fine_tuned_model)
    t_smoke = time.time()
    fine_tuned_model.fit(train_objectives=[(smoke_loader, smoke_loss)], epochs=1, warmup_steps=2, show_progress_bar=False)
    smoke_dur = time.time() - t_smoke
    smoke_speed = len(smoke_triplets) / smoke_dur if smoke_dur > 0 else 0
    print(f"[SMOKE TEST] Trained 100 pairs in {smoke_dur:.2f}s ({smoke_speed:.1f} pairs/sec on {device})")

    # Full training
    train_loader = DataLoader(train_triplets, batch_size=args.batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(fine_tuned_model)
    t_train = time.time()
    print(f"Training on full {len(train_triplets)} triplets...")
    fine_tuned_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(train_loader) * 0.1),
        show_progress_bar=False,
    )
    train_dur = time.time() - t_train
    print(f"Fine-tuning completed in {train_dur:.2f}s ({train_dur/60.0:.2f} min, budget: 15 min)")

    # Save fine-tuned model
    FINE_TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    fine_tuned_model.save(str(FINE_TUNED_MODEL_PATH))
    print(f"Saved fine-tuned model to {FINE_TUNED_MODEL_PATH.relative_to(ROOT)}")

    # Save train manifest IDs for test isolation proof
    manifest_path = MATCHER_DIR / "training_id_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"train_resume_ids": train_res_ids, "train_jd_ids": train_jd_ids}, f)

    # Evaluate on VAL
    print("\nEvaluating on VALIDATION split (tuning)...")
    val_metrics = evaluate_split_models(
        tfidf_vec, zero_shot_model, fine_tuned_model, resumes["val"], jds["val"]
    )
    print_comparison_table("VAL EVALUATION (3-ROW COMPARISON TABLE)", val_metrics)
    print_per_category_table(val_metrics["fine_tuned"]["per_category"])

    # Save val metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    val_out_path = METRICS_DIR / "matcher_val.json"
    with open(val_out_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved validation metrics to {val_out_path.relative_to(ROOT)}")

    # Final eval on TEST if requested
    if args.final_eval:
        print("\n" + "=" * 75)
        print("RUNNING FINAL EVALUATION ON TEST SPLIT")
        print("=" * 75)
        test_metrics = evaluate_split_models(
            tfidf_vec, zero_shot_model, fine_tuned_model, resumes["test"], jds["test"]
        )
        print_comparison_table("TEST EVALUATION (3-ROW COMPARISON TABLE)", test_metrics)
        print_per_category_table(test_metrics["fine_tuned"]["per_category"])

        full_eval = {
            "val": val_metrics,
            "test": test_metrics,
            "training_stats": {
                "num_triplets": len(train_triplets),
                "num_resumes": len(train_res_ids),
                "num_jds": len(train_jd_ids),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "training_time_seconds": round(train_dur, 2),
            },
        }
        eval_out_path = METRICS_DIR / "matcher_eval.json"
        with open(eval_out_path, "w") as f:
            json.dump(full_eval, f, indent=2)
        print(f"\nFinal evaluation saved to {eval_out_path.relative_to(ROOT)}")


def run_only_final_eval() -> None:
    """Load existing trained model and run final evaluation on TEST split."""
    set_all_seeds(settings.seed)
    device = settings.device
    print(f"Running Matcher Final Evaluation | Device: {device}")

    resumes, jds = load_datasets()

    # Load TF-IDF
    with open(TFIDF_MATCHER_PATH, "rb") as f:
        tfidf_vec = pickle.load(f)["vectorizer"]

    from sentence_transformers import SentenceTransformer

    zero_shot_model = SentenceTransformer(settings.sbert_model_name, device=device)
    zero_shot_model.max_seq_length = 256

    fine_tuned_model = SentenceTransformer(str(FINE_TUNED_MODEL_PATH), device=device)
    fine_tuned_model.max_seq_length = 256

    print("\nEvaluating on VALIDATION split...")
    val_metrics = evaluate_split_models(
        tfidf_vec, zero_shot_model, fine_tuned_model, resumes["val"], jds["val"]
    )
    print_comparison_table("VAL EVALUATION (3-ROW COMPARISON TABLE)", val_metrics)

    print("\nEvaluating on TEST split...")
    test_metrics = evaluate_split_models(
        tfidf_vec, zero_shot_model, fine_tuned_model, resumes["test"], jds["test"]
    )
    print_comparison_table("TEST EVALUATION (3-ROW COMPARISON TABLE)", test_metrics)
    print_per_category_table(test_metrics["fine_tuned"]["per_category"])

    full_eval = {
        "val": val_metrics,
        "test": test_metrics,
    }
    eval_out_path = METRICS_DIR / "matcher_eval.json"
    with open(eval_out_path, "w") as f:
        json.dump(full_eval, f, indent=2)
    print(f"\nSaved final evaluation to {eval_out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate résumé-JD matcher.")
    parser.add_argument("--final-eval", action="store_true", help="Run final evaluation on TEST split.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--jd-cap", type=int, default=20, help="Cap on JDs per category in training pairs.")
    args = parser.parse_args()

    # If --final-eval is passed and model is already trained, we can run eval directly or full pipeline
    if args.final_eval and FINE_TUNED_MODEL_PATH.exists() and TFIDF_MATCHER_PATH.exists():
        run_only_final_eval()
    else:
        run_training_pipeline(args)

