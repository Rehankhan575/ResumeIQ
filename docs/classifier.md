# Stage 3a: Resume Category Classifier

## 1. Overview & Problem Definition

The Stage 3a Classifier maps unstructured or semi-structured resume text to one of **24 job categories** from the LiveCareer taxonomy:
`ACCOUNTANT`, `ADVOCATE`, `AGRICULTURE`, `APPAREL`, `ARTS`, `AUTOMOBILE`, `AVIATION`, `BANKING`, `BPO`, `BUSINESS-DEVELOPMENT`, `CHEF`, `CONSTRUCTION`, `CONSULTANT`, `DESIGNER`, `DIGITAL-MEDIA`, `ENGINEERING`, `FINANCE`, `FITNESS`, `HEALTHCARE`, `HR`, `INFORMATION-TECHNOLOGY`, `PUBLIC-RELATIONS`, `SALES`, `TEACHER`.

To prevent trivial label leakage from explicit job titles in headers, all models were trained and evaluated exclusively on the **masked text column** (`text_masked`). Splits were grouped by content fingerprint with 0 duplicate texts across splits (train: 1,739, val: 370, test: 374).

---

## 2. Experimental Setup & Time Budget

Under a 20-minute strict compute budget on an Apple Silicon machine (M4 with PyTorch MPS acceleration):
1. **Initial Smoke Testing**: Measured 1 batch / 1 epoch inference/training throughput across all 4 architectures.
   - TF-IDF + LR: 0.11s per 128 samples
   - TF-IDF + MLP: 1.92s per 128 samples
   - BiLSTM: 0.89s per 128 samples
   - DistilBERT: 2.94s per 16 samples (extrapolating to ~230s per full epoch on 1,739 samples)
2. **Selected Hyperparameters**:
   - **TF-IDF + LR**: Max features 5,000, n-grams (1, 2), sublinear TF, validation grid search over $C \in \{0.5, 1.0, 2.0, 5.0\}$ (best: $C=5.0$).
   - **TF-IDF + MLP**: 2 hidden layers (5,000 $\to$ 256 $\to$ 128 $\to$ 24), BatchNorm1d, Dropout(0.3), AdamW ($lr=10^{-3}$, weight decay $10^{-4}$), early stopping on validation loss (patience 5, stopped at epoch 18).
   - **BiLSTM**: Word vocabulary 20,000, embedding dimension 128, hidden dimension 128, bidirectional, max sequence length 400 tokens, 8 epochs, AdamW ($lr=10^{-3}$, weight decay $10^{-4}$).
   - **DistilBERT**: `distilbert-base-uncased`, max sequence length 256 tokens, batch size 16, 3 epochs, AdamW ($lr=3 \times 10^{-5}$, weight decay 0.01), linear warmup with decay. Total DistilBERT training time was 12.5 minutes on MPS.

---

## 3. Results & 4-Model Comparison

All models were tuned strictly on the validation set. The test set was touched **only once** by running `python scripts/train_classifier.py --final-eval` at the conclusion of experiments.

| Model | Train Macro-F1 | Val Macro-F1 | Test Macro-F1 | Train Acc | Val Acc | Test Acc | Train-Val Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LR** | 0.9882 | 0.5913 | 0.5959 | 0.9902 | 0.6405 | 0.6631 | +0.3969 |
| **TF-IDF + MLP** | 1.0000 | **0.6001** | 0.6181 | 1.0000 | 0.6459 | 0.6524 | +0.3999 |
| **BiLSTM** | 0.7091 | 0.1860 | 0.1966 | 0.7430 | 0.2351 | 0.2513 | +0.5231 |
| **DistilBERT** | 0.6748 | 0.5675 | **0.6454** | 0.7585 | **0.6514** | **0.7273** | **+0.1073** |

### Paired Hypothesis Testing & Statistical Analysis
Evaluated across 1,000 bootstrap resamples (seed 42) and paired McNemar tests on the 374 test set samples (`artifacts/reports/classifier_paired_comparison.csv`):

| Comparison | Metric | Diff (A − B) | 95% CI Lower | 95% CI Upper | Bootstrap p-value | McNemar $\chi^2$ (df=1) | McNemar p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT vs. LR** | Macro-F1 | **+0.0495** | +0.0056 | +0.0931 | 0.0360 | — | — |
| **DistilBERT vs. LR** | Accuracy | **+0.0642** | +0.0160 | +0.1096 | 0.0220 | 6.6125 | 0.0101 |
| **DistilBERT vs. MLP** | Macro-F1 | +0.0273 | -0.0250 | +0.0901 | 0.2800 | — | — |
| **DistilBERT vs. MLP** | Accuracy | **+0.0749** | +0.0241 | +0.1257 | 0.0100 | 7.0096 | 0.0081 |
| **MLP vs. LR** | Macro-F1 | +0.0222 | -0.0220 | +0.0569 | 0.3480 | — | — |
| **MLP vs. LR** | Accuracy | -0.0107 | -0.0428 | +0.0214 | 0.5980 | 0.2368 | 0.6265 |

### Findings & Caveats
- **LR and MLP are indistinguishable on test** across both macro-F1 (diff +0.0222, 95% CI `[-0.0220, +0.0569]`, $p=0.3480$) and accuracy (diff -0.0107, 95% CI `[-0.0428, +0.0214]`, $p=0.5980$, McNemar $p=0.6265$).
- **DistilBERT vs. LR:** Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely.
  - vs. LR: macro-F1 +0.050 (95% CI `[+0.0056, +0.0931]`, $p=0.0360$) and accuracy +6.4 pts (+0.0642, 95% CI `[+0.0160, +0.1096]`, $p=0.0220$, McNemar $p=0.0101$).
  - vs. MLP: accuracy +7.5 pts (+0.0749, 95% CI `[+0.0241, +0.1257]`, $p=0.0100$, McNemar $p=0.0081$), but its macro-F1 gain over MLP (+0.0273, 95% CI `[-0.0250, +0.0901]`, $p=0.2800$) is not significant.
- **Caveats stated in the same place:**
  - Six paired tests were conducted with no multiplicity correction.
  - All comparisons rest on a single test split of 374 resumes across 24 classes (average ~15, smallest class 3, largest 18).
  - On validation, DistilBERT did NOT outperform (val macro-F1 0.5675 vs. MLP 0.6001), so validation and test disagree.
  - DistilBERT was trained for only 3 epochs due to compute limits; the BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).
- **Production Model Selection:**
  - The served classifier is the MLP because the selection rule (val macro-F1) was fixed before the test set was opened.
  - Choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.

---

## 4. Visual Artifacts

Generated metrics and plots are stored in `artifacts/metrics/classifier/`:
- Confusion Matrices:
  - `cm_tfidf_lr.png`
  - `cm_tfidf_mlp.png`
  - `cm_bilstm.png`
  - `cm_distilbert.png`
- Training & Validation Loss Curves:
  - `loss_curve_tfidf_mlp.png`
  - `loss_curve_bilstm.png`
  - `loss_curve_distilbert.png`
- Metrics payload: `metrics.json`

---

## 5. Production Serving & Fallback

The entrypoint `app.models.classifier.classify(text, top_k=3, normalize=True)` provides:
1. **Top-3 Category Predictions**: Returns ranked `CategoryPrediction(category=..., probability=...)` objects.
2. **Normalized Probabilities**: Probabilities sum to 1.0.
3. **Graceful Fallback**: If model weights in `artifacts/models/classifier/` are missing or corrupted, the system automatically trains a lightweight on-the-fly TF-IDF + Logistic Regression model from `data/processed/classifier_train.parquet` in ~1.4 seconds, caches it in memory, and completes inference without crashing.
4. **Pipeline Integration**: `classify_resume(parsed: ParsedResume)` consumes extracted resume objects directly from the parser stage.

---

## 6. Limitations & Error Analysis

1. **Title Masking vs Vocabulary Drift**: Masking explicit job title strings forces models to infer categories from verbs, domain tools, and responsibility context. High-overlap categories (e.g., `FINANCE` vs `BANKING`, or `BUSINESS-DEVELOPMENT` vs `SALES`) exhibit mutual confusion.
2. **BiLSTM Embedding Sparsity**: Training word embeddings from scratch on only 1,739 domain resumes with a 20k vocabulary leads to high out-of-vocabulary rates and severe underfitting (0.1966 test macro-F1) compared to subword transformer representations.
3. **Sequence Length Truncation**: DistilBERT was evaluated with max length 256 to fit within the 20-minute training budget. Resumes longer than ~350 words have later sections (education, prior historical experience) truncated.
