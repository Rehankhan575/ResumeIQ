# End-to-End Pipeline Evaluation & Empirical Results

This document presents the complete empirical benchmark across all stages of the Resume Screening and Job Fit platform, compiled directly from stored metrics in `artifacts/metrics/`.

---

## 1. Resume Category Classifier (24 Classes)

Evaluated on 2,483 stratified resumes across 24 job categories (per-class counts range from 22 in BPO to 120 in INFORMATION-TECHNOLOGY; Train: 1,739, Val: 370, Test: 374):

| Architecture | Train Acc | Train Macro F1 | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Gen Gap (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 99.02% | 0.9882 | 64.05% | 0.5913 | 66.31% | 0.5959 | 0.3969 |
| **TF-IDF + MLP (2-layer)** | 100.00% | 1.0000 | 64.59% | 0.6001 | 65.24% | 0.6181 | 0.3999 |
| **BiLSTM + Pretrained Embeddings** | 74.30% | 0.7091 | 23.51% | 0.1860 | 25.13% | 0.1966 | 0.5231 |
| **DistilBERT (Fine-tuned, 3 epochs)** | 75.85% | 0.6748 | 65.14% | 0.5675 | 72.73% | 0.6454 | **0.1073** |

### Bootstrap 95% Confidence Intervals & Paired Hypothesis Tests
Evaluated via 1,000 bootstrap resamples (seed 42) on the 374 test set samples without retraining (`scripts/classifier_ci.py`):

| Model | Test Macro F1 | 95% CI Lower | 95% CI Upper | CI Width | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 0.5959 | 0.5465 | 0.6352 | 0.0887 | 66.31% |
| **TF-IDF + MLP (2-layer)** | 0.6181 | 0.5606 | 0.6587 | 0.0981 | 65.24% |
| **BiLSTM + Pretrained Embeddings** | 0.1966 | 0.1610 | 0.2290 | 0.0680 | 25.13% |
| **DistilBERT (Fine-tuned, 3 epochs)** | 0.6454 | 0.5970 | 0.6874 | 0.0904 | 72.73% |

#### Paired Comparisons: DistilBERT vs. LR, DistilBERT vs. MLP, and MLP vs. LR
Evaluated via paired bootstrap differences (1,000 resamples, seed 42) and paired McNemar tests on the 374 test set samples (`scripts/classifier_ci.py`):

| Comparison | Metric | Diff (A − B) | 95% CI Lower | 95% CI Upper | Boot p-value | McNemar $\chi^2$ (df=1) | McNemar p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT vs. LR** | Macro-F1 | **+0.0495** | +0.0056 | +0.0931 | 0.0360 | — | — |
| **DistilBERT vs. LR** | Accuracy | **+0.0642** | +0.0160 | +0.1096 | 0.0220 | 6.6125 | 0.0101 |
| **DistilBERT vs. MLP** | Macro-F1 | +0.0273 | -0.0250 | +0.0901 | 0.2800 | — | — |
| **DistilBERT vs. MLP** | Accuracy | **+0.0749** | +0.0241 | +0.1257 | 0.0100 | 7.0096 | 0.0081 |
| **MLP vs. LR** | Macro-F1 | +0.0222 | -0.0220 | +0.0569 | 0.3480 | — | — |
| **MLP vs. LR** | Accuracy | -0.0107 | -0.0428 | +0.0214 | 0.5980 | 0.2368 | 0.6265 |

### Key Findings, Statistical Hypothesis Tests & Model Selection
- **LR and MLP are indistinguishable on test** across paired tests for both macro-F1 (diff +0.0222, 95% CI `[-0.0220, +0.0569]`, $p=0.3480$) and accuracy (diff -0.0107, 95% CI `[-0.0428, +0.0214]`, $p=0.5980$, McNemar $\chi^2 = 0.2368, p=0.6265$).
- **DistilBERT scores higher on test:**
  - vs. LR: macro-F1 +0.050 (+0.0495, 95% CI `[+0.0056, +0.0931]`, $p=0.0360$) and accuracy +6.4 pts (+0.0642, 95% CI `[+0.0160, +0.1096]`, $p=0.0220$, McNemar $p=0.0101$).
  - vs. MLP: accuracy +7.5 pts (+0.0749, 95% CI `[+0.0241, +0.1257]`, $p=0.0100$, McNemar $p=0.0081$), but its macro-F1 gain over MLP (+0.0273, 95% CI `[-0.0250, +0.0901]`, $p=0.2800$) is not statistically significant.
- **Caveats stated in the same place:**
  - Six paired hypothesis tests were evaluated with no multiplicity correction.
  - All results derive from a single test split of 374 resumes (~15 samples per class across 24 classes).
  - On validation, DistilBERT did NOT outperform (val macro-F1 0.5675 vs. MLP 0.6001), so validation and test disagree.
  - DistilBERT trained for only 3 epochs due to compute constraints; the BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).
- **Served Model Selection Rationale:**
  - The served classifier is the MLP because the selection rule (val macro-F1) was fixed before the test set was opened.
  - Choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.

---

## 2. Siamese Resume-JD Matcher (15 Target Categories)

> [!WARNING]
> **Weak Labels & Circularity:** Ground truth labels are weak, title-derived category pairings. The evaluation metric is partly circular, measuring separation between occupational clusters rather than human recruiter qualification.

| Model Architecture | Val MRR | Val nDCG@10 | Val Recall@5 | Val Recall@10 | Test MRR | Test nDCG@10 | Test Recall@5 | Test Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.6121 | 0.4537 | 0.2209 | 0.3239 | 0.6333 | 0.4854 | 0.2337 | 0.3456 |
| **Zero-Shot SBERT (all-MiniLM-L6-v2)** | 0.6377 | 0.4414 | 0.2107 | 0.3090 | 0.6311 | 0.4532 | 0.1964 | 0.3007 |
| **Fine-Tuned Siamese SBERT (MNRL)** | **0.8567** | **0.6821** | **0.3286** | **0.4671** | **0.7798** | **0.7286** | **0.3755** | **0.5338** |

*Takeaway:* Fine-tuned Siamese SBERT achieves **0.7798 Test MRR**, representing a **+14.87 point increase vs zero-shot** (0.7798 vs 0.6311) and **+14.65 points vs TF-IDF baseline** (0.7798 vs 0.6333). Note that labels are weak and title-derived, and the retrieval metric is partly circular. Broad domains such as BPO (test MRR 0.0326) and CONSULTANT (test MRR 0.2032) suffer from high cross-category overlap.

### Per-Category Reciprocal Rank (MRR)
*Evaluated across 15 target categories; labels are weak, title-derived and metrics are partly circular.*

| Category | Val N | Test N | TF-IDF Val | Zero-Shot Val | Fine-Tuned Val | TF-IDF Test | Zero-Shot Test | Fine-Tuned Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ACCOUNTANT | 17 | 18 | 0.8693 | 1.0000 | 1.0000 | 0.9008 | 0.9352 | 1.0000 |
| ADVOCATE | 17 | 18 | 0.1799 | 0.1808 | 0.7967 | 0.1696 | 0.3047 | 0.8926 |
| BANKING | 17 | 18 | 0.2214 | 0.2319 | 0.6308 | 0.4874 | 0.4869 | 0.6001 |
| BPO | 4 | 3 | 0.0333 | 0.3331 | 0.1429 | 0.0967 | 0.1241 | 0.0326 |
| BUSINESS-DEVELOPMENT | 17 | 18 | 0.6231 | 0.6814 | 0.9559 | 0.4045 | 0.1949 | 0.9630 |
| CHEF | 17 | 18 | 0.9069 | 0.9706 | 1.0000 | 0.6011 | 0.7900 | 0.7585 |
| CONSTRUCTION | 17 | 17 | 0.7420 | 0.7813 | 0.8703 | 0.8003 | 0.7739 | 0.8049 |
| CONSULTANT | 17 | 17 | 0.1282 | 0.4045 | 0.9496 | 0.0818 | 0.1779 | 0.2032 |
| DIGITAL-MEDIA | 15 | 14 | 0.7612 | 0.5835 | 0.6508 | 0.5239 | 0.5649 | 0.4648 |
| ENGINEERING | 17 | 17 | 0.5324 | 0.6816 | 0.8627 | 0.8140 | 0.8379 | 1.0000 |
| HEALTHCARE | 17 | 17 | 0.5543 | 0.5742 | 0.7095 | 0.7208 | 0.5267 | 0.8144 |
| HR | 16 | 18 | 0.7339 | 0.5947 | 0.9688 | 0.9099 | 0.7396 | 0.5926 |
| INFORMATION-TECHNOLOGY | 18 | 18 | 0.8588 | 0.6023 | 0.8421 | 0.8796 | 0.8333 | 0.9259 |
| SALES | 17 | 18 | 0.7819 | 0.8549 | 1.0000 | 0.8889 | 0.9722 | 1.0000 |
| TEACHER | 16 | 15 | 0.8359 | 0.8646 | 0.9104 | 0.7634 | 0.7801 | 0.9667 |

---

## 3. ATS Scoring Ablation & Explainability

Ablation of each of the 9 scoring components. Scores for unrelated pairs sit near 50 because formatting and contact components give every resume free points, and weights are transparent heuristics with no ground truth.

| Component | Default Weight | Mean Original Score | Mean Ablated Score | Mean Score Delta (Δ) |
| :--- | :---: | :---: | :---: | :---: |
| `semantic_match` | 0.25 | 79.64 | 76.11 | **-3.53** |
| `keyword_coverage` | 0.20 | 79.64 | 79.41 | **-0.23** |
| `section_completeness` | 0.10 | 79.64 | 77.56 | **-2.08** |
| `formatting` | 0.10 | 79.64 | 80.89 | **+1.25** |
| `experience_fit` | 0.10 | 79.64 | 77.38 | **-2.26** |
| `quantified_achievements` | 0.10 | 79.64 | 87.01 | **+7.37** |
| `action_verbs` | 0.05 | 79.64 | 79.21 | **-0.43** |
| `readability` | 0.05 | 79.64 | 79.27 | **-0.37** |
| `contact_completeness` | 0.05 | 79.64 | 79.36 | **-0.28** |

### Explainability Architecture & Faithfulness Tests
- **Methods:** Leave-one-sentence-out (LOSO) for sentence-level contribution and TF-IDF + Logistic Regression linear model coefficients for classifier word importance.
- **Sentence-Level Erasure (ATS Score):** Inconclusive (top-3 most important sentence removal dropped scores by 0.07 pts vs 0.11 pts for 3 random sentences across 20 validation resumes, k=3, seed 42). Composite ATS score explanations are inconclusive and unvalidated.
- **Classifier Word Deletion Test (Feature Attribution):** Evaluated across 50 validation resumes ($k=10$ words, seed 42, `scripts/eval_scoring.py`):
  - **Coefficients Picker:** Words are selected strictly by highest positive linear coefficients from the TF-IDF + Logistic Regression model (`clf_lr.coef_[pred_idx]`).
  - **Random Baseline:** 10 random words that occur in the document AND in the TF-IDF vocabulary, strictly excluding English stopwords (`ENGLISH_STOP_WORDS`).
  - **Variant 1 — LR Model (Self-Attribution):**
    - Top-10 LR words mean probability drop: **0.3590** (35.90 percentage points drop)
    - Random-10 non-stopwords mean probability drop: **-0.0021** (-0.21 percentage points drop)
    - Net attribution gap: **+0.3611** (36.11 percentage points degradation)
    - Verdict: **faithful to the linear model (verified by deletion)**
  - **Variant 2 — Served Classifier `classify()` [TF-IDF + 2-layer MLP] (Cross-Model Transfer):**
    - Top-10 LR words mean probability drop: **0.4921** (49.21 percentage points drop)
    - Random-10 non-stopwords mean probability drop: **0.0058** (0.58 percentage points drop)
    - Net attribution gap: **+0.4863** (48.63 percentage points degradation)
    - Verdict: **words picked by the LR also reduce the served model's probability (cross-model transfer)**.
- **Sub-Score Decomposition (UI Result Hero):** The composite ATS score is partitioned into two orthogonal sub-scores:
  - **Job fit** ($w=0.55$): weighted combination of `semantic_match` (0.25), `keyword_coverage` (0.20), and `experience_fit` (0.10) renormalized to 0–100.
  - **Resume quality** ($w=0.45$): weighted combination of the remaining 6 components (`section_completeness` 0.10, `formatting` 0.10, `quantified_achievements` 0.10, `action_verbs` 0.05, `readability` 0.05, `contact_completeness` 0.05) renormalized to 0–100.
  - Invariant: $0.55 \cdot \text{Job fit} + 0.45 \cdot \text{Resume quality} \equiv \text{Total ATS Score}$.
- **Scoring Latency:** Mean 46.05 ms, Median 43.36 ms, P95 57.50 ms.

---

## 4. Multi-Format Parsing & Extraction Yield

- **Parsing Precision/Recall:** 100% on 5 synthetic resumes (and 0 JDs in `parsing_eval.json`: `ds_clean.pdf`, `swe_two_column.pdf`, `accountant_table.pdf`, `chef_plain.docx`, `nurse_scanned.pdf`).
- **Extraction Micro F1:** 0.8980 with 0 trap violations on 5 synthetic resumes and 3 JDs. NER fine-tuning was not implemented; extraction is taxonomy-based with 36 disambiguation rules.
- **Real-Text Extraction Yield:** 95.95% experience parsing yield, 74.86% education parsing yield on collapsed val text (370 raw resumes); mean extraction latency 4.45 ms.

---

## 5. Parser & Extraction Robustness CLI (`scripts/eval_robustness.py`)

A privacy-safe evaluation utility for real-world document sets (.pdf and .docx; default folder `data/real_samples/`, gitignored). Outputs per-file metadata only: pages, parse method, seconds, sections found, skills count, years experience, and warnings count. Strictly zero candidate text, names, emails, or phone numbers are logged.

### Fixture Suite Benchmark (5 documents: 4 PDF, 1 DOCX)
- **Processing Success Rate:** 5/5 (100.0%)
- **OCR Fallback Invocations:** 1 (`nurse_scanned.pdf`, 2.66s via Tesseract)
- **Native Document Speeds:** Native PDF: 0.085s–0.240s; Native DOCX: 0.011s
- **Average Skills Extracted:** 5.2 skills per document
- **Average Experience Detected:** 9.2 years
- **Total Warnings Recorded:** 1 (OCR fallback notification)
