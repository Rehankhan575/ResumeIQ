# End-to-End Pipeline Evaluation & Empirical Results

This document presents the complete empirical benchmark across all 7 stages of the Resume Screening and Job Fit platform, compiled directly from stored metrics in `artifacts/metrics/`.

---

## 1. Resume Category Classifier (24 Classes)

Evaluated on 2,483 stratified resumes (Train: 1,739, Val: 370, Test: 374):

| Architecture | Train Acc | Train Macro F1 | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Gen Gap (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 99.02% | 0.9882 | 64.05% | 0.5913 | 66.31% | 0.5959 | 0.3969 |
| **TF-IDF + MLP (2-layer)** | 100.00% | 1.0000 | 64.59% | 0.6001 | 65.24% | 0.6181 | 0.3999 |
| **BiLSTM + Pretrained Embeddings** | 74.30% | 0.7091 | 23.51% | 0.1860 | 25.13% | 0.1966 | 0.5231 |
| **DistilBERT (Fine-tuned, 3 epochs)** | 75.85% | 0.6748 | 65.14% | 0.5675 | 72.73% | 0.6454 | **0.1073** |

---

## 2. Siamese Resume-JD Matcher (15 Target Categories)

Evaluated against gold category pairings:

| Model Architecture | Val MRR | Val nDCG@10 | Val Recall@5 | Val Recall@10 | Test MRR | Test nDCG@10 | Test Recall@5 | Test Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.6121 | 0.4537 | 0.2209 | 0.3239 | 0.6333 | 0.4854 | 0.2337 | 0.3456 |
| **Zero-Shot SBERT (all-MiniLM-L6-v2)** | 0.6377 | 0.4414 | 0.2107 | 0.3090 | 0.6311 | 0.4532 | 0.1964 | 0.3007 |
| **Fine-Tuned Siamese SBERT (MNRL)** | **0.8567** | **0.6821** | **0.3286** | **0.4671** | **0.7798** | **0.7286** | **0.3755** | **0.5338** |

### Per-Category Reciprocal Rank (MRR)
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

## 3. ATS Scoring Ablation & XAI

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

### XAI Latency & Faithfulness
- **Scoring Latency:** Mean 55.63 ms, Median 47.45 ms, P95 73.42 ms.
- **Top-k Keyword Drop:** 0.07 vs Random Word Drop 0.11 (`is_faithful: False` due to high document length vs small perturbation size).

---

## 4. Multi-Format Parsing & Extraction Yield

- **Parsing Precision/Recall:** 100% on 6 evaluation fixtures (`ds_clean.pdf`, `swe_two_column.pdf`, `accountant_table.pdf`, `chef_plain.docx`, `hr_docx.docx`, `nurse_scanned.pdf`).
- **Extraction Micro F1:** 0.8980 with 0 trap violations.
- **Real-Text Extraction Yield:** 95.95% experience parsing yield, 74.86% education parsing yield on 370 raw resumes.
