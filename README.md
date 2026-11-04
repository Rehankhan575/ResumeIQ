# AI-Powered Resume Screening & Job Fit Analytics Platform

An end-to-end, privacy-preserving deep learning pipeline and interactive dashboard for resume parsing, skill extraction, industry classification, semantic job matching, transparent ATS scoring, perturbation-based explainability, and targeted suggestions.

---

## Architecture Overview

```mermaid
flowchart LR
    A["Stage 1: Multi-Format Parsing (PDF / DOCX / Scanned OCR)"] --> B["Stage 2: Taxonomy & Rule-Based NER Extraction"]
    B --> C["Stage 3: Resume Classification (DistilBERT / LR / MLP / BiLSTM)"]
    B --> D["Stage 4: Siamese Semantic Matcher (Fine-Tuned SBERT)"]
    C --> E["Stage 5: Transparent ATS Scoring (9-Component Weighted Engine)"]
    D --> E
    E --> F["Stage 6: Perturbation & Counterfactual Explainability (XAI)"]
    E --> G["Stage 7: Generative Suggestions (LLM + Rule Fallback)"]
```

The system executes a strictly sequential 7-stage pipeline orchestrated by `app.pipeline.run_pipeline`:
1. **Multi-Format Parsing (`app.parsing`)**: Extracts raw text, detects document structure (multi-column, tables, scanned images), and extracts contact info using pdfplumber, python-docx, and Tesseract OCR fallback.
2. **Entity & Skill Extraction (`app.extraction`)**: Maps skills against a curated taxonomy of 1,668 skills (488 with aliases, 2,404 total canonical and alias terms, 833 active skills in `skills_stats.csv`) with longest-match-first token scanning and 36 ambiguous term disambiguation rules in `ambiguous_terms.csv`. Extracts education and date-normalized experience spans. (NER fine-tuning was not implemented).
3. **Resume Classification (`app.classifier`)**: Classifies candidate resumes into 24 industry categories using TF-IDF + MLP (selected by validation F1), with DistilBERT and fast Logistic Regression available.
4. **Semantic Resume-JD Matching (`app.matcher`)**: Encodes resumes and job descriptions into dense embeddings via fine-tuned Siamese SBERT (`sentence-transformers/all-MiniLM-L6-v2` trained with MultipleNegativesRankingLoss).
5. **Transparent ATS Scoring (`app.scoring`)**: Computes a deterministic 0–100 score across 9 weighted dimensions: Semantic Match (25%), Keyword Coverage (20%), Section Completeness (10%), Experience Fit (10%), Formatting (10%), Quantified Achievements (10%), Action Verbs (5%), Readability (5%), and Contact Info (5%). Unrelated pairs sit near ~50 because formatting and contact award free points.
6. **Sentence & Word Explainability (`app.explainability`)**: Uses leave-one-sentence-out (LOSO) for sentence importance and TF-IDF + Logistic Regression coefficients for classifier word importance.
7. **Targeted Suggestions (`app.suggestions`)**: Generates actionable, ATS-improving suggestions using Google Gemini / Anthropic / OpenAI APIs when configured, falling back seamlessly to deterministic rule-based advice offline.

---

## Syllabus Unit Mapping

| Deep Learning Curriculum Unit | Implemented System Modules | Technical Concepts & Artifacts |
| :--- | :--- | :--- |
| **Unit I: Foundations of Neural Networks** | `app/classifier/tfidf_mlp.py`, `app/classifier/tfidf_lr.py` | Feedforward neural architectures, cross-entropy loss, Adam optimization, regularization, learning curves, train-val-test splitting. |
| **Unit II: Deep Representations & Metric Learning** | `app/matcher/`, `app/explainability/` | Siamese networks, cosine similarity spaces, MultipleNegativesRankingLoss (MNRL), contrastive representations, perturbation-based feature attribution. |
| **Unit III: Recurrent Neural Architectures** | `app/classifier/bilstm.py` | Bidirectional LSTMs, sequence tokenization, padded mini-batching, out-of-vocabulary handling, hidden-state pooling, underfitting analysis. |
| **Unit IV: Transformers & Pretrained Models** | `app/classifier/distilbert.py`, `app/matcher/model.py` | Self-attention mechanisms, transfer learning, Hugging Face `transformers` fine-tuning, linear warmup learning rate schedules, subword tokenization. |
| **Unit V: Generative AI & Explainability** | `app/suggestions/`, `app/explainability/`, `app/ui/` | Prompt engineering, multi-turn LLM generation, grounded hallucination guards, counterfactual explanation, model interpretability. |

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or 3.11 (tested on Python 3.11.15)
- macOS, Linux, or Windows WSL2
- Optional OCR Engine (for scanned image PDFs):
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

### 2. Virtual Environment Setup
```bash
# Clone or navigate to the repository
cd /path/to/DL_PROJECT

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional LLM API Configuration
The pipeline is 100% functional locally and offline. If you wish to enable GenAI suggestions via cloud providers, configure any of the following environment variables in a `.env` file:
```bash
# Supported providers (pipeline detects whichever is set, prioritizing Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```
When no API key is provided, the platform automatically activates deterministic, rule-based suggestion engines.

### 4. Offline Verification
To run strictly offline without network access to Hugging Face Hub:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## How to Run

### Interactive Streamlit Dashboard
Launch the custom-styled dark theme analytics dashboard:
```bash
streamlit run app/ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Preflight Sanity Check
Verify environment packages, hardware acceleration, taxonomy paths, model checkpoints, and offline operation:
```bash
python scripts/preflight.py
```

### Collect Metrics and Reports
Regenerate all markdown and CSV metric tables from stored evaluation files:
```bash
python scripts/collect_results.py
```

### Run Unit & Integration Test Suite
```bash
pytest
```

---

## Empirical Benchmark Results

All metrics below are drawn directly from saved artifacts in `artifacts/metrics/` and aggregated via `scripts/collect_results.py`.

### 1. Classifier Model Comparison (24 Classes)
Evaluated on 2,483 stratified resumes across 24 job categories (per-class resume counts range from 22 in BPO to 120 in INFORMATION-TECHNOLOGY; Train: 1,739, Val: 370, Test: 374).

| Architecture | Train Acc | Train Macro F1 | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Gen Gap (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 99.02% | 0.9882 | 64.05% | 0.5913 | 66.31% | 0.5959 | 0.3969 |
| **TF-IDF + MLP (2-layer)** | 100.00% | 1.0000 | 64.59% | 0.6001 | 65.24% | 0.6181 | 0.3999 |
| **BiLSTM + Pretrained Embeddings** | 74.30% | 0.7091 | 23.51% | 0.1860 | 25.13% | 0.1966 | 0.5231 |
| **DistilBERT (Fine-tuned, 3 epochs)** | 75.85% | 0.6748 | 65.14% | 0.5675 | 72.73% | 0.6454 | **0.1073** |

*Takeaways & Bootstrap 95% Confidence Intervals:*
- **Validation Macro-F1:** Logistic Regression: 0.5913, MLP: 0.6001, DistilBERT: 0.5675. Differences are within noise on about 370 validation resumes (~15 resumes per class across 24 classes).
- **Test Macro-F1 with 95% Bootstrap CIs (1,000 resamples, seed 42):**
  - **TF-IDF + LR:** 0.5959 (95% CI: `[0.5465, 0.6352]`, test accuracy: 66.31%)
  - **TF-IDF + MLP:** 0.6181 (95% CI: `[0.5606, 0.6587]`, test accuracy: 65.24%)
  - **BiLSTM:** 0.1966 (95% CI: `[0.1610, 0.2290]`, test accuracy: 25.13%)
  - **DistilBERT:** 0.6454 (95% CI: `[0.5970, 0.6874]`, test accuracy: 72.73%)
- **Paired Hypothesis Testing (MLP vs. LR):** Point difference +0.0222 macro-F1, bootstrap 95% CI `[-0.0220, +0.0569]` (spans zero, bootstrap $p = 0.3480$). Paired McNemar test on discordant sample predictions: continuity-corrected $\chi^2 = 0.2368$ (df=1), $p = 0.6265$. The difference is statistically indistinguishable from zero.
- The production model was selected by validation F1 (MLP at 0.6001). DistilBERT achieved the best test numbers but was not selected on test metrics and was trained for only 3 epochs due to compute budgets. BiLSTM underfits significantly at this sample size.

### 2. Semantic Matcher Retrieval Performance (15 Categories)

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

### 3. ATS Scoring Ablation Study & Explainability
Ablation of each of the 9 scoring components (tested across candidate-job pairs). Scores for unrelated pairs sit near ~50 because formatting and contact components award free points, and weights are transparent heuristics with no ground truth.

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

### XAI Explainability & Faithfulness Tests
- **Explainer Architecture:** Leave-one-sentence-out (LOSO) for sentence-level contribution and TF-IDF + Logistic Regression linear model coefficients for classifier word importance.
- **Faithfulness Benchmark (Sentence Erasure on ATS Total):** Inconclusive (top-3 sentence removal caused a mean drop of 0.07 pts vs 3 random sentence removal drop of 0.11 pts across 20 validation resumes, k=3, seed 42). Explanations for the composite ATS score are not experimentally validated.
- **Classifier Word Deletion Test (Feature Importance):** Evaluated across 50 validation resumes (k=10 words, seed 42): removing the top-10 positive LR words dropped the predicted class probability by **0.3590** (35.90 percentage points), whereas deleting 10 random words produced a drop of **-0.0021** (-0.21 percentage points), establishing a net attribution gap of **+0.3611**. Linear coefficient feature importance is verified to drive classifier predictions directly.
- **Sub-Score Decomposition (UI Result Hero):** The UI presents two sub-scores beside the total:
  - **Job fit** ($w=0.55$): `semantic_match` (0.25), `keyword_coverage` (0.20), `experience_fit` (0.10) renormalized to 0–100.
  - **Resume quality** ($w=0.45$): `section_completeness` (0.10), `formatting` (0.10), `quantified_achievements` (0.10), `action_verbs` (0.05), `readability` (0.05), `contact_completeness` (0.05) renormalized to 0–100.
  - Invariant: $0.55 \cdot \text{Job fit} + 0.45 \cdot \text{Resume quality} \equiv \text{Total ATS Score}$.
- **Scoring Latency:** Mean 46.05 ms, Median 43.36 ms, P95 57.50 ms.

### 4. Parsing & Extraction Accuracy
- **Parsing Section Accuracy:** 100.0% precision and 100.0% recall on 5 synthetic resumes (and 0 JDs in `parsing_eval.json`: `ds_clean.pdf`, `swe_two_column.pdf`, `accountant_table.pdf`, `chef_plain.docx`, `nurse_scanned.pdf`).
- **Extraction Micro F1:** 0.8980 (Micro Precision: 81.48%, Micro Recall: 100.00%) with 0 trap violations on 5 synthetic resumes and 3 JDs. NER fine-tuning was not implemented; extraction is taxonomy-based with 36 disambiguation rules.
- **Real-Text Extraction Yield:** 95.95% experience section parsing yield, 74.86% education section parsing yield on collapsed val text (370 resumes); mean extraction latency 4.45 ms.
- **Document Robustness Evaluation:** `scripts/eval_robustness.py` provides a privacy-safe audit CLI for real resume collections (.pdf/.docx), logging per-file metadata without candidate PII (100% processed successfully across fixture test suite with 1 OCR fallback on `nurse_scanned.pdf`).

---

## Limitations

We document the real constraints and trade-offs of this platform plainly:
1. **Resume NER Fine-Tuning Not Implemented:** Token-level entity extraction relies on a curated skill taxonomy of 1,668 skills (488 with aliases, 2,404 total terms, 833 active skills) with 36 context disambiguation rules in `ambiguous_terms.csv` rather than a custom token-classification model (e.g., token-level BERT-NER).
2. **Scope Boundaries:** Interview-question generation, FastAPI REST deployment, Docker packaging, and cloud hosting are deliberately out of scope; the application is designed to operate locally.
3. **Classifier Selection & Data Noise:** On validation macro-F1, Logistic Regression (0.5913), MLP (0.6001), and DistilBERT (0.5675) are statistically close on about 370 validation resumes (~15 per class across 24 classes). The production model was selected by validation F1 (MLP). DistilBERT achieved the highest test macro-F1 (0.6454) and test accuracy (72.73%), but was not selected on test metrics and was trained for only 3 epochs due to compute constraints. BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy) at this data volume. Paired bootstrap difference (95% CI `[-0.0220, +0.0569]`, $p=0.3480$) and McNemar test ($p=0.6265$) confirm MLP and LR test differences are within statistical noise.
4. **Matcher Weak Labels & Circularity:** Matcher ground truth uses title-derived category matching; labels are weak and the retrieval evaluation metric is partly circular. While fine-tuned SBERT achieved 0.7798 test MRR (+14.87 points vs zero-shot, +14.65 points vs TF-IDF), broad categories (BPO test MRR 0.0326, CONSULTANT test MRR 0.2032) perform poorly. For categories with >10 true positives, Recall@5 is mathematically capped below 50%; MRR is the primary evaluation metric.
5. **Heuristic ATS Baseline & Free Points:** No objective universal ATS ground-truth score exists in industry. Weights are transparent heuristics. Scores for unrelated resume-job pairs sit near 50 (e.g., Data Scientist vs Executive Chef scores 55.5) because formatting (10%) and contact information (5%) award every resume free points regardless of job fit.
6. **XAI Distinction & ATS Unvalidated:** The explainer uses leave-one-sentence-out for sentence importance and TF-IDF+LR coefficients for word importance. While the classifier word-deletion test demonstrated a validated +0.3611 net probability gap for category attribution, the composite ATS sentence erasure test was inconclusive (top-3 sentence drop 0.07 pts vs random-3 sentence drop 0.11 pts on 20 validation resumes, k=3), and ATS score sentence explanations are not experimentally validated.
7. **Optional GenAI Dependency:** Personalized rewriting suggestions depend on third-party cloud APIs (Gemini/Claude/OpenAI). When no key is provided, the platform gracefully switches to deterministic rule-based advice.
8. **OCR Quality Dependency:** Tesseract OCR performance depends directly on image DPI and scan resolution. Scanned documents with non-standard fonts or low contrast may experience minor word corruption.
9. **Single Stratified Split:** Models were evaluated on a deterministic 70/15/15 train/val/test split; k-fold cross-validation was not run due to training budget constraints.

---

## Privacy & Safety Commitments
- **Zero Raw PII Logging:** Candidate names, phone numbers, email addresses, and full resume texts are never logged to console, disk logs, or error telemetry.
- **Local Isolation:** Resumes are processed in volatile memory. No documents are uploaded to third-party endpoints unless the user explicitly enables GenAI suggestions.
- **Air-Gapped Operation:** The platform runs fully functional with `HF_HUB_OFFLINE=1` using locally cached model weights.

