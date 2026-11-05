# ResumeIQ: Transparent, Explainable & Privacy-Preserving ATS Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18_Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/Tailwind-CSS_3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Privacy-100%25_In--Memory-10B981?style=for-the-badge&logo=shield&logoColor=white" alt="Privacy First" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
</p>

---

## 📌 Executive Summary

Over **75% of job seekers** are eliminated by commercial Applicant Tracking Systems (ATS) before a human recruiter ever reviews their application. However, prevailing ATS software functions as an opaque black box:
1. They rely on brittle, literal keyword matching that rewards deceptive keyword stuffing while penalizing conceptually qualified applicants.
2. They provide zero actionable feedback or score interpretability.
3. They inadvertently perpetuate unconscious demographic discrimination (ageism, gender bias, and marital profiling).

**ResumeIQ** is an end-to-end, production-grade ATS optimization and diagnostic platform powered by deep learning. It replaces arbitrary single-percentage scores with an interpretable **9-Rubric Scoring Engine**, provides **sub-10ms deterministic token-level explainability**, executes **automated bias audits**, and generates **grounded, active-power bullet rewrites**—all running 100% in-memory with strict zero-data-retention guarantees.

---

## 📸 Application Screenshots

### 1. ATS Evaluation & 9-Rubric Decomposition
> Comprehensive diagnostic scorecard featuring the composite ATS grade, weighted rubric breakdown, prioritized point-gain action items, and real-time category inference.

![ResumeIQ Overview](docs/screenshots/react_analyze_84_9.png)

---

### 2. Token-Level Explainability Overlay
> Reader-style token inspection snapped strictly to whole-word boundaries. Visualizes exact keyword hits (yellow), high-weight transformer attributions (green), and missing core qualifications.

![Explainability Overlay](docs/screenshots/react_highlight_fixed.png)

---

### 3. Grounded AI Rewrites & Actionable Recommendations
> Side-by-side Before/After diff cards transforming passive candidate statements into quantified, active-power statements guarded against LLM hallucinations.

![AI Rewrites](docs/screenshots/react_suggestions_enhanced.png)

---

### 4. Deep Learning Model Benchmarks & Comparison
> Side-by-side empirical benchmark comparing fine-tuned DistilBERT against baseline Linear and Multi-Layer Perceptron architectures across inference latency and Macro-F1.

![Model Results](docs/screenshots/react_model_results.png)

---

### 5. Multi-Job Fit Ranking & Analytics
> Cross-evaluates a single resume across multiple job requisitions, ranking opportunities by semantic alignment and highlighting critical competency gaps.

![Multi Job Ranking](docs/screenshots/react_multi_rank.png)

---

### 6. Automated Ethical Bias Auditor
> Proactively audits resumes against Title VII and EEOC protected attributes (DOB, marital status, graduation dates, gendered phrasing).

![Bias Check](docs/screenshots/react_bias_clean.png)

---

## 🚀 Key Features

* **Dual-Track Matching Architecture:** Combines dense contextual semantics via fine-tuned Siamese Sentence-Transformers (`all-MiniLM-L6-v2`) with sparse domain skill extraction across a curated 1,668-competency taxonomy.
* **Fine-Tuned 24-Class Classifier:** DistilBERT transformer fine-tuned over 2,484 real-world resumes spanning 24 occupational categories, delivering sub-50ms CPU inference.
* **Deterministic 9-Rubric Scoring Engine:** Computes an interpretable 100-point score with mathematical point-gain transparency across semantic fit, keyword density, quantified metrics, action verbs, and formatting.
* **Fast Sub-10ms Explainability (XAI):** Implements Leave-One-Sentence-Out (LOSO) batch perturbation and linear surrogate coefficient attribution—deliberately replacing slow, stochastic SHAP/LIME with deterministic, instant token highlights snapped to whole-word boundaries.
* **Demographic Bias Auditor:** Scans for five categories of demographic trap indicators (age, gender phrasing, marital status, graduation year proxies) to safeguard candidates from algorithmic bias.
* **Fact-Grounded Bullet Rewriter:** Employs action verb taxonomy substitution and metric placeholder generation to boost bullet impact without hallucinating non-existent candidate achievements.
* **Zero-Retention Privacy Guarantee:** 100% in-memory processing. Resumes are parsed, scored, and discarded from memory without writing candidate data to disk, databases, or third-party storage.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["Stage 1: Multi-Format Ingestion (In-Memory)"]
        A["Candidate Resume (PDF / DOCX)"] --> B["Document Parser (PyMuPDF / pdfplumber)"]
        B --> C["PII & Layout Normalizer"]
    end

    subgraph DUAL_NLP ["Dual-Track Deep Learning & NLP"]
        C --> D["Taxonomy Matcher (1,668 Skills + Aliases)"]
        C --> E["Domain Classifier (Fine-Tuned DistilBERT)"]
        C --> F["Dense Embedding Engine (SBERT all-MiniLM-L6-v2)"]
        JD["Job Description (Text)"] --> F
        JD --> D
    end

    subgraph SCORING ["Stage 5: Deterministic 9-Rubric Engine"]
        D --> G["Keyword Coverage (20 pts)"]
        F --> H["Semantic Cosine Fit (25 pts)"]
        C --> I["Section Completeness (10 pts)"]
        C --> J["Formatting & Structure (10 pts)"]
        C --> K["Experience Duration (10 pts)"]
        C --> L["Quantified Metrics (10 pts)"]
        C --> M["Action Power Verbs (5 pts)"]
        C --> N["Readability & Flow (5 pts)"]
        C --> O["Contact Completeness (5 pts)"]
        G & H & I & J & K & L & M & N & O --> P["Composite ATS Score (0 - 100)"]
    end

    subgraph AUDIT_XAI ["Explainability & Ethical Auditing"]
        P --> Q["LOSO Sentence Perturbation (Sub-10ms)"]
        P --> R["Linear Coefficient Token Highlights"]
        C --> S["Ethical Bias Auditor (EEOC Demographic Traps)"]
    end

    subgraph GENERATION ["Stage 7: Grounded Optimization"]
        P --> T["Prioritized Action Items"]
        C --> U["Active-Power Bullet Rewriter"]
    end

    subgraph UI ["Interactive Dark-Glass Dashboard"]
        P & Q & R & S & T & U --> V["React 18 + Vite + Tailwind CSS SPA"]
    end
```

---

## 📊 9-Rubric ATS Decomposition Formula

The overall ATS evaluation is computed as a deterministic weighted linear combination bounded in $[0, 100]$:

$$\text{ATS Score} = \sum_{i=1}^{9} w_i \times S_i$$

| Rubric Component | Weight ($w_i$) | Max Points | Evaluation Methodology |
| :--- | :---: | :---: | :--- |
| **Semantic Match** | 25% | 25.0 | Cosine similarity between SBERT `all-MiniLM-L6-v2` 384-d embeddings |
| **Keyword Coverage** | 20% | 20.0 | Exact and alias matches of required JD skills from curated taxonomy |
| **Section Completeness**| 10% | 10.0 | Presence of core sections (Experience, Education, Skills, Summary) |
| **Formatting & Layout** | 10% | 10.0 | Page count budget (1–2 pgs), contact placement, standard fonts |
| **Experience Fit** | 10% | 10.0 | Normalized career duration alignment against target job seniority |
| **Quantified Metrics** | 10% | 10.0 | Percentage of bullets containing empirical KPIs (%, \$, metrics) |
| **Action Verbs** | 5% | 5.0 | Ratio of experience bullets initiating with high-impact power verbs |
| **Readability & Flow** | 5% | 5.0 | Flesch-Kincaid / Automated Readability Index target (Grade 9–12) |
| **Contact Details** | 5% | 5.0 | Presence of Email, Phone Number, and professional links (LinkedIn/GitHub) |

---

## 🔬 Dataset & Preprocessing

* **Primary Corpus:** LiveCareer Resume Corpus containing **2,484 resumes** across **24 industry categories** (Information Technology, Engineering, Finance, Healthcare, Sales, etc.).
* **Job Description Corpus:** Over 2,000 real-world tech and corporate job postings scraped from Kaggle, standardized for skill extraction.
* **Data Splitting:** Stratified **80% Training (1,987 resumes)**, **10% Validation (248 resumes)**, and **10% Test (249 resumes)** to ensure zero category imbalance or evaluation leakage.
* **Preprocessing Pipeline:**
  * Regex-based PII scrubbing (anonymizing phone, email, and candidate identity during model training).
  * Header leakage elimination (removing explicit category banners to prevent shortcut learning).
  * Longest-match-first token scanning with 36 domain-specific disambiguation rules (e.g., distinguishing *Go* programming language from common verb *go*).

---

## 🧠 Deep Learning Models & Empirical Benchmarks

### Resume Category Classifier (24 Occupational Classes)

| Model Architecture | Features / Embeddings | Test Macro-F1 | Test Accuracy | CPU Latency | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | TF-IDF (10,000 max n-grams) | 0.5841 | 63.85% | **0.8 ms** | 240K |
| **TF-IDF + Multi-Layer Perceptron (MLP)** | TF-IDF (2-Layer: 512, 128, ReLU, Dropout) | 0.6120 | 66.26% | 3.2 ms | 5.2M |
| **Bidirectional LSTM (BiLSTM)** | FastText 300-d Word Embeddings | 0.4912 | 54.21% | 28.5 ms | 1.8M |
| **Fine-Tuned DistilBERT** *(Selected)* | Pretrained `distilbert-base-uncased` | **0.6240** | **67.87%** | **42.1 ms** | 66.3M |

> **Design Choice:** DistilBERT was selected for domain classification due to its bidirectional contextual attention over ambiguous multi-discipline technical terminology, while retaining a sub-50ms CPU execution time.

### Semantic Matching Engine
* **Model:** Siamese Sentence-Transformers fine-tuned on sentence pairs using `MultipleNegativesRankingLoss`.
* **Base Architecture:** `sentence-transformers/all-MiniLM-L6-v2`.
* **Embedding Dimension:** 384 dimensions.
* **Vector Metric:** Unit-normalized Cosine Similarity:
  $$\text{Sim}(R, JD) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$

---

## 🔍 Explainability Without SHAP or LIME

Standard XAI libraries like SHAP (KernelSHAP) and LIME are ill-suited for real-time ATS screening: they require hundreds of stochastic forward passes (taking 5–30+ seconds per resume) and introduce random sampling variance.

ResumeIQ implements two deterministic, sub-10ms explainability algorithms:

1. **Leave-One-Sentence-Out (LOSO) Perturbation:**  
   Computes the delta contribution of each grammatical sentence by vectorizing the resume into a single batch matrix where sentence $s_i$ is omitted:
   $$\Delta_i = \text{Cosine}(\vec{V}_{\text{full}}, \vec{V}_{\text{JD}}) - \text{Cosine}(\vec{V}_{\text{full} \setminus \{s_i\}}, \vec{V}_{\text{JD}})$$
   Executes in **$< 10$ milliseconds** on CPU with zero stochastic noise.

2. **Linear Surrogate Feature Attribution:**  
   Attributes category classification decisions using exact trained linear model coefficients:
   $$\text{Attribution}(w) = \text{TF-IDF}(w) \times \beta_{c, w}$$
   * **Faithfulness Deletion Test:** In our 50-resume validation benchmark ($k=10$ words), removing the top-10 attributed words caused a **49.21 percentage point drop** in model classification confidence, versus only a **0.58 percentage point drop** for random words (Net Attribution Gap: **$+0.4863$**).
   * **Boundary Snapping:** Word offsets automatically expand via boundary snapping (`\b`) to guarantee highlights never split tokens mid-character.

---

## 🛡️ Ethical Bias & Anti-Discrimination Auditor

ResumeIQ implements an active compliance scanner auditing against EEOC and Title VII prohibited hiring criteria:

* **Age Indicators:** Flags direct dates of birth (`DOB: DD/MM/YYYY`) and graduation year proxies older than 10 years that invite ageist screening.
* **Marital Status:** Detects marital flags (`Single`, `Married`, `Divorced`).
* **Gender Phrasing & Pronouns:** Identifies gendered honorifics (`she/her`, `he/him`, `Chairman`, `Salesman`) and recommends gender-neutral alternatives.
* **Nationality & Religion:** Flags native language/nationality declarations that have no bearing on job competence.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & Core Engine** | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| **Deep Learning & NLP** | PyTorch, Hugging Face Transformers, Sentence-Transformers, Scikit-learn, SpaCy, NumPy |
| **Document Parsing** | pdfplumber, PyMuPDF (fitz), python-docx, Tesseract OCR |
| **Frontend SPA** | React 18, TypeScript, Vite, Tailwind CSS, Lucide React, Canvas Confetti |
| **Testing & CI** | Pytest, Pytest-Asyncio, Vitest, React Testing Library |

---

## 📦 Installation & Setup

### 1. Prerequisites
* Python 3.10 or 3.11
* Node.js 18+ and npm
* Optional: Tesseract OCR (for scanned document fallback)
  * **macOS:** `brew install tesseract`
  * **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Rehankhan575/ResumeIQ.git
cd ResumeIQ

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup & Build Frontend
```bash
# Install frontend packages
npm --prefix frontend install

# Build production assets
npm --prefix frontend run build
```

### 4. Run the Platform

#### Unified Production Server (Recommended)
Launches the FastAPI backend serving both the REST API and the compiled React production application:
```bash
./scripts/run_demo.sh
```
* **Dashboard:** [http://localhost:8000](http://localhost:8000)
* **Interactive OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

#### Developer Mode (Hot-Reloading)
Run backend and frontend independently:
```bash
# Terminal 1: FastAPI Backend
./venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Vite Dev Server
npm --prefix frontend run dev
```

---

## 🧪 Testing & Verification

Run the full dual-ecosystem test suite:

```bash
# Run Backend Python Tests (24 unit & integration tests)
./venv/bin/python -m pytest tests/test_api.py tests/test_scoring.py

# Run Frontend React Tests (16 UI & XSS verification tests)
npm --prefix frontend run test -- --run
```

---

## 📂 Project Structure

```text
ResumeIQ/
├── app/                           # Core Backend Application
│   ├── api/                       # FastAPI endpoints, routers, and CORS
│   │   └── main.py                # REST API routes (/api/analyze, /api/rank, /api/health)
│   ├── extraction/                # Skill & entity extraction
│   │   ├── taxonomy.py            # 1,668-skill longest-match scanner
│   │   └── cleaner.py             # Ambiguous term disambiguation
│   ├── models/                    # ML / DL Model Inferences
│   │   ├── classifier.py          # DistilBERT & MLP category inference
│   │   └── matcher.py             # SBERT dense semantic embeddings
│   ├── parsing/                   # In-memory document ingestion
│   │   ├── parser.py              # PDFMiner / PyMuPDF / docx loaders
│   │   └── ocr.py                 # Tesseract OCR fallback
│   ├── scoring/                   # 9-Rubric Scoring & Explainability
│   │   ├── engine.py              # Deterministic weighted scoring calculator
│   │   ├── explain.py             # LOSO perturbation & linear attribution
│   │   └── bias.py                # Demographic & Title VII bias auditor
│   ├── genai/                     # Bullet optimization & rewrites
│   │   └── generator.py           # Grounded power statement generation
│   ├── pipeline.py                # 7-stage master orchestration pipeline
│   └── schemas.py                 # Pydantic v2 domain schemas
├── frontend/                      # React 18 + Vite Web Application
│   ├── src/
│   │   ├── components/            # UI Components (ScoreRing, Rubric, Highlights, Suggestions)
│   │   ├── services/api.ts        # Axios API client
│   │   ├── types/api.ts           # TypeScript interfaces matching backend
│   │   └── App.tsx                # Master Dashboard SPA shell
│   ├── package.json               # Node dependencies
│   └── vite.config.ts             # Vite build & proxy configuration
├── artifacts/                     # Evaluation metrics, confusion matrices & curves
├── docs/                          # Architecture documentation & screenshots
│   └── screenshots/               # High-resolution dashboard captures
├── tests/                         # End-to-end integration & unit test suite
│   ├── test_api.py                # FastAPI endpoint tests
│   ├── test_scoring.py            # Scoring & explainability tests
│   └── samples/                   # Sample resumes (clean & bias-trap test cases)
├── scripts/                       # Startup & automation scripts
│   └── run_demo.sh                # Single-command launcher
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🔒 Privacy & Compliance

ResumeIQ was engineered from the ground up to respect candidate data privacy:
1. **100% In-Memory Processing:** Resumes are processed strictly in RAM buffers and garbage collected immediately after scoring.
2. **Zero Database Retention:** No candidate resumes, names, contact numbers, or parsed histories are ever stored in a database or logged to disk.
3. **Local Storage Opt-In:** Analysis history in the frontend is stored strictly inside the candidate's browser `localStorage` and never contains raw resume text.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
