# AI Resume Analyzer & ATS Scorer — Project Context

## Goal
Build a full deep-learning pipeline that ingests a résumé (PDF/DOCX/image) and an optional
Job Description, then outputs: a parsed structured representation, a category label, a
semantic JD-match score, a composite ATS score with explanations, and generative improvement
suggestions. The system must be reproducible, explainable, and deployable on CPU/MPS (no
mandatory GPU). It is a college Deep-Learning project; every component must be experimentally
validated (NER F1, classifier accuracy/F1, ranking MRR/nDCG, baseline vs. SBERT comparison).

---

## Hardware & Runtime
| Key | Value |
|-----|-------|
| Machine | MacBook Air M4 |
| Chip | Apple Silicon M4 |
| OS | macOS |
| Python | 3.11.15 (venv at `venv/`) |
| Compute | CPU + MPS (Apple GPU via PyTorch MPS backend) |
| CUDA | Not available |
| Tesseract | **Not installed** — install with `brew install tesseract` before OCR features |

Device selection order in code: **CUDA → MPS → CPU** (see `app/config.py`).

---

## Pipeline

```
File bytes (PDF / DOCX / image)
        │
        ▼
┌─────────────────┐
│  1. PARSING     │  PyMuPDF · pdfplumber · python-docx · Tesseract OCR fallback
│  app/parsing/   │  → raw_text, sections dict, layout metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. EXTRACTION  │  Rule-based NER + fine-tuned DistilBERT NER
│  app/extraction/│  → skills, entities, education, experience, projects, contact
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  3. CLASSIFICATION      │  TF-IDF+MLP (baseline) → BiLSTM → fine-tuned BERT
│  app/models/classifier  │  → CategoryPrediction list
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  4. JD MATCHING         │  Siamese Sentence-BERT + keyword/skill-gap matching
│  app/models/matcher     │  → MatchResult (semantic score + skill gap)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  5. ATS SCORING         │  Weighted composite score (see weights in config.py)
│  app/scoring/ats        │  → ScoreResult.total, .components, .issues
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  6. EXPLAINABILITY      │  SHAP / LIME / attention highlighting
│  app/scoring/explain    │  → ScoreResult.highlights
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  7. GENERATIVE FEEDBACK │  Flan-T5 / LLM API + rule-based fallback
│  app/genai/             │  → ScoreResult.suggestions
└────────┬────────────────┘
         │
         ▼
    ScoreResult (app/schemas.py)   →   API (FastAPI)  →  UI (Streamlit)
```

---

## Folder Map

```
resume-ai/  (root = /Users/rehan/Desktop/DL_PROJECT)
├── app/
│   ├── config.py          ← ALL settings, paths, weights, device detection
│   ├── schemas.py         ← ALL Pydantic v2 models (DO NOT CHANGE without updating tests + this file)
│   ├── pipeline.py        ← run_pipeline() chains every stage
│   ├── parsing/           ← Stage 1: PDF/DOCX/OCR parsing
│   ├── extraction/        ← Stage 2: NER, skill/education/exp extraction
│   ├── models/
│   │   ├── classifier.py  ← Stage 3: resume category classifier
│   │   └── matcher.py     ← Stage 4: JD matching (Siamese SBERT)
│   ├── scoring/
│   │   ├── ats.py         ← Stage 5: ATS composite scorer
│   │   └── explain.py     ← Stage 6: SHAP/LIME/attention highlights
│   ├── genai/
│   │   └── generator.py   ← Stage 7: generative suggestions
│   ├── extras/            ← Bias checker, interview Q generator, course recommender
│   ├── api/               ← FastAPI routes
│   └── ui/                ← Streamlit pages
├── data/
│   ├── raw/               ← Original Kaggle/LinkedIn datasets (gitignored)
│   ├── processed/         ← Tokenised, split datasets (gitignored)
│   └── taxonomies/        ← ESCO / O*NET skill lists
├── artifacts/
│   ├── models/            ← Saved checkpoints (gitignored)
│   ├── metrics/           ← Eval JSON files (F1, accuracy, MRR, nDCG)
│   └── reports/           ← SHAP plots, confusion matrices
├── scripts/               ← Training, eval, data-prep scripts
├── tests/
│   ├── fixtures/          ← Sample PDFs, DOCX, JD texts for tests
│   ├── test_schemas.py    ← Schema validation & range-rejection tests
│   └── test_pipeline.py   ← End-to-end stub pipeline tests
├── notebooks/             ← Exploratory notebooks
├── docs/                  ← Report, architecture diagrams
├── deploy/                ← Dockerfile, HuggingFace Spaces config
├── PROJECT_CONTEXT.md     ← THIS FILE
├── app/config.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Coding Rules (enforced in all PRs / commits)

1. **Type hints everywhere** — every function signature, every field.
2. **Docstrings on every public function/class** — Google style.
3. **All settings in `app/config.py`** — no magic strings, no hardcoded paths anywhere else.
4. **No hardcoded paths** — use `pathlib.Path` and `Settings` objects.
5. **Device auto-detection** — check CUDA → MPS → CPU; never hardcode `"cpu"`.
6. **Seed 42** — set in config; apply to torch, numpy, random at pipeline entry.
7. **`logging` not `print`** — use `loguru` logger throughout.
8. **Privacy** — resume text is held in memory only; never written to disk, never logged,
   never stored in a database. All uploads processed in-memory and discarded after response.
9. **File-size limit** — enforce `Settings.max_file_bytes` at API ingress.
10. **`import pymupdf`** — not `import fitz` (deprecated in PyMuPDF 2.0).

---

## Schema Change Rule

> **Any change to `app/schemas.py` REQUIRES:**
> 1. Bumping `SCHEMA_VERSION` in `schemas.py`.
> 2. Updating all affected tests in `tests/test_schemas.py`.
> 3. Updating the "Schemas" section of this file.
> 4. Adding an entry to the Decisions Log below.

---

## Datasets & Licenses (to be verified before use)

| Dataset | Source | License | Status |
|---------|--------|---------|--------|
| Resume Dataset (LiveCareer ~2,400) | Kaggle | CC BY-SA 4.0 | To download |
| Resume NER Dataset | Kaggle | Check per dataset | To download |
| LinkedIn Job Postings | Kaggle | Check per dataset | To download |
| ESCO Skill Taxonomy | EU/ESCO | CC BY 4.0 | To download |
| O*NET Occupation Data | US DOL | Public Domain | To download |

---

## Validation Strategy

| Component | Metric |
|-----------|--------|
| NER (DistilBERT) | Entity-level F1, Precision, Recall (seqeval) |
| Classifier | Accuracy, Macro-F1 (sklearn) |
| JD Matching (SBERT) | MRR, nDCG@10 (ranx) |
| Baseline comparison | TF-IDF cosine vs. SBERT cosine on same pairs |
| ATS Score | Component-level ablation; no ground-truth ATS label |

---

## Decisions Log

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 1 | 2026-11-02 | Use `import pymupdf` not `import fitz` | PyMuPDF 2.0 deprecated `fitz` API; future-proofing |
| 2 | 2026-11-02 | Drop `spacy-transformers` from active use | Pins `transformers<4.53.3`, conflicts with `transformers 5.22.1`; HuggingFace pipeline used for NER instead |
| 3 | 2026-11-02 | CPU-only PyTorch installed; MPS enabled via M4 chip | No CUDA on MacBook Air M4; MPS backend available for acceleration |
| 4 | 2026-11-02 | Device order: CUDA → MPS → CPU | Ensures code runs on any hardware without modification |
| 5 | 2026-11-02 | Tesseract binary NOT installed yet | OCR fallback deferred; install with `brew install tesseract` before Stage 1 OCR work |
| 6 | 2026-11-02 | `SCHEMA_VERSION = "1.0.0"` established | Baseline; any schema change increments this and triggers test + context updates |
| 7 | 2026-11-02 | All stubs return valid dummy data typed to schemas | Enables end-to-end pytest validation before any model is trained |
| 8 | 2026-11-02 | Resume text never logged or stored | Privacy rule; enforced at pipeline and API layer |

---

## CHANGELOG

### v0.1.0 — 2026-11-02
- Repo scaffold created: all folders, `__init__.py` files, placeholder files
- `app/schemas.py` v1.0.0: 14 Pydantic v2 models with range validators
- `app/config.py`: Settings class with CUDA→MPS→CPU device detection, seed, paths, score weights
- `app/pipeline.py`: `run_pipeline()` stub chain returning valid `ScoreResult`
- Stub modules: `parsing/parser.py`, `extraction/extractor.py`, `models/classifier.py`,
  `models/matcher.py`, `scoring/ats.py`, `scoring/explain.py`, `genai/generator.py`
- `tests/test_schemas.py`: schema validation + range rejection tests
- `tests/test_pipeline.py`: end-to-end pipeline smoke test
- `requirements.txt` pinned, `.gitignore`, `.env.example` added
- Initial git commit
