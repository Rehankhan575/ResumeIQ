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
| Tesseract | **Installed** via `brew install tesseract` (v5.5.4) |

Device selection order in code: **CUDA → MPS → CPU** (see `app/config.py`).
Execution rule: Always run commands via `source venv/bin/activate` or `venv/bin/python -m ...`.

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
│   ├── raw/               ← Original Kaggle/LinkedIn datasets (gitignored content)
│   ├── processed/         ← Tokenised, split datasets (gitignored content)
│   └── taxonomies/        ← ESCO / O*NET skill lists
├── artifacts/
│   ├── models/            ← Saved checkpoints (gitignored content)
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
├── requirements.txt       ← Direct runtime dependencies only (pinned)
├── requirements-dev.txt   ← Test & development dependencies
├── pytest.ini             ← pytest configuration
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
11. **Environment execution** — Always run commands via `source venv/bin/activate` or `venv/bin/python -m ...`.
12. **Dependency management** — Each step adds its own new dependencies to `requirements.txt`, pinned, in the same commit.

---

## Schema Change Rule

> **Any change to `app/schemas.py` REQUIRES:**
> 1. Bumping `SCHEMA_VERSION` in `schemas.py`.
> 2. Updating all affected tests in `tests/test_schemas.py`.
> 3. Updating the "Schemas" section of this file.
> 4. Adding an entry to the Decisions Log below.

---

## Schemas

`SCHEMA_VERSION = "1.3.0"`

Core models defined in `app/schemas.py`:
- `ContactInfo`: `name`, `email`, `phone`, `linkedin`, `github`, `other_links`
- `LayoutMeta`: `file_type`, `page_count`, `word_count`, `has_tables`, `has_images`, `is_multi_column`, `font_count`, `ocr_used`, `needs_ocr`, `extraction_method`
- `EducationEntry`: `degree`, `field`, `institution`, `start_year`, `end_year`, `graduation_year`, `gpa`
- `ExperienceEntry`: `title`, `company`, `start_date`, `end_date`, `start`, `end`, `is_current`, `duration_months`, `bullets`
- `ProjectEntry`: `name`, `description`, `technologies`
- `Entity`: `text`, `label`, `start`, `end`, `score`, `source`
- `ParsedResume`: `raw_text`, `sections`, `contact`, `layout`, `skills`, `entities`, `education`, `experience`, `projects`, `certifications`, `total_years_experience`, `warnings`
- `CategoryPrediction`, `JobDescription` (`title`, `text`, `required_skills`, `preferred_skills`, `min_years`, `min_years_experience`), `MatchResult`, `Issue`, `Highlight`, `Suggestion`, `ScoreResult` (`total`, `grade`, `components`, `missing_skills`, `issues`, `highlights`, `predicted_categories`, `match`, `suggestions`, `timings_ms`, `warnings`)

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

> [!NOTE]
> **Matcher Weak Labels & Evaluation Circularity**: JD category relevance labels are derived via title-matching heuristic rules (`title_category_map.csv`), not human recruiter curation. Because category membership defines both positive training pairs and evaluation relevance judgments, ranking metrics (MRR, nDCG) partly reflect category separation rather than individual applicant qualification.

---

## Decisions Log

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 1 | 2026-11-02 | Use `import pymupdf` not `import fitz` | PyMuPDF 2.0 deprecated `fitz` API; future-proofing |
| 2 | 2026-11-02 | Drop `spacy-transformers` from active use | Pins `transformers<4.53.3`, conflicts with `transformers 5.22.1`; HuggingFace pipeline used for NER instead |
| 3 | 2026-11-02 | CPU-only PyTorch installed; MPS enabled via M4 chip | No CUDA on MacBook Air M4; MPS backend available for acceleration |
| 4 | 2026-11-02 | Device order: CUDA → MPS → CPU | Ensures code runs on any hardware without modification |
| 5 | 2026-11-02 | Tesseract installed via `brew install tesseract` | OCR fallback available for Stage 1 parser |
| 6 | 2026-11-02 | `SCHEMA_VERSION = "1.0.0"` established | Baseline; any schema change increments this and triggers test + context updates |
| 7 | 2026-11-02 | All stubs return valid dummy data typed to schemas | Enables end-to-end pytest validation before any model is trained |
| 8 | 2026-11-02 | Resume text never logged or stored | Privacy rule; enforced at pipeline and API layer |
| 9 | 2026-11-02 | `extractor.py` is the extraction orchestrator; Steps 3a/3b add `skills.py` and `ner.py` next to it | Modular design separating rule-based and neural extraction under a unified orchestrator |
| 10 | 2026-11-03 | `ner_real_skills` REJECTED as ground truth | Dictionary-style auto-tags; 12,695 distinct strings, only 1.5% in O*NET. Kept on disk, never used for training or evaluation. |
| 11 | 2026-11-03 | NER training uses `ner_synthetic` with template-disjoint splits | Honest evaluation on hand-corrected gold set of real LiveCareer resumes (IDs reserved in data/splits/ner_gold_ids.csv). |
| 12 | 2026-11-03 | `job_resume_fit` (23 JDs) NOT used for matcher training | Only 23 distinct JDs; used solely as external validation (Spearman vs ai_match_score on test-split resumes). Training/eval JDs come from `linkedin_sample` mapped by title rules. |
| 13 | 2026-11-03 | One shared resume split keyed on LiveCareer ID, group-aware | Groups defined by exact-duplicate Resume_str OR TF-IDF cosine ≥ 0.90. Split serves classifier, matcher, and NER gold selection. |
| 14 | 2026-11-03 | Classifier reports results with and without the leading title header | LiveCareer resumes begin with job title (div.PARAGRAPH_NAME); masking it removes the dominant signal to measure label leak. |
| 15 | 2026-11-03 | `title_category_map` is frozen, labels are title-rule weak labels | Precedence rules establish domain keyword priority over generic role words; frozen to preserve split stability. |
| 16 | 2026-11-03 | O*NET skills are kept in taxonomy but flagged inactive when unused in train text, not deleted | Preserves occupational taxonomy integrity while cleanly separating active vocabulary (DF > 0 or curated) from unobserved tools. |
| 17 | 2026-11-03 | `SCHEMA_VERSION` bumped to 1.1.0 (additive: `ContactInfo.name`, `LayoutMeta.needs_ocr`, `LayoutMeta.extraction_method`) | Stage 1 parser requires extracted contact name and layout OCR/method signals |
| 18 | 2026-11-03 | OCR triggered when chars/page < 50; rasterised at 300 DPI with grayscale + autocontrast | Prevents silent blank extraction on scanned or image-based résumés while maintaining high character recognition accuracy |
| 19 | 2026-11-03 | Two-column layout detected via x0 clustering (gutter ≥ 50pt, vertical overlap ≥ 20%); read left then right | Prevents interleaving of sidebar metadata (contact, skills, education) into main work experience streams |
| 20 | 2026-11-03 | Typed parser errors subclassing `ParseError` (empty, oversize, corrupted, locked, unsupported) | Prevents unhandled bare library exceptions from leaking through the ingestion boundary |
| 21 | 2026-11-03 | `SCHEMA_VERSION` bumped to 1.2.0 (additive: `JobDescription.min_years`, `EducationEntry.graduation_year`, `ExperienceEntry.start`, `ExperienceEntry.end`) | Stage 2 extraction requires graduation year for education and compact start/end alias keys for experience and JD minimum experience |
| 22 | 2026-11-03 | In full-text fallback mode, ambiguous terms require strong list evidence (2+ non-ambiguous skills within 6 tokens) | Prevents soft skill and verb leaks (`make`, `word`, `access`, `project`, `management`, `sales`) when document layout collapses without section headings |
| 23 | 2026-11-03 | Classifier trained across 4 models (TF-IDF+LR, TF-IDF+MLP, BiLSTM, DistilBERT) on masked text; best val F1 saved, test set accessed strictly via --final-eval | Prevents title leak, adheres to 20-min compute budget, provides graceful on-the-fly fallback if checkpoints missing |
| 24 | 2026-11-03 | Composite ATS score (0-100) across 9 weighted components; issues prioritized by estimated gain; explainability via batched LOSO and TF-IDF+LR linear attribution | Reflects absence of ground-truth ATS scores; guarantees deterministic sub-second CPU explainability without SHAP/LIME overhead |
| 25 | 2026-11-04 | Siamese bi-encoder (`all-MiniLM-L6-v2`) fine-tuned with `MultipleNegativesRankingLoss` on MPS; fallback to TF-IDF | Optimizes dense embeddings for resume-JD matching within 15-min budget; centroid-based cross-category hard negative mining and per-category JD caps prevent class skew |
| 26 | 2026-11-04 | Matcher evaluation explicitly discloses weak label circularity | Shared title-rule category definitions govern both training triplets and evaluation qrels; metrics measure category separation rather than recruiter qualification |
| 27 | 2026-11-04 | Stage 5 ATS scorer routes to public `app.models.matcher.match()` and `app.models.classifier.classify()` with cached neural bi-encoder and TF-IDF fallback | Connects fine-tuned Siamese bi-encoder to ATS scoring while maintaining sub-second inference caching and robust TF-IDF fallback for missing checkpoints |
| 28 | 2026-11-04 | Stage 7 GenAI provider abstraction (Gemini, Groq, Ollama) with 8s timeout, 2 retries, in-memory SHA-256 caching, and Grounding Guard | Prevents API stalls, guarantees zero disk/log credential leakage, and strictly drops ungrounded hallucinated numbers, orgs, and skills |
| 29 | 2026-11-04 | Modular extras: Multi-JD ranking, 70-skill URL-free courseware mapping, privacy-preserving bias audit, and opt-in SQLite history | Provides comprehensive resume optimization while preserving candidate privacy (no personal text in logs, bias labels, or SQLite) |
| 30 | 2026-11-04 | `SCHEMA_VERSION` bumped to 1.3.0 (additive: `ScoreResult.warnings`) | Pipeline graceful degradation surfaces non-fatal stage warnings in result object without raising exceptions |

---

## CHANGELOG

### v1.0.0 — 2026-11-04
- `app/pipeline.py`: implemented full 7-stage chain with `warm_up()` lazy singleton, rich `run_pipeline(file_bytes, filename, jd_text, jd_title="", use_genai=True)` supporting graceful degradation with `warnings` tracking, 5MB file-size and type validation, and segfault-mitigation env vars (`TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `PYTORCH_ENABLE_MPS_FALLBACK=1`)
- `app/extras/ranking.py`: fixed item 0 bug where empty `required_skills` inflated keyword coverage by +20; now populates skills via `extract_jd_skills`
- `app/schemas.py`: bumped `SCHEMA_VERSION` to 1.3.0 with additive `ScoreResult.warnings`
- `app/ui/`: polished developer-tools Streamlit dashboard with custom CSS tokens (`app/ui/styles.css`), self-hosted Inter & JetBrains Mono WOFF2 fonts (`app/ui/static/fonts/`), SVG circular score ring, component breakdown bars, stat tiles, gain-sorted issues, 7 underline tabs, and no default Streamlit chrome
- `scripts/preflight.py`: preflight verification runner evaluating all 15 fixture × JD combinations, measuring cold-start warm-up time (7.87s–8.38s), and asserting zero failures
- `tests/test_ui.py`: headless `streamlit.testing.v1.AppTest` suite testing initial state, sample analysis, empty input, corrupted files, and multiple JDs
- `tests/test_pipeline.py`: 30 comprehensive pipeline tests covering ranking score gap, graceful degradation, and log privacy
- `docs/ui.md`: documentation detailing execution instructions, design tokens, tab overviews, and system limitations
- `docs/screenshots/`: 30 visual verification screenshots captured via automated headless Chrome CDP across 1440x900, 1280x720, and 1920x1080

### v0.9.0 — 2026-11-04
- `app/genai/`: implemented Stage 7 Generative AI provider abstraction supporting Google Gemini, Groq, and Ollama with strict 8s timeout, max 2 retries, transient in-memory caching, and rule-based fallback
- `app/genai/grounding.py`: implemented `check_grounding(original, rewritten)` detecting ungrounded numbers, capitalized organizations, and taxonomy skills, immediately dropping hallucinated bullets in favor of safe placeholders
- `app/genai/prompts/`: prompt templates for bullet rewriting, cover letter drafting, and prioritized improvement plans enforcing hard anti-hallucination rules
- `data/courses.json`: curated learning catalog spanning 70 in-demand skills with 1-2 resources (title, provider, level, estimated hours) and zero external URLs
- `app/extras/`: implemented multi-JD ranking (`rank_jds`), skill gap courseware recommendation (`skill_gap`), unconscious demographic bias auditing (`bias_check` with abstract privacy labels), and strictly opt-in SQLite history tracking (`record_history` with zero text fields)
- `tests/test_genai.py` & `tests/test_extras.py`: 12 comprehensive unit and integration tests (215 total passing tests across repository)
- `docs/genai.md` & `docs/extras.md`: comprehensive documentation for generative pipelines, grounding guard, bias checks, and course recommendations

### v0.8.1 — 2026-11-04
- `app/scoring/ats.py`: connected Stage 5 scorer to public `app.models.matcher.match()` and `app.models.classifier.classify()` with in-memory model caching and graceful TF-IDF cosine fallback; updated score matrix and ablation metrics
- `docs/scoring.md`: updated score matrix results (100% matched-pair pass with fine-tuned bi-encoder), component ablation table, and honesty disclosure for XAI faithfulness feature erasure
- `artifacts/metrics/scoring/`: updated `score_matrix.json`, `ablation.csv`, `scoring_eval.json`, and `artifacts/reports/xai_faithfulness.png`

### v0.8.0 — 2026-11-04
- `app/models/matcher.py`: implemented Stage 4 Résumé–JD matcher supporting TF-IDF cosine baseline and fine-tuned Siamese bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`), with `match(resume_text, jd_text)` computing composite score (0-1) and section similarities (skills overlap via `TaxonomySkillExtractor`, experience vs JD text), and graceful TF-IDF fallback
- `scripts/train_matcher.py`: end-to-end training and evaluation pipeline for the 15 categories with >=20 JDs, centroid-based hard negative mining across categories, capped JDs per category to prevent class skew, MPS/GPU accelerated fine-tuning with `MultipleNegativesRankingLoss`, validation evaluation, isolated `--final-eval` test evaluation, and metrics export to `artifacts/metrics/matcher/`
- `tests/test_matcher.py`: 6 tests verifying 0-1 bounded scoring, empty input resilience, missing-model TF-IDF fallback, zero test resume/JD leak into training pairs, fixture discrimination sanity check (`ds_clean` vs `jd_data_scientist` vs `jd_chef`, and `chef_plain` reversed), and pipeline compatibility (203 total passing tests)
- `artifacts/models/matcher/`: trained bi-encoder model weights and TF-IDF matcher baseline
- `artifacts/metrics/matcher/`: validation (`matcher_val.json`) and test (`matcher_eval.json`) metrics
- `docs/matcher.md`: comprehensive documentation covering architecture, 15 target categories, ranking metrics, section similarity formulation, and explicit disclosure of weak label circularity

### v0.7.0 — 2026-11-03
- `data/taxonomies/ambiguous_terms.csv`: added rule `rn,require_context` preventing clinical "RN" (Registered Nurse) from colliding with software framework React Native without context
- `app/config.py`: defined `SCORE_WEIGHTS` across all 9 components summing to 1.0 (`semantic_match: 0.25`, `keyword_coverage: 0.20`, `section_completeness: 0.10`, `formatting: 0.10`, `experience_fit: 0.10`, `quantified_achievements: 0.10`, `action_verbs: 0.05`, `readability: 0.05`, `contact_completeness: 0.05`), plus curated `ACTION_VERBS` set
- `app/scoring/formatting.py`: layout scoring engine deriving parsability deductions from `LayoutMeta` (multi-column, tables, images, page count, word count, OCR)
- `app/scoring/ats.py`: Stage 5 ATS composite scorer `score_resume(parsed_resume, jd)` generating 0-100 score, letter grade (A-F), components dict, missing skills, issues sorted by estimated point gain, graceful empty/corrupted document handling, and calibrated TF-IDF cosine matcher fallback
- `app/scoring/explain.py`: Stage 6 explainability engine combining batched Leave-One-Sentence-Out (LOSO) sentence contributions (capped at 40 sentences), sub-millisecond TF-IDF + Logistic Regression linear coefficient attribution, keyword highlights, and safe `render_html()` with entity escaping
- `scripts/eval_scoring.py`: comprehensive benchmark evaluating 5x3 score matrix (100% matched pair pass), component ablation table (`artifacts/metrics/scoring/ablation.csv`), XAI feature erasure faithfulness on 20 val resumes (`artifacts/reports/xai_faithfulness.png`, top-3 drop 0.26 vs random-3 drop 0.00), and latency benchmarking (17.33 ms/resume)
- `tests/test_scoring.py`: 12 tests covering weight sum to 1.0, range bounds, determinism, keyword addition monotonicity, empty input resilience, issue sorting, offset accuracy, XSS escaping, and "rn" rule verification (197 total passing tests)
- `docs/scoring.md`: comprehensive documentation detailing mathematical formulations, point gain derivations, LOSO/linear XAI algorithms, experimental results, and ATS ground-truth limitations

### v0.6.0 — 2026-11-03
- `scripts/train_classifier.py`: implemented 4-model classifier training pipeline (TF-IDF+LR, TF-IDF+MLP, BiLSTM, DistilBERT) with smoke testing, val-only hyperparameter tuning, early stopping, confusion matrix plotting, loss curve generation, and isolated `--final-eval` test evaluation
- `app/models/classifier.py`: implemented `classify(text, top_k=3, normalize=True)` returning top-3 `CategoryPrediction` objects with probabilities summing to 1.0, cached model inference for the best validation model (`tfidf_mlp`), pipeline integration `classify_resume(parsed)`, and on-the-fly TF-IDF+LR fallback
- `tests/test_classifier.py`: 6 tests verifying output shape, probability normalization to 1.0, missing model directory fallback, zero text leakage across train/val/test splits, schema integration with `ParsedResume`, and empty/whitespace input handling (185 total tests passing)
- `artifacts/metrics/classifier/`: generated confusion matrices (`cm_*.png`), loss curves (`loss_curve_*.png`), and metrics payload (`metrics.json`) across all 4 models
- `docs/classifier.md`: comprehensive Stage 3a documentation detailing experimental setup, throughput measurements, 4-model comparison table, serving fallback, and error analysis

### v0.5.1 — 2026-11-03
- `app/extraction/skills.py`: added acceptance reason tracking (`get_reason_counts()`, `reset_reason_counts()`) and implemented strict fallback filter: in full-text fallback mode (no detected skills section), ambiguous, `exclude_standalone`, and `require_context` terms are rejected unless situated in a delimited run with ≥2 other non-ambiguous active taxonomy skills within a 6-token window; behavior inside detected skills sections remains unchanged
- `app/extraction/entries.py`: cleaned up fixture-specific comment examples in `clean_title_company` to use generic placeholders (`Senior Specialist — Core Platform — Example Corporation, Chicago, IL`)
- `scripts/eval_extraction.py`: added dual-mode stress test evaluating collapsed `Resume_str` vs generic block-flattened `Resume_html` (`div`, `p`, `li`, `br`, `tr` only) across 370 val IDs, added acceptance reason histogram table, detailed FP classifications for `chef_plain` and `jd_chef`, and before/after ambiguity comparison
- `tests/test_extraction.py`: added 3 unit tests verifying rejection of ambiguous terms (`make`, `word`, `access`, `project`, `management`) in collapsed prose, acceptance inside delimited runs with 2+ non-ambiguous skills, and reason counter lifecycle (16 unit tests, 179 total tests passing)
- `docs/extraction.md`: documented fallback ambiguity filter design, reason histograms, dual-mode section detection rates, and HTML approximation caveats

### v0.5.0 — 2026-11-03
- `app/extraction/`: implemented Stage 2 text tokenization (`text_utils.py`, 100% parity with `taxonomy_stats.py`), rule-based taxonomy skill extractor (`skills.py`, active-only, 1-5 n-grams, longest-match first, non-overlapping spans, `ambiguous_terms.csv` context & case validation, JD required/preferred splitting and `min_years` parsing), structured entry extractor (`entries.py`, normalized education, date-anchored experience, non-overlapping interval union `total_years_experience`), and unified orchestrator (`extractor.py`, `extract_entities` preserving pipeline signature)
- `app/schemas.py`: bumped `SCHEMA_VERSION` to 1.2.0 with additive fields (`JobDescription.min_years`, `EducationEntry.graduation_year`, `ExperienceEntry.start`, `ExperienceEntry.end`)
- `app/config.py`: added Stage 2 extraction settings (`skills_csv_path`, `ambiguous_terms_csv_path`, `skills_stats_csv_path`, `default_as_of_date`)
- `tests/test_schemas.py`: updated schema tests for v1.2.0 version assertion and defaults of new additive fields
- `tests/test_extraction.py`: 13 comprehensive tests covering 50-string tokenizer parity, negative traps ("go-to-market", "R&D", "Grade C", "excelled at"), positive contexts, alias normalizations, longest match, inactive skill filtering, all date formats, fixed as_of determinism, overlapping interval union, empty inputs, and JD required/preferred splitting
- `scripts/eval_extraction.py`: Stage 2 evaluation benchmark suite testing 5 resume fixtures, 3 JDs, and 370-resume real-text stress test on `classifier_val.parquet`, exporting `artifacts/metrics/extraction_eval.json` (micro-averaged skill F1: 0.8889, recall: 1.0000, 0 trap violations, 100% education/experience/years pass)
- `docs/extraction.md`: Stage 2 documentation detailing architecture, disambiguation rules, entry normalization, evaluation benchmark results, and limitations

### v0.4.0 — 2026-11-03
- `app/parsing/`: implemented Stage 1 parser orchestrator (`parser.py`), format detection (`detect_file_type` via magic bytes first, extension second), column-aware layout ordering (`_order_page_blocks`), table extraction, Tesseract OCR fallback (`min_chars_per_page=50`, 300 DPI, grayscale + autocontrast), text cleaning (`cleaner.py`), canonical section splitting (`sections.py`, 11 sections, exact + rapidfuzz >= 92), and contact info extraction (`contact.py`)
- `app/parsing/errors.py`: typed parsing exception hierarchy (`ParseError`, `UnsupportedFileTypeError`, `FileTooLargeError`, `CorruptedFileError`, `PasswordProtectedError`, `EmptyDocumentError`)
- `data/taxonomies/section_headings.csv`: 97 heading synonym rules mapping to 11 canonical sections
- `app/config.py`: added Stage 1 parsing configurations (`min_chars_per_page=50`, `max_pages=10`, `ocr_dpi=300`)
- `app/schemas.py`: bumped `SCHEMA_VERSION` to 1.1.0 with additive fields (`ContactInfo.name`, `LayoutMeta.needs_ocr`, `LayoutMeta.extraction_method`)
- `tests/test_schemas.py`: updated schema tests for v1.1.0 defaults, valid payloads, round-trip, and v1.0.0 backward compatibility
- `tests/test_parsing.py`: 23 comprehensive tests covering cleaning, sectioning, contact extraction, in-memory edge cases (empty bytes, truncated PDF, encrypted PDF, oversize, max pages, mime swap, unsupported format), and all 5 test fixtures
- `scripts/eval_parsing.py`: benchmark evaluation suite testing all 5 fixtures and 8 edge cases, saving metrics to `artifacts/metrics/parsing_eval.json` (100% precision/recall across all sections, 100% contact extraction accuracy, 100% layout detection accuracy, 100% OCR skill recall on nurse_scanned, 0 interleaving on swe_two_column)
- `docs/parsing.md`: comprehensive Stage 1 documentation detailing architecture, thresholds, design, and known limitations
- `requirements.txt`: pinned `rapidfuzz==3.14.6`, `pdfplumber==0.11.10`, `pytesseract==0.3.13`

### v0.3.0 — 2026-11-03
- `data/taxonomies/title_category_map.csv`: frozen title precedence rules (243 rules), `scripts/build_jd_pool.py` updated (1,796 mapped JDs, 26 ambiguous, 3,011 unmapped), `docs/splits.md` updated
- `scripts/build_skills_taxonomy.py`: built unified taxonomy (`data/taxonomies/skills.csv`, 1,667 rows), curated non-IT skills (`data/taxonomies/skills_curated.csv`, 487 rows), ambiguity rules (`data/taxonomies/ambiguous_terms.csv`, 35 rows), unit tests (`tests/test_taxonomy.py`, 16 tests), and documentation (`docs/taxonomy.md`)
- `scripts/make_fixtures.py`: generated 5 synthetic resume fixtures (`ds_clean.pdf`, `swe_two_column.pdf`, `accountant_table.pdf`, `chef_plain.docx`, `nurse_scanned.pdf`), 3 synthetic JDs, and 8 `.expected.json` sidecars in `tests/fixtures/`
- `scripts/taxonomy_stats.py`: fast 1-5 n-gram taxonomy stats over unmasked train text -> `data/taxonomies/skills_stats.csv`, reporting 833 active skills, distribution percentiles, suspects, and O*NET breakdown in `docs/taxonomy.md`
- `scripts/make_eda_notebook.py`: programmatic build and execution of `notebooks/01_eda.ipynb` via `nbformat` and `nbclient` using `dl_project_venv`, producing figures in `artifacts/reports/eda_*.png`
- `tests/test_fixtures.py`: comprehensive fixture tests verifying file integrity, scanned PDF zero-text layer (<20 chars, >=1 image), sidecar schema compliance, taxonomy skill validation, and `@example.com` domain isolation
- `requirements.txt`: pinned `reportlab==5.1.0`, `python-docx==1.2.0`, `pymupdf==2.0`, `Pillow==13.0.0`, `nbformat==5.11.1`, `nbclient==0.11.0`, `ipykernel==7.4.0`, `matplotlib==3.11.2`

### v0.2.0 — 2026-11-03
- `scripts/make_splits.py`: group-aware 70/15/15 resume split (seed 42), TF-IDF dedup,
  header-masking with leak measurement, NER gold candidate selection
- `scripts/build_jd_pool.py`: LinkedIn JD dedup, title→category mapping, JD split, jrf_eval
- `scripts/verify_splits.py`: independent PASS/FAIL integrity check for all splits
- `tests/test_splits.py`: offline unit tests for union-find, group split, masking, title map
- `data/splits/`: classifier_split.csv, jd_split.csv, ner_gold_ids.csv (IDs only, no text)
- `data/processed/`: classifier_{train,val,test}.parquet, jd_{train,val,test}.parquet, jrf_eval.parquet
- `data/taxonomies/title_category_map.csv`: keyword→category rules for 24 categories
- `docs/splits.md`: split design, counts, limitations
- `requirements.txt`: added `beautifulsoup4`, `scipy` (for sparse cosine)

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
- Scaffolding hygiene update: `.gitignore` refined to preserve `.gitkeep` files, `requirements.txt` trimmed to direct imports, `requirements-dev.txt` & `pytest.ini` created, Tesseract installed and documented, tagged `step-00`
