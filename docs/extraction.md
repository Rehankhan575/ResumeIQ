# Stage 2: Structured Entity & Skill Extraction

## Architecture Overview

Stage 2 transforms raw document text and segmented sections into structured resume and job description entities. It operates deterministically using rule-based and taxonomy-driven approaches:

```
                      ┌────────────────────────┐
                      │ Stage 1: ParsedResume  │
                      └───────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       ┌─────────────────────┐         ┌─────────────────────┐
       │     skills.py       │         │     entries.py      │
       │ (Taxonomy Extractor)│         │ (Edu & Experience)  │
       └──────────┬──────────┘         └──────────┬──────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                      ┌────────────────────────┐
                      │      extractor.py      │
                      │ (Enriched ParsedResume)│
                      └────────────────────────┘
```

The system components are modularly organized in `app/extraction/`:
- `text_utils.py`: Text normalization and tokenization logic mirrored from `scripts/taxonomy_stats.py` to ensure taxonomy statistics and extraction never drift.
- `skills.py`: Longest-match n-gram extractor with ambiguity filtering, alias resolution, and JD requirement extraction.
- `entries.py`: Normalization of degree titles, institutions, graduation dates, work experience dates, and interval union experience calculation.
- `extractor.py`: High-level orchestrator called by `app/pipeline.py`.

---

## 1. Skill Extraction & Disambiguation Rules

### Active Taxonomy Loading
The extractor loads active vocabulary from `data/taxonomies/skills.csv` filtered strictly by `active == True` in `data/taxonomies/skills_stats.csv` (833 active skills). Inactive O*NET skills unobserved in the training distribution are discarded at initialization.

### Longest-Match First Resolution
Text is tokenized using sliding windows of length 1 to 5 tokens. Candidate matches are evaluated by length descending, ensuring compound skills (e.g., `scikit-learn`, `machine learning`, `relational database`) take precedence over individual sub-tokens. Matched token spans are marked as consumed to enforce non-overlapping spans.

### Ambiguity Rules (`data/taxonomies/ambiguous_terms.csv`)

| Rule Type | Terms | Disambiguation Mechanism |
|---|---|---|
| `case_sensitive` | `java`, `jira` | Requires exact capitalization (`Java`, `Jira`, `JAVA`, `JIRA`). Rejects lowercase occurrences in non-skills sections. |
| `require_context` | `c`, `r`, `go`, `swift`, `rust`, `spark`, `excel`, `word`, `access`, `chef`, `puppet`, `spring`, `react`, `bash`, `git`, `rest`, `dart`, `sass`, `yarn`, `triage`, `welding`, `baking`, `grilling`, `roasting` | 1. **Exact-Case & Token Boundary**: Single letters (`C`, `R`) must be uppercase. `Go` must be capitalized.<br/>2. **Negative Traps**: Rejects `go-to-market`, `R&D`, `Grade C`, `excelled at`, `word` preceded by articles, `access` as a verb.<br/>3. **Context Validation**: Must appear inside a skills section, in list format (bullets, commas, pipes), or adjacent to domain context keywords (e.g., healthcare terms for `triage`, culinary for `baking`). |
| `exclude_standalone` | `sales`, `project`, `management`, `make`, `communication`, `leadership`, `teamwork`, `carpentry`, `grading` | Rejects standalone occurrences in narrative experience or summary prose (e.g. "project timeline", "food cost management"). Only accepted when inside dedicated skills sections or structured bullet lists. |

---

## 2. Structured Entries & Experience Calculation

### Education Normalization
Education text is scanned for degree designations and institutions:
- **Degrees**: Standardized into canonical categories: `Bachelor of Science`, `Bachelor of Arts`, `Bachelor of Technology`, `Bachelor of Engineering`, `Bachelor of Science in Nursing`, `Master of Science`, `Master of Business Administration`, `Master of Arts`, `Doctor of Philosophy`, `Associate of Applied Science`, `Associate Degree`, `Diploma`.
- **Institution**: Identified via academic keywords (`University`, `College`, `Institute`, `School`, `Academy`, `UC Berkeley`, `Stanford`, `MIT`) on the degree line or immediately adjacent lines.
- **Graduation Year**: Extracted from 4-digit years or `Expected YYYY` patterns.

### Experience Anchoring & Interval Union
Work experience entries are anchored on date-range lines:
- Supported date formats: `YYYY-MM`, `MM/YYYY`, `Month YYYY`, `YYYY`, and ongoing tokens (`Present`, `Current`, `Ongoing`, `Till date`).
- Delimiters: `-`, `–` (en dash), `—` (em dash), `to`.
- **Title and Company**: Parsed from delimiters (`Title — Company`, `Title | Company`, `Title at Company`, `Company - Title`) or adjacent lines, with location information automatically stripped.
- **`total_years_experience`**: Calculated as the **union of monthly intervals** across all non-overlapping positions. Past positions include full start and end months; active ongoing positions are computed up to the reference month (`as_of`). Granularity is monthly, divided by 12.0 and rounded to 1 decimal place.

---

## 3. Evaluation Benchmark Results

Metrics generated from `scripts/eval_extraction.py` and saved to `artifacts/metrics/extraction_eval.json`:

### Fixture Precision & Recall

| Fixture | Type | Precision | Recall | F1 | Trap Hits | Fields / Years Pass |
|---|---|---|---|---|---|---|
| `accountant_table.pdf` | Resume | 1.00 | 1.00 | 1.00 | 0 | Edu: Pass, Exp: Pass, Yrs: Pass (diff=0.0) |
| `chef_plain.docx` | Resume | 0.43 | 1.00 | 0.60 | 0 | Edu: Pass, Exp: Pass, Yrs: Pass (diff=0.1) |
| `ds_clean.pdf` | Resume | 0.83 | 1.00 | 0.91 | 0 | Edu: Pass, Exp: Pass, Yrs: Pass (diff=0.0) |
| `nurse_scanned.pdf` | Resume | 0.71 | 1.00 | 0.83 | 0 | Edu: Pass, Exp: Pass, Yrs: Pass (diff=0.0) |
| `swe_two_column.pdf` | Resume | 1.00 | 1.00 | 1.00 | 0 | Edu: Pass, Exp: Pass, Yrs: Pass (diff=0.0) |
| `jd_data_scientist.txt` | JD | Req P: 1.00 | Req R: 1.00 | Req F1: 1.00 | 0 | Min Years: Pass (val=4) |
| `jd_software_engineer.txt` | JD | Req P: 1.00 | Req R: 1.00 | Req F1: 1.00 | 0 | Min Years: Pass (val=3) |
| `jd_chef.txt` | JD | Req P: 0.67 | Req R: 1.00 | Req F1: 0.80 | 0 | Min Years: Pass (val=5) |

- **Micro-Averaged Skill Precision**: 0.8000
- **Micro-Averaged Skill Recall**: 1.0000
- **Micro-Averaged Skill F1**: 0.8889
- **Trap Violations**: 0 across all fixtures

### Real-Text Stress Test (370 Resumes from `classifier_val.parquet`)

- **Skills per resume**: Mean 12.95, Median 12.0, p5 5.0, <5 hits: 4.05%
- **Education parse yield**: 74.86% of resumes yielded at least 1 education entry
- **Experience parse yield**: 95.95% of resumes yielded at least 1 experience entry
- **Experience duration**: Mean 10.63 years, Median 11.8 years, p95 20.4 years, >40 years sanity flag 0.28%
- **Extraction latency**: Mean 4.7 ms, p95 7.3 ms per resume

---

## 4. Known Limitations & Next Steps

1. **Unformatted Text Layouts**:
   In unsegmented raw text dumps (such as single-line LiveCareer OCR outputs where newlines are stripped), section boundaries cannot be reliably identified by layout headers alone. The system falls back to full-text scanning, which maintains high recall but increases exposure for general soft skills.
2. **Emerging & Novel Skills**:
   Taxonomy matching relies on canonical entries in `data/taxonomies/skills.csv`. Unseen tools or newly coined frameworks are addressed in Stage 3b via fine-tuned Transformer NER.
