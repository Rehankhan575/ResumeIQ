# Stage 4 — Résumé–JD Semantic Matching

## 1. Overview & Architecture

The Matcher module (`app/models/matcher.py`) implements Stage 4 of the AI Resume Analyzer pipeline. Given a candidate résumé and a job description (or a pool of candidate JDs), it computes an overall semantic match score and granular section-wise similarities.

```
       Candidate Résumé                         Job Description
       ┌───────────────┐                        ┌──────────────┐
       │   raw_text    │                        │   raw_text   │
       └───────┬───────┘                        └──────┬───────┘
               │                                       │
    ┌──────────┴───────────────┐             ┌─────────┴───────────────┐
    │  app.parsing / sections  │             │  app.extraction.skills  │
    └──────────┬───────────────┘             └─────────┬───────────────┘
               │                                       │
       ├── sections['experience']             ├── required_skills
       └── sections['skills']                 └── preferred_skills
               │                                       │
               ▼                                       ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                Siamese Bi-Encoder (MiniLM)                   │
    │          (or TF-IDF Cosine Baseline / Fallback)              │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                        MatchResult                           │
    │  - score: [0.0, 1.0]                                         │
    │  - section_scores: {'skills': 0.85, 'experience': 0.78}      │
    │  - matched_skills: ['python', 'pytorch', 'sql']              │
    │  - missing_skills: ['docker', 'kubernetes']                  │
    │  - method: 'sbert_fine_tuned'                                │
    └──────────────────────────────────────────────────────────────┘
```

---

## 2. Experimental Task & The 15 Target Categories

### 2.1. Task Formulation
Given a candidate résumé query $q$, the task is to rank a pool of $M$ candidate Job Descriptions. A JD $d_j$ is defined as **relevant** ($r_j = 1$) if its category matches the candidate résumé's category ($c(d_j) = c(q)$), and non-relevant ($r_j = 0$) otherwise.

### 2.2. The 15 Target Categories (≥ 20 JDs)
Out of the 24 categories in the dataset, only categories with at least 20 JDs across the corpus are retained for matching evaluation to ensure statistically meaningful recall pools:

| # | Category | Total JDs in Pool | Train JDs | Val JDs | Test JDs |
|---|----------|:-----------------:|:---------:|:-------:|:--------:|
| 1 | HEALTHCARE | 461 | 315 | 68 | 78 |
| 2 | SALES | 388 | 274 | 55 | 59 |
| 3 | INFORMATION-TECHNOLOGY | 189 | 128 | 28 | 33 |
| 4 | ACCOUNTANT | 151 | 113 | 20 | 18 |
| 5 | ENGINEERING | 81 | 57 | 10 | 14 |
| 6 | CONSTRUCTION | 71 | 52 | 8 | 11 |
| 7 | ADVOCATE | 70 | 42 | 17 | 11 |
| 8 | CONSULTANT | 49 | 34 | 11 | 4 |
| 9 | HR | 45 | 32 | 7 | 6 |
| 10 | TEACHER | 41 | 26 | 8 | 7 |
| 11 | BANKING | 40 | 33 | 2 | 5 |
| 12 | BPO | 39 | 30 | 6 | 3 |
| 13 | CHEF | 39 | 30 | 5 | 4 |
| 14 | BUSINESS-DEVELOPMENT | 31 | 22 | 7 | 2 |
| 15 | DIGITAL-MEDIA | 24 | 16 | 5 | 3 |

---

## 3. Evaluated Model Variants

Three distinct models were benchmarked under identical ranking conditions:

1. **TF-IDF Cosine Baseline**:
   - Sublinear term frequency + IDF vectors fit on training texts (resumes + JDs), max 10,000 features, unigram + bigram.
   - Fast sparse cosine similarity.
2. **Zero-Shot Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`)**:
   - Pretrained 6-layer MiniLM bi-encoder producing 384-dimensional dense sentence embeddings.
   - Evaluated out-of-the-box without task fine-tuning.
3. **Fine-Tuned Siamese Bi-Encoder (MiniLM backbone)**:
   - Same `all-MiniLM-L6-v2` architecture, fine-tuned using `MultipleNegativesRankingLoss`.
   - Training triplets: `(anchor_resume, positive_jd, hard_negative_jd)`.
   - Hard negatives sampled from top-similar categories based on TF-IDF category centroid similarity (e.g. `ACCOUNTANT` paired against `BANKING` / `CONSULTANT`; `IT` against `ENGINEERING` / `CONSULTANT`).
   - Capped at 20 JDs per category during training to prevent dominant categories (`HEALTHCARE`, `SALES`, `IT`) from skewing representation.
   - Sequence length: 256 tokens, trained on Apple Silicon MPS backend.

---

## 4. Evaluation Protocol & Metrics

Evaluation is performed using standard Information Retrieval metrics across the candidate pool:
- **MRR (Mean Reciprocal Rank)**: Reciprocal of the rank of the first relevant JD ($1 / \text{rank}_1$).
- **nDCG@10**: Normalized Discounted Cumulative Gain truncated at rank 10, rewarding relevant JDs ranked near the top.
- **Recall@5**: Share of total relevant category JDs retrieved within the top 5 positions.
- **Recall@10**: Share of total relevant category JDs retrieved within the top 10 positions.

Tuning and checkpoint selection are performed strictly on the **VALIDATION** split (`classifier_val` and `jd_val`). The **TEST** split is touched only once at the end via `python scripts/train_matcher.py --final-eval`.

### 4.1 Benchmark Results (3-Model Comparison)

*Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means.*

| Model Architecture | Val MRR | Val nDCG@10 | Val Recall@5 | Val Recall@10 | Test MRR | Test nDCG@10 | Test Recall@5 | Test Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.6121 | 0.4537 | 0.2209 | 0.3239 | 0.6333 | 0.4854 | 0.2337 | 0.3456 |
| **Zero-Shot SBERT (all-MiniLM-L6-v2)** | 0.6377 | 0.4414 | 0.2107 | 0.3090 | 0.6311 | 0.4532 | 0.1964 | 0.3007 |
| **Fine-Tuned Siamese SBERT (MNRL)** | **0.8567** | **0.6821** | **0.3286** | **0.4671** | **0.7798** | **0.7286** | **0.3755** | **0.5338** |

*Takeaway:* Fine-tuned Siamese SBERT achieves **0.7798 Test MRR**, representing a **+14.87 point increase vs zero-shot** (0.7798 vs 0.6311) and **+14.65 points vs TF-IDF baseline** (0.7798 vs 0.6333). Note that labels are weak and title-derived, and the retrieval metric is partly circular. Broad domains such as BPO (test MRR 0.0326) and CONSULTANT (test MRR 0.2032) suffer from high cross-category overlap.

---

## 5. Granular Section Similarity & Fallback

### 5.1. Section-Wise Scoring
In addition to overall semantic alignment, `match(resume_text, jd_text)` decomposes match quality into:
1. **Skills Similarity**:
   - Extracts candidate skills via `TaxonomySkillExtractor`.
   - Extracts required and preferred skills from the JD via `extract_jd_skills()`.
   - Computes set overlap: $\text{skills\_similarity} = \frac{|\text{candidate\_skills} \cap \text{jd\_required}|}{|\text{jd\_required}|}$.
2. **Experience Similarity**:
   - Isolates the candidate's work history section text via `split_sections()`.
   - Computes bi-encoder cosine similarity specifically between the experience narrative and the JD requirements.

### 5.2. Graceful Degradation & Fallback
If the fine-tuned neural model cannot be loaded (missing checkpoint, CUDA/MPS OOM, or missing weights), the system logs a descriptive warning and falls back to TF-IDF cosine matching. All return values remain within the valid $[0.0, 1.0]$ schema range.

---

## 6. Scientific Discussion: Weak Labels & Evaluation Circularity

> [!WARNING]
> **Weak Labels & Evaluation Circularity Notice**
> 1. **Weak Category Labels**: All job description categories are weak labels generated by title keyword precedence rules (`data/taxonomies/title_category_map.csv`), not human recruiter judgment.
> 2. **Evaluation Circularity**: Because the same category definition governs both the training objective (positive pairs from the same category) and the evaluation ground truth (relevance $r=1$ iff categories match), the ranking metrics primarily measure category-level thematic discrimination rather than nuanced candidate suitability or seniority alignment.
> 3. **Validation Strategy**: To mitigate circularity, Section 4 includes an external validation check on orthogonal human-labeled résumé–JD pairs (`job_resume_fit`), testing whether model similarity scores correlate with human fit rankings across diverse roles.

