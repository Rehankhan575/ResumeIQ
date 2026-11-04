# Comprehensive Viva Voce Questions & Answers

Rigorous, evidence-backed answers citing exact project metrics, architectural decisions, and honest empirical limitations.

---

## Group 1: Dataset & Data Pipeline

### Q1. What dataset was used, and what was your splitting strategy?
**Answer:** We used the Kaggle Resume Dataset comprising 2,483 resumes across 24 balanced job categories. We applied a stratified 70/15/15 train/val/test split resulting in 1,739 training, 370 validation, and 374 testing resumes. Stratification was locked with a fixed random seed to guarantee class representation across splits.

### Q2. How did you prevent data leakage during preprocessing?
**Answer:** Resumes often contain explicit title headers (e.g., "HR Manager Resume"). We implemented automated header detection and category token sanitization, stripping explicit target labels prior to text vectorization. Additionally, TF-IDF vectorizers and label encoders were fit strictly on the 1,739 training resumes and only applied as transforms to validation and test splits.

### Q3. Why did you not perform k-fold cross-validation?
**Answer:** Given compute budgets and deep learning model training times (particularly DistilBERT fine-tuning and Siamese SBERT training), a deterministic 70/15/15 stratified split was used. We document this plainly as an empirical limitation; all reported test numbers reflect completely held-out, unseen samples.

---

## Group 2: Stage 3 Classifier Comparison (Units I, III, IV)

### Q4. Why did you benchmark four different classification architectures?
**Answer:** To systematically evaluate the trade-off between model complexity, inference latency, and generalization across syllabus units: Linear models (Unit I: TF-IDF + Logistic Regression), Feedforward Networks (Unit I: TF-IDF + 2-layer MLP), Recurrent Architectures (Unit III: BiLSTM), and Pretrained Transformers (Unit IV: DistilBERT).

### Q5. What were the empirical results across the four models?
**Answer:** On the held-out test split of 374 resumes across 24 classes:
- **DistilBERT:** 72.73% Accuracy | 0.6454 Macro F1 (Lowest generalization gap: 0.1073)
- **TF-IDF + Logistic Regression:** 66.31% Accuracy | 0.5959 Macro F1 (Trained in < 1 second)
- **TF-IDF + MLP:** 65.24% Accuracy | 0.6181 Macro F1 (Overfitted: 1.0000 Train F1 vs 0.6001 Val F1)
- **BiLSTM:** 25.13% Accuracy | 0.1966 Macro F1 (Severely underfitted)

### Q6. Why did the BiLSTM perform so poorly compared to other models?
**Answer:** The training split contained 1,739 samples across 24 classes (~72 samples per class). Recurrent networks lack the strong inductive bias of pretrained self-attention and have high parameter complexity. Without a massive corpus to learn sequence transitions, the BiLSTM failed to optimize effectively, achieving only 25.13% test accuracy.

### Q7. Why is TF-IDF Logistic Regression so competitive with DistilBERT?
**Answer:** Professional resumes are heavily keyword-driven: distinctive domain vocabulary (e.g., "culinary", "GAAP", "litigation") provides linearly separable bag-of-words features for most categories. DistilBERT outperformed it by +6.42% accuracy by modeling contextual semantics, but was constrained by a 3-epoch training budget.

---

## Group 3: Stage 4 Siamese Semantic Matcher (Unit II)

### Q8. What architecture and loss function were used for resume-JD matching?
**Answer:** We used a Siamese bi-encoder based on `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense representations) fine-tuned with `MultipleNegativesRankingLoss` (MNRL). Resumes and job descriptions are encoded independently into a shared metric space where cosine distance reflects domain and responsibility fit.

### Q9. How much did fine-tuning improve over the zero-shot baseline?
**Answer:** Fine-tuning yielded major retrieval gains across 15 target categories:
- **Test MRR:** Improved from 0.6311 (zero-shot) to **0.7798 (+14.87 points)**.
- **Test nDCG@10:** Improved from 0.4532 to **0.7286 (+27.54 points)**.
- **Test Recall@10:** Increased from 0.3007 to **0.5338 (+23.31 points)**.

### Q10. Why is Recall@k relatively low for certain categories despite high MRR?
**Answer:** In our evaluation setup, categories contain 14 to 18 resumes. Because $k=5$ can retrieve at most 5 documents, the maximum possible Recall@5 is mathematically capped at $5 / 18 \approx 27.78\%$. Mean Reciprocal Rank (MRR) evaluates whether the first true positive ranks at position 1, which reached 1.0000 for Chef, Accountant, Engineering, and Sales.

### Q11. Which categories perform worst in matching, and why?
**Answer:** BPO (Test MRR: 0.0326) and Consultant (Test MRR: 0.2032). These categories have broad, generic job descriptions (e.g., "customer handling", "strategic advice") that heavily overlap with Information Technology, Sales, and Business Development, creating semantic interference in vector space.

---

## Group 4: Stage 5 ATS Scoring & Explainability (Units II & V)

### Q12. What are the 9 components of your ATS scoring formula?
**Answer:** The score is a deterministic weighted linear combination bounded in $[0, 100]$:
1. Semantic Match (25%)
2. Keyword Coverage (20%)
3. Section Completeness (10%)
4. Experience Fit (10%)
5. Formatting & Layout (10%)
6. Quantified Achievements (10%)
7. Action Verbs (5%)
8. Readability (5%)
9. Contact Completeness (5%)

### Q13. Why does an unrelated resume-JD pair score around 50 rather than 0?
**Answer:** ATS scoring models real-world document hygiene. An unrelated pair (e.g., Data Scientist vs Executive Chef) scores ~55.5 because the candidate possesses intact contact info (5%), standard sections (10%), accredited degrees (10%), and clean formatting (10%). Pure job-fit components (semantic match and keyword coverage) plunge to baseline, dropping the grade to F.

### Q14. What did the ablation study reveal about individual scoring components?
**Answer:** Removing Semantic Match caused the largest intended drop (-3.53 pts). Removing Section Completeness dropped scores by -2.08 pts and Experience Fit by -2.26 pts. Conversely, ablating Quantified Achievements increased mean scores by +7.37 pts, proving that the rubric aggressively penalizes resumes that lack numeric metrics.

### Q15. Why did your LIME/perturbation test flag `is_faithful: False`?
**Answer:** Masking the top 5 keywords produced a mean score drop of 0.07 compared to 0.11 for random word masking. In a 500-word resume, removing 5 words has minimal impact on dense cosine similarity or global TF-IDF vectors. Rather than disguising this, we document it as an honest limitation of word-level perturbation on long-form documents.

---

## Group 5: Stage 2 NER & Disambiguation Rules

### Q16. How does your entity extraction work without fine-tuned BERT-NER?
**Answer:** We implemented a taxonomy-based extractor utilizing 2,800+ canonical skills with alias mapping, longest-match-first n-gram tokenization (up to 5-grams), and 19 ambiguous term disambiguation rules. This yields a deterministic 4.45 ms extraction latency with zero GPU dependency.

### Q17. How was the false positive "RN" $\rightarrow$ "React Native" resolved?
**Answer:** In `nurse_scanned.pdf`, "RN" (Registered Nurse) was erroneously mapped to React Native via an alias. We tested an ambiguous context rule, but discovered skills-section extraction bypassed alias checks. We cleanly resolved it by removing `rn` from React Native aliases in `data/taxonomies/skills.csv`, eliminating the false positive while preserving "React Native" in software contexts.

### Q18. How do you disambiguate single-letter programming languages like "C" or "R"?
**Answer:** Case-sensitivity and context guards: "C" and "R" must be strictly uppercase and located in a designated skills section or list context. "R" is rejected if adjacent to "&" or "D" (R&D trap). "Go" requires capitalization ("Go", "GO", "Golang") and is rejected if adjacent to "-to-market".

---

## Group 6: GenAI & Suggestions (Unit V)

### Q19. How does your suggestion module prevent LLM hallucinations?
**Answer:** We employ a Grounded Hallucination Guard. The LLM prompt is injected only with extracted candidate skills and verified missing JD skills. Generated suggestions are parsed and checked against the candidate's actual background; the model is prohibited from inventing unearned credentials or degrees.

### Q20. What happens if the user runs the system without an LLM API key?
**Answer:** The system features graceful degradation: if no `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` is present, it executes a deterministic rule-based engine. This analyzes the lowest-scoring components (e.g., missing metrics, low action verb count, missing core skills) and generates actionable bullet points offline.

---

## Group 7: System Design, Privacy & Operations

### Q21. How is candidate privacy protected across the platform?
**Answer:** Zero PII is logged. Candidate names, phone numbers, email addresses, and full resume texts are never written to disk logs, terminal stdout, or git artifacts. Resumes are processed in volatile memory. Documents are never transmitted to external APIs unless the user explicitly enables GenAI suggestions.

### Q22. How are multi-column and table-based resumes handled?
**Answer:** The parser uses `pdfplumber` bounding-box coordinates to detect columns and tables. Multi-column text is extracted column-by-column rather than line-by-line across columns, preventing sentences from interleaving. Scanned image documents trigger an automatic Tesseract OCR fallback.

### Q23. What is the total latency of the end-to-end pipeline?
**Answer:** After singleton model loading via `warm_up()`, total pipeline execution for a clean PDF takes ~200–500 ms:
- Parsing: ~80–120 ms
- Skill Extraction: ~4.5 ms
- DistilBERT Classification: ~40 ms
- Siamese Embedding & Scoring: ~55 ms
- Rule-based suggestions: < 5 ms

### Q24. How does the pipeline handle corrupted or oversized uploads?
**Answer:** `app.pipeline.run_pipeline` enforces an upfront 5 MB size limit and checks MIME magic bytes. If a file is oversized or corrupted, it returns a structured `ScoreResult` with a descriptive message in `warnings`, ensuring the Streamlit UI or API never crashes with an unhandled exception.

### Q25. What is the single biggest engineering takeaway from this project?
**Answer:** Deep learning model accuracy is only one piece of an effective hiring platform. Transparent heuristic scoring, robust multi-format document parsing, context-aware disambiguation, and strict privacy guards are what transform raw neural predictions into a reliable, fair, and actionable decision-support tool.
