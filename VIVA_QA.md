# Comprehensive Viva Voce Questions & Answers

Rigorous, evidence-backed answers citing exact project metrics, architectural decisions, and honest empirical limitations.

---

## Group 1: Dataset & Data Pipeline

### Q1. What dataset was used, and what was your splitting strategy?
**Answer:** We used the Kaggle Resume Dataset comprising 2,483 resumes across 24 job categories (per-class resume counts range from 22 in BPO to 120 in INFORMATION-TECHNOLOGY, a 445.5% difference; not balanced). We applied a stratified 70/15/15 train/val/test split resulting in 1,739 training, 370 validation, and 374 testing resumes. Stratification was locked with a fixed random seed to guarantee class representation across splits.

### Q2. How did you prevent data leakage during preprocessing?
**Answer:** Resumes often contain explicit title headers (e.g., "HR Manager Resume"). We implemented automated header detection and category token sanitization, stripping explicit target labels prior to text vectorization. Additionally, TF-IDF vectorizers and label encoders were fit strictly on the 1,739 training resumes and only applied as transforms to validation and test splits.

### Q3. Why did you not perform k-fold cross-validation?
**Answer:** Given compute budgets and deep learning model training times (particularly DistilBERT fine-tuning and Siamese SBERT training), a deterministic 70/15/15 stratified split was used. We document this plainly as an empirical limitation; all reported test numbers reflect completely held-out, unseen samples.

---

## Group 2: Stage 3 Classifier Comparison (Units I, III, IV)

### Q4. Why did you benchmark four different classification architectures?
**Answer:** To systematically evaluate the trade-off between model complexity, inference latency, and generalization across syllabus units: Linear models (Unit I: TF-IDF + Logistic Regression), Feedforward Networks (Unit I: TF-IDF + 2-layer MLP), Recurrent Architectures (Unit III: BiLSTM), and Pretrained Transformers (Unit IV: DistilBERT).

### Q5. What were the empirical results across the four models?
**Answer:** On validation macro-F1: Logistic Regression achieved 0.5913, MLP achieved 0.6001, and DistilBERT achieved 0.5675. On test macro-F1: Logistic Regression achieved 0.5959, MLP achieved 0.6181, and DistilBERT achieved 0.6454. LR and MLP are indistinguishable on test across paired tests for both macro-F1 (diff +0.0222, $p=0.3480$) and accuracy (diff -0.0107, $p=0.5980$, McNemar $p=0.6265$). DistilBERT vs. LR is suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely. Specifically: vs. LR macro-F1 +0.050 (95% CI `[+0.0056, +0.0931]`, $p=0.0360$) and accuracy +6.4 pts (+0.0642, $p=0.0220$, McNemar $p=0.0101$); vs. MLP accuracy +7.5 pts (+0.0749, $p=0.0100$, McNemar $p=0.0081$), but its macro-F1 gain over MLP (+0.0273, 95% CI `[-0.0250, +0.0901]`, $p=0.2800$) is not significant. Caveats: six paired tests with no multiplicity correction; one test split of 374 resumes across 24 classes (average ~15, smallest class 3, largest 18); on validation DistilBERT did NOT outperform (val macro-F1 0.5675 vs. MLP 0.6001), so validation and test disagree; DistilBERT trained only 3 epochs; the BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy). The served classifier is the MLP because the selection rule (val macro-F1) was fixed before the test set was opened; choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.

### Q6. Why did the BiLSTM perform so poorly compared to other models?
**Answer:** The training split contained 1,739 samples across 24 classes (~72 samples per class). Recurrent networks lack the strong inductive bias of pretrained self-attention and have high parameter complexity. Without a massive corpus to learn sequence transitions from scratch, the BiLSTM failed to optimize effectively, achieving only 25.13% test accuracy.

### Q7. Why are TF-IDF models so competitive with DistilBERT on validation F1, and what do paired tests show?
**Answer:** Professional resumes are heavily keyword-driven: distinctive domain vocabulary (e.g., "culinary", "GAAP", "litigation") provides linearly separable bag-of-words features for most categories. On validation macro-F1, LR achieved 0.5913, MLP 0.6001, and DistilBERT 0.5675 across 370 validation resumes (average ~15, smallest class 3, largest 18). Furthermore, bootstrap 95% confidence intervals (1,000 resamples, seed 42) on test macro-F1 show:
- TF-IDF + LR: 0.5959 (95% CI: `[0.5465, 0.6352]`, Acc: 66.31%)
- TF-IDF + MLP: 0.6181 (95% CI: `[0.5606, 0.6587]`, Acc: 65.24%)
- DistilBERT: 0.6454 (95% CI: `[0.5970, 0.6874]`, Acc: 72.73%)

Paired hypothesis tests on the 374 test set samples (`scripts/classifier_ci.py`):
1. **MLP vs. LR:** Macro-F1 diff +0.0222 (95% CI `[-0.0220, +0.0569]`, $p = 0.3480$); Accuracy diff -0.0107 (95% CI `[-0.0428, +0.0214]`, $p=0.5980$, McNemar $p=0.6265$). LR and MLP are indistinguishable on test across both metrics.
2. **DistilBERT vs. LR:** Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); macro-F1 diff +0.0495 (95% CI `[+0.0056, +0.0931]`, $p = 0.0360$); Accuracy diff +0.0642 (95% CI `[+0.0160, +0.1096]`, $p=0.0220$, McNemar $p=0.0101$).
3. **DistilBERT vs. MLP:** Macro-F1 diff +0.0273 (95% CI `[-0.0250, +0.0901]`, $p = 0.2800$, spans zero); Accuracy diff +0.0749 (95% CI `[+0.0241, +0.1257]`, $p=0.0100$, McNemar $p=0.0081$). Only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears the Bonferroni threshold (about 0.008), barely, but its macro-F1 gain over MLP is not significant.

### Q8. Why do you serve the MLP when DistilBERT scored higher on test?
**Answer:** We serve the TF-IDF + MLP classifier because the selection protocol was strictly committed before unblinding test data:
1. **Model Selection Protocol:** The selection rule (validation macro-F1) was locked before the test set was opened. On validation, TF-IDF + MLP achieved 0.6001 macro-F1 vs. DistilBERT's 0.5675. Choosing DistilBERT post-hoc because it scored higher on test would be test-set selection (data leakage of test information into model selection). A fresh, unadulterated holdout set would be strictly required to justify a switch.
2. **Paired Test Findings:**
   - LR and MLP are indistinguishable on test (macro-F1 diff +0.0222, $p=0.3480$; accuracy diff -0.0107, $p=0.5980$, McNemar $p=0.6265$).
   - DistilBERT vs. LR is suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely (vs. LR macro-F1 +0.050, 95% CI `[+0.0056, +0.0931]`, $p=0.0360$, accuracy +6.4 pts, $p=0.0220$, McNemar $p=0.0101$; vs. MLP accuracy +7.5 pts, $p=0.0100$, McNemar $p=0.0081$, while its macro-F1 gain of +0.0273, 95% CI `[-0.0250, +0.0901]`, $p=0.2800$ is not significant).
3. **Caveats Stated in the Same Place:**
   - Six paired tests with no multiplicity correction.
   - One test split of 374 resumes across 24 classes (average ~15, smallest class 3, largest 18).
   - On validation, DistilBERT did NOT outperform (val macro-F1 0.5675 vs. MLP 0.6001), so validation and test disagree.
   - DistilBERT was trained for only 3 epochs due to compute limits; the BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).

---

## Group 3: Stage 4 Siamese Semantic Matcher (Unit II)

### Q9. What architecture and loss function were used for resume-JD matching?
**Answer:** We used a Siamese bi-encoder based on `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense representations) fine-tuned with `MultipleNegativesRankingLoss` (MNRL). Resumes and job descriptions are encoded independently into a shared metric space where cosine distance reflects domain and responsibility fit.

### Q10. How much did fine-tuning improve retrieval performance?
**Answer:** Fine-tuned Siamese SBERT achieved **0.7798 Test MRR**, representing **+14.87 points vs zero-shot** (0.7798 vs 0.6311) and **+14.65 points vs TF-IDF baseline** (0.7798 vs 0.6333); Test nDCG@10 reached **0.7286** (+27.54 points vs zero-shot, +24.32 points vs TF-IDF). **Note on Evaluation:** Labels are weak, title-derived category pairings and the retrieval metric is partly circular, measuring separation between occupational clusters rather than human recruiter qualification. Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means.

### Q11. Why is Recall@k relatively low for certain categories despite high MRR?
**Answer:** In our evaluation setup, categories contain 14 to 18 resumes. Because $k=5$ can retrieve at most 5 documents, the maximum possible Recall@5 is mathematically capped at $5 / 18 \approx 27.78\%$. Mean Reciprocal Rank (MRR) evaluates whether the first true positive ranks at position 1, which reached 1.0000 for Chef, Accountant, Engineering, and Sales.

### Q12. Which categories perform worst in matching, and why?
**Answer:** BPO (Test MRR: 0.0326) and Consultant (Test MRR: 0.2032). These categories have broad, generic job descriptions (e.g., "customer handling", "strategic advice") that heavily overlap with Information Technology, Sales, and Business Development, creating semantic interference in vector space.

---

## Group 4: Stage 5 ATS Scoring & Explainability (Units II & V)

### Q13. What are the 9 components of your ATS scoring formula, and how are sub-scores computed?
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

In the UI result hero, the score is decomposed into two orthogonal sub-scores:
- **Job fit (55% total weight):** $(0.25 \cdot c_{\text{sem}} + 0.20 \cdot c_{\text{kw}} + 0.10 \cdot c_{\text{exp}}) / 0.55 \times 100$. Measures role-specific qualification.
- **Resume quality (45% total weight):** Renormalized combination of the remaining 6 structural and readability components. Measures formatting and presentation clarity.
- **Invariant:** $0.55 \cdot \text{Job fit} + 0.45 \cdot \text{Resume quality} \equiv \text{Total ATS Score}$.

### Q14. Why does an unrelated resume-JD pair score around 50 rather than 0?
**Answer:** Scores for unrelated pairs sit near 50 (e.g., Data Scientist vs Executive Chef scores 55.5) because the formatting (10%) and contact components (5%) give every resume free points, and the weights are transparent heuristics with no ground truth. Pure job-fit components (semantic match and keyword coverage) plunge to zero, bringing the grade to F.

### Q15. What did the ablation study reveal about individual scoring components?
**Answer:** The ablation study removes one component at a time, renormalizing the remaining weights to 1.0 across test pairs (mean baseline: 79.64). Removing Semantic Match drops scores by -3.53 pts (to 76.11), Section Completeness by -2.08 pts (to 77.56), and Experience Fit by -2.26 pts (to 77.38). Conversely, ablating Quantified Achievements causes a +7.37 pt change (to 87.01) and Formatting +1.25 pts (to 80.89). This positive sign does not imply metrics hurt resume quality; it is a purely mathematical consequence of weight renormalization when removing a component where test resumes scored 0% (zero quantified bullet points), shifting its 10% weight into the remaining high-scoring components (which average ~88%).

### Q16. What explainability methods does the platform use, and how did the faithfulness tests turn out?
**Answer:** The platform uses two complementary XAI techniques:
1. **Sentence Importance for ATS Score (LOSO):** Leave-one-sentence-out on semantic match. In faithfulness testing (20 validation resumes, k=3), removing the top-3 sentences caused a 0.07 pt drop compared to 0.11 pts for 3 random sentences. This test was inconclusive, and sentence-level ATS explanations are inconclusive and unvalidated.
2. **Word Importance for Classifier (Linear Coefficients):** Feature attribution uses TF-IDF + Logistic Regression linear model coefficients (`clf_lr.coef_`) for the predicted category. In a 50-resume validation deletion test ($k=10$ words, seed 42) evaluated against a strong random baseline of 10 in-document words excluding `ENGLISH_STOP_WORDS`:
   - *LR Model (Self-Attribution):* Removing top-10 LR words dropped class probability by **0.3590** (35.90 percentage points) versus **-0.0021** (-0.21 percentage points) for random non-stopwords, yielding a net attribution gap of **+0.3611**.
   - *Served Classifier `classify()` [TF-IDF + MLP] (Cross-Model Transfer):* Removing the same top-10 LR words dropped the served MLP's probability by **0.4921** (49.21 percentage points) versus **0.0058** (0.58 percentage points) for random non-stopwords, yielding a net attribution gap of **+0.4863**.
   - *Verdict:* For LR, word importance is **faithful to the linear model (verified by deletion)**. For the served MLP, **words picked by the LR also reduce the served model's probability (cross-model transfer)**. Composite ATS score sentence explanations remain inconclusive and unvalidated.

---

## Group 5: Stage 2 Taxonomy Skill Extraction & Disambiguation Rules (NER Fine-Tuning Not Implemented)

### Q17. How does your entity extraction work without fine-tuned BERT-NER?
**Answer:** NER fine-tuning was NOT implemented; entity extraction is taxonomy-based utilizing 1,668 skills (488 with aliases, 2,404 total terms, 833 active skills in skills_stats.csv), longest-match-first n-gram tokenization (up to 5-grams), and 36 ambiguous term disambiguation rules in ambiguous_terms.csv. This yields a deterministic 4.45 ms extraction latency on collapsed val text with zero GPU dependency.

### Q18. How was the false positive "RN" $\rightarrow$ "React Native" resolved?
**Answer:** In `nurse_scanned.pdf`, "RN" (Registered Nurse) was erroneously mapped to React Native via an alias. We tested an ambiguous context rule, but discovered skills-section extraction bypassed alias checks. We cleanly resolved it by removing `rn` from React Native aliases in `data/taxonomies/skills.csv`, eliminating the false positive while preserving "React Native" in software contexts.

### Q19. How do you disambiguate single-letter programming languages like "C" or "R"?
**Answer:** Case-sensitivity and context guards: "C" and "R" must be strictly uppercase and located in a designated skills section or list context. "R" is rejected if adjacent to "&" or "D" (R&D trap). "Go" requires capitalization ("Go", "GO", "Golang") and is rejected if adjacent to "-to-market".

---

## Group 6: GenAI & Suggestions (Unit V)

### Q20. How does your suggestion module prevent LLM hallucinations?
**Answer:** We employ a Grounded Hallucination Guard. The LLM prompt is injected only with extracted candidate skills and verified missing JD skills. Generated suggestions are parsed and checked against the candidate's actual background; the model is prohibited from inventing unearned credentials or degrees.

### Q21. What happens if the user runs the system without an LLM API key?
**Answer:** The system features graceful degradation: if no `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` is present, it executes a deterministic rule-based engine. This analyzes the lowest-scoring components (e.g., missing metrics, low action verb count, missing core skills) and generates actionable bullet points offline.

---

## Group 7: System Design, Privacy & Operations

### Q22. How is candidate privacy protected across the platform?
**Answer:** Zero PII is logged. Candidate names, phone numbers, email addresses, and full resume texts are never written to disk logs, terminal stdout, or git artifacts. Resumes are processed in volatile memory. Documents are never transmitted to external APIs unless the user explicitly enables GenAI suggestions.

### Q23. How are multi-column and table-based resumes handled?
**Answer:** The parser uses `pdfplumber` bounding-box coordinates to detect columns and tables. Multi-column text is extracted column-by-column rather than line-by-line across columns, preventing sentences from interleaving. Scanned image documents trigger an automatic Tesseract OCR fallback.

### Q24. What is the total latency of the end-to-end pipeline?
**Answer:** Cold-start model loading takes about 8 s (measured 8.31 s online, 7.95 s offline). Once warm, end-to-end processing takes 0.06 to 0.6 s per resume (measured 0.05–0.07 s for clean DOCX, 0.13–0.17 s for clean PDF, 0.20–0.59 s for table-based PDF). Scanned image PDFs take about 2.8 s (measured 2.70–2.88 s) because of Tesseract OCR rasterization at 300 DPI.

### Q25. How does the pipeline handle corrupted or oversized uploads?
**Answer:** `app.pipeline.run_pipeline` enforces an upfront 5 MB size limit and checks MIME magic bytes. If a file is oversized or corrupted, it returns a structured `ScoreResult` with a descriptive message in `warnings`, ensuring the Streamlit UI or API never crashes with an unhandled exception.

### Q26. What is the single biggest engineering takeaway from this project?
**Answer:** Deep learning model accuracy is only one piece of an effective hiring platform. Transparent heuristic scoring, robust multi-format document parsing, context-aware disambiguation, and strict privacy guards are what transform raw neural predictions into a reliable, fair, and actionable decision-support tool.

### Q27. How do you evaluate parser and entity extraction robustness on arbitrary candidate documents?
**Answer:** We implemented `scripts/eval_robustness.py`, an audit script that processes folders of PDF and DOCX files (default: `data/real_samples/`). It extracts pages, detection method (native vs OCR fallback), latency, sections detected, skill count, and years of experience without printing or logging any candidate text, names, emails, or phone numbers. Across the test fixture suite, it achieved 100% extraction success with 0 unhandled exceptions.
