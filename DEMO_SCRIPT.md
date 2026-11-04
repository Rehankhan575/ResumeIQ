# 5-Minute Live Demonstration Script

An exact step-by-step walkthrough for demonstrating the AI-Powered Resume Screening & Job Fit Analytics Platform.

---

## Pre-Demo Checklist (T-minus 5 Minutes)

1. **Terminal 1 (Environment & Preflight):**
   ```bash
   cd /Users/rehan/Desktop/DL_PROJECT
   source venv/bin/activate
   python scripts/preflight.py
   ```
   *Verify:* Confirm all 7/7 checks pass with `Preflight passed!`.

2. **Terminal 2 (Launch UI):**
   ```bash
   streamlit run app/ui/app.py
   ```
   *Verify:* Browser opens at `http://localhost:8501`. Dark obsidian theme (`#0e1117`) and font styles render cleanly.

3. **Sample Fixture Files Ready (in `tests/fixtures/`):**
   - Resumes: `ds_clean.pdf`, `accountant_table.pdf`, `nurse_scanned.pdf`
   - Job Descriptions: `jd_data_scientist.txt`, `jd_chef.txt`

---

## Timed Walkthrough (5 Minutes Total)

### Minute 0:00 – 1:00 | Architecture & Problem Statement
- **Speaker Script:**
  > "Traditional Applicant Tracking Systems (ATS) rely on brittle keyword string matching that candidates can game and that penalizes non-standard formatting while offering candidates zero feedback. We engineered an end-to-end deep learning platform that combines transformer-based semantic matching, robust multi-format document extraction, and transparent explainability across 7 distinct pipeline stages."
- **Action:**
  - Show the running dashboard at `http://localhost:8501`.
  - Point out the clean layout: Sidebar (Resume upload & JD selection), Main View (Score Overview, Highlights, Suggestions, Skill Gap, Ranking, and Model Benchmarks).

---

### Minute 1:00 – 2:15 | Single Candidate Fit: Data Scientist Resume
- **Action:**
  1. In the sidebar, under **Upload Resume**, click **Browse files** and select `tests/fixtures/ds_clean.pdf`.
  2. Under **Select Job Description**, choose **Data Scientist** (or paste `tests/fixtures/jd_data_scientist.txt`).
  3. Click **Analyze Resume**.
- **Expected Results on Screen:**
  - **ATS Score:** **`84.9 / 100`**
  - **Letter Grade:** **`B`** (High Match)
  - **Pipeline Latency:** ~250–500 ms logged in timing card.
- **Key Features to Highlight:**
  - **Component Radar / Progress:** Show that **Semantic Match (25%)** and **Keyword Coverage (20%)** are strong, while **Formatting (10/10)** received full points for clean structure.
  - **Highlights Panel:** Identified 7.4 years of experience, core skills (Python, Machine Learning, SQL, Pandas).
  - **Targeted Suggestions Panel:** Note the provider badge (`[Rule-Based Engine]` or `[Gemini 1.5 Pro]`). Shows specific suggestions (e.g., adding metrics to bullet points, addressing minor preferred skill gaps).

---

### Minute 2:15 – 3:15 | Cross-Domain Negative Control & Skill Gap Detection
- **Action:**
  1. Keep `ds_clean.pdf` uploaded.
  2. In the JD selector, switch to **Executive Chef** (or paste `tests/fixtures/jd_chef.txt`).
  3. Click **Analyze Resume**.
- **Expected Results on Screen:**
  - **ATS Score:** Plunges to **`55.5 / 100`** (A sharp **29.4 point drop**).
  - **Letter Grade:** **`F`** (Unqualified / Mismatch).
  - **Missing Skills Card:** Immediately flags **6 missing culinary skills**:
    - `culinary arts`, `kitchen sanitation`, `food safety`, `menu planning`, `haccp`, `inventory management`.
- **Speaker Script:**
  > "Notice how the system rejects domain mismatches. The score drops by nearly 30 points. Why doesn't it drop to zero? Because our 9-component scoring is transparent: the candidate still has valid contact info, an accredited degree, and clean ATS formatting, which form baseline document hygiene."

---

### Minute 3:15 – 4:00 | Multi-JD Candidate Ranking
- **Action:**
  1. Navigate to the **Multi-JD Ranking** tab in the main panel.
  2. Click **Rank Against Standard Pool**.
- **Expected Results on Screen:**
  - Interactive table displaying the candidate ranked across roles:
    1. **Data Scientist:** `84.9 / 100` (`B` Grade, 0 Missing Core Skills)
    2. **Software Engineer:** `~74.0 / 100` (`C` Grade, moderate skill overlap)
    3. **Executive Chef:** `55.5 / 100` (`F` Grade, 6 missing culinary skills)
- **Speaker Script:**
  > "A recruiter can immediately rank a single candidate against multiple open requisitions in real time using our fine-tuned vector space, preventing misplaced applications."

---

### Minute 4:00 – 4:30 | Multi-Format Robustness & Privacy Guard
- **Action:**
  1. Upload `tests/fixtures/accountant_table.pdf`.
  2. Select the **Accountant** JD.
  3. View the **Document Layout Diagnostics** in the sidebar / tabs.
- **Expected Results on Screen:**
  - Layout Diagnostics flags `has_tables: True`.
  - Sections (`experience`, `education`, `skills`) are extracted with **100% precision**.
  - Candidate PII (name, email, phone) is securely parsed into private memory fields; terminal console logs and disk logs show zero raw candidate text or personal identifiers.
- **Speaker Script:**
  > "Our hybrid parser detects tables and multi-column formats using pdfplumber coordinate analysis, ensuring table-based resumes don't corrupt text flow. Privacy is strictly preserved: no raw candidate PII is ever logged to stdout or saved in unencrypted artifacts."

---

### Minute 4:30 – 5:00 | Empirical Model Results & Viva Wrap-up
- **Action:**
  1. Click the **Model Results & Benchmarks** tab.
  2. Point to the **Classifier Comparison Table** and **Siamese Matcher Retrieval Table**.
- **Key Talking Points:**
  - **Classifier:** DistilBERT achieves **72.73% test accuracy** (0.6454 macro F1) across 24 categories. Logistic Regression provides a near-instant baseline at **66.31%**. BiLSTM underfits on this dataset size (25.13%).
  - **Matcher:** Siamese fine-tuning with Multiple Negatives Ranking Loss (MNRL) boosted **Test MRR from 0.6311 to 0.7798 (+14.87 pts)** and **nDCG@10 from 0.4532 to 0.7286 (+27.54 pts)** over zero-shot SBERT.

---

## Fallback Recovery Plan

| Scenario | Immediate Action |
| :--- | :--- |
| **Streamlit server port in use or unresponsive** | Run `pkill -f streamlit` and restart: `streamlit run app/ui/app.py --server.port 8502`. |
| **Model loading latency / timeout** | Cold-start model weights are cached in memory after first load. Pre-warm once via `python -c "import app.pipeline as p; p.warm_up()"`. |
| **Offline / No Internet Connection** | The entire system operates in local air-gapped mode. Model checkpoints reside in `artifacts/models/`. Set `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. |
| **GenAI API Key Quota Exhausted** | The platform automatically defaults to rule-based suggestions with zero crash or degraded experience. |
| **Host machine graphic failure** | Present the pre-rendered screenshots in `docs/screenshots/` and report in `artifacts/reports/results_summary.md`. |
