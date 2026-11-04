# 5-Minute Live Demonstration Script

An exact step-by-step walkthrough for demonstrating the AI-Powered Resume Screening & Job Fit Analytics Platform. Every click and visual described matches the actual Streamlit interface implemented in `app/ui/app.py`.

---

## Pre-Demo Checklist (T-minus 5 Minutes)

1. **Terminal 1 (Environment & Preflight):**
   ```bash
   cd /Users/rehan/Desktop/DL_PROJECT
   source venv/bin/activate
   python scripts/preflight.py
   ```
   *Verify:* Confirm all 15 fixture-JD combinations pass with 0 failures (`All preflight checks passed.`). Cold-start time is ~8 s (measured 8.31s); warm inference is 0.05s–0.59s per resume.

2. **Terminal 2 (Launch UI):**
   ```bash
   streamlit run app/ui/app.py
   ```
   *Verify:* Browser opens at `http://localhost:8501`. Dark obsidian theme (`#0e1117`), header bar, and two-card layout render cleanly.

3. **Sample Fixture Files Ready (in `tests/fixtures/`):**
   - Resumes: `ds_clean.pdf`, `accountant_table.pdf`, `nurse_scanned.pdf`
   - Job Descriptions: `jd_data_scientist.txt`, `jd_chef.txt`

---

## Timed Walkthrough (5 Minutes Total)

### Minute 0:00 – 1:00 | Architecture & Interface Layout
- **Speaker Script:**
  > "Traditional Applicant Tracking Systems (ATS) rely on brittle keyword string matching that candidates can game and that penalizes non-standard formatting while offering candidates zero feedback. We engineered an end-to-end deep learning platform that combines Siamese metric learning, multi-format document extraction, and transparent heuristic scoring across 7 sequential stages."
- **Action:**
  - Show the running dashboard at `http://localhost:8501`.
  - Point out the interface layout:
    - **Header Bar:** Wordmark "Resume Analyzer" with status badge ("Models loaded").
    - **Inline Settings Expander:** Below the header, click the `Settings` expander (displays "Save score history (opt-in)" checkbox and "Enable AI suggestions" checkbox).
    - **Two-Card Ingestion Area:** Side-by-side cards:
      - Left card: **Resume** (`Sample resume` selectbox with preselected `ds_clean.pdf`, plus file uploader).
      - Right card: **Job Description** (`Sample JD` selectbox with preselected `jd_data_scientist.txt`, paste text area, and optional job title input).
    - **Action Button:** Center `Analyze` button (`st.button("Analyze", type="primary")`).

---

### Minute 1:00 – 2:15 | Single Candidate Fit: Data Scientist Resume
- **Action:**
  1. In the **Resume** card, verify that `ds_clean.pdf` is selected in the `Sample resume` selectbox.
  2. In the **Job Description** card, verify that `jd_data_scientist.txt` is selected in the `Sample JD` selectbox.
  3. Click the primary **Analyze** button.
- **Expected Results on Screen:**
  - **Score Ring Card:** Numerical score **`85`** (exact total 84.88) with green grade chip **`B`**.
  - **Component Breakdown Card:** Nine **horizontal bars** with proportional fills and weights:
    - Semantic Match: `88%` fill (`25pt` weight)
    - Keyword Coverage: `100%` fill (`20pt` weight)
    - Formatting: `100%` fill (`10pt` weight)
    - Section Completeness: `100%` fill (`10pt` weight)
    - Experience Fit: `80%` fill (`10pt` weight)
  - **Three Stat Tiles:**
    - `Experience`: `7.4 yr`
    - `Skills Matched`: `4 of 4`
    - `Top Category`: `INFORMATION-TECHNOLOGY` (subtitle: `100% confidence`)
  - **Issues & Opportunities:** Lists prioritized improvements with estimated gain chips (e.g. `+15 pt`, `+10 pt`).
  - **Custom Tab Bar:** Click the **Suggestions** tab to view suggestion cards with provider badge (e.g. `[Rule-based Engine]` or `[Gemini]`), bullet rewrites, cover letter, and improvement plan.

---

### Minute 2:15 – 3:15 | Cross-Domain Negative Control & Heuristic Score Disclosure
- **Action:**
  1. Keep `ds_clean.pdf` selected in the **Resume** card.
  2. In the **Job Description** card, change the `Sample JD` selectbox to `jd_chef.txt`.
  3. Click **Analyze**.
- **Expected Results on Screen:**
  - **Score Ring Card:** Numerical score drops to **`56`** (exact total 55.50) with red grade chip **`F`** (A sharp **29.4 point drop**).
  - **Horizontal Component Breakdown:** Semantic match and keyword coverage bars plunge to zero.
  - **Three Stat Tiles:**
    - `Experience`: `7.4 yr`
    - `Skills Matched`: `0 of 6`
    - `Top Category`: `CHEF`
  - **Skill gap Tab:** Click the **Skill gap** tab to view **6 missing culinary skills**:
    - `culinary arts`, `kitchen sanitation`, `food safety`, `menu planning`, `haccp`, `inventory management`.
- **Speaker Script:**
  > "Notice how the score drops from 84.9 (B) to 55.5 (F). Why does an entirely unrelated resume score near 50 instead of zero? Because scores for unrelated pairs sit near 50 because the formatting and contact components give every resume free points, and the weights are transparent heuristics with no ground truth."

---

### Minute 3:15 – 4:00 | Multi-JD Candidate Ranking
- **Action:**
  1. Click the **Ranking** tab on the tab bar.
  2. View the interactive JD Ranking table ranking the candidate across open positions:
- **Expected Results on Screen:**
  - Table columns: `Rank`, `Job Title`, `Score`, `Grade`, `Missing Skills`:
    1. **Data Scientist:** Score `84.9`, Grade `B`, Missing Skills: None
    2. **Software Engineer:** Score `60.5`, Grade `D`, Missing Skills: `react`, `typescript`
    3. **Executive Chef:** Score `55.5`, Grade `F`, Missing Skills: `culinary arts`, `kitchen sanitation`, `food safety`, `menu planning`, `haccp`, `inventory management`
- **Speaker Script:**
  > "The ranking engine evaluates a candidate simultaneously against open job descriptions using our fine-tuned metric space. When JD skill requirements are unspecified, the system dynamically extracts requirements directly from JD text."

---

### Minute 4:00 – 4:30 | Layout Diagnostics, Table Handling & Privacy Audit
- **Action:**
  1. In the **Resume** card, select `accountant_table.pdf` from the `Sample resume` selectbox.
  2. In the **Job Description** card, choose `jd_data_scientist.txt`.
  3. Click **Analyze**, then click the **Bias check** tab.
- **Expected Results on Screen:**
  - **Demographics & Contact Privacy:** Confirms candidate name, phone number, and email are scrubbed before model analysis. Console logs output zero raw resume text or PII.
  - **Format Diagnostics:** Displays layout metadata: `has_tables: True`. Parser coordinate analysis preserved table cells on synthetic resume fixtures without interleaving text.

---

### Minute 4:30 – 5:00 | Empirical Model Results & Viva Wrap-up
- **Action:**
  1. Click the **Model results** tab.
  2. Review the three embedded benchmark tables:
- **Key Talking Points:**
  - **Classifier Comparison:**
    - On validation macro-F1: Logistic Regression is **0.5913**, MLP is **0.6001**, DistilBERT is **0.5675**.
    - On test macro-F1: Logistic Regression is **0.5959**, MLP is **0.6181**, DistilBERT is **0.6454**.
    - Differences are within noise on about 370 validation resumes (~15 per class across 24 classes).
    - The production model was selected by validation F1 (MLP). DistilBERT achieved the highest test number (0.6454 macro-F1, 72.73% test accuracy) but was not selected on it and was trained for only 3 epochs due to compute limits. BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).
  - **Siamese Matcher:**
    - Fine-tuned SBERT achieved **0.7798 Test MRR**, representing **+14.87 points vs zero-shot** (0.7798 vs 0.6311) and **+14.65 points vs TF-IDF baseline** (0.7798 vs 0.6333); Test nDCG@10 is **0.7286** (+27.54 points vs zero-shot).
    - *Caveat:* Labels are weak, title-derived and the evaluation metric is partly circular. Broad domains like BPO (test MRR 0.0326) and Consultant (test MRR 0.2032) suffer from high overlap.
  - **Explainability (XAI):**
    - The explainer uses leave-one-sentence-out (LOSO) for sentence importance and TF-IDF+LR coefficients for word importance.
    - The faithfulness test was inconclusive (top-3 sentence removal dropped scores by 0.07 pts vs 0.11 pts for 3 random sentences on 20 validation resumes, k=3), and the explanations are not validated.

---

## Fallback Recovery Plan

| Scenario | Immediate Action |
| :--- | :--- |
| **Streamlit server port in use or unresponsive** | Run `pkill -f streamlit` and restart: `streamlit run app/ui/app.py --server.port 8502`. |
| **Model loading latency / timeout** | Cold-start model weights take ~8 s on first load. Pre-warm once via `python -c "import app.pipeline as p; p.warm_up()"`. |
| **Offline / No Internet Connection** | The entire system operates in local air-gapped mode. Model checkpoints reside in `artifacts/models/`. Set `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. |
| **GenAI API Key Quota Exhausted** | The platform automatically defaults to rule-based suggestions with zero crash or degraded experience. |
| **Host machine graphic failure** | Present the pre-rendered screenshots in `docs/screenshots/` and report in `artifacts/reports/results_summary.md`. |
