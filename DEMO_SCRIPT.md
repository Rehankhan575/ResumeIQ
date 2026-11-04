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
   - **Option A (Primary — Production React Dashboard):**
     ```bash
     ./scripts/run_demo.sh
     ```
     *Verify:* Server warms up models and opens on `http://localhost:8000`. Premium dark glassmorphism dashboard with sidebar navigation renders cleanly.
   - **Option B (Legacy Streamlit Dashboard Fallback):**
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
    - **Two-Card Ingestion Area:** Side-by-side cards with explicit mode segmented controls:
      - Left card: **Resume** (`Resume mode` segmented control toggling between `Sample` and `Upload`, selectbox preselected to `ds_clean.pdf`, showing caption `Using: ds_clean.pdf`).
      - Right card: **Job Description** (`JD mode` segmented control toggling between `Sample` and `Paste`, selectbox preselected to `jd_data_scientist.txt`, optional job title input, showing caption `Using: jd_data_scientist.txt`).
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
  - **Four Stat Tiles:**
    - `Roles held`: `3`
    - `Experience`: `7.4 yr`
    - `Matched skills`: `4 of 4`
    - `Top category`: `INFORMATION-TECHNOLOGY` (subtitle: `100% confidence`)
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
  - **Four Stat Tiles:**
    - `Roles held`: `3`
    - `Experience`: `7.4 yr`
    - `Matched skills`: `0 of 6`
    - `Top category`: `CHEF`
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
  - **Classifier Comparison & Selection:**
    - On validation macro-F1: Logistic Regression is **0.5913**, MLP is **0.6001**, DistilBERT is **0.5675**.
    - On test macro-F1: Logistic Regression is **0.5959**, MLP is **0.6181**, DistilBERT is **0.6454**.
    - **LR and MLP are indistinguishable on test** across paired tests for macro-F1 (diff +0.0222, $p=0.3480$) and accuracy (diff -0.0107, $p=0.5980$, McNemar $p=0.6265$).
    - **DistilBERT vs. LR:** Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely. Specifically: vs. LR macro-F1 +0.050 (95% CI `[+0.0056, +0.0931]`, $p=0.0360$) and accuracy +6.4 pts (+0.0642, $p=0.0220$, McNemar $p=0.0101$); vs. MLP accuracy +7.5 pts (+0.0749, $p=0.0100$, McNemar $p=0.0081$), but its macro-F1 gain over MLP (+0.0273, 95% CI `[-0.0250, +0.0901]`, $p=0.2800$) is not significant.
    - **Caveats:** Six paired tests with no multiplicity correction; one test split of 374 resumes across 24 classes (average ~15, smallest class 3, largest 18); on validation DistilBERT did NOT outperform (val macro-F1 0.5675 vs. MLP 0.6001), so validation and test disagree; DistilBERT trained only 3 epochs; BiLSTM underfits (0.1966 test macro-F1, 25.13% test accuracy).
    - **Why We Serve the MLP:** The served classifier is the MLP because the selection rule (val macro-F1) was fixed before the test set was opened; choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.
  - **Siamese Matcher:**
    - Fine-tuned SBERT achieved **0.7798 Test MRR**, representing **+14.87 points vs zero-shot** (0.7798 vs 0.6311) and **+14.65 points vs TF-IDF baseline** (0.7798 vs 0.6333); Test nDCG@10 is **0.7286** (+27.54 points vs zero-shot).
    - *Averaging method:* Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means.
    - *Caveat:* Labels are weak, title-derived and the evaluation metric is partly circular. Broad domains like BPO (test MRR 0.0326) and Consultant (test MRR 0.2032) suffer from high overlap.
  - **Scoring & Ablation:**
    - Baseline matched score is 79.64. Removing semantic match causes a -3.53 pt drop, experience fit -2.26 pts, and section completeness -2.08 pts.
    - Ablating quantified achievements yields a +7.37 pt change and formatting +1.25 pts. This positive delta is an expected mathematical consequence of weight renormalization when dropping a 0-scoring component on high-scoring resumes, not an indication that metrics harm resumes.
  - **Explainability (XAI):**
    - The explainer uses leave-one-sentence-out (LOSO) for sentence importance and TF-IDF+LR coefficients for word importance.
    - Word importance on LR is faithful to the linear model (verified by deletion) with a +0.3611 drop gap, and words picked by the LR also reduce the served model's probability (cross-model transfer) with a +0.4863 drop gap on the served MLP.
    - The composite ATS sentence faithfulness test was inconclusive (top-3 sentence removal dropped scores by 0.07 pts vs 0.11 pts for 3 random sentences on 20 validation resumes, k=3), and sentence explanations remain inconclusive and unvalidated.
  - **Closing Line:** "In summary, the platform serves the TF-IDF + MLP classifier because our selection rule was locked on validation macro-F1 before unblinding test data; switching to DistilBERT purely because it scored higher on test would constitute test-set selection. With our Siamese matcher, explainability verified through feature deletion, and deterministic privacy guards, this system provides a rigorous, transparent recruitment decision tool."

---

## Fallback Recovery Plan

| Scenario | Immediate Action |
| :--- | :--- |
| **Streamlit server port in use or unresponsive** | Run `pkill -f streamlit` and restart: `streamlit run app/ui/app.py --server.port 8502`. |
| **Model loading latency / timeout** | Cold-start model weights take ~8 s on first load. Pre-warm once via `python -c "import app.pipeline as p; p.warm_up()"`. |
| **Offline / No Internet Connection** | The entire system operates in local air-gapped mode. Model checkpoints reside in `artifacts/models/`. Set `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. |
| **GenAI API Key Quota Exhausted** | The platform automatically defaults to rule-based suggestions with zero crash or degraded experience. |
| **Host machine graphic failure** | Present the pre-rendered screenshots in `docs/screenshots/` and report in `artifacts/reports/results_summary.md`. |
