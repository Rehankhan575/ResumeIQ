# ResumeIQ — UI Documentation

## Overview

ResumeIQ provides a clean, projector-readable interface for analyzing candidate résumés against job descriptions, inspecting ATS score breakdowns, exploring highlighted evidence, reviewing sentence rewrites, and auditing for demographic bias.

The user interface uses native Streamlit components styled via `.streamlit/config.toml` and a concise stylesheet (`app/ui/styles.css` under 40 lines). The light theme is optimized for presentation visibility on projectors and high-resolution monitors.

---

## How to Run

### Prerequisites
1. Activate the Python 3.11 virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Verify dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Launching the Application
```bash
streamlit run app/ui/app.py
```
The interface opens at `http://localhost:8501`.

### Pre-flight Verification
Run the end-to-end verification across all 5 test resume fixtures and 3 job descriptions:
```bash
python scripts/preflight.py
```

### Automated UI Visual & DOM Audit
Run the automated headless Chrome CDP visual and layout audit:
```bash
python scripts/ui_audit.py
```

---

## Theme & Design System

The application styling is configured in `.streamlit/config.toml` and `app/ui/styles.css`:

| Token / Setting | Value | Purpose |
| :--- | :--- | :--- |
| **`backgroundColor`** | `#FFFFFF` | Main canvas background |
| **`secondaryBackgroundColor`** | `#F3F5FA` | Card backgrounds, inputs, table zebra rows |
| **`textColor`** | `#111827` | Primary dark body text (high contrast) |
| **`primaryColor`** | `#4F46E5` | Indigo accent: primary button, progress bars, active tab |
| **Semantic Green** | `#15803D` | Grade A and B, positive impact, matched keywords |
| **Semantic Amber** | `#B45309` | Grade C, medium warnings |
| **Semantic Red** | `#B91C1C` | Grade D and F, high severity issues, missing keywords |

### Strict Typography & Contrast Rules
- **Body Text**: 16px default font-size across all narrative text.
- **Minimum Font-Size**: Every visible text node in the interface computes to at least **14px** (enforced on captions, table cells, metric labels, and helper text).
- **Headings & Titles**:
  - Page Title: 36px / 700 bold (`ResumeIQ`).
  - Section Headings: 24px semibold.
  - Card Titles: 18px / 600 semibold in sentence case.
- **Contrast**: All text achieves at least **4.5:1** contrast ratio against its nearest opaque background (WCAG AA compliant).
- **Prohibitions**: No emojis anywhere in the interface, no uppercase letter-spaced labels, no custom font downloads, no drop shadows, and no height hacks.

---

## Page Structure

### 0. Application Header & Sidebar
- **Product Name Constant**: `APP_NAME = "ResumeIQ"`.
- **Sidebar (Collapsed by default)**:
  - Settings: Opt-in checkbox for SQLite score history, toggle for GenAI bullet suggestions.
  - Privacy disclosure: "Privacy: resume text is processed in-memory only and never stored."
  - System status indicator: "System status: Models loaded".
  - No custom header bar or expander drawer on the main canvas.

### 1. Hero Section
- **Page Title**: `ResumeIQ` in 36px bold.
- **Subtitle**: "See how an applicant tracking system (ATS) scores your resume against a job description, and exactly what to fix."
- **Three Step Cards** in a horizontal row (`st.columns(3)` with `st.container(border=True)`):
  1. `1. Choose`: Upload a resume and paste a job description, or try a sample.
  2. `2. Analyze`: We read the resume, match it to the job and score it out of 100.
  3. `3. Improve`: Fix the top issues, then run it again to see your score change.

### 2. Inputs Section
Two bordered cards (`st.columns(2)` with `st.container(border=True)`):
- **Your resume**: `st.file_uploader` (PDF, DOCX, PNG, JPG; help: "Processed in memory, never stored") plus `st.selectbox` ("Or try a sample resume").
- **Job description**: `st.text_area` (height 220, placeholder "Paste the job description here") plus `st.selectbox` ("Or try a sample job description").
- **Precedence Rule**: An upload or pasted text **always wins** over a selected sample.
- **Primary Action**: Full-width primary button `Analyze my resume` (`width="stretch"`).
- **First Load**: Auto-runs the preselected sample pair (`ds_clean.pdf` + `jd_data_scientist.txt`) and displays an info callout: "You are seeing a sample analysis. Upload your resume and a job description, then click Analyze."

### 3. Verdict Row
A two-column verdict section (`st.columns([1, 2])`):
- **Left Card: "Your ATS score"**:
  - Circular SVG score ring with grade-band progress stroke: `#15803D` (A/B), `#B45309` (C), or `#B91C1C` (D/F).
  - Centered 36px score and letter grade chip.
  - Verdict word under the ring: `Excellent` (A), `Good` (B), `Fair` (C), `Needs work` (D), or `Poor` (F).
  - Top fix highlight: `Top fix: <first issue message> (+<gain> pts)`.
  - Sub-scores with plain-language explanations:
    - **Job match** (`NN / 100`): Matches skills, experience, and semantic fit for this specific job (55% of composite score).
    - **Resume quality** (`NN / 100`): Evaluates formatting, metrics, action verbs, sections, and readability (45% of composite score).
- **Right Card: "What makes up your score"**:
  - Native `st.dataframe` with columns `Component | Your score | Worth`.
  - `Your score` rendered via `st.column_config.ProgressColumn` (0–100) using primary `#4F46E5`.
  - Rows sorted by `Worth` descending.
  - Caption: "Worth is how many of the 100 points each part carries."

### 4. Summary Metric Tiles
Four equal bordered tiles via `st.metric(border=True)`:
1. **Experience**: e.g. `7.4 yrs`.
2. **Required skills found**: e.g. `4 of 4`.
3. **Closest job category**: e.g. `INFORMATION-…` with full title and confidence in the caption (`INFORMATION-TECHNOLOGY (39%)`).
4. **Analysis time**: e.g. `0.98s`.

### 5. Tabs
Eight native tabs (`st.tabs`) in exact order, each starting with ONE plain intro sentence:
1. **What to fix**: Table of issues with columns `Gain (pts) | What to fix | Area` sorted by estimated score gain descending, followed by the `Download report` button.
2. **Resume highlights**: Evidence overlay preview with colored markdown legend (`:green[■ Helps the match]`, `:red[■ Missing or weak]`, `:green[━━ High-impact sentence]`).
3. **Rewrite suggestions**: Side-by-side Before/After cards for bullet points, attribution disclosure ("Built-in rules (no AI model used)..."), and a collapsed expander for "Step-by-step plan (same fixes as above)".
4. **Compare jobs**: Multi-JD ranking table evaluating suitability across all sample JDs.
5. **Skill gaps**: Targeted courseware recommendations mapping missing required skills to learning resources.
6. **Fairness check**: Privacy screening for age, marital, or gendered wording indicators.
7. **History**: Opt-in historical score tracking backed by local SQLite storage.
8. **Model results**: Intro sentence ("How the models behind ResumeIQ were tested on held-out data") and native `st.dataframe` tables for Classifier performance (with CIs), Paired statistical tests, Matcher retrieval metrics, Scoring ablation, and XAI faithfulness evaluations (LOSO sentence deletion noted as inconclusive).

### 6. Pipeline Summary
A bottom expander `How ResumeIQ works` summarizing the end-to-end pipeline in one line using served models from `PROJECT_CONTEXT.md`:
- **Parse** (PyMuPDF / python-docx / Tesseract OCR) →
- **Extract** (Rule-based taxonomy & regex entities) →
- **Classify** (TF-IDF + MLP) →
- **Match** (Fine-Tuned Siamese SBERT) →
- **Score** (9-component heuristic ATS composite scoring) →
- **Explain** (Leave-One-Sentence-Out deletion & linear attribution) →
- **Suggest** (Rule-based bullet rewriter & template cover letter)

---

## Automated CDP Audit Results (`scripts/ui_audit.py`)

Headless Google Chrome connected via Chrome DevTools Protocol (CDP) evaluated all design and layout criteria at `1440x900` and `1280x720`:

| Audit Check | 1440x900 Evaluation | 1280x720 Evaluation | Status |
| :--- | :--- | :--- | :--- |
| **a) Minimum font-size $\ge 14$px** | 0 text nodes $< 14$px | 0 text nodes $< 14$px | **PASS** |
| **b) Text contrast ratio $\ge 4.5:1$** | 0 low-contrast text nodes | 0 low-contrast text nodes | **PASS** |
| **c) 1440x900 fold contains Hero + Inputs + Analyze** | Button bottom: `770.4px` ($\le 900$px) | N/A (tested at 1440x900) | **PASS** |
| **d1) No horizontal scroll** | `scrollWidth=1440px <= 1440px` | `scrollWidth=1280px <= 1280px` | **PASS** |
| **d2) Zero exceptions** | 0 `.stException` elements | 0 `.stException` elements | **PASS** |
| **d3) First load not empty** | Score ring & verdict tables present | Score ring & verdict tables present | **PASS** |

---

## Screenshot Review & Quality Scoring

| View / Tab | Screenshot File | Score (1–10) | Visual Observations & Defects |
| :--- | :--- | :---: | :--- |
| **First Load Viewport** | `initial_1440x900.png` | **9.0 / 10** | High contrast, crisp 36px title, clean 3 step cards, inputs and Analyze button completely above the 900px fold. Left card height slightly shorter than right card due to textarea height. |
| **First Load (1280x720)** | `initial_1280x720.png` | **9.5 / 10** | Scales cleanly to standard 720p laptop resolution; Analyze button sits right at bottom fold. |
| **Verdict Row & Top** | `results_top_1440x900.png` | **9.5 / 10** | Score ring is rendered with semantic green `#15803D`, centered 85 score, "Good" verdict, top fix callout, and ProgressColumn dataframe with Worth sorting. |
| **What to fix** | `tab_what_to_fix_1440x900.png` | **9.5 / 10** | Clear 3-column table (`Gain (pts) \| What to fix \| Area`), clean download button, 4 stat tiles neatly aligned above. |
| **Resume highlights** | `tab_highlights_1440x900.png` | **9.0 / 10** | Coloured markdown legend renders cleanly; transient memory privacy mode indicated. |
| **Rewrite suggestions** | `tab_suggestions_1440x900.png` | **9.5 / 10** | Side-by-side Before/After containers, proper rule-based attribution note, collapsible step-by-step plan. |
| **Compare jobs** | `tab_ranking_1440x900.png` | **9.5 / 10** | Multi-JD ranking dataframe displays ATS scores, letter grades, and missing skill badges clearly. |
| **Skill gaps** | `tab_skill_gap_1440x900.png` | **9.0 / 10** | Clean status callout indicating all required skills are present for this fixture. |
| **Fairness check** | `tab_bias_check_1440x900.png` | **9.0 / 10** | Light green status banner confirming no demographic bias signals detected. |
| **History** | `tab_history_1440x900.png` | **9.0 / 10** | Explanatory note indicating history is opt-in via Settings. |
| **Model results** | `tab_model_results_1440x900.png` | **9.5 / 10** | Native dataframes for classifier metrics, paired statistical tests, matcher retrieval, scoring ablation, and XAI evaluations. High contrast and easily readable on a projector. |
