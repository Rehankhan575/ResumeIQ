# Resume Analyzer — UI Documentation

## Overview

The Resume Analyzer UI is built with Streamlit and styled with a custom developer-tools design system. It presents a dark, calm, and precise interface for analyzing candidate résumés against job descriptions, inspecting ATS score breakdowns, exploring highlighted evidence, reviewing AI-generated suggestions, and auditing for demographic bias.

---

## How to Run

### Prerequisites
1. Python 3.11 with virtual environment activated:
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
The application will open at `http://localhost:8501`.

### Pre-flight Verification
To verify the end-to-end pipeline across all 5 test resume fixtures and 3 job descriptions:
```bash
python scripts/preflight.py
```

### Automated UI Audit
To run the automated headless Chrome CDP layout and visual audit across 1440x900 and 1280x720:
```bash
python scripts/ui_audit.py
```

---

## Design System & Tokens

All tokens are defined in `app/ui/styles.css` as CSS variables on `:root` and applied consistently across all UI components:

| Category | Token | Value | Purpose |
| :--- | :--- | :--- | :--- |
| **Surfaces** | `--bg` | `#0A0E14` | Application canvas background |
| | `--surface` | `#111720` | Primary cards, header bar |
| | `--surface-raised` | `#161D28` | Inputs, dropdowns, code/diff blocks |
| | `--border` | `#212B38` | Hairline 1px borders |
| **Typography** | `--text` | `#E8EDF3` | Primary body text |
| | `--muted` | `#8A97A8` | Secondary labels, captions |
| | `--faint` | `#5B6878` | Component weights, metadata tags |
| **Accent** | `--accent` | `#5EEAD4` | Teal: primary button, score ring, active tab |
| | `--chart-blue` | `#7AA2F7` | Soft blue: chart data only |
| **Semantics** | `--good` | `#6EE7A0` | Green: Grade A/A-, positive deltas |
| | `--warn` | `#F5B454` | Amber: Grade B/C, medium issues |
| | `--bad` | `#F2777A` | Red/coral: Grade D/F, high severity issues |
| **Radius** | `--r-card` | `12px` | Content cards |
| | `--r-input` | `8px` | Buttons, text areas, input fields |
| **Spacing** | 8px Grid | `4, 8, 16, 24, 32, 48px` | Card padding (24px), section spacing (48px) |

### Strict P5 Design Constraints & Banned List
The interface strictly adheres to the developer-tool design philosophy:
- **No Emojis**: Emojis are strictly prohibited throughout the interface, code, and documentation. All status and category badges use text or SVG indicators.
- **No Drop Shadows**: All card, button, and popover box-shadows are disabled (`box-shadow: none !important;`). Elevation is communicated solely through surface shades (`#0A0E14` $\rightarrow$ `#111720` $\rightarrow$ `#161D28`) and hairline borders (`#212B38`).
- **No Gradients**: All backgrounds and fills are solid flat colors (`linear-gradient` forbidden).
- **No Glow / Blur**: Text and border glows are forbidden (`text-shadow: none !important; filter: drop-shadow(none) !important;`).
- **Single Teal Accent**: Interactive states, active tabs, and primary actions exclusively use the teal accent token `#5EEAD4`.

### Typography & Fonts
- **UI Text:** Inter (400 regular, 600 semibold), self-hosted as WOFF2 in `app/ui/static/fonts/Inter-Regular.woff2` and `Inter-SemiBold.woff2`. Fallback stack: `system-ui, -apple-system, sans-serif`.
- **Numbers / Code:** JetBrains Mono (400 regular), self-hosted in `app/ui/static/fonts/JetBrainsMono-Regular.woff2`. `font-variant-numeric: tabular-nums` is enforced on all numbers, chips, and table data cells.

---

## Card Container Architecture & Sibling Height Matching

To guarantee visual balance across viewport resolutions:
1. **Container Keying**: All cards are instantiated via `st.container(key="card_xxx")` and targeted directly in CSS via `.st-key-card_xxx`.
2. **Flex Row Equalization**: Streamlit `st.columns` columns are styled with `align-self: stretch !important; height: auto !important; min-height: 100% !important;` and container elements with `flex: 1 1 auto; height: 100%;`.
3. **Card Matching Targets**:
   - **Input Section**: `card_resume` and `card_jd` are guaranteed equal height within 1px (210.0px vs 210.0px at standard viewports).
   - **Hero Section**: `card_score_ring` and `card_breakdown` match height within 1px (318.8px vs 318.8px).
   - **Stat Summary Tiles**: 4 stat cards (`card_stat_0` through `card_stat_3`) match height within 1px (136.0px).

---

## Header Bar & Settings Controls

The application top bar features a clean, slim header:
- **Left**: Wordmark `Resume Analyzer` in 18px Inter semibold (`#E8EDF3`).
- **Right**: Status pill (`• Models loaded` with green indicator) and a quiet `st.popover("Settings")` button.
- **Privacy Settings**: Clicking `Settings` opens an overlay card containing transparency disclosures ("Privacy: resume text is processed in-memory only and never stored") and an opt-in checkbox for local SQLite score history tracking.

---

## Input Precedence & Modes

Both input cards provide explicit mode selection to prevent pasted or uploaded text from overriding selections silently:
- **Resume Input**: Explicit toggle (`st.segmented_control` or horizontal `st.radio`) between `Sample` and `Upload`. The active mode determines the document source; the inactive widget is hidden and never read. A status label `Using: <name>` is displayed beneath the card. Max upload size is enforced at 5 MB (`server.maxUploadSize = 5` in `.streamlit/config.toml`).
- **Job Description Input**: Explicit toggle between `Sample` and `Paste`. The active mode decides the JD source; the inactive widget is hidden and never evaluated. A status label `Using: <name>` is displayed beneath the card.

---

## Score Ring & Component Breakdown

### Score Ring
- **SVG Rendering**: Inline SVG circle with explicit background track (`#212B38`) and animated progress stroke matched to grade semantics (`#6EE7A0` for A, `#F5B454` for B/C, `#F2777A` for D/F).
- **Center Metric**: Large 48px JetBrains Mono overall composite score centered vertically and horizontally.
- **Letter Grade Chip**: Compact grade badge (`A`, `B`, `C`, `D`, `F`) positioned directly under the score within the ring geometry.
- **Sub-scores**: Below the SVG ring, two monospace metric tiles display `Job fit <score>` and `Resume quality <score>` with 11px uppercase muted labels.

### Component Breakdown
Nine horizontal progress bars detail the constituent score components:
- Quantified Achievements (10pt max)
- Semantic Match (25pt max)
- Formatting (10pt max)
- Action Verbs (5pt max)
- Contact Completeness (5pt max)
- Readability (5pt max)
- Keyword Coverage (20pt max)
- Section Completeness (10pt max)
- Experience Fit (10pt max)

---

## Unmeasurable Components & Warning Chip

When evaluating incomplete or non-standard job descriptions:
- If a JD has no extractable required skills, `keyword_coverage` is excluded from scoring and the remaining component weights are renormalised to sum to 100%.
- If a JD has no specified minimum years of experience (`min_years`), `experience_fit` is excluded and weights are renormalised.
- A descriptive warning is appended to `ScoreResult.warnings`.
- If the JD is shorter than 30 words, has 0 detectable skills, or has an unmeasurable warning, a warning chip is displayed prominently above the results:
  `JD too short or no skills detected; keyword coverage not measured`

---

## Stat Summary Tiles

Below the hero score ring and breakdown bars, four equal stat tiles present key summary metrics:
1. **Experience**: Candidate's total years of experience (e.g. `7.4 yrs`), with sub-label `N roles` (or `1 role`).
2. **Skills matched**: Count of overlapping skills vs. required skills (e.g. `12 / 16`), with sub-label `required skills in JD` (or `none detected in JD` if 0 required skills).
3. **Top category**: Full predicted occupational category name (e.g. `DATA-SCIENCE`), with sub-label `NN% confidence`.
4. **Processing time**: End-to-end pipeline execution time in seconds (e.g. `0.85s`), with sub-label `parse to score`.

---

## Navigation & Tabs

The UI provides 7 native tabs (`st.tabs`) styled with underline navigation (14px muted text, 2px teal active underline, zero radio inputs, zero pill backgrounds):

1. **Highlights**: Character-level evidence overlays displaying matched keywords in teal tint (`rgba(94, 234, 212, 0.18)`), high-impact sentences with a 2px teal underline, missing JD keywords as outlined chips, and a visual legend.
2. **Suggestions**: Generative bullet point rewrites formatted as side-by-side Before/After cards, copy-friendly cover letter in a textarea card, and an improvement plan structured by estimated score gain.
3. **Ranking**: Multi-JD ranking table sorting candidate suitability across all sample JDs, displaying score, grade chip, and missing skill badges.
4. **Skill gap**: Maps candidate missing skills to curated, URL-free courseware resources detailing title, provider, level, and estimated study hours.
5. **Bias check**: Privacy-preserving demographic bias audit displaying flags for age indicators, marital status, and gendered wording using abstract category labels without personal text leakage.
6. **History**: Opt-in historical score tracking backed by a local SQLite database storing only numeric scores, letter grades, and timestamps (zero resume text stored).
7. **Model results**: Rebuilt directly from `artifacts/reports/` and `artifacts/metrics/` with `@st.cache_data` caching:
   - **Classifier**: 4-model performance table (TF-IDF + LR, TF-IDF + MLP with `SERVED` badge, BiLSTM, DistilBERT) showing validation macro-F1, test macro-F1 with 95% bootstrap CIs, test accuracy, and generalization gap. Paired-test statistical comparison table (McNemar $\chi^2$, bootstrap $p$-values, Bonferroni threshold caveats).
   - **Matcher**: Retrieval metrics (Val/Test MRR, nDCG@10, Recall@5, Recall@10) across TF-IDF, Zero-Shot SBERT, and Fine-Tuned SBERT (`SERVED` badge), with expandable per-category breakdown.
   - **Scoring**: Component weight and sensitivity ablation table (`quantified_achievements` +7.37, `formatting` +1.25, `experience_fit` -2.26).
   - **Explainability (XAI)**: ATS sentence-level deletion (inconclusive / unvalidated) and classifier word-level deletion (faithful to linear model with cross-model transfer to served MLP). Missing files display a fallback card: `Results file missing: run scripts/collect_results.py`. Never displays internal raw keys like `matched_pair_pass`.

---

## Automated UI Layout & Visual Audit (`scripts/ui_audit.py`)

Layout and visual styling are audited via an automated test harness in `scripts/ui_audit.py` that launches a headless Google Chrome instance connected via the Chrome DevTools Protocol (CDP) to evaluate 8 distinct layout and rendering criteria across both desktop (`1440x900`) and standard laptop (`1280x720`) resolutions:

| # | Audit Criterion | Method / DOM Evaluation | 1440x900 Result | 1280x720 Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **No horizontal scroll** | `scrollWidth <= innerWidth` | `1440px <= 1440px` | `1280px <= 1280px` | **PASS** |
| **2** | **Sibling card heights equal** | Sibling height diff $\le 1$px | Inputs: 210.0px vs 210.0px (diff 0.0px)<br>Hero: 318.8px vs 318.8px (diff 0.0px) | Inputs: 210.0px vs 210.0px (diff 0.0px)<br>Hero: 318.8px vs 318.8px (diff 0.0px) | **PASS** |
| **3** | **4 stat tiles equal height** | Max height minus min height $\le 1$px | `[136.0, 136.0, 136.0, 136.0]px` (max diff 0.0px) | `[136.0, 136.0, 136.0, 136.0]px` (max diff 0.0px) | **PASS** |
| **4** | **Typography compliance** | `window.getComputedStyle(el).fontFamily` starts with Inter | All tested labels, tabs, tables compute with Inter | All tested labels, tabs, tables compute with Inter | **PASS** |
| **5** | **Zero radio tabs** | Native `[role="tab"]` count = 7, `[data-testid="stTabs"] input[type="radio"]` = 0 | 0 radio inputs, 7 native tabs | 0 radio inputs, 7 native tabs | **PASS** |
| **6** | **Score ring geometry** | SVG bbox $> 0$, stroke matches grade color | Box: `136x136px`, stroke: `#F5B454` | Box: `136x136px`, stroke: `#F5B454` | **PASS** |
| **7** | **Section left edge alignment** | Left bounding edges equal within 1px | Left edges: `[80, 80, 80, 80]` (max diff 0.0px) | Left edges: `[80, 80, 80, 80]` (max diff 0.0px) | **PASS** |
| **8** | **No clipped text containers** | `scrollHeight <= clientHeight` for single-line text containers | 0 clipped containers | 0 clipped containers | **PASS** |

Screenshots for all views and tabs are captured automatically and saved to `docs/screenshots/`:
- `initial_1440x900.png` & `initial_1280x720.png`
- `results_top_1440x900.png` & `results_top_1280x720.png`
- `tab_highlights_1440x900.png` & `tab_highlights_1280x720.png`
- `tab_suggestions_1440x900.png` & `tab_suggestions_1280x720.png`
- `tab_ranking_1440x900.png` & `tab_ranking_1280x720.png`
- `tab_skill_gap_1440x900.png` & `tab_skill_gap_1280x720.png`
- `tab_bias_check_1440x900.png` & `tab_bias_check_1280x720.png`
- `tab_history_1440x900.png` & `tab_history_1280x720.png`
- `tab_model_results_1440x900.png` & `tab_model_results_1280x720.png`

---

## UI Layout & P5 Spec Alignment (Deviations Report)

The implemented Streamlit interface in `app/ui/app.py` organizes the workflow as follows:

| Component | P5 Prototype Spec | Implemented UI (`app/ui/app.py`) | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion Controls** | Dedicated Streamlit Sidebar (`st.sidebar`) | Two-column card layout on main canvas (`col_resume`, `col_jd`) with explicit mode toggles | **Deviation:** Main canvas side-by-side cards replace sidebar to optimize screen width for split comparison |
| **Settings Controls** | Popover or Sidebar widget | Popover widget (`st.popover("Settings")`) in the top right header bar | **Aligned** |
| **Primary Action** | Button | Primary `Analyze` button (`st.button("Analyze", type="primary")`) | **Aligned** |
| **Stat Summary Tiles** | 4 Tiles (e.g. including Latency) | 4 Stat Tiles (`Experience`, `Skills matched`, `Top category`, `Processing time`) via `st.columns(4)` | **Aligned** |
| **Score Breakdown** | Visual breakdown | Circular SVG score ring with grade chip + 9 horizontal progress bars | **Aligned** |
| **Tab Navigation** | 7 Tabs | 7 native underline tabs (`st.tabs`): `Highlights`, `Suggestions`, `Ranking`, `Skill gap`, `Bias check`, `History`, `Model results` | **Aligned** |

---

## Known Limitations & Disclosures

1. **Weak Labels & Circularity**: Job description category relevance labels are derived via title-matching heuristic rules rather than human recruiter annotation; matching metrics partly reflect category separation.
2. **ATS Ground Truth**: No universal ground-truth ATS score exists; composite weights (0–100) are transparent heuristics designed for candidate actionable feedback.
3. **XAI Faithfulness**: Leave-One-Sentence-Out (LOSO) and linear feature attribution provide fast sub-second attributions; however, empirical faithfulness erasure benchmarks remain approximate on short résumés.
4. **Candidate Privacy**: Uploaded résumé text and contact information are held in transient memory only; no personal details are written to disk, SQLite, or server log files.
