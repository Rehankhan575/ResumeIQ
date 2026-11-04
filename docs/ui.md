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

---

## Design System & Tokens

All tokens are defined in `app/ui/styles.css` as CSS variables on `:root` and applied consistently across all UI components:

| Category | Token | Value | Purpose |
|----------|-------|-------|---------|
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

### Typography & Fonts
- **UI Text:** Inter (400 regular, 600 semibold), self-hosted as WOFF2 in `app/ui/static/fonts/Inter-Regular.woff2` and `Inter-SemiBold.woff2`. Fallback stack: `system-ui, -apple-system, sans-serif`.
- **Numbers / Code:** JetBrains Mono (400 regular), self-hosted in `app/ui/static/fonts/JetBrainsMono-Regular.woff2`. `font-variant-numeric: tabular-nums` is enforced on all numbers and table data cells.

---

## Navigation & Tabs

The UI provides 7 tabs styled with underline navigation (muted text, 2px teal active underline, zero pill backgrounds):

1. **Highlights**: Character-level evidence overlays displaying matched keywords in teal tint (`rgba(94, 234, 212, 0.18)`), high-impact sentences with a 2px teal underline, missing JD keywords as outlined chips, and a visual legend.
2. **Suggestions**: Generative bullet point rewrites formatted as side-by-side Before/After cards, copy-friendly cover letter in a textarea card, and an improvement plan structured by estimated score gain.
3. **Ranking**: Multi-JD ranking table sorting candidate suitability across all sample JDs, displaying score, grade chip, and missing skill badges.
4. **Skill gap**: Maps candidate missing skills to curated, URL-free courseware resources detailing title, provider, level, and estimated study hours.
5. **Bias check**: Privacy-preserving demographic bias audit displaying flags for age indicators, marital status, and gendered wording using abstract category labels without personal text leakage.
6. **History**: Opt-in historical score tracking backed by a local SQLite database storing only numeric scores, letter grades, and timestamps (zero resume text stored).
7. **Model results**: Benchmark metric tables for Classifier, Matcher, and ATS Scorer with right-aligned tabular numbers, 11px uppercase headers, and transparent honesty disclosures.

---

## UI Layout & P5 Spec Alignment (Deviations Report)

The implemented Streamlit interface in `app/ui/app.py` organizes the workflow as follows:

| Component | P5 Prototype Spec | Implemented UI (`app/ui/app.py`) | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion Controls** | Dedicated Streamlit Sidebar (`st.sidebar`) | Two-column card layout on main canvas (`col_resume`, `col_jd`) | **Deviation:** Main canvas side-by-side cards replace sidebar to optimize screen width for split comparison |
| **Settings Controls** | Popover or Sidebar widget | Inline expandable drawer (`st.expander("Settings", expanded=False)`) directly below header | **Deviation:** Inline expander below header preserves single-column vertical flow |
| **Primary Action** | Button | Primary `Analyze` button (`st.button("Analyze", type="primary")`) | Aligned |
| **Stat Summary Tiles** | 4 Tiles (e.g. including Latency) | 3 Stat Tiles (`Experience`, `Skills Matched`, `Top Category`) via `st.columns(3)` | **Deviation:** 3 tiles rendered; latency logged in execution toast/diagnostics |
| **Score Breakdown** | Visual breakdown | Circular SVG score ring with grade chip + 9 horizontal progress bars | Aligned |
| **Tab Navigation** | 7 Tabs | 7 custom styled tabs: `Highlights`, `Suggestions`, `Ranking`, `Skill gap`, `Bias check`, `History`, `Model results` | Aligned |

---

## Known Limitations & Disclosures

1. **Weak Labels & Circularity**: Job description category relevance labels are derived via title-matching heuristic rules rather than human recruiter annotation; matching metrics partly reflect category separation.
2. **ATS Ground Truth**: No universal ground-truth ATS score exists; composite weights (0–100) are transparent heuristics designed for candidate actionable feedback.
3. **XAI Faithfulness**: Leave-One-Sentence-Out (LOSO) and linear feature attribution provide fast sub-second attributions; however, empirical faithfulness erasure benchmarks remain approximate on short résumés.
4. **Candidate Privacy**: Uploaded résumé text and contact information are held in transient memory only; no personal details are written to disk, SQLite, or server log files.

