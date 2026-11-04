# ResumeIQ — React Design System & UI Architecture

## 1. Overview & Aesthetics
ResumeIQ features a quiet, restrained, monochrome SaaS dashboard interface inspired by high-end dark hardware interfaces and glassmorphic dashboards.
The design avoids loud neon colors, multi-color gradients, and rainbow glows, focusing on deep obsidian surfaces, subtle 1px borders, crisp typography, and an understated sand accent (`#D9C7A0`).

---

## 2. Design Tokens

### Color Palette
- **Canvas / Root Background:** `#0A0A0B`
- **Surfaces:**
  - Base Surface: `#111113`
  - Card Surface: `#161618`
  - Subtle Surface: `#1C1C1F`
- **Borders:**
  - Standard Border: `rgba(255, 255, 255, 0.08)` (1px solid)
  - Strong Border: `rgba(255, 255, 255, 0.14)`
  - Active / Focus Border: `#D9C7A0` (sand)
- **Typography:**
  - Primary Text: `#F5F5F4` (WCAG AAA contrast against `#0A0A0B`)
  - Secondary Text: `#A1A1A6` (WCAG AA contrast)
  - Muted Text: `#6B6B72`
- **Brand Accent:**
  - Sand: `#D9C7A0` (used strictly for active navigation indicator bars, focus rings, and metric gains)
  - Sand Dim: `rgba(217, 199, 160, 0.12)`
- **Semantic Muted Statuses:**
  - Good (Grades A/B): `#7FB28A` (surface: `rgba(127, 178, 138, 0.12)`)
  - Warning (Grade C): `#D4A95A` (surface: `rgba(212, 169, 90, 0.12)`)
  - Critical (Grades D/F): `#C97B7B` (surface: `rgba(201, 123, 123, 0.12)`)

### Typography & Fonts
- **UI & Body:** `Geist Sans` (`@fontsource/geist-sans`), self-hosted
- **Display Accent:** `Instrument Serif` (`@fontsource/instrument-serif`), italic single-word accent in hero headings
- **Code & Numerics:** `Geist Mono` (`@fontsource/geist-mono`), self-hosted for scores, timestamps, and chips
- **Type Scale:** 12 / 13 / 14 / 16 / 20 / 28 / 40 / 54 px
- **Tracking:** `-0.02em` on headings; standard mono on data values

### Glass & Lighting Effects
- **Frosted Glass Panels:**
  ```css
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.015) 100%);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.12), 0 24px 48px -12px rgba(0, 0, 0, 0.75);
  ```
- **Top Ambient Spotlight:** Very dim radial light beam (`rgba(255, 255, 255, 0.05)`) radiating from the top center.
- **Film Grain Overlay:** 3% SVG fractal noise overlay on canvas with `mix-blend-mode: overlay`.

---

## 3. App Shell Structure
- **Left Navigation:**
  - 64px slim pill icon rail for persistent single-click tool switching.
  - 240px expanded navigation panel collapsible to the rail on narrow viewports.
  - Active item: highlighted pill with a vertical sand (`#D9C7A0`) indicator bar on its left edge.
  - Live count badges (e.g. `3` issues).
  - Bottom "Files are never stored" status badge.
- **Top Bar:** Breadcrumb hierarchy, global command search input (`[⌘K]`), user profile placeholder, and thin border.
- **Responsive Reflow:** On screens `< 720px`, the desktop navigation collapses to a fixed bottom navigation bar with active dot indicators.

---

## 4. Screen Specifications

1. **Analyze (Landing):**
   - Hero block with Instrument Serif italic accent word (*does*), subcopy, and white pill CTA.
   - Right-hand Evaluation Workspace card with drag-and-drop zone (5 MB limit), JD textarea with character/word counter, and sample selectors.
   - 3-step row (`01. Choose`, `02. Analyze`, `03. Improve`) and 3-tile real project verification strip.
   - 44px solid white pill button (`#F5F5F4` background, `#0A0A0B` text) with integrated loading spinner.

2. **Results (12-Column Bento Grid):**
   - **Score Card (5 cols):** Big circular ring (thin 6px stroke, white arc on dark track, centered Geist Mono numeral), letter grade badge, verdict, top fix callout (`+N pts`), and sub-score stat tiles (Job Match 55% vs Resume Quality 45%).
   - **Component Breakdown (7 cols):** 9-rubric breakdown with thin 4px bars, percentage, and points pill.
   - **Prioritized Action Items (7 cols):** Gain-sorted issues with severity dots, message, `+N pts` badge, and expandable guidance.
   - **Keywords Card (5 cols):** Matched vs missing skill chips.
   - **Classifier Inference Card (5 cols):** Softmax probabilities for domain categories.
   - **Execution Latency Card (5 cols):** Collapsible per-stage pipeline millisecond breakdown.
   - **Explainability Overlay (12 cols):** Reader-style document inspection with soft tinted token underlines and weight tooltips.
   - **Client-Side Export:** Markdown report generator with confirmation toast.

3. **Rank Jobs:** Multi-JD candidate ranking leaderboard with comparative horizontal score bars and skill coverage chips.
4. **Skill Gap:** Matched skills vs missing skills cross-referenced with a curated, URL-free courseware catalog.
5. **Bias Check:** Neutrality audit detecting ageist proxies, gendered terminology, and demographic indicators.
6. **History:** Local opt-in evaluation history table. Only stores timestamps, scores, and JD titles; zero resume text stored.
7. **Privacy:** Full architectural statement of in-memory RAM processing and zero disk persistence.

---

## 5. Security & Zero-XSS Architecture
- **Safe Text Rendering:** Resume text and highlights are rendered strictly through native React virtual DOM text nodes (`<span>{text}</span>`). `dangerouslySetInnerHTML` is strictly prohibited, ensuring any malicious `<script>` tags are rendered harmlessly as literal strings.
- **Privacy Guarantees:** File processing occurs purely in-memory. No resume payload or extracted section text is ever serialized to `localStorage` or disk logs.

---

## 6. How to Run

### Backend (FastAPI)
Run with the project virtual environment on port 8000:
```bash
./venv/bin/python -m uvicorn app.api.main:app --port 8000 --reload
```
Available endpoints:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/health`
- `http://localhost:8000/api/samples`
- `http://localhost:8000/api/analyze`
- `http://localhost:8000/api/rank`

### Frontend (React + Vite)
In a separate terminal:
```bash
cd frontend
npm run dev
```
Navigate to `http://localhost:5173`. The Vite development server automatically proxies `/api` and `/health` requests to the FastAPI backend at `http://localhost:8000`.

### Running Tests
- **Frontend Vitest Suite:**
  ```bash
  cd frontend
  npm run test
  ```
- **Frontend Type & Production Build Check:**
  ```bash
  cd frontend
  npm run build
  ```
- **Backend Pytest Suite:**
  ```bash
  ./venv/bin/pytest tests/test_api.py tests/test_schemas.py tests/test_pipeline.py
  ```

### Streamlit Fallback
The original Streamlit interface remains fully functional in the repository as a fallback:
```bash
./venv/bin/python -m streamlit run app/ui/app.py
```
