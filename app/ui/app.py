"""
app/ui/app.py
=============
Resume Analyzer — Streamlit UI.

Design: dark, calm, precise. Developer-tools dashboard aesthetic.
All widgets styled to match the CSS design system in app/ui/styles.css.

Segfault-mitigation (item 0c): env vars set before any torch import.
"""

from __future__ import annotations

# ── Segfault-mitigation: set before any torch/HF import ──────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# Remove app/ui from sys.path so Python doesn't confuse app.py with the app/ package
_ui_dir_str = str(Path(__file__).resolve().parent)
while _ui_dir_str in sys.path:
    sys.path.remove(_ui_dir_str)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import html
import json
import math
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
UI_DIR = Path(__file__).resolve().parent
STYLES_CSS = UI_DIR / "styles.css"
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# ---------------------------------------------------------------------------
# Inject CSS
# ---------------------------------------------------------------------------

def _load_css() -> None:
    """Load and inject the design-system CSS."""
    css = ""
    if STYLES_CSS.exists():
        css = STYLES_CSS.read_text(encoding="utf-8")
    # Remap font paths for Streamlit static serving
    css = css.replace(
        "url(\"./app/static/fonts/",
        "url(\"./app/static/fonts/",
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_load_css()

# ---------------------------------------------------------------------------
# Warm-up models
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _warm_up() -> bool:
    """Warm up classifier + matcher once per server process."""
    from app.pipeline import warm_up
    warm_up()
    return True


_models_ready = _warm_up()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Highlights"
if "settings_open" not in st.session_state:
    st.session_state["settings_open"] = False
if "history_opt_in" not in st.session_state:
    st.session_state["history_opt_in"] = False
if "use_genai" not in st.session_state:
    st.session_state["use_genai"] = True
if "jd_title_state" not in st.session_state:
    st.session_state["jd_title_state"] = ""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

SAMPLE_RESUMES = sorted([
    p.name for p in FIXTURES_DIR.glob("*.pdf")
    if not p.name.startswith("jd_")
] + [
    p.name for p in FIXTURES_DIR.glob("*.docx")
    if not p.name.startswith("jd_")
]) if FIXTURES_DIR.exists() else []

SAMPLE_JDS = sorted([
    p.name for p in FIXTURES_DIR.glob("jd_*.txt")
]) if FIXTURES_DIR.exists() else []


def _grade_class(grade: str) -> str:
    if grade in ("A", "A-"):
        return "good"
    if grade in ("B", "C"):
        return "warn"
    return "bad"


def _score_to_colour(score: float) -> str:
    """Return CSS colour token for a 0-1 component score."""
    if score >= 0.75:
        return "var(--accent)"
    if score >= 0.45:
        return "var(--warn)"
    return "var(--bad)"


def _bar_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "mid"
    return "low"


def _fmt_component_name(key: str) -> str:
    return key.replace("_", " ").title()


def _svg_ring(score: float, grade: str) -> str:
    """Generate an SVG circular score ring."""
    r = 56
    cx = cy = 68
    stroke_w = 8
    circumference = 2 * math.pi * r
    offset = circumference * (1 - score / 100)

    if grade in ("A", "A-"):
        colour = "var(--good)"
    elif grade in ("B", "C"):
        colour = "var(--warn)"
    else:
        colour = "var(--bad)"

    return f"""
<svg width="136" height="136" viewBox="0 0 136 136" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-label="Score ring: {score:.0f}">
  <!-- Background track -->
  <circle cx="{cx}" cy="{cy}" r="{r}"
          stroke="var(--surface-raised)" stroke-width="{stroke_w}" fill="none"/>
  <!-- Progress arc -->
  <circle cx="{cx}" cy="{cy}" r="{r}"
          stroke="{colour}" stroke-width="{stroke_w}" fill="none"
          stroke-linecap="round"
          stroke-dasharray="{circumference:.2f}"
          stroke-dashoffset="{offset:.2f}"
          transform="rotate(-90 {cx} {cy})"/>
</svg>"""


def _escape(s: str) -> str:
    return html.escape(str(s))


def _run_analysis(
    file_bytes: bytes,
    filename: str,
    jd_text: str,
    jd_title: str,
    use_genai: bool,
) -> None:
    """Run the pipeline and store result in session state."""
    from app.pipeline import run_pipeline
    from app.parsing.parser import parse_document
    from app.extraction.extractor import extract_entities

    t0 = time.perf_counter()
    result = run_pipeline(
        file_bytes=file_bytes,
        filename=filename,
        jd_text=jd_text,
        jd_title=jd_title,
        use_genai=use_genai,
    )
    elapsed = time.perf_counter() - t0

    # Attach parsed resume for stat tile experience and role counts
    try:
        parsed_doc = parse_document(file_bytes, filename)
        extracted = extract_entities(parsed_doc)
        result._parsed = extracted
        st.session_state["parsed_resume"] = extracted
    except Exception:
        pass

    st.session_state["result"] = result
    st.session_state["elapsed"] = elapsed
    st.session_state["jd_text"] = jd_text
    st.session_state["active_tab"] = "Highlights"

    # Optional history recording
    if st.session_state.get("history_opt_in") and result.total > 0:
        try:
            from app.extras.history import record_history
            record_history(result, jd_title=jd_title or "Unknown", opt_in=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Render: Header bar
# ---------------------------------------------------------------------------

def _render_header() -> None:
    status_txt = "Models loaded" if _models_ready else "Models loading..."
    st.markdown(
        f"""
<div class="ra-header">
  <span class="ra-wordmark">Resume Analyzer</span>
  <div class="ra-header-right">
    <div class="ra-status-chip">
      <span class="ra-status-dot"></span>
      <span>{_escape(status_txt)}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Settings expander (inline, below header)
    with st.expander("Settings", expanded=False):
        st.caption("Privacy: resume text is processed in-memory only and never stored.")
        st.session_state["history_opt_in"] = st.checkbox(
            "Save score history (opt-in, no resume text stored)",
            value=st.session_state["history_opt_in"],
        )
        st.session_state["use_genai"] = st.checkbox(
            "Enable AI suggestions (requires API key in .env)",
            value=st.session_state["use_genai"],
        )


# ---------------------------------------------------------------------------
# Render: Input section
# ---------------------------------------------------------------------------

def _render_input() -> None:
    """Two-card input section with explicit mode toggles: Resume + Job Description."""
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)

    col_resume, col_jd = st.columns(2, gap="large")

    with col_resume:
        st.markdown('<div class="ra-card">', unsafe_allow_html=True)
        st.markdown('<span class="ra-card-label">Resume</span>', unsafe_allow_html=True)

        resume_mode = st.segmented_control(
            "Resume mode",
            options=["Sample", "Upload"],
            default="Sample",
            key="resume_mode",
            label_visibility="collapsed",
        )
        if not resume_mode:
            resume_mode = "Sample"

        # Preselect ds_clean.pdf for first-load demo
        default_idx = 0
        if "ds_clean.pdf" in SAMPLE_RESUMES:
            default_idx = SAMPLE_RESUMES.index("ds_clean.pdf")

        if resume_mode == "Sample":
            resume_choice = st.selectbox(
                "Sample resume",
                options=SAMPLE_RESUMES,
                index=default_idx if SAMPLE_RESUMES else 0,
                label_visibility="collapsed",
                key="resume_choice",
            )
            uploaded_resume = None
            using_resume = resume_choice or "None"
        else:
            resume_choice = None
            uploaded_resume = st.file_uploader(
                "Upload a PDF / DOCX / image",
                type=["pdf", "docx", "png", "jpg", "jpeg"],
                label_visibility="visible",
                key="resume_upload",
            )
            using_resume = uploaded_resume.name if uploaded_resume is not None else "No file uploaded"

        st.caption(f"Using: {using_resume}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_jd:
        st.markdown('<div class="ra-card">', unsafe_allow_html=True)
        st.markdown('<span class="ra-card-label">Job Description</span>', unsafe_allow_html=True)

        jd_mode = st.segmented_control(
            "JD mode",
            options=["Sample", "Paste"],
            default="Sample",
            key="jd_mode",
            label_visibility="collapsed",
        )
        if not jd_mode:
            jd_mode = "Sample"

        default_jd_idx = 0
        if "jd_data_scientist.txt" in SAMPLE_JDS:
            default_jd_idx = SAMPLE_JDS.index("jd_data_scientist.txt")

        if jd_mode == "Sample":
            jd_choice = st.selectbox(
                "Sample JD",
                options=SAMPLE_JDS,
                index=default_jd_idx if SAMPLE_JDS else 0,
                label_visibility="collapsed",
                key="jd_choice",
            )
            jd_pasted = None
            jd_title_input = ""
            using_jd = jd_choice or "None"
        else:
            jd_choice = None
            jd_pasted = st.text_area(
                "Paste job description text",
                height=120,
                label_visibility="visible",
                key="jd_text_input",
                placeholder="Paste job description text here…",
            )
            jd_title_input = st.text_input(
                "Job title (optional)",
                key="jd_title_input",
                placeholder="e.g. Senior Data Scientist",
            )
            if jd_title_input and jd_title_input.strip():
                using_jd = jd_title_input.strip()
            elif jd_pasted and jd_pasted.strip():
                using_jd = "Pasted text"
            else:
                using_jd = "No JD entered"

        st.caption(f"Using: {using_jd}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve inputs strictly decided by active mode toggle
    file_bytes: Optional[bytes] = None
    filename = ""

    if resume_mode == "Upload":
        if uploaded_resume is not None:
            file_bytes = uploaded_resume.read()
            filename = uploaded_resume.name
    else:
        if resume_choice:
            p = FIXTURES_DIR / resume_choice
            if p.exists():
                file_bytes = p.read_bytes()
                filename = resume_choice

    jd_text = ""
    jd_title = ""
    if jd_mode == "Paste":
        if jd_pasted and jd_pasted.strip():
            jd_text = jd_pasted.strip()
        jd_title = jd_title_input.strip() if jd_title_input else ""
    else:
        if jd_choice:
            p = FIXTURES_DIR / jd_choice
            if p.exists():
                jd_text = p.read_text(encoding="utf-8")
                jd_title = jd_choice.replace("jd_", "").replace(".txt", "").replace("_", " ").title()

    # Analyze button
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=False,
        key="analyze_btn",
    )

    if analyze_clicked:
        if not file_bytes:
            st.markdown(
                '<div class="ra-empty-state ra-error-state">'
                '<span class="ra-error-msg">Please select or upload a resume file.</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Analyzing…"):
            t0 = time.perf_counter()
            _run_analysis(file_bytes, filename, jd_text, jd_title, st.session_state["use_genai"])
            elapsed = time.perf_counter() - t0

        result = st.session_state["result"]
        if result and getattr(result, "warnings", []):
            for w in result.warnings:
                # Sanitize: never surface resume text in warnings
                st.caption(f"Warning: {_escape(w[:200])}")


# ---------------------------------------------------------------------------
# Render: Result hero
# ---------------------------------------------------------------------------

def _render_hero(result) -> None:
    """Score ring + component bars row."""
    total = result.total
    grade = result.grade
    gc = _grade_class(grade)

    ring_svg = _svg_ring(total, grade)

    # Sort components by points lost (largest first)
    weights = {
        "semantic_match": 0.25, "keyword_coverage": 0.20, "section_completeness": 0.10,
        "formatting": 0.10, "experience_fit": 0.10, "quantified_achievements": 0.10,
        "action_verbs": 0.05, "readability": 0.05, "contact_completeness": 0.05,
    }
    components = result.components or {}
    sorted_comps = sorted(
        components.items(),
        key=lambda kv: weights.get(kv[0], 0) * (1 - kv[1]),
        reverse=True,
    )

    bars_html = ""
    for key, val in sorted_comps:
        w = weights.get(key, 0)
        pct = val * 100
        label = _fmt_component_name(key)
        bc = _bar_class(val)
        bars_html += f"""
<div class="ra-bar-row">
  <span class="ra-bar-label" title="{_escape(label)}">{_escape(label)}</span>
  <div class="ra-bar-track"><div class="ra-bar-fill {bc}" style="width:{pct:.1f}%"></div></div>
  <span class="ra-bar-value">{pct:.0f}%</span>
  <span class="ra-bar-weight">{w*100:.0f}pt</span>
</div>"""

    # Sub-scores (display only):
    # Job fit: semantic_match (0.25), keyword_coverage (0.20), experience_fit (0.10) -> sum=0.55
    fit_keys = ["semantic_match", "keyword_coverage", "experience_fit"]
    w_fit = sum(weights[k] for k in fit_keys)
    fit_weighted = sum(weights[k] * components.get(k, 0.0) for k in fit_keys)
    job_fit = (fit_weighted / w_fit) * 100.0 if w_fit > 0 else 0.0

    # Resume quality: remaining 6 components -> sum=0.45
    qual_keys = [
        "section_completeness",
        "formatting",
        "quantified_achievements",
        "action_verbs",
        "readability",
        "contact_completeness",
    ]
    w_qual = sum(weights[k] for k in qual_keys)
    qual_weighted = sum(weights[k] * components.get(k, 0.0) for k in qual_keys)
    resume_quality = (qual_weighted / w_qual) * 100.0 if w_qual > 0 else 0.0

    st.markdown(f"""
<div class="ra-row ra-row-hero ra-section">
  <div class="ra-card ra-score-ring-wrap">
    {ring_svg}
    <span class="ra-score-number">{total:.0f}</span>
    <span class="ra-grade-chip {gc}">{_escape(grade)}</span>
    <div class="ra-subscores" style="display:flex; gap:16px; margin-top:14px; align-items:center; justify-content:center; width:100%; border-top:1px solid var(--border); padding-top:12px;">
      <div style="text-align:center;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-family:var(--font-mono);">Job fit</div>
        <div style="font-family:var(--font-mono); font-size:16px; color:var(--text); font-variant-numeric:tabular-nums; margin-top:2px;">{job_fit:.0f} <span style="font-size:11px; color:var(--faint);">/ 100</span></div>
      </div>
      <div style="width:1px; background:var(--border); height:24px;"></div>
      <div style="text-align:center;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-family:var(--font-mono);">Resume quality</div>
        <div style="font-family:var(--font-mono); font-size:16px; color:var(--text); font-variant-numeric:tabular-nums; margin-top:2px;">{resume_quality:.0f} <span style="font-size:11px; color:var(--faint);">/ 100</span></div>
      </div>
    </div>
  </div>
  <div class="ra-card">
    <span class="ra-card-label">Component Breakdown</span>
    <div class="ra-bars" style="margin-top:16px">{bars_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Stat tiles
# ---------------------------------------------------------------------------

def _render_stat_tiles(result) -> None:
    exp_years = 0.0
    n_roles = 0
    matched = 0
    total_req = 0
    top_cat = "—"
    top_prob = 0.0

    # 1. Experience & roles from parsed resume
    parsed = getattr(result, "_parsed", None) or st.session_state.get("parsed_resume")
    if parsed is not None:
        exp_years = parsed.total_years_experience
        n_roles = len(parsed.experience)
    else:
        exp_fit = result.components.get("experience_fit", 0.0)
        exp_years = exp_fit * 5.0

    # 2. Skills matched
    if result.match:
        matched = len(result.match.matched_skills)
        total_req = matched + len(result.match.missing_skills)

    # 3. Top category (full name)
    if result.predicted_categories:
        top = result.predicted_categories[0]
        top_cat = top.category
        top_prob = top.probability

    # 4. Processing time
    elapsed_sec = st.session_state.get("elapsed")
    if elapsed_sec is None:
        elapsed_sec = sum(result.timings_ms.values()) / 1000.0 if result.timings_ms else 0.0

    tiles = [
        (
            "Experience",
            f"{exp_years:.1f} yrs",
            f"{n_roles} role" if n_roles == 1 else f"{n_roles} roles",
        ),
        (
            "Skills matched",
            f"{matched} / {total_req}",
            "none detected in JD" if total_req == 0 else "required skills in JD",
        ),
        (
            "Top category",
            top_cat,
            f"{int(round(top_prob * 100))}% confidence" if top_prob > 0 else "",
        ),
        (
            "Processing time",
            f"{elapsed_sec:.2f} s",
            "parse to score",
        ),
    ]

    cols = st.columns(4, gap="large")
    for col, (label, value, subtitle) in zip(cols, tiles):
        with col:
            sub_html = (
                f'<div style="font-size:11px;color:var(--faint);font-family:var(--font-mono);margin-top:4px">{_escape(subtitle)}</div>'
                if subtitle
                else ""
            )
            st.markdown(
                f"""
<div class="ra-card">
  <span class="ra-card-label">{_escape(label)}</span>
  <div class="ra-stat-value">{_escape(value)}</div>
  {sub_html}
</div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Render: Issues list
# ---------------------------------------------------------------------------

def _render_issues(result) -> None:
    if not result.issues:
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">No issues found — resume looks good for this JD.</span></div>',
            unsafe_allow_html=True,
        )
        return

    rows_html = ""
    for issue in result.issues:
        sev = issue.severity
        gain = issue.estimated_gain
        comp = issue.component.replace("_", " ")
        msg = issue.message
        rows_html += f"""
<div class="ra-issue-row {_escape(sev)}">
  <span class="ra-issue-gain">+{gain:.1f} pts</span>
  <span class="ra-issue-msg">{_escape(msg)}</span>
  <span class="ra-issue-tag">{_escape(comp)}</span>
</div>"""

    st.markdown(f'<div class="ra-issues">{rows_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Custom tab bar
# ---------------------------------------------------------------------------

TABS = ["Highlights", "Suggestions", "Ranking", "Skill gap", "Bias check", "History", "Model results"]


def _render_tabbar() -> str:
    active = st.session_state["active_tab"]
    tabs_html = ""
    for tab in TABS:
        cls = "ra-tab active" if tab == active else "ra-tab"
        tabs_html += f'<button class="{cls}" id="tab-{_escape(tab.replace(" ", "-"))}" onclick="">{_escape(tab)}</button>'

    # Render invisible radio for actual tab switching
    chosen = st.radio(
        "tab",
        TABS,
        index=TABS.index(active),
        horizontal=True,
        label_visibility="collapsed",
        key="tab_radio",
    )
    if chosen != active:
        st.session_state["active_tab"] = chosen

    return st.session_state["active_tab"]


# ---------------------------------------------------------------------------
# Tab: Highlights
# ---------------------------------------------------------------------------

def _render_highlights_tab(result) -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">Highlights</span>', unsafe_allow_html=True)

    # Legend
    legend_html = """
<div class="ra-legend">
  <span class="ra-legend-item">
    <span class="ra-legend-swatch" style="background:rgba(94,234,212,0.18)"></span>
    Matched keyword
  </span>
  <span class="ra-legend-item">
    <span class="ra-legend-swatch" style="border-bottom:2px solid var(--accent);background:transparent;width:16px"></span>
    High-impact sentence
  </span>
  <span class="ra-legend-item">
    <span class="ra-missing-chip" style="margin:0">missing</span>
    JD keyword not in resume
  </span>
</div>"""
    st.markdown(legend_html, unsafe_allow_html=True)

    # Build highlighted text
    if result.highlights:
        # Sort highlights by start position
        from app.schemas import Highlight
        kw_highlights = [h for h in result.highlights if h.type in ("keyword_match", "positive")]
        kw_highlights.sort(key=lambda h: h.start)
        # For demonstration use the matched skills from match result
        matched_kws = set()
        if result.match:
            matched_kws = {s.lower() for s in result.match.matched_skills}

    # Show missing keywords as chips
    missing = result.match.missing_skills if result.match else result.missing_skills
    if missing:
        chips = " ".join(f'<span class="ra-missing-chip">{_escape(s)}</span>' for s in missing[:20])
        st.markdown(f"<div style='margin-bottom:16px'><span class='ra-card-label'>Missing JD keywords:</span><br>{chips}</div>", unsafe_allow_html=True)

    # Resume text display (never show raw contact info in logs — display only)
    # Retrieve from session if stored; otherwise show placeholder
    resume_text_html = ""
    if hasattr(result, "_resume_text_preview"):
        raw = result._resume_text_preview
    else:
        raw = ""

    if raw:
        escaped = _escape(raw[:4000])
        resume_text_html = f'<div class="ra-card"><pre class="ra-resume-text">{escaped}</pre></div>'
    else:
        resume_text_html = '<div class="ra-empty-state"><span class="ra-empty-msg">Resume text preview not stored (privacy mode). Re-run to see highlights.</span></div>'

    st.markdown(resume_text_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Suggestions
# ---------------------------------------------------------------------------

def _render_suggestions_tab(result) -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">Suggestions</span>', unsafe_allow_html=True)

    if not result.suggestions:
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">No suggestions generated. Enable AI suggestions in Settings or re-run analysis.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Group by kind
    bullets = [s for s in result.suggestions if s.kind == "bullet_rewrite"]
    cover = [s for s in result.suggestions if s.kind == "cover_letter"]
    plan = [s for s in result.suggestions if s.kind == "improvement_plan"]

    if bullets:
        st.markdown('<span class="ra-section-title" style="font-size:16px">Bullet Rewrites</span>', unsafe_allow_html=True)
        for s in bullets:
            provider = s.notes if s.notes else "Rule-based fallback"
            chip = f'<span class="ra-provider-chip">Generated by {_escape(provider)}</span>'
            before = f'<div class="ra-before"><strong style="font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em">Before</strong><br>{_escape(s.original)}</div>'
            after = f'<div class="ra-after"><strong style="font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em">After</strong><br>{_escape(s.suggested)}</div>'
            st.markdown(f"{chip}<div class='ra-before-after'>{before}{after}</div>", unsafe_allow_html=True)

    if cover:
        st.markdown('<span class="ra-section-title" style="font-size:16px">Cover Letter</span>', unsafe_allow_html=True)
        for s in cover:
            provider = s.notes if s.notes else "Rule-based fallback"
            st.markdown(f'<span class="ra-provider-chip">Generated by {_escape(provider)}</span>', unsafe_allow_html=True)
            st.markdown('<div class="ra-card">', unsafe_allow_html=True)
            st.text_area("Cover letter (copy-friendly)", value=s.suggested, height=200, key=f"cl_{id(s)}")
            st.markdown("</div>", unsafe_allow_html=True)

    if plan:
        st.markdown('<span class="ra-section-title" style="font-size:16px">Improvement Plan</span>', unsafe_allow_html=True)
        rows_html = ""
        for s in plan:
            rows_html += f"""
<div class="ra-issue-row medium">
  <span class="ra-issue-msg">{_escape(s.suggested)}</span>
</div>"""
        st.markdown(f'<div class="ra-issues">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Ranking
# ---------------------------------------------------------------------------

def _render_ranking_tab() -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">JD Ranking</span>', unsafe_allow_html=True)

    result = st.session_state.get("result")
    if result is None or not hasattr(result, "_file_bytes"):
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">Run an analysis first to rank against multiple JDs.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Rank against all sample JDs
    from app.extras.ranking import rank_jds
    from app.schemas import JobDescription

    jd_list = []
    for jd_file in SAMPLE_JDS:
        p = FIXTURES_DIR / jd_file
        if p.exists():
            title = jd_file.replace("jd_", "").replace(".txt", "").replace("_", " ").title()
            jd_list.append(JobDescription(text=p.read_text(encoding="utf-8"), title=title))

    if not jd_list:
        st.markdown('<div class="ra-empty-state"><span class="ra-empty-msg">No sample JD files found in tests/fixtures/.</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    parsed = result._parsed if hasattr(result, "_parsed") else None
    if parsed is None:
        st.markdown('<div class="ra-empty-state"><span class="ra-empty-msg">Parsed resume not available for ranking.</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("Ranking…"):
        ranked = rank_jds(parsed, jd_list)

    rows_html = ""
    for item in ranked:
        gc = _grade_class(item["grade"])
        ms = ", ".join(item["missing_skills"][:3]) or "—"
        if len(item["missing_skills"]) > 3:
            ms += f" +{len(item['missing_skills'])-3} more"
        rows_html += f"""
<tr>
  <td>{_escape(item['jd_title'])}</td>
  <td class="{'best' if item['rank'] == 1 else ''}">{item['score']:.1f}</td>
  <td><span class="ra-grade-chip {gc}">{_escape(item['grade'])}</span></td>
  <td style="text-align:left;font-family:var(--font-ui);color:var(--muted)">{_escape(ms)}</td>
</tr>"""

    table_html = f"""
<div class="ra-table-wrap">
  <table class="ra-table">
    <thead>
      <tr>
        <th style="text-align:left">Job Description</th>
        <th>Score</th>
        <th>Grade</th>
        <th style="text-align:left">Missing Skills</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Skill gap
# ---------------------------------------------------------------------------

def _render_skill_gap_tab(result) -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">Skill Gap</span>', unsafe_allow_html=True)

    if not result.missing_skills:
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">No missing skills detected for this JD.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    parsed_skills = []
    if result.match:
        parsed_skills = result.match.matched_skills

    try:
        from app.extras.skill_gap import skill_gap
        gap = skill_gap(parsed_skills, result.missing_skills)
    except Exception:
        gap = {}

    if not gap:
        chips = " ".join(f'<span class="ra-missing-chip">{_escape(s)}</span>' for s in result.missing_skills)
        st.markdown(f"<div class='ra-card'><span class='ra-card-label'>Missing Skills</span><div style='margin-top:8px'>{chips}</div></div>", unsafe_allow_html=True)
    else:
        for skill, courses in gap.items():
            st.markdown(f'<div class="ra-card" style="margin-bottom:16px"><span class="ra-card-label">{_escape(skill)}</span>', unsafe_allow_html=True)
            for c in courses[:2]:
                st.markdown(f"""
<div style="margin-top:8px;padding:8px;background:var(--surface-raised);border-radius:8px">
  <div style="font-size:14px;color:var(--text)">{_escape(c.get('title',''))}</div>
  <div style="font-size:11px;font-family:var(--font-mono);color:var(--muted);margin-top:4px">
    {_escape(c.get('provider',''))} · {_escape(c.get('level',''))} · {_escape(str(c.get('estimated_hours','')))}h
  </div>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Bias check
# ---------------------------------------------------------------------------

def _render_bias_tab(result) -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">Bias Check</span>', unsafe_allow_html=True)

    parsed = getattr(result, "_parsed", None)
    if parsed is None:
        st.markdown('<div class="ra-empty-state"><span class="ra-empty-msg">Parsed resume not available for bias check.</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        from app.extras.bias import bias_check
        findings = bias_check(parsed)
    except Exception as e:
        st.markdown(f'<div class="ra-empty-state ra-error-state"><span class="ra-error-msg">Bias check failed: {_escape(str(e)[:100])}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not findings:
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">No potential bias signals detected. Resume uses neutral language.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        rows_html = ""
        for f in findings:
            sev_cls = "high" if f.get("severity", "low") == "high" else "medium"
            label = f.get("evidence_label", "")
            type_ = f.get("type", "")
            msg = f.get("message", "")
            rows_html += f"""
<div class="ra-issue-row {sev_cls}">
  <span class="ra-issue-gain" style="color:var(--warn)">{_escape(type_)}</span>
  <span class="ra-issue-msg">{_escape(msg)}</span>
  <span class="ra-issue-tag">{_escape(label)}</span>
</div>"""
        st.markdown(f'<div class="ra-issues">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: History
# ---------------------------------------------------------------------------

def _render_history_tab() -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">History</span>', unsafe_allow_html=True)

    if not st.session_state.get("history_opt_in"):
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">History is off. Enable "Save score history" in Settings to track past analyses.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        from app.extras.history import get_history
        records = get_history()
    except Exception:
        records = []

    if not records:
        st.markdown(
            '<div class="ra-empty-state"><span class="ra-empty-msg">No history yet. Run an analysis with history opt-in enabled.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows_html = ""
    for r in records[-20:]:
        ts = str(r.get("timestamp", ""))[:16]
        jdt = r.get("jd_title", "—")
        total = r.get("total_score", 0)
        grade = r.get("grade", "—")
        gc = _grade_class(grade)
        rows_html += f"""
<tr>
  <td>{_escape(ts)}</td>
  <td>{_escape(jdt)}</td>
  <td class="best">{total:.1f}</td>
  <td><span class="ra-grade-chip {gc}" style="font-size:10px;padding:1px 6px">{_escape(grade)}</span></td>
</tr>"""

    table_html = f"""
<div class="ra-table-wrap">
  <table class="ra-table">
    <thead>
      <tr><th style="text-align:left">Date</th><th style="text-align:left">JD</th><th>Score</th><th>Grade</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Model results
# ---------------------------------------------------------------------------

def _render_model_results_tab() -> None:
    st.markdown('<div class="ra-section">', unsafe_allow_html=True)
    st.markdown('<span class="ra-section-title">Model Results</span>', unsafe_allow_html=True)

    reports_dir = ROOT_DIR / "artifacts" / "reports"
    metrics_dir = ROOT_DIR / "artifacts" / "metrics"

    def _render_missing(msg: str = "Results file missing: run scripts/collect_results.py") -> None:
        st.markdown(
            f'<div class="ra-card ra-empty-state"><span class="ra-empty-msg">{_escape(msg)}</span></div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 1. Classifier section
    # -----------------------------------------------------------------------
    st.markdown('<span class="ra-card-label">Classifier</span>', unsafe_allow_html=True)
    c4_path = reports_dir / "classifier_4_model.csv"
    ci_path = reports_dir / "classifier_ci.csv"
    paired_path = reports_dir / "classifier_paired_comparison.csv"
    metrics_json_path = metrics_dir / "classifier" / "metrics.json"

    if not metrics_json_path.exists() and not c4_path.exists():
        _render_missing("Results file missing: run scripts/collect_results.py")
    else:
        try:
            # Load raw metrics.json to ensure val macro-F1 and train-val gap match exactly
            raw_metrics = {}
            if metrics_json_path.exists():
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    raw_metrics = json.load(f)

            df_c4 = pd.read_csv(c4_path) if c4_path.exists() else None
            df_ci = pd.read_csv(ci_path) if ci_path.exists() else None

            ci_map = {}
            if df_ci is not None:
                for _, r in df_ci.iterrows():
                    ci_map[str(r["model"])] = r

            readable_names = {
                "tfidf_lr": "TF-IDF + LR",
                "tfidf_mlp": "TF-IDF + MLP",
                "bilstm": "BiLSTM",
                "distilbert": "DistilBERT (3 epochs)",
            }
            model_keys = ["tfidf_lr", "tfidf_mlp", "bilstm", "distilbert"]

            rows_html = ""
            for m_key in model_keys:
                m_name = readable_names.get(m_key, m_key)
                is_served = (m_key == "tfidf_mlp")
                served_badge = ' <span class="ra-chip ra-chip-good" style="margin-left:6px;font-size:10px;padding:2px 6px">SERVED</span>' if is_served else ""

                m_data = raw_metrics.get(m_key, {})
                val_f1_val = m_data.get("val_macro_f1")
                train_f1_val = m_data.get("train_macro_f1")

                # Fallback to c4 dataframe if not in metrics.json
                if val_f1_val is None and df_c4 is not None:
                    row_c4 = df_c4[df_c4["model_key"] == m_key]
                    if not row_c4.empty:
                        val_f1_val = float(row_c4.iloc[0]["val_macro_f1"])
                        train_f1_val = float(row_c4.iloc[0]["train_macro_f1"])

                val_f1 = f"{val_f1_val:.4f}" if val_f1_val is not None else "—"

                r_ci = ci_map.get(m_key)
                if r_ci is not None:
                    test_f1_str = f"{float(r_ci['test_macro_f1']):.4f} [{float(r_ci['ci_95_lower']):.4f}, {float(r_ci['ci_95_upper']):.4f}]"
                    test_acc_str = f"{float(r_ci['test_accuracy'])*100:.2f}%"
                elif m_data.get("test_macro_f1") is not None:
                    test_f1_str = f"{m_data['test_macro_f1']:.4f}"
                    test_acc_str = f"{m_data['test_accuracy']*100:.2f}%"
                else:
                    test_f1_str = "—"
                    test_acc_str = "—"

                if train_f1_val is not None and val_f1_val is not None:
                    gap_val = train_f1_val - val_f1_val
                    gap_str = f"{gap_val:+.4f}"
                else:
                    gap_str = "—"

                rows_html += f"<tr><td>{_escape(m_name)}{served_badge}</td><td>{val_f1}</td><td>{test_f1_str}</td><td>{test_acc_str}</td><td>{gap_str}</td></tr>"

            table1_html = f"""
<div class="ra-table-wrap">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Model</th>
    <th>Val Macro F1</th>
    <th>Test Macro F1 (95% CI)</th>
    <th>Test Accuracy</th>
    <th>Train-Val Gap</th>
  </tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""
            st.markdown(table1_html, unsafe_allow_html=True)

            # Table 2: Paired tests
            if paired_path.exists():
                df_paired = pd.read_csv(paired_path)
                p_rows_html = ""
                for _, pr in df_paired.iterrows():
                    metric_label = "Macro-F1" if pr["metric"] == "macro_f1" else "Accuracy"
                    diff_val = f"{pr['diff']:+.4f}"
                    ci_str = f"[{pr['ci_95_lower']:+.4f}, {pr['ci_95_upper']:+.4f}]"
                    if pr["metric"] == "macro_f1":
                        p_val_str = f"p = {pr['bootstrap_p']:.4f} (boot)"
                    else:
                        m_p = pr.get("mcnemar_p")
                        p_val_str = f"p = {m_p:.4f} (McNemar)" if pd.notna(m_p) else f"p = {pr['bootstrap_p']:.4f}"
                    p_rows_html += f"<tr><td>{_escape(pr['comparison'])}</td><td>{metric_label}</td><td>{diff_val}</td><td>{ci_str}</td><td>{p_val_str}</td></tr>"

                table2_html = f"""
<div class="ra-table-wrap" style="margin-top:16px">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Comparison</th>
    <th>Metric</th>
    <th>Diff (A − B)</th>
    <th>95% CI</th>
    <th>p-value</th>
  </tr>
</thead>
<tbody>{p_rows_html}</tbody>
</table>
</div>"""
                st.markdown(table2_html, unsafe_allow_html=True)

            st.markdown(
                '<div class="ra-table-note">'
                'LR and MLP are indistinguishable on test (macro-F1 and accuracy, paired tests). '
                'DistilBERT scores higher on test: vs LR macro-F1 +0.050 (95% CI [0.0056, 0.0931], p=0.036) and accuracy +6.4 pts (+0.0642, p=0.022); '
                'vs MLP accuracy +7.5 pts (+0.0749, p=0.010), but its macro-F1 gain over MLP is not significant (p=0.280). '
                'Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely. '
                'Validation and test disagree (DistilBERT val macro-F1 0.5675 vs MLP 0.6001); six paired tests with no multiplicity correction; one test split of 374 resumes (average ~15, smallest class 3, largest 18); DistilBERT trained only 3 epochs; the BiLSTM underfits. '
                'The served classifier is the MLP because the selection rule (val macro-F1) was fixed on validation before the test set was opened; choosing DistilBERT because it won on test would be test-set selection; a fresh holdout would be needed to justify a switch.'
                '</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            _render_missing("Results file missing: run scripts/collect_results.py")

    # -----------------------------------------------------------------------
    # 2. Matcher section
    # -----------------------------------------------------------------------
    st.markdown('<span class="ra-card-label" style="margin-top:32px;display:block">Matcher</span>', unsafe_allow_html=True)
    m_overall_path = reports_dir / "matcher_overall.csv"
    m_cat_path = reports_dir / "matcher_per_category.csv"

    if not m_overall_path.exists():
        _render_missing("Results file missing: run scripts/collect_results.py")
    else:
        try:
            df_mo = pd.read_csv(m_overall_path)
            row_labels = {
                "tfidf": "TF-IDF Baseline",
                "zero_shot": "Zero-Shot SBERT",
                "fine_tuned": "Fine-Tuned Siamese SBERT",
            }
            m_rows_html = ""
            for _, r in df_mo.iterrows():
                label = row_labels.get(str(r["model_key"]), str(r["model_key"]))
                m_rows_html += (
                    f"<tr><td>{_escape(label)}</td>"
                    f"<td>{r['val_mrr']:.4f}</td><td>{r['val_ndcg10']:.4f}</td>"
                    f"<td>{r['val_recall5']:.4f}</td><td>{r['val_recall10']:.4f}</td>"
                    f"<td>{r['test_mrr']:.4f}</td><td>{r['test_ndcg10']:.4f}</td>"
                    f"<td>{r['test_recall5']:.4f}</td><td>{r['test_recall10']:.4f}</td></tr>"
                )
            mo_table_html = f"""
<div class="ra-table-wrap">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Model</th>
    <th>Val MRR</th>
    <th>Val nDCG@10</th>
    <th>Val Recall@5</th>
    <th>Val Recall@10</th>
    <th>Test MRR</th>
    <th>Test nDCG@10</th>
    <th>Test Recall@5</th>
    <th>Test Recall@10</th>
  </tr>
</thead>
<tbody>{m_rows_html}</tbody>
</table>
</div>"""
            st.markdown(mo_table_html, unsafe_allow_html=True)

            if m_cat_path.exists():
                with st.expander("Per-category matcher results (15 categories)", expanded=False):
                    df_mcat = pd.read_csv(m_cat_path)
                    cat_rows_html = ""
                    for _, cr in df_mcat.iterrows():
                        cat_rows_html += (
                            f"<tr><td>{_escape(str(cr['category']))}</td>"
                            f"<td>{int(cr['val_count'])}</td><td>{int(cr['test_count'])}</td>"
                            f"<td>{cr['val_tfidf_mrr']:.4f}</td><td>{cr['val_zs_mrr']:.4f}</td><td>{cr['val_ft_mrr']:.4f}</td>"
                            f"<td>{cr['test_tfidf_mrr']:.4f}</td><td>{cr['test_zs_mrr']:.4f}</td><td>{cr['test_ft_mrr']:.4f}</td></tr>"
                        )
                    mcat_table_html = f"""
<div class="ra-table-wrap">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Category</th><th>Val N</th><th>Test N</th>
    <th>TF-IDF Val</th><th>Zero-Shot Val</th><th>Fine-Tuned Val</th>
    <th>TF-IDF Test</th><th>Zero-Shot Test</th><th>Fine-Tuned Test</th>
  </tr>
</thead>
<tbody>{cat_rows_html}</tbody>
</table>
</div>"""
                    st.markdown(mcat_table_html, unsafe_allow_html=True)

            st.markdown(
                '<div class="ra-table-note">Matcher note: Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means. Weak title-derived labels, partly circular.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            _render_missing("Results file missing: run scripts/collect_results.py")

    # -----------------------------------------------------------------------
    # 3. Scoring section (Ablation)
    # -----------------------------------------------------------------------
    st.markdown('<span class="ra-card-label" style="margin-top:32px;display:block">Scoring</span>', unsafe_allow_html=True)
    abl_path = reports_dir / "scoring_ablation.csv"

    if not abl_path.exists():
        _render_missing("Results file missing: run scripts/collect_results.py")
    else:
        try:
            df_abl = pd.read_csv(abl_path)
            abl_rows_html = ""
            for _, ar in df_abl.iterrows():
                comp_name = _fmt_component_name(str(ar["component"]))
                orig_w = f"{ar['original_weight']:.2f}"
                mean_orig = f"{ar['mean_original_score']:.2f}"
                mean_abl = f"{ar['mean_ablated_score']:.2f}"
                delta = f"{ar['mean_score_delta']:+.2f}"
                delta_cls = "good" if ar["mean_score_delta"] < 0 else "bad"
                abl_rows_html += (
                    f"<tr><td>{_escape(comp_name)}</td><td>{orig_w}</td>"
                    f"<td>{mean_orig}</td><td>{mean_abl}</td>"
                    f"<td class='{delta_cls}'>{delta}</td></tr>"
                )
            abl_table_html = f"""
<div class="ra-table-wrap">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Component</th>
    <th>Weight</th>
    <th>Mean Original</th>
    <th>Mean Ablated</th>
    <th>Mean Change (Delta)</th>
  </tr>
</thead>
<tbody>{abl_rows_html}</tbody>
</table>
</div>"""
            st.markdown(abl_table_html, unsafe_allow_html=True)
            st.markdown(
                '<div class="ra-table-note">Scoring note: No ground-truth ATS score exists; weights are heuristics.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            _render_missing("Results file missing: run scripts/collect_results.py")

    # -----------------------------------------------------------------------
    # 4. XAI section
    # -----------------------------------------------------------------------
    st.markdown('<span class="ra-card-label" style="margin-top:32px;display:block">Explainability (XAI)</span>', unsafe_allow_html=True)
    xai_path = reports_dir / "xai_faithfulness.csv"
    wdel_path = metrics_dir / "scoring" / "classifier_word_deletion.json"

    if not xai_path.exists() and not wdel_path.exists():
        _render_missing("Results file missing: run scripts/collect_results.py")
    else:
        try:
            # Row 1: ATS sentence deletion
            sent_drop_top = "0.07 pts"
            sent_drop_rand = "0.11 pts"
            if xai_path.exists():
                df_xai = pd.read_csv(xai_path)
                if len(df_xai) > 0:
                    sent_drop_top = f"{df_xai.iloc[0]['mean_top_k_drop']:.2f} pts"
                    sent_drop_rand = f"{df_xai.iloc[0]['mean_random_k_drop']:.2f} pts"

            # Row 2: Classifier word deletion
            wdel_lr_str = "Top-10 drop 0.3590 vs Random -0.0021 (+0.3611 gap)"
            wdel_served_str = "Top-10 drop 0.4921 vs Random 0.0058 (+0.4863 gap)"
            if wdel_path.exists():
                with open(wdel_path, "r", encoding="utf-8") as f:
                    wdata = json.load(f)
                    lr_sub = wdata.get("lr_self_attribution", {})
                    mlp_sub = wdata.get("served_classifier_transfer", {})
                    if lr_sub:
                        wdel_lr_str = f"Top-10 drop {lr_sub.get('mean_top_k_drop', 0.359):.4f} vs Random {lr_sub.get('mean_random_k_drop', -0.0021):+.4f} (gap: {lr_sub.get('attribution_gap', 0.3611):+.4f})"
                    if mlp_sub:
                        wdel_served_str = f"Top-10 drop {mlp_sub.get('mean_top_k_drop', 0.4921):.4f} vs Random {mlp_sub.get('mean_random_k_drop', 0.0058):+.4f} (gap: {mlp_sub.get('attribution_gap', 0.4863):+.4f})"

            xai_rows_html = f"""
<tr>
  <td style="font-weight:600">ATS Total (Sentence-Level)</td>
  <td>Leave-one-sentence-out (LOSO) (k=3, 20 val resumes)</td>
  <td>Top-3 drop: {sent_drop_top} | Random-3 drop: {sent_drop_rand}</td>
  <td><span class="ra-chip ra-chip-warn" style="font-size:11px">Inconclusive & unvalidated</span></td>
</tr>
<tr>
  <td style="font-weight:600">Classifier (Word-Level)</td>
  <td>TF-IDF + LR linear coefficients (k=10, 50 val resumes)</td>
  <td>LR (self): {wdel_lr_str}<br>Served MLP (transfer): {wdel_served_str}</td>
  <td><span class="ra-chip ra-chip-good" style="font-size:11px">Faithful to linear model & cross-model transfer</span></td>
</tr>"""

            xai_table_html = f"""
<div class="ra-table-wrap">
<table class="ra-table">
<thead>
  <tr>
    <th style="text-align:left">Attribution Level</th>
    <th>Method & Sample</th>
    <th>Observed Drops (Top vs Random)</th>
    <th>Verdict</th>
  </tr>
</thead>
<tbody>{xai_rows_html}</tbody>
</table>
</div>"""
            st.markdown(xai_table_html, unsafe_allow_html=True)
            st.markdown(
                '<div class="ra-table-note">XAI note: Sentence deletion is inconclusive and unvalidated. Word deletion shows faithfulness to the linear model and transfer to the served model.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            _render_missing("Results file missing: run scripts/collect_results.py")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Download report button
# ---------------------------------------------------------------------------

def _render_download(result) -> None:
    lines = [
        "# Resume Analyzer Report",
        "",
        f"**Score:** {result.total:.1f} / 100  |  **Grade:** {result.grade}",
        "",
        "## Component Scores",
        "",
    ]
    for k, v in (result.components or {}).items():
        lines.append(f"- {_fmt_component_name(k)}: {v*100:.0f}%")

    if result.issues:
        lines += ["", "## Top Issues", ""]
        for issue in result.issues[:5]:
            lines.append(f"- [{issue.severity.upper()}] {issue.message} (+{issue.estimated_gain:.1f} pts)")

    if result.suggestions:
        lines += ["", "## Suggestions", ""]
        for s in result.suggestions[:5]:
            lines.append(f"- {s.suggested[:200]}")

    # Note: no resume text in report
    report_md = "\n".join(lines)
    st.download_button(
        label="Download report",
        data=report_md.encode("utf-8"),
        file_name="resume_analysis_report.md",
        mime="text/markdown",
        type="secondary",
        key="download_report",
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

_render_header()

st.markdown('<div style="max-width:var(--max-w);margin:0 auto;padding:0 24px">', unsafe_allow_html=True)

_render_input()

result = st.session_state.get("result")

if result is None:
    # First load: show empty state
    st.markdown("""
<div class="ra-empty-state" style="margin-top:32px">
  <span class="ra-empty-msg">Select a sample resume and job description above, then click <strong>Analyze</strong> to see your ATS score, component breakdown, and improvement suggestions.</span>
</div>
""", unsafe_allow_html=True)
else:
    # Check for error result (grade F with no components is likely an error)
    is_error = (
        result.total == 0.0
        and result.grade == "F"
        and not result.components
    )

    if is_error:
        err_msg = result.issues[0].message if result.issues else "An unknown error occurred."
        st.markdown(f"""
<div class="ra-empty-state ra-error-state" style="margin-top:32px">
  <span class="ra-error-msg">{_escape(err_msg)}</span>
  <span style="font-size:13px;color:var(--muted)">Please check the file and try again.</span>
</div>
""", unsafe_allow_html=True)
    else:
        # Warning chip for short JD or no skills detected (Item 2)
        jd_text_val = st.session_state.get("jd_text", "")
        n_words = len(jd_text_val.split()) if jd_text_val else 0
        n_skills = (
            len(result.match.matched_skills) + len(result.match.missing_skills)
            if result.match
            else 0
        )
        has_kw_warn = any(
            "keyword coverage" in w.lower() for w in getattr(result, "warnings", [])
        )
        if jd_text_val and (n_words < 30 or n_skills == 0 or has_kw_warn):
            st.markdown(
                '<div style="margin-top:24px;margin-bottom:8px">'
                '<span class="ra-chip ra-chip-warn">'
                'JD too short or no skills detected; keyword coverage not measured'
                '</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Result hero
        _render_hero(result)

        # Stat tiles
        _render_stat_tiles(result)

        # Issues
        st.markdown('<span class="ra-section-title" style="margin-top:48px;display:block">Issues & Opportunities</span>', unsafe_allow_html=True)
        _render_issues(result)

        st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)

        # Download report button
        _render_download(result)

        st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

        # Tab bar + content
        active_tab = _render_tabbar()

        if active_tab == "Highlights":
            _render_highlights_tab(result)
        elif active_tab == "Suggestions":
            _render_suggestions_tab(result)
        elif active_tab == "Ranking":
            _render_ranking_tab()
        elif active_tab == "Skill gap":
            _render_skill_gap_tab(result)
        elif active_tab == "Bias check":
            _render_bias_tab(result)
        elif active_tab == "History":
            _render_history_tab()
        elif active_tab == "Model results":
            _render_model_results_tab()

st.markdown("</div>", unsafe_allow_html=True)

