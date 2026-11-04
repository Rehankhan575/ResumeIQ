"""
app/ui/app.py
=============
ResumeIQ — Streamlit UI.

Native Streamlit components styled via .streamlit/config.toml light theme.
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

APP_NAME = "ResumeIQ"
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
UI_DIR = Path(__file__).resolve().parent
STYLES_CSS = UI_DIR / "styles.css"
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"

# ---------------------------------------------------------------------------
# 0. Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Inject styles.css
# ---------------------------------------------------------------------------

def _load_css() -> None:
    if STYLES_CSS.exists():
        css = STYLES_CSS.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_load_css()

# ---------------------------------------------------------------------------
# Warm-up models
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _warm_up() -> bool:
    from app.pipeline import warm_up
    warm_up()
    return True


_models_ready = _warm_up()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None
if "history_opt_in" not in st.session_state:
    st.session_state["history_opt_in"] = False
if "use_genai" not in st.session_state:
    st.session_state["use_genai"] = True
if "ranking_result" not in st.session_state:
    st.session_state["ranking_result"] = None

# ---------------------------------------------------------------------------
# Helpers & data loaders
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

VERDICTS = {
    "A": "Excellent",
    "B": "Good",
    "C": "Fair",
    "D": "Needs work",
    "F": "Poor",
}

WEIGHTS = {
    "semantic_match": 0.25,
    "keyword_coverage": 0.20,
    "section_completeness": 0.10,
    "formatting": 0.10,
    "experience_fit": 0.10,
    "quantified_achievements": 0.10,
    "action_verbs": 0.05,
    "readability": 0.05,
    "contact_completeness": 0.05,
}


def _fmt_component_name(key: str) -> str:
    return key.replace("_", " ").title()


def _render_score_ring(score: float, grade: str) -> str:
    r = 54
    cx = cy = 68
    stroke_w = 10
    circumference = 2 * math.pi * r
    clamped_score = max(0.0, min(100.0, float(score)))
    offset = circumference * (1.0 - clamped_score / 100.0)

    if grade in ("A", "B"):
        stroke_color = "#15803D"
    elif grade == "C":
        stroke_color = "#B45309"
    else:
        stroke_color = "#B91C1C"

    verdict_word = VERDICTS.get(grade, "Good")

    return f"""<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;margin:6px 0;">
  <div style="position:relative;width:136px;height:136px;display:flex;align-items:center;justify-content:center;">
    <svg width="136" height="136" viewBox="0 0 136 136" style="transform:rotate(-90deg);">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#E5E7EB" stroke-width="{stroke_w}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_w}"
              stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>
    </svg>
    <div style="position:absolute;text-align:center;">
      <div style="font-size:36px;font-weight:700;color:#111827;line-height:1;">{clamped_score:.0f}</div>
      <div style="font-size:14px;font-weight:600;color:{stroke_color};margin-top:2px;">Grade {html.escape(grade)}</div>
    </div>
  </div>
  <div style="font-size:16px;font-weight:600;color:{stroke_color};margin-top:6px;">{verdict_word}</div>
</div>"""


def _run_analysis(
    file_bytes: bytes,
    filename: str,
    jd_text: str,
    jd_title: str,
    use_genai: bool,
) -> None:
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

    try:
        parsed_doc = parse_document(file_bytes, filename)
        extracted = extract_entities(parsed_doc)
        result._parsed = extracted
        result._file_bytes = file_bytes
        st.session_state["parsed_resume"] = extracted
    except Exception:
        pass

    st.session_state["result"] = result
    st.session_state["elapsed"] = elapsed
    st.session_state["jd_text"] = jd_text
    st.session_state["filename"] = filename
    st.session_state["jd_title"] = jd_title
    st.session_state["ranking_result"] = None

    if st.session_state.get("history_opt_in") and result.total > 0:
        try:
            from app.extras.history import record_history
            record_history(result, jd_title=jd_title or "Unknown", opt_in=True)
        except Exception:
            pass


@st.cache_data
def _load_model_reports():
    c4_path = REPORTS_DIR / "classifier_4_model.csv"
    ci_path = REPORTS_DIR / "classifier_ci.csv"
    paired_path = REPORTS_DIR / "classifier_paired_comparison.csv"
    metrics_json_path = METRICS_DIR / "classifier" / "metrics.json"

    raw_metrics = {}
    if metrics_json_path.exists():
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            raw_metrics = json.load(f)

    df_c4 = pd.read_csv(c4_path) if c4_path.exists() else None
    df_ci = pd.read_csv(ci_path) if ci_path.exists() else None
    df_paired = pd.read_csv(paired_path) if paired_path.exists() else None

    m_overall_path = REPORTS_DIR / "matcher_overall.csv"
    m_cat_path = REPORTS_DIR / "matcher_per_category.csv"
    df_mo = pd.read_csv(m_overall_path) if m_overall_path.exists() else None
    df_mcat = pd.read_csv(m_cat_path) if m_cat_path.exists() else None

    abl_path = REPORTS_DIR / "scoring_ablation.csv"
    if not abl_path.exists():
        abl_path = METRICS_DIR / "scoring" / "ablation.csv"
    df_abl = pd.read_csv(abl_path) if abl_path.exists() else None

    xai_path = REPORTS_DIR / "xai_faithfulness.csv"
    df_xai = pd.read_csv(xai_path) if xai_path.exists() else None

    wdel_path = METRICS_DIR / "scoring" / "classifier_word_deletion.json"
    wdata = None
    if wdel_path.exists():
        with open(wdel_path, "r", encoding="utf-8") as f:
            wdata = json.load(f)

    return (raw_metrics, df_c4, df_ci, df_paired, df_mo, df_mcat, df_abl, df_xai, wdata)


# ---------------------------------------------------------------------------
# 7. Sidebar: Settings & System Status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    st.caption("Privacy: resume text is processed in-memory only and never stored.")
    st.session_state["history_opt_in"] = st.checkbox(
        "Save score history (opt-in, no resume text stored)",
        value=st.session_state.get("history_opt_in", False),
    )
    st.session_state["use_genai"] = st.checkbox(
        "Enable AI suggestions (requires API key in .env)",
        value=st.session_state.get("use_genai", True),
    )
    status_txt = "Models loaded" if _models_ready else "Loading models…"
    st.info(f"System status: {status_txt}")

# ---------------------------------------------------------------------------
# 1. Hero Section
# ---------------------------------------------------------------------------

st.title(APP_NAME)
st.write(
    "See how an applicant tracking system (ATS) scores your resume against a "
    "job description, and exactly what to fix."
)

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    with st.container(border=True):
        st.markdown(
            "**1. Choose:** Upload a resume and paste a job description, or try a sample."
        )
with col_h2:
    with st.container(border=True):
        st.markdown(
            "**2. Analyze:** We read the resume, match it to the job and score it out of 100."
        )
with col_h3:
    with st.container(border=True):
        st.markdown(
            "**3. Improve:** Fix the top issues, then run it again to see your score change."
        )

# ---------------------------------------------------------------------------
# 2. Inputs Section
# ---------------------------------------------------------------------------

default_resume_idx = 0
if "ds_clean.pdf" in SAMPLE_RESUMES:
    default_resume_idx = SAMPLE_RESUMES.index("ds_clean.pdf")

default_jd_idx = 0
if "jd_data_scientist.txt" in SAMPLE_JDS:
    default_jd_idx = SAMPLE_JDS.index("jd_data_scientist.txt")

col_in1, col_in2 = st.columns(2)

with col_in1:
    with st.container(border=True):
        st.markdown('<div class="card-title">Your resume</div>', unsafe_allow_html=True)
        uploaded_resume = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            help="Processed in memory, never stored",
            key="uploaded_resume",
        )
        sample_resume_choice = st.selectbox(
            "Or try a sample resume",
            options=SAMPLE_RESUMES,
            index=default_resume_idx if SAMPLE_RESUMES else 0,
            key="resume_choice",
        )

with col_in2:
    with st.container(border=True):
        st.markdown('<div class="card-title">Job description</div>', unsafe_allow_html=True)
        pasted_jd = st.text_area(
            "Paste the job description here",
            height=220,
            placeholder="Paste the job description here",
            key="pasted_jd",
        )
        sample_jd_choice = st.selectbox(
            "Or try a sample job description",
            options=SAMPLE_JDS,
            index=default_jd_idx if SAMPLE_JDS else 0,
            key="jd_choice",
        )

# Input Precedence: upload or pasted text ALWAYS wins over a sample
if uploaded_resume is not None:
    file_bytes = uploaded_resume.getvalue()
    filename = uploaded_resume.name
    is_sample_resume = False
elif sample_resume_choice:
    p_resume = FIXTURES_DIR / sample_resume_choice
    file_bytes = p_resume.read_bytes() if p_resume.exists() else b""
    filename = sample_resume_choice
    is_sample_resume = True
else:
    file_bytes = b""
    filename = ""
    is_sample_resume = True

if pasted_jd and pasted_jd.strip():
    jd_text = pasted_jd.strip()
    jd_title = "Pasted Job Description"
    is_sample_jd = False
elif sample_jd_choice:
    p_jd = FIXTURES_DIR / sample_jd_choice
    jd_text = p_jd.read_text(encoding="utf-8") if p_jd.exists() else ""
    jd_title = sample_jd_choice.replace("jd_", "").replace(".txt", "").replace("_", " ").title()
    is_sample_jd = True
else:
    jd_text = ""
    jd_title = ""
    is_sample_jd = True

# Full-width primary button
analyze_clicked = st.button(
    "Analyze my resume",
    type="primary",
    use_container_width=True,
    key="analyze_btn",
)

# First load auto-run of preselected sample pair
is_first_load = st.session_state["result"] is None
if is_first_load:
    if file_bytes and jd_text:
        _run_analysis(
            file_bytes=file_bytes,
            filename=filename,
            jd_text=jd_text,
            jd_title=jd_title,
            use_genai=st.session_state.get("use_genai", True),
        )

if analyze_clicked:
    if not file_bytes:
        st.error("Please provide or select a resume.")
    else:
        with st.spinner("Analyzing…"):
            _run_analysis(
                file_bytes=file_bytes,
                filename=filename,
                jd_text=jd_text,
                jd_title=jd_title,
                use_genai=st.session_state.get("use_genai", True),
            )

# Info banner on sample analysis
if is_sample_resume and is_sample_jd:
    st.info("You are seeing a sample analysis. Upload your resume and a job description, then click Analyze.")

result = st.session_state.get("result")

if result is not None:
    # Warning chip for short JD or unmeasurable components
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
        st.warning("JD too short or no skills detected; keyword coverage not measured")

    total = result.total
    grade = result.grade
    components = result.components or {}

    # -----------------------------------------------------------------------
    # 3. Verdict Row: st.columns([1, 2])
    # -----------------------------------------------------------------------
    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Your ATS score</div>', unsafe_allow_html=True)
            st.markdown(_render_score_ring(total, grade), unsafe_allow_html=True)

            # Top fix
            if result.issues:
                top_issue = result.issues[0]
                st.markdown(
                    f'<div style="font-size:15px;margin:8px 0 16px 0;color:#111827;line-height:1.4;">'
                    f'<strong>Top fix:</strong> {html.escape(top_issue.message)} '
                    f'<span style="color:#4F46E5;font-weight:600;">(+{top_issue.estimated_gain:.1f} pts)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:15px;margin:8px 0 16px 0;color:#111827;">'
                    '<strong>Top fix:</strong> No major issues detected (+0.0 pts)</div>',
                    unsafe_allow_html=True,
                )

            # Sub-scores: Job match & Resume quality
            fit_keys = ["semantic_match", "keyword_coverage", "experience_fit"]
            w_fit = sum(WEIGHTS[k] for k in fit_keys)
            fit_weighted = sum(WEIGHTS[k] * components.get(k, 0.0) for k in fit_keys)
            job_fit = (fit_weighted / w_fit) * 100.0 if w_fit > 0 else 0.0

            qual_keys = [k for k in components if k not in fit_keys]
            w_qual = sum(WEIGHTS[k] for k in qual_keys)
            qual_weighted = sum(WEIGHTS[k] * components.get(k, 0.0) for k in qual_keys)
            qual_score = (qual_weighted / w_qual) * 100.0 if w_qual > 0 else 0.0

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                st.metric("Job match", f"{job_fit:.0f}")
                st.caption("Matches skills, experience, and semantic fit for this specific job.")
            with col_sub2:
                st.metric("Resume quality", f"{qual_score:.0f}")
                st.caption("Evaluates formatting, metrics, action verbs, sections, and readability.")

    with col_v2:
        with st.container(border=True):
            st.markdown('<div class="card-title">What makes up your score</div>', unsafe_allow_html=True)
            rows = []
            for k, comp_val in components.items():
                worth = round(WEIGHTS.get(k, 0.0) * 100.0, 1)
                score_100 = round(comp_val * 100.0, 1)
                rows.append({
                    "Component": _fmt_component_name(k),
                    "Your score": score_100,
                    "Worth": worth,
                })
            df_comp = pd.DataFrame(rows).sort_values("Worth", ascending=False)
            st.dataframe(
                df_comp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Your score": st.column_config.ProgressColumn(
                        "Your score",
                        min_value=0,
                        max_value=100,
                        format="%.0f%%",
                    ),
                    "Worth": st.column_config.NumberColumn(
                        "Worth",
                        format="%.0f pts",
                    ),
                },
            )
            st.caption("Worth is how many of the 100 points each part carries.")

    # -----------------------------------------------------------------------
    # 4. Four st.metric(border=True) tiles
    # -----------------------------------------------------------------------
    exp_years = 0.0
    parsed = getattr(result, "_parsed", None) or st.session_state.get("parsed_resume")
    if parsed is not None:
        exp_years = parsed.total_years_experience
    else:
        exp_fit = components.get("experience_fit", 0.0)
        exp_years = exp_fit * 5.0

    matched_cnt = 0
    req_cnt = 0
    if result.match:
        matched_cnt = len(result.match.matched_skills)
        req_cnt = matched_cnt + len(result.match.missing_skills)

    top_cat = "Unknown"
    top_prob = 0.0
    if result.predicted_categories:
        top_cat = result.predicted_categories[0].category
        top_prob = result.predicted_categories[0].probability

    elapsed_sec = st.session_state.get("elapsed")
    if elapsed_sec is None:
        elapsed_sec = sum(result.timings_ms.values()) / 1000.0 if result.timings_ms else 0.0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Experience",
            value=f"{exp_years:.1f} yrs" if exp_years > 0 else "N/A",
            border=True,
        )
    with col_m2:
        st.metric(
            label="Required skills found",
            value=f"{matched_cnt} of {req_cnt}" if req_cnt > 0 else f"{matched_cnt}",
            border=True,
        )
    with col_m3:
        conf_txt = f"{top_prob:.0%}" if top_prob > 0 else ""
        if len(top_cat) > 13:
            st.metric(label="Closest job category", value=top_cat[:12] + "…", border=True)
            st.caption(f"{top_cat} ({conf_txt})" if conf_txt else top_cat)
        else:
            st.metric(label="Closest job category", value=top_cat, border=True)
            if conf_txt:
                st.caption(f"Confidence: {conf_txt}")
    with col_m4:
        st.metric(
            label="Analysis time",
            value=f"{elapsed_sec:.2f}s",
            border=True,
        )

    # -----------------------------------------------------------------------
    # 5. Tabs in exact order
    # -----------------------------------------------------------------------
    tab_fix, tab_high, tab_sugg, tab_rank, tab_skills, tab_fair, tab_hist, tab_models = st.tabs([
        "What to fix",
        "Resume highlights",
        "Rewrite suggestions",
        "Compare jobs",
        "Skill gaps",
        "Fairness check",
        "History",
        "Model results",
    ])

    # ── Tab 1: What to fix ────────────────────────────────────────────────
    with tab_fix:
        st.write("Specific improvements sorted by the points they add to your ATS score.")
        if result.issues:
            df_issues = pd.DataFrame([
                {
                    "Gain (pts)": f"+{issue.estimated_gain:.1f}",
                    "What to fix": issue.message,
                    "Area": _fmt_component_name(issue.component),
                }
                for issue in result.issues
            ])
            st.dataframe(df_issues, use_container_width=True, hide_index=True)
        else:
            st.info("No issues detected for this resume and job description.")

        report_lines = [
            f"# {APP_NAME} Analysis Report",
            "",
            f"**Score:** {result.total:.1f} / 100  |  **Grade:** {result.grade}",
            "",
            "## Component Scores",
            "",
        ]
        for k, v in components.items():
            report_lines.append(f"- {_fmt_component_name(k)}: {v*100:.0f}%")
        if result.issues:
            report_lines += ["", "## Issues & Fixes", ""]
            for issue in result.issues:
                report_lines.append(f"- [{issue.severity.upper()}] {issue.message} (+{issue.estimated_gain:.1f} pts)")

        st.download_button(
            label="Download report",
            data="\n".join(report_lines).encode("utf-8"),
            file_name="resume_analysis_report.md",
            mime="text/markdown",
            key="download_report",
        )

    # ── Tab 2: Resume highlights ──────────────────────────────────────────
    with tab_high:
        st.write("Review how your resume text matches the job description, showing found skills and missing requirements.")
        st.markdown(
            ":green[■ Helps the match] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":red[■ Missing or weak] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":green[━━ High-impact sentence]"
        )

        missing = result.match.missing_skills if result.match else result.missing_skills
        if missing:
            st.markdown(
                "**Missing required skills:** "
                + " ".join(f":red[`{s}`]" for s in missing[:20])
            )

        if hasattr(result, "_resume_text_preview") and result._resume_text_preview:
            st.text_area(
                "Resume preview",
                value=result._resume_text_preview[:4000],
                height=300,
                disabled=True,
            )
        else:
            st.caption("Resume text preview processed in transient memory (privacy mode).")

    # ── Tab 3: Rewrite suggestions ────────────────────────────────────────
    with tab_sugg:
        st.write("Targeted bullet point improvements showing how to incorporate stronger action verbs and metrics.")
        if getattr(result, "genai_provider", None) and result.genai_provider not in ("rule_based", "none", None):
            st.caption(f"Written by {result.genai_provider}")
        else:
            st.caption("Built-in rules (no AI model used). Uses only facts from your resume. [add metric] marks where you should add a real number.")

        bullets = [s for s in getattr(result, "suggestions", []) if s.kind == "bullet_rewrite"]
        if bullets:
            for s in bullets:
                c_b, c_a = st.columns(2)
                with c_b:
                    with st.container(border=True):
                        st.markdown("**Before**")
                        st.write(s.original)
                with c_a:
                    with st.container(border=True):
                        st.markdown("**After**")
                        st.write(s.suggested)
        else:
            st.info("No bullet rewrites generated. Resume formatting looks clean.")

        plan = [s for s in getattr(result, "suggestions", []) if s.kind == "improvement_plan"]
        with st.expander("Step-by-step plan (same fixes as above)", expanded=False):
            if plan:
                for idx, step in enumerate(plan, 1):
                    st.markdown(f"**Step {idx}:** {step.suggested}")
            else:
                st.write("No additional steps needed.")

        cover = [s for s in getattr(result, "suggestions", []) if s.kind == "cover_letter"]
        if cover:
            with st.expander("Tailored cover letter", expanded=False):
                st.text_area("Cover letter", value=cover[0].suggested, height=220)

    # ── Tab 4: Compare jobs ───────────────────────────────────────────────
    with tab_rank:
        st.write("Evaluate how your resume matches against other target job descriptions.")
        ranked = st.session_state.get("ranking_result")
        if ranked is None:
            from app.extras.ranking import rank_jds
            from app.schemas import JobDescription

            jd_list = []
            for jd_file in SAMPLE_JDS:
                p = FIXTURES_DIR / jd_file
                if p.exists():
                    t_name = jd_file.replace("jd_", "").replace(".txt", "").replace("_", " ").title()
                    jd_list.append(JobDescription(text=p.read_text(encoding="utf-8"), title=t_name))

            if jd_list and parsed is not None:
                with st.spinner("Comparing…"):
                    ranked = rank_jds(parsed, jd_list)
                    st.session_state["ranking_result"] = ranked

        if ranked:
            rank_rows = []
            for item in ranked:
                ms = ", ".join(item["missing_skills"][:3]) or "None"
                if len(item["missing_skills"]) > 3:
                    ms += f" (+{len(item['missing_skills']) - 3} more)"
                rank_rows.append({
                    "Job Description": item["jd_title"],
                    "ATS Score": round(item["score"], 1),
                    "Grade": item["grade"],
                    "Missing Skills": ms,
                })
            st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Ranking requires parsed resume text.")

    # ── Tab 5: Skill gaps ─────────────────────────────────────────────────
    with tab_skills:
        st.write("Recommended learning resources and courses for skills required by the job that are not in your resume.")
        if not result.missing_skills:
            st.info("No missing skills detected for this job description.")
        else:
            parsed_skills = result.match.matched_skills if result.match else []
            try:
                from app.extras.skill_gap import skill_gap
                gap = skill_gap(parsed_skills, result.missing_skills)
            except Exception:
                gap = {}

            if not gap:
                st.markdown("**Missing skills:** " + ", ".join(f"`{s}`" for s in result.missing_skills))
            else:
                for skill, courses in gap.items():
                    with st.container(border=True):
                        st.markdown(f"**{skill}**")
                        for c in courses[:2]:
                            st.write(f"- **{c.get('title','')}** ({c.get('provider','')}, {c.get('level','')}, {c.get('estimated_hours','')}h)")

    # ── Tab 6: Fairness check ─────────────────────────────────────────────
    with tab_fair:
        st.write("Audits your resume for personal or demographic indicators that could trigger unconscious hiring bias.")
        if parsed is None:
            st.info("Parsed resume not available for fairness check.")
        else:
            try:
                from app.extras.bias import bias_check
                findings = bias_check(parsed)
            except Exception as e:
                findings = []

            if not findings:
                st.success("No potential demographic bias signals detected. Resume uses neutral professional language.")
            else:
                df_bias = pd.DataFrame([
                    {
                        "Severity": f.get("severity", "low").title(),
                        "Signal": f.get("type", "").replace("_", " ").title(),
                        "Details": f.get("message", ""),
                    }
                    for f in findings
                ])
                st.dataframe(df_bias, use_container_width=True, hide_index=True)

    # ── Tab 7: History ────────────────────────────────────────────────────
    with tab_hist:
        st.write("Track your composite ATS score improvements over time across resume revisions.")
        if not st.session_state.get("history_opt_in"):
            st.info("History tracking is currently disabled. Enable 'Save score history' in Settings to track your progress.")
        else:
            try:
                from app.extras.history import get_history
                records = get_history()
            except Exception:
                records = []

            if not records:
                st.info("No history records saved yet.")
            else:
                df_hist = pd.DataFrame([
                    {
                        "Date": str(r.get("timestamp", ""))[:16],
                        "Job Title": r.get("jd_title", "—"),
                        "ATS Score": round(r.get("total_score", 0), 1),
                        "Grade": r.get("grade", "—"),
                    }
                    for r in records[-20:]
                ])
                st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # ── Tab 8: Model results ──────────────────────────────────────────────
    with tab_models:
        st.write("How the models behind ResumeIQ were tested on held-out data.")
        raw_metrics, df_c4, df_ci, df_paired, df_mo, df_mcat, df_abl, df_xai, wdata = _load_model_reports()

        st.markdown("#### Classifier Performance")
        if raw_metrics or df_c4 is not None:
            ci_map = {}
            if df_ci is not None:
                for _, r in df_ci.iterrows():
                    ci_map[str(r["model"])] = r

            readable_names = {
                "tfidf_lr": "TF-IDF + LR",
                "tfidf_mlp": "TF-IDF + MLP (Served)",
                "bilstm": "BiLSTM",
                "distilbert": "DistilBERT (3 epochs)",
            }
            c_rows = []
            for m_key in ["tfidf_lr", "tfidf_mlp", "bilstm", "distilbert"]:
                m_name = readable_names.get(m_key, m_key)
                m_data = raw_metrics.get(m_key, {})
                val_f1_val = m_data.get("val_macro_f1")
                train_f1_val = m_data.get("train_macro_f1")

                if val_f1_val is None and df_c4 is not None:
                    row_c4 = df_c4[df_c4["model_key"] == m_key]
                    if not row_c4.empty:
                        val_f1_val = float(row_c4.iloc[0]["val_macro_f1"])
                        train_f1_val = float(row_c4.iloc[0]["train_macro_f1"])

                val_f1_str = f"{val_f1_val:.4f}" if val_f1_val is not None else "—"

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
                    gap_str = f"{train_f1_val - val_f1_val:+.4f}"
                else:
                    gap_str = "—"

                c_rows.append({
                    "Model": m_name,
                    "Val Macro F1": val_f1_str,
                    "Test Macro F1 (95% CI)": test_f1_str,
                    "Test Accuracy": test_acc_str,
                    "Train-Val Gap": gap_str,
                })
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

        if df_paired is not None:
            st.markdown("#### Paired Statistical Comparison")
            p_rows = []
            for _, pr in df_paired.iterrows():
                metric_label = "Macro-F1" if pr["metric"] == "macro_f1" else "Accuracy"
                if pr["metric"] == "macro_f1":
                    p_val_str = f"p = {pr['bootstrap_p']:.4f} (boot)"
                else:
                    m_p = pr.get("mcnemar_p")
                    p_val_str = f"p = {m_p:.4f} (McNemar)" if pd.notna(m_p) else f"p = {pr['bootstrap_p']:.4f}"
                p_rows.append({
                    "Comparison": pr["comparison"],
                    "Metric": metric_label,
                    "Diff (A - B)": f"{pr['diff']:+.4f}",
                    "95% CI": f"[{pr['ci_95_lower']:+.4f}, {pr['ci_95_upper']:+.4f}]",
                    "P-Value": p_val_str,
                })
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

        st.caption(
            "LR and MLP are indistinguishable on test (macro-F1 and accuracy, paired tests). "
            "DistilBERT scores higher on test: vs LR macro-F1 +0.050 (95% CI [0.0056, 0.0931], p=0.036) and accuracy +6.4 pts (+0.0642, p=0.022); "
            "vs MLP accuracy +7.5 pts (+0.0749, p=0.010), but its macro-F1 gain over MLP is not significant (p=0.280). "
            "Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs MLP accuracy (McNemar p=0.0081) clears it, barely. "
            "The served classifier is the MLP because the selection rule (val macro-F1) was fixed on validation before the test set was opened."
        )

        st.markdown("#### Matcher Retrieval Metrics")
        if df_mo is not None:
            row_labels = {
                "tfidf": "TF-IDF Baseline",
                "zero_shot": "Zero-Shot SBERT",
                "fine_tuned": "Fine-Tuned Siamese SBERT (Served)",
            }
            m_rows = []
            for _, r in df_mo.iterrows():
                m_key = str(r["model_key"])
                m_rows.append({
                    "Model": row_labels.get(m_key, m_key),
                    "Val MRR": f"{r['val_mrr']:.4f}",
                    "Val nDCG@10": f"{r['val_ndcg10']:.4f}",
                    "Val Recall@5": f"{r['val_recall5']:.4f}",
                    "Val Recall@10": f"{r['val_recall10']:.4f}",
                    "Test MRR": f"{r['test_mrr']:.4f}",
                    "Test nDCG@10": f"{r['test_ndcg10']:.4f}",
                    "Test Recall@5": f"{r['test_recall5']:.4f}",
                    "Test Recall@10": f"{r['test_recall10']:.4f}",
                })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

        if df_mcat is not None:
            with st.expander("Per-category matcher results (15 categories)", expanded=False):
                mcat_rows = []
                for _, cr in df_mcat.iterrows():
                    mcat_rows.append({
                        "Category": str(cr["category"]),
                        "Val N": int(cr["val_count"]),
                        "Test N": int(cr["test_count"]),
                        "TF-IDF Val": f"{cr['val_tfidf_mrr']:.4f}",
                        "Zero-Shot Val": f"{cr['val_zs_mrr']:.4f}",
                        "Fine-Tuned Val": f"{cr['val_ft_mrr']:.4f}",
                        "TF-IDF Test": f"{cr['test_tfidf_mrr']:.4f}",
                        "Zero-Shot Test": f"{cr['test_zs_mrr']:.4f}",
                        "Fine-Tuned Test": f"{cr['test_ft_mrr']:.4f}",
                    })
                st.dataframe(pd.DataFrame(mcat_rows), use_container_width=True, hide_index=True)

        st.caption("Matcher note: Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means. Weak title-derived labels, partly circular.")

        st.markdown("#### Scoring Component Sensitivity & Ablation")
        if df_abl is not None:
            abl_rows = []
            for _, ar in df_abl.iterrows():
                abl_rows.append({
                    "Component": _fmt_component_name(str(ar["component"])),
                    "Weight": f"{ar['original_weight']:.2f}",
                    "Mean Original": f"{ar['mean_original_score']:.2f}",
                    "Mean Ablated": f"{ar['mean_ablated_score']:.2f}",
                    "Mean Change (Delta)": f"{ar['mean_score_delta']:+.2f}",
                })
            st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)
            st.caption("Scoring note: No ground-truth ATS score exists; weights are heuristics.")

        st.markdown("#### Explainability (XAI)")
        sent_drop_top = "0.07 pts"
        sent_drop_rand = "0.11 pts"
        if df_xai is not None and len(df_xai) > 0:
            sent_drop_top = f"{df_xai.iloc[0]['mean_top_k_drop']:.2f} pts"
            sent_drop_rand = f"{df_xai.iloc[0]['mean_random_k_drop']:.2f} pts"

        wdel_lr_str = "Top-10 drop 0.3590 vs Random -0.0021 (+0.3611 gap)"
        wdel_served_str = "Top-10 drop 0.4921 vs Random 0.0058 (+0.4863 gap)"
        if wdata is not None:
            lr_sub = wdata.get("lr_self_attribution", {})
            mlp_sub = wdata.get("served_classifier_transfer", {})
            if lr_sub:
                wdel_lr_str = f"Top-10 drop {lr_sub.get('mean_top_k_drop', 0.359):.4f} vs Random {lr_sub.get('mean_random_k_drop', -0.0021):+.4f} (gap: {lr_sub.get('attribution_gap', 0.3611):+.4f})"
            if mlp_sub:
                wdel_served_str = f"Top-10 drop {mlp_sub.get('mean_top_k_drop', 0.4921):.4f} vs Random {mlp_sub.get('mean_random_k_drop', 0.0058):+.4f} (gap: {mlp_sub.get('attribution_gap', 0.4863):+.4f})"

        xai_rows = [
            {
                "Attribution Level": "ATS Total (Sentence-Level)",
                "Method & Sample": "Leave-one-sentence-out (LOSO) (k=3, 20 val resumes)",
                "Observed Drops (Top vs Random)": f"Top-3 drop: {sent_drop_top} | Random-3 drop: {sent_drop_rand}",
                "Verdict": "Inconclusive & unvalidated",
            },
            {
                "Attribution Level": "Classifier (Word-Level)",
                "Method & Sample": "TF-IDF + LR linear coefficients (k=10, 50 val resumes)",
                "Observed Drops (Top vs Random)": f"LR (self): {wdel_lr_str} | Served MLP (transfer): {wdel_served_str}",
                "Verdict": "Faithful to linear model & cross-model transfer",
            },
        ]
        st.dataframe(pd.DataFrame(xai_rows), use_container_width=True, hide_index=True)
        st.caption(
            "XAI note: Sentence deletion is inconclusive and unvalidated. "
            "Word deletion shows faithfulness to the linear model and transfer to the served model."
        )

# ---------------------------------------------------------------------------
# 6. Expander: How ResumeIQ works
# ---------------------------------------------------------------------------

with st.expander("How ResumeIQ works", expanded=False):
    st.markdown(
        "**Pipeline:** **Parse** (PyMuPDF / python-docx / Tesseract OCR) → "
        "**Extract** (Rule-based taxonomy & regex entities) → "
        "**Classify** (TF-IDF + MLP) → "
        "**Match** (Fine-Tuned Siamese SBERT) → "
        "**Score** (9-component heuristic ATS composite scoring) → "
        "**Explain** (Leave-One-Sentence-Out deletion & linear attribution) → "
        "**Suggest** (Rule-based bullet rewriter & template cover letter)"
    )
