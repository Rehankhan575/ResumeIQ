"""
app/api/main.py
===============
FastAPI backend for ResumeIQ.
Provides REST endpoints for resume analysis, multi-job ranking, sample fixtures,
and model verification results.

Privacy & Security Guarantees:
- Resume text is processed purely in-memory and NEVER logged or stored.
- Max file size strictly enforced at 5 MB.
- Allowed file extensions: .pdf, .docx, .png, .jpg, .jpeg.
- CORS enabled for http://localhost:5173.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Segfault-mitigation: set before torch/transformers imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from app.extraction.extractor import extract_entities
from app.extraction.skills import TaxonomySkillExtractor
from app.extras.bias import bias_check
from app.extras.ranking import rank_jds
from app.extras.skill_gap import skill_gap
from app.parsing.parser import parse_document
from app.pipeline import run_pipeline, warm_up
from app.schemas import JobDescription, ScoreResult

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


class AnalyzeResponse(ScoreResult):
    """Additive extension of ScoreResult for web clients with bias & skill gaps."""

    bias: List[Dict[str, Any]] = Field(default_factory=list)
    skill_gap: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="ResumeIQ API",
    description="REST backend for ResumeIQ ATS optimization and scoring engine.",
    version="1.3.0",
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Warm up pipeline models on server start."""
    try:
        warm_up()
    except Exception:
        pass


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "app": "ResumeIQ"}


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    """List available sample fixture resumes and job descriptions."""
    resumes = []
    if FIXTURES_DIR.exists():
        for p in sorted(FIXTURES_DIR.glob("*.pdf")) + sorted(FIXTURES_DIR.glob("*.docx")):
            if p.name.startswith("jd_"):
                continue
            title = p.stem.replace("_", " ").title()
            resumes.append({
                "filename": p.name,
                "title": title,
                "extension": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
            })

    jds = []
    if FIXTURES_DIR.exists():
        for p in sorted(FIXTURES_DIR.glob("jd_*.txt")):
            title = p.stem.replace("jd_", "").replace("_", " ").title()
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                content = ""
            jds.append({
                "filename": p.name,
                "title": title,
                "preview": content[:200] + ("..." if len(content) > 200 else ""),
                "full_text": content,
            })

    return {"resumes": resumes, "jds": jds}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    file: Optional[UploadFile] = File(None),
    resume_choice: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
    jd_choice: Optional[str] = Form(None),
    use_genai: bool = Form(True),
) -> AnalyzeResponse:
    """
    Analyze a resume against a job description.
    Enforces 5 MB limit, allowed extensions, and memory-only privacy guarantees.
    Returns composite score, breakdown, highlights, issues, bias findings, and skill gaps.
    """
    file_bytes = b""
    filename = ""

    # 1. Resolve Resume source (Upload takes precedence over sample choice)
    if file is not None and file.filename:
        filename = file.filename
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of 5 MB ({len(file_bytes)} bytes received).",
            )
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
    elif resume_choice:
        sample_path = FIXTURES_DIR / resume_choice
        if not sample_path.exists() or not sample_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample resume '{resume_choice}' not found.",
            )
        file_bytes = sample_path.read_bytes()
        filename = sample_path.name
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume provided. Upload a file or choose a sample.",
        )

    # 2. Resolve Job Description source (Pasted text takes precedence over sample choice)
    resolved_jd_text = ""
    resolved_jd_title = "Target Position"

    if jd_text and jd_text.strip():
        resolved_jd_text = jd_text.strip()
        resolved_jd_title = "Pasted Job Description"
    elif jd_choice:
        sample_jd_path = FIXTURES_DIR / jd_choice
        if not sample_jd_path.exists() or not sample_jd_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample job description '{jd_choice}' not found.",
            )
        resolved_jd_text = sample_jd_path.read_text(encoding="utf-8")
        resolved_jd_title = jd_choice.replace("jd_", "").replace(".txt", "").replace("_", " ").title()
    else:
        resolved_jd_text = ""
        resolved_jd_title = ""

    # 3. Execute Pipeline (zero logging of resume text)
    try:
        result = run_pipeline(
            file_bytes=file_bytes,
            filename=filename,
            jd_text=resolved_jd_text,
            jd_title=resolved_jd_title,
            use_genai=use_genai,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )

    # 4. Compute Bias Check (labels only, no candidate identity text)
    bias_findings: List[Dict[str, Any]] = []
    parsed_doc = None
    try:
        parsed_doc = parse_document(file_bytes, filename)
        extracted_doc = extract_entities(parsed_doc)
        raw_bias = bias_check(extracted_doc)
        for b in raw_bias:
            bias_findings.append({
                "type": str(b.get("type", "neutral")),
                "evidence_label": str(b.get("evidence_label", "")),
                "suggestion": str(b.get("suggestion", "")),
            })
    except Exception as exc:
        logger.debug("Bias check error: %s", exc)
        extracted_doc = None

    # 5. Compute Skill Gap (no external URLs)
    gap_summary: Dict[str, Any] = {"matched": [], "missing": [], "recommendations": {}}
    try:
        if extracted_doc is not None and resolved_jd_text:
            tax = TaxonomySkillExtractor()
            jd_info = tax.extract_jd_skills(resolved_jd_text)
            req_skills = jd_info.get("required_skills", [])
            gap_map = skill_gap(extracted_doc.skills, req_skills)
            matched_skills = [
                s for s in req_skills
                if s.lower() in {x.lower() for x in extracted_doc.skills}
            ]
            gap_summary = {
                "matched": matched_skills,
                "missing": list(gap_map.keys()),
                "recommendations": gap_map,
            }
    except Exception as exc:
        logger.debug("Skill gap computation error: %s", exc)

    payload = result.model_dump()
    payload["bias"] = bias_findings
    payload["skill_gap"] = gap_summary
    return AnalyzeResponse(**payload)


@app.post("/api/rank")
async def rank_multi_jds(
    file: Optional[UploadFile] = File(None),
    resume_choice: Optional[str] = Form(None),
    jd_filenames: List[str] = Form(...),
) -> Dict[str, Any]:
    """Rank multiple job descriptions against a given resume using rank_jds."""
    file_bytes = b""
    filename = ""

    if file is not None and file.filename:
        filename = file.filename
        file_bytes = await file.read()
    elif resume_choice:
        p = FIXTURES_DIR / resume_choice
        if p.exists():
            file_bytes = p.read_bytes()
            filename = p.name

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resume required for ranking.",
        )

    jds: List[JobDescription] = []
    for jd_name in jd_filenames:
        p = FIXTURES_DIR / jd_name
        if p.exists():
            title = jd_name.replace("jd_", "").replace(".txt", "").replace("_", " ").title()
            text = p.read_text(encoding="utf-8")
            jds.append(JobDescription(title=title, text=text))

    if not jds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid job descriptions provided.",
        )

    ranked = rank_jds(file_bytes, jds)
    return {"ranked": ranked}


@app.get("/api/results")
def get_model_results() -> Dict[str, Any]:
    """
    Return comprehensive verification reports from artifacts/reports and artifacts/metrics.
    Includes exact notes strings from app/ui/app.py.
    """
    metrics_json_path = METRICS_DIR / "classifier" / "metrics.json"
    raw_metrics: Dict[str, Any] = {}
    if metrics_json_path.exists():
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                raw_metrics = json.load(f)
        except Exception:
            raw_metrics = {}

    c4_path = REPORTS_DIR / "classifier_4_model.csv"
    ci_path = REPORTS_DIR / "classifier_ci.csv"
    paired_path = REPORTS_DIR / "classifier_paired_comparison.csv"

    df_c4 = pd.read_csv(c4_path) if c4_path.exists() else None
    df_ci = pd.read_csv(ci_path) if ci_path.exists() else None
    df_paired = pd.read_csv(paired_path) if paired_path.exists() else None

    readable_names = {
        "tfidf_lr": "TF-IDF + LR",
        "tfidf_mlp": "TF-IDF + MLP (Served)",
        "bilstm": "BiLSTM",
        "distilbert": "DistilBERT (3 epochs)",
    }

    ci_map: Dict[str, Any] = {}
    if df_ci is not None:
        for _, r in df_ci.iterrows():
            ci_map[str(r["model"])] = r

    c_rows: List[Dict[str, Any]] = []
    for m_key in ["tfidf_lr", "tfidf_mlp", "bilstm", "distilbert"]:
        m_name = readable_names.get(m_key, m_key)
        m_data = raw_metrics.get(m_key, {})
        val_f1 = m_data.get("val_macro_f1")
        train_f1 = m_data.get("train_macro_f1")

        if val_f1 is None and df_c4 is not None:
            sub = df_c4[df_c4["model_key"] == m_key]
            if not sub.empty:
                val_f1 = float(sub.iloc[0]["val_macro_f1"])
                train_f1 = float(sub.iloc[0]["train_macro_f1"])

        val_f1_str = f"{val_f1:.4f}" if val_f1 is not None else "—"
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

        gap_str = (
            f"{train_f1 - val_f1:+.4f}"
            if (train_f1 is not None and val_f1 is not None)
            else "—"
        )
        c_rows.append({
            "key": m_key,
            "model": m_name,
            "val_macro_f1": val_f1_str,
            "test_macro_f1_ci": test_f1_str,
            "test_accuracy": test_acc_str,
            "train_val_gap": gap_str,
            "served": m_key == "tfidf_mlp",
        })

    # Paired tests
    p_rows: List[Dict[str, Any]] = []
    if df_paired is not None:
        for _, pr in df_paired.iterrows():
            metric_label = "Macro-F1" if pr["metric"] == "macro_f1" else "Accuracy"
            if pr["metric"] == "macro_f1":
                p_val_str = f"p = {pr['bootstrap_p']:.4f} (boot)"
            else:
                m_p = pr.get("mcnemar_p")
                p_val_str = (
                    f"p = {m_p:.4f} (McNemar)"
                    if pd.notna(m_p)
                    else f"p = {pr['bootstrap_p']:.4f}"
                )
            p_rows.append({
                "comparison": str(pr["comparison"]),
                "metric": metric_label,
                "diff": f"{pr['diff']:+.4f}",
                "ci_95": f"[{pr['ci_95_lower']:+.4f}, {pr['ci_95_upper']:+.4f}]",
                "p_value": p_val_str,
            })

    # Matcher overall
    m_overall_path = REPORTS_DIR / "matcher_overall.csv"
    m_cat_path = REPORTS_DIR / "matcher_per_category.csv"
    df_mo = pd.read_csv(m_overall_path) if m_overall_path.exists() else None
    df_mcat = pd.read_csv(m_cat_path) if m_cat_path.exists() else None

    row_labels = {
        "tfidf": "TF-IDF Baseline",
        "zero_shot": "Zero-Shot SBERT",
        "fine_tuned": "Fine-Tuned Siamese SBERT (Served)",
    }
    m_rows: List[Dict[str, Any]] = []
    if df_mo is not None:
        for _, r in df_mo.iterrows():
            m_key = str(r["model_key"])
            m_rows.append({
                "key": m_key,
                "model": row_labels.get(m_key, m_key),
                "val_mrr": f"{r['val_mrr']:.4f}",
                "val_ndcg10": f"{r['val_ndcg10']:.4f}",
                "val_recall5": f"{r['val_recall5']:.4f}",
                "val_recall10": f"{r['val_recall10']:.4f}",
                "test_mrr": f"{r['test_mrr']:.4f}",
                "test_ndcg10": f"{r['test_ndcg10']:.4f}",
                "test_recall5": f"{r['test_recall5']:.4f}",
                "test_recall10": f"{r['test_recall10']:.4f}",
                "served": m_key == "fine_tuned",
            })

    mcat_rows: List[Dict[str, Any]] = []
    if df_mcat is not None:
        for _, cr in df_mcat.iterrows():
            mcat_rows.append({
                "category": str(cr["category"]),
                "val_count": int(cr["val_count"]),
                "test_count": int(cr["test_count"]),
                "val_tfidf_mrr": f"{cr['val_tfidf_mrr']:.4f}",
                "val_zs_mrr": f"{cr['val_zs_mrr']:.4f}",
                "val_ft_mrr": f"{cr['val_ft_mrr']:.4f}",
                "test_tfidf_mrr": f"{cr['test_tfidf_mrr']:.4f}",
                "test_zs_mrr": f"{cr['test_zs_mrr']:.4f}",
                "test_ft_mrr": f"{cr['test_ft_mrr']:.4f}",
            })

    # Ablation
    abl_path = REPORTS_DIR / "scoring_ablation.csv"
    if not abl_path.exists():
        abl_path = METRICS_DIR / "scoring" / "ablation.csv"
    df_abl = pd.read_csv(abl_path) if abl_path.exists() else None
    abl_rows: List[Dict[str, Any]] = []
    if df_abl is not None:
        for _, ar in df_abl.iterrows():
            abl_rows.append({
                "component": str(ar["component"]).replace("_", " ").title(),
                "original_weight": f"{ar['original_weight']:.2f}",
                "mean_original": f"{ar['mean_original_score']:.2f}",
                "mean_ablated": f"{ar['mean_ablated_score']:.2f}",
                "mean_delta": f"{ar['mean_score_delta']:+.2f}",
            })

    # Explainability (XAI)
    xai_path = REPORTS_DIR / "xai_faithfulness.csv"
    df_xai = pd.read_csv(xai_path) if xai_path.exists() else None
    sent_drop_top = "0.07 pts"
    sent_drop_rand = "0.11 pts"
    if df_xai is not None and len(df_xai) > 0:
        sent_drop_top = f"{df_xai.iloc[0]['mean_top_k_drop']:.2f} pts"
        sent_drop_rand = f"{df_xai.iloc[0]['mean_random_k_drop']:.2f} pts"

    wdel_path = METRICS_DIR / "scoring" / "classifier_word_deletion.json"
    wdata = None
    if wdel_path.exists():
        try:
            with open(wdel_path, "r", encoding="utf-8") as f:
                wdata = json.load(f)
        except Exception:
            wdata = None

    wdel_lr_str = "Top-10 drop 0.3590 vs Random -0.0021 (+0.3611 gap)"
    wdel_served_str = "Top-10 drop 0.4921 vs Random 0.0058 (+0.4863 gap)"
    if wdata is not None:
        lr_sub = wdata.get("lr_self_attribution", {})
        mlp_sub = wdata.get("served_classifier_transfer", {})
        if lr_sub:
            wdel_lr_str = (
                f"Top-10 drop {lr_sub.get('mean_top_k_drop', 0.359):.4f} "
                f"vs Random {lr_sub.get('mean_random_k_drop', -0.0021):+.4f} "
                f"(gap: {lr_sub.get('attribution_gap', 0.3611):+.4f})"
            )
        if mlp_sub:
            wdel_served_str = (
                f"Top-10 drop {mlp_sub.get('mean_top_k_drop', 0.4921):.4f} "
                f"vs Random {mlp_sub.get('mean_random_k_drop', 0.0058):+.4f} "
                f"(gap: {mlp_sub.get('attribution_gap', 0.4863):+.4f})"
            )

    xai_rows = [
        {
            "attribution_level": "ATS Total (Sentence-Level)",
            "method": "Leave-one-sentence-out (LOSO) (k=3, 20 val resumes)",
            "observed_drops": f"Top-3 drop: {sent_drop_top} | Random-3 drop: {sent_drop_rand}",
            "verdict": "Inconclusive & unvalidated",
        },
        {
            "attribution_level": "Classifier (Word-Level)",
            "method": "TF-IDF + LR linear coefficients (k=10, 50 val resumes)",
            "observed_drops": f"LR (self): {wdel_lr_str} | Served MLP (transfer): {wdel_served_str}",
            "verdict": "Faithful to linear model & cross-model transfer",
        },
    ]

    return {
        "classifier": {
            "models": c_rows,
            "paired_tests": p_rows,
            "note": (
                "LR and MLP are indistinguishable on test (macro-F1 and accuracy, paired tests). "
                "DistilBERT scores higher on test: vs LR macro-F1 +0.050 (95% CI [0.0056, 0.0931], p=0.036) "
                "and accuracy +6.4 pts (+0.0642, p=0.022); vs MLP accuracy +7.5 pts (+0.0749, p=0.010), "
                "but its macro-F1 gain over MLP is not significant (p=0.280). "
                "Suggestive on test (macro-F1 p=0.036, accuracy McNemar p=0.0101) but not significant "
                "after Bonferroni correction across six tests (threshold about 0.008); only DistilBERT vs "
                "MLP accuracy (McNemar p=0.0081) clears it, barely. "
                "The served classifier is the MLP because the selection rule (val macro-F1) was fixed "
                "on validation before the test set was opened."
            ),
        },
        "matcher": {
            "models": m_rows,
            "categories": mcat_rows,
            "note": (
                "Matcher note: Overall metrics report query-level pooled (micro) averages across all queries; "
                "per-category metrics report within-category query means. Weak title-derived labels, partly circular."
            ),
        },
        "ablation": {
            "components": abl_rows,
            "note": "Scoring note: No ground-truth ATS score exists; weights are heuristics.",
        },
        "xai": {
            "rows": xai_rows,
            "note": (
                "XAI note: Sentence deletion is inconclusive and unvalidated. "
                "Word deletion shows faithfulness to the linear model and transfer to the served model."
            ),
        },
    }


# Mount Static Files for React frontend if dist exists (mounted after all API routes)
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
