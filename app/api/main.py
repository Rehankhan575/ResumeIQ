"""
app/api/main.py
===============
FastAPI backend for ResumeIQ.
Provides REST endpoints for resume analysis, multi-job ranking, and sample fixtures.

Privacy & Security Guarantees:
- Resume text is processed purely in-memory and NEVER logged or stored.
- Max file size strictly enforced at 5 MB.
- Allowed file extensions: .pdf, .docx, .png, .jpg, .jpeg.
- CORS enabled for http://localhost:5173.
"""

from __future__ import annotations

import os
# Segfault-mitigation: set before torch/transformers imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import ScoreResult
from app.pipeline import run_pipeline, warm_up

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="ResumeIQ API",
    description="REST backend for ResumeIQ ATS optimization and scoring engine.",
    version="1.0.0",
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
    except Exception as e:
        # Non-fatal if models load lazily
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


@app.post("/api/analyze", response_model=ScoreResult)
async def analyze_resume(
    file: Optional[UploadFile] = File(None),
    resume_choice: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
    jd_choice: Optional[str] = Form(None),
    use_genai: bool = Form(False),
) -> ScoreResult:
    """
    Analyze a resume against a job description.
    Enforces 5 MB limit, allowed extensions, and memory-only privacy guarantees.
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
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )


@app.post("/api/rank")
async def rank_multi_jds(
    file: Optional[UploadFile] = File(None),
    resume_choice: Optional[str] = Form(None),
    jd_filenames: List[str] = Form(...),
) -> Dict[str, Any]:
    """Rank multiple job descriptions against a given resume."""
    from app.extras.ranking import rank_jds
    from app.schemas import JobDescription

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

