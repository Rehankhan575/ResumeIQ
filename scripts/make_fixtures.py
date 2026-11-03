"""
scripts/make_fixtures.py
========================
Deterministically generate synthetic test fixtures and expected sidecar JSONs.

Outputs in tests/fixtures/:
  1. ds_clean.pdf: Data scientist, single column, standard headings, Present job,
     skills: Python, sklearn (alias), PyTorch, SQL, Power BI.
  2. swe_two_column.pdf: Software engineer, two-column layout with sidebar,
     non-standard headings ("Work History", "Academics", "Tech Stack"), embedded image,
     skills: JS (alias), React, Docker, Git, Go.
  3. accountant_table.pdf: Skills inside a table, personal-details block (DOB, marital status),
     graduation year shown, gendered wording, skills: Excel, QuickBooks, GAAP.
  4. chef_plain.docx: DOCX, headings "Work Experience" & "Certifications" (ServSafe),
     date formats "Jan 2019 - Present" and "03/2016 - 12/2018",
     skills: HACCP, menu planning, food safety.
  5. nurse_scanned.pdf: Image-only PDF rasterised at 200 dpi with slight rotation and noise.
     No extractable text layer (<20 chars, >=1 image).
     Skills: patient care, vital signs, basic life support, triage, wound care.

  3 JDs as .txt (250-400 words each with required/preferred sections and min-years):
     - jd_data_scientist.txt
     - jd_software_engineer.txt
     - jd_chef.txt

  Sidecars <name>.expected.json for all 8 fixtures.
  Fails if any expected skill is not in data/taxonomies/skills.csv.

Privacy: All personas and text are 100% synthetic. Emails @example.com, phones 555-01xx.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

import docx
import numpy as np
import pandas as pd
import pymupdf
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Paths
ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
TAXONOMY_PATH = ROOT / "data" / "taxonomies" / "skills.csv"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

AS_OF_DATE = "2026-11-01"


# -- Taxonomy Verification Helper -------------------------------------------

def load_taxonomy_skills() -> Tuple[Set[str], Dict[str, str]]:
    """Load valid canonical skills and alias mapping from skills.csv."""
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Missing taxonomy file at {TAXONOMY_PATH}")

    df = pd.read_csv(TAXONOMY_PATH, keep_default_na=False)
    canonical = set(df["skill"].str.strip().str.lower())
    alias_to_canonical: Dict[str, str] = {}

    for _, row in df.iterrows():
        c_skill = row["skill"].strip().lower()
        aliases_str = row["aliases"]
        if aliases_str:
            for a in aliases_str.split("|"):
                a_clean = a.strip().lower()
                if a_clean:
                    alias_to_canonical[a_clean] = c_skill

    return canonical, alias_to_canonical


def validate_expected_skills(
    skills_list: List[str],
    canonical_skills: Set[str],
    alias_map: Dict[str, str],
    fixture_name: str,
) -> List[str]:
    """Validate and resolve expected skill names to canonical taxonomy keys."""
    resolved: List[str] = []
    for s in skills_list:
        sl = s.strip().lower()
        if sl in canonical_skills:
            resolved.append(sl)
        elif sl in alias_map:
            resolved.append(alias_map[sl])
        else:
            raise ValueError(
                f"Fixture '{fixture_name}' expects skill '{s}', but it does NOT exist in {TAXONOMY_PATH}"
            )
    return resolved


# -- Fixture 1: ds_clean.pdf ------------------------------------------------

def make_ds_clean(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate ds_clean.pdf and sidecar."""
    pdf_path = FIXTURES_DIR / "ds_clean.pdf"
    json_path = FIXTURES_DIR / "ds_clean.expected.json"

    expected_skills = validate_expected_skills(
        ["python", "sklearn", "pytorch", "sql", "power bi"],
        canonical_skills,
        alias_map,
        "ds_clean",
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
    )

    story = [
        Paragraph("Alex Mercer", title_style),
        Paragraph("alex.mercer@example.com | 555-0142 | San Francisco, CA", body_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=10),
        Paragraph("Summary", heading_style),
        Paragraph(
            "Experienced Data Scientist with 5 years of hands-on expertise building production predictive models "
            "and machine learning pipelines. Recognized for technical excellence, having excelled at delivering "
            "scalable algorithms in complex R&D environments.",
            body_style,
        ),
        Paragraph("Experience", heading_style),
        Paragraph("<b>Lead Data Scientist</b> — Apex Analytics Corp, San Francisco, CA", body_style),
        Paragraph("<i>March 2021 – Present</i>", body_style),
        Paragraph(
            "• Spearheaded deep learning model training using PyTorch for customer retention forecasting.<br/>"
            "• Architected distributed feature engineering pipelines using Python and SQL data warehouses.<br/>"
            "• Built automated predictive modeling toolkits leveraging sklearn for regression benchmarks.<br/>"
            "• Designed interactive executive dashboards in Power BI for cross-departmental reporting.",
            body_style,
        ),
        Paragraph("<b>Data Scientist</b> — BluePeak Insights, San Jose, CA", body_style),
        Paragraph("<i>June 2019 – February 2021</i>", body_style),
        Paragraph(
            "• Conducted SQL queries over massive relational tables to extract feature datasets.<br/>"
            "• Evaluated statistical model stability and supported algorithm R&D.",
            body_style,
        ),
        Paragraph("Education", heading_style),
        Paragraph("<b>Master of Science in Computer Science</b> — Stanford University, 2019", body_style),
        Paragraph("<b>Bachelor of Science in Statistics</b> — UC Berkeley, 2017", body_style),
        Paragraph("Skills", heading_style),
        Paragraph(
            "Python, SQL, PyTorch, sklearn, Power BI, Statistical Modeling, Machine Learning",
            body_style,
        ),
    ]

    doc.build(story)

    sidecar = {
        "as_of": AS_OF_DATE,
        "contact": {
            "name": "Alex Mercer",
            "email": "alex.mercer@example.com",
            "phone": "555-0142",
            "location": "San Francisco, CA",
        },
        "expected_sections": ["summary", "experience", "education", "skills"],
        "skills": expected_skills,
        "education": [
            {
                "degree": "Master of Science",
                "field": "Computer Science",
                "institution": "Stanford University",
                "graduation_year": 2019,
            },
            {
                "degree": "Bachelor of Science",
                "field": "Statistics",
                "institution": "UC Berkeley",
                "graduation_year": 2017,
            },
        ],
        "experience": [
            {
                "title": "Lead Data Scientist",
                "company": "Apex Analytics Corp",
                "start": "2021-03",
                "end": "Present",
                "is_current": True,
            },
            {
                "title": "Data Scientist",
                "company": "BluePeak Insights",
                "start": "2019-06",
                "end": "2021-02",
                "is_current": False,
            },
        ],
        "total_years_experience": 7.4,
        "layout": {
            "page_count": 1,
            "is_multi_column": False,
            "has_tables": False,
            "has_images": False,
            "needs_ocr": False,
        },
        "traps": ["excelled at", "R&D"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Generated ds_clean.pdf and ds_clean.expected.json")


# -- Fixture 2: swe_two_column.pdf ------------------------------------------

def make_swe_two_column(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate swe_two_column.pdf and sidecar."""
    pdf_path = FIXTURES_DIR / "swe_two_column.pdf"
    json_path = FIXTURES_DIR / "swe_two_column.expected.json"
    img_temp_path = FIXTURES_DIR / "_temp_avatar.png"

    expected_skills = validate_expected_skills(
        ["js", "react", "docker", "git", "go"],
        canonical_skills,
        alias_map,
        "swe_two_column",
    )

    # 1. Create a tiny square avatar image
    avatar = Image.new("RGB", (64, 64), color=(43, 108, 176))
    draw = ImageDraw.Draw(avatar)
    draw.rectangle([16, 16, 48, 48], fill=(237, 242, 247))
    avatar.save(img_temp_path)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "HeaderName",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#2D3748"),
    )
    side_heading = ParagraphStyle(
        "SideHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#3182CE"),
        spaceBefore=8,
        spaceAfter=2,
    )
    side_body = ParagraphStyle(
        "SideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
    )
    main_heading = ParagraphStyle(
        "MainHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=6,
        spaceAfter=3,
    )
    main_body = ParagraphStyle(
        "MainBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
    )

    # Sidebar Content
    sidebar = [
        RLImage(str(img_temp_path), width=48, height=48),
        Spacer(1, 8),
        Paragraph("<b>Contact</b>", side_heading),
        Paragraph("jordan.lee@example.com<br/>555-0187<br/>Seattle, WA", side_body),
        Spacer(1, 6),
        Paragraph("<b>Academics</b>", side_heading),
        Paragraph("B.S. Software Engineering<br/>University of Washington<br/>Graduated: Grade C honor", side_body),
        Spacer(1, 6),
        Paragraph("<b>Tech Stack</b>", side_heading),
        Paragraph("• React<br/>• JS / ES6<br/>• Go (Golang)<br/>• Docker<br/>• Git", side_body),
    ]

    # Main Column Content
    main_col = [
        Paragraph("Jordan Lee", header_style),
        Paragraph("Full Stack Software Engineer | Cloud & Distributed Systems", side_body),
        Spacer(1, 8),
        Paragraph("<b>Work History</b>", main_heading),
        Paragraph("<b>Senior Backend Engineer</b> — CloudScale Systems", main_body),
        Paragraph("<i>2022-04 – Present</i>", main_body),
        Paragraph(
            "• Developed high-throughput microservices using Go and deployed containerised pods via Docker.<br/>"
            "• Led team transition to modern Git branching workflows and automated CI pipelines.<br/>"
            "• Supported company go-to-market initiatives with reliable cloud APIs.",
            main_body,
        ),
        Spacer(1, 6),
        Paragraph("<b>Frontend Developer</b> — WebSphere Digital", main_body),
        Paragraph("<i>2019-08 – 2022-03</i>", main_body),
        Paragraph(
            "• Built dynamic, responsive user portals with React and modular JS components.<br/>"
            "• Reduced client-side bundle size by 35% through effective code splitting.",
            main_body,
        ),
    ]

    # Two column layout table
    two_col_table = Table(
        [[sidebar, main_col]],
        colWidths=[160, 380],
    )
    two_col_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("RIGHTPADDING", (0, 0), (0, -1), 14),
            ("LEFTPADDING", (1, 0), (1, -1), 14),
            ("LINEAFTER", (0, 0), (0, -1), 1, colors.HexColor("#E2E8F0")),
        ])
    )

    doc.build([two_col_table])

    # Clean temporary image
    if img_temp_path.exists():
        img_temp_path.unlink()

    sidecar = {
        "as_of": AS_OF_DATE,
        "contact": {
            "name": "Jordan Lee",
            "email": "jordan.lee@example.com",
            "phone": "555-0187",
            "location": "Seattle, WA",
        },
        "expected_sections": ["academics", "tech stack", "work history"],
        "skills": expected_skills,
        "education": [
            {
                "degree": "Bachelor of Science",
                "field": "Software Engineering",
                "institution": "University of Washington",
                "graduation_year": None,
            }
        ],
        "experience": [
            {
                "title": "Senior Backend Engineer",
                "company": "CloudScale Systems",
                "start": "2022-04",
                "end": "Present",
                "is_current": True,
            },
            {
                "title": "Frontend Developer",
                "company": "WebSphere Digital",
                "start": "2019-08",
                "end": "2022-03",
                "is_current": False,
            },
        ],
        "total_years_experience": 7.2,
        "layout": {
            "page_count": 1,
            "is_multi_column": True,
            "has_tables": True,
            "has_images": True,
            "needs_ocr": False,
        },
        "traps": ["go-to-market", "Grade C"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Generated swe_two_column.pdf and swe_two_column.expected.json")


# -- Fixture 3: accountant_table.pdf ----------------------------------------

def make_accountant_table(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate accountant_table.pdf and sidecar."""
    pdf_path = FIXTURES_DIR / "accountant_table.pdf"
    json_path = FIXTURES_DIR / "accountant_table.expected.json"

    expected_skills = validate_expected_skills(
        ["excel", "quickbooks", "gaap"],
        canonical_skills,
        alias_map,
        "accountant_table",
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AccTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#2C3E50"),
    )
    heading_style = ParagraphStyle(
        "AccHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2C3E50"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "AccBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#333333"),
    )

    # Personal details block with gendered text and protected fields
    personal_data = [
        ["Date of Birth:", "1988-04-12", "Marital Status:", "Married"],
        ["Gender:", "Female", "Citizenship:", "United States"],
    ]
    personal_table = Table(personal_data, colWidths=[100, 150, 100, 150])
    personal_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#555555")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ])
    )

    # Skills inside a structured table
    skills_table_data = [
        ["Core Competency", "Proficiency Level", "Practical Application"],
        ["Excel", "Expert", "Financial modeling, advanced pivot tables, vlookups"],
        ["QuickBooks", "Advanced", "GL bookkeeping, accounts payable, ledger reconciliation"],
        ["GAAP", "Comprehensive", "Statutory audit compliance, financial statement preparation"],
    ]
    skills_table = Table(skills_table_data, colWidths=[120, 110, 270])
    skills_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ])
    )

    story = [
        Paragraph("Morgan Vance", title_style),
        Paragraph("morgan.vance@example.com | 555-0193 | Chicago, IL", body_style),
        Spacer(1, 6),
        Paragraph("Personal Information", heading_style),
        personal_table,
        Spacer(1, 6),
        Paragraph("Professional Profile", heading_style),
        Paragraph(
            "She is a dedicated and highly organized Senior Accountant with over 10 years of general ledger "
            "and reconciliation experience. Throughout her career, she has excelled at maintaining immaculate "
            "financial records adhering strictly to GAAP guidelines.",
            body_style,
        ),
        Paragraph("Professional Experience", heading_style),
        Paragraph("<b>Senior Accountant</b> — Highland Financial Group, Chicago, IL", body_style),
        Paragraph("<i>2018-05 – Present</i>", body_style),
        Paragraph(
            "• Managed month-end close processes and general ledger entries using QuickBooks.<br/>"
            "• Prepared quarterly financial statements under US GAAP standards.",
            body_style,
        ),
        Paragraph("<b>Staff Accountant</b> — Prairie View Associates, Peoria, IL", body_style),
        Paragraph("<i>2014-06 – 2018-04</i>", body_style),
        Paragraph(
            "• Reconciled accounts receivable and payable reports using complex Excel spreadsheets.<br/>"
            "• Earned Grade C rating in internal audit review during sophomore year transition.",
            body_style,
        ),
        Paragraph("Technical Skills & Competencies", heading_style),
        skills_table,
        Paragraph("Education", heading_style),
        Paragraph("<b>Bachelor of Science in Accounting</b> — University of Illinois Urbana-Champaign, 2010", body_style),
    ]

    doc.build(story)

    sidecar = {
        "as_of": AS_OF_DATE,
        "contact": {
            "name": "Morgan Vance",
            "email": "morgan.vance@example.com",
            "phone": "555-0193",
            "location": "Chicago, IL",
        },
        "expected_sections": ["personal information", "professional profile", "professional experience", "technical skills & competencies", "education"],
        "skills": expected_skills,
        "education": [
            {
                "degree": "Bachelor of Science",
                "field": "Accounting",
                "institution": "University of Illinois Urbana-Champaign",
                "graduation_year": 2010,
            }
        ],
        "experience": [
            {
                "title": "Senior Accountant",
                "company": "Highland Financial Group",
                "start": "2018-05",
                "end": "Present",
                "is_current": True,
            },
            {
                "title": "Staff Accountant",
                "company": "Prairie View Associates",
                "start": "2014-06",
                "end": "2018-04",
                "is_current": False,
            },
        ],
        "total_years_experience": 12.4,
        "layout": {
            "page_count": 1,
            "is_multi_column": False,
            "has_tables": True,
            "has_images": False,
            "needs_ocr": False,
        },
        "traps": ["excelled at", "Grade C"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Generated accountant_table.pdf and accountant_table.expected.json")


# -- Fixture 4: chef_plain.docx ---------------------------------------------

def make_chef_plain(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate chef_plain.docx and sidecar."""
    docx_path = FIXTURES_DIR / "chef_plain.docx"
    json_path = FIXTURES_DIR / "chef_plain.expected.json"

    expected_skills = validate_expected_skills(
        ["haccp", "menu planning", "food safety"],
        canonical_skills,
        alias_map,
        "chef_plain",
    )

    doc = docx.Document()

    # Title & Contact
    h1 = doc.add_heading("Marcus Bennett", level=0)
    p_contact = doc.add_paragraph("marcus.bennett@example.com | 555-0164 | Austin, TX")

    # Summary
    doc.add_heading("Professional Summary", level=1)
    doc.add_paragraph(
        "Passionate Executive Chef with over 8 years directing high-volume kitchen operations, banquet catering, "
        "and culinary R&D. Proficient in menu planning and food safety standards with thorough HACCP execution. "
        "Familiar with kitchen reporting tools including Microsoft Word documentation."
    )

    # Work Experience
    doc.add_heading("Work Experience", level=1)
    p1 = doc.add_paragraph()
    p1.add_run("Executive Chef").bold = True
    p1.add_run(" — The Salt Lick Bistro, Austin, TX\n")
    p1.add_run("Jan 2019 - Present").italic = True
    doc.add_paragraph(
        "• Led back-of-house culinary brigade of 18 cooks and kitchen staff across breakfast and dinner services.\n"
        "• Conducted seasonal menu planning emphasizing farm-to-table locally sourced Texas ingredients.\n"
        "• Enforced strict food safety compliance and daily sanitation routines according to HACCP protocols."
    )

    p2 = doc.add_paragraph()
    p2.add_run("Sous Chef").bold = True
    p2.add_run(" — Riverbend Grille, San Antonio, TX\n")
    p2.add_run("03/2016 - 12/2018").italic = True
    doc.add_paragraph(
        "• Managed prep station, inventory rotation, and butchery execution under the head chef.\n"
        "• Oversaw food safety audits and trained new culinary apprentices."
    )

    # Certifications
    doc.add_heading("Certifications", level=1)
    doc.add_paragraph("• ServSafe Food Protection Manager Certification, 2022")
    doc.add_paragraph("• HACCP Implementation and Compliance Certificate, 2020")

    # Education
    doc.add_heading("Education", level=1)
    doc.add_paragraph("Associate of Applied Science in Culinary Arts — Culinary Institute of America, 2015")

    doc.save(str(docx_path))

    sidecar = {
        "as_of": AS_OF_DATE,
        "contact": {
            "name": "Marcus Bennett",
            "email": "marcus.bennett@example.com",
            "phone": "555-0164",
            "location": "Austin, TX",
        },
        "expected_sections": ["professional summary", "work experience", "certifications", "education"],
        "skills": expected_skills,
        "education": [
            {
                "degree": "Associate of Applied Science",
                "field": "Culinary Arts",
                "institution": "Culinary Institute of America",
                "graduation_year": 2015,
            }
        ],
        "experience": [
            {
                "title": "Executive Chef",
                "company": "The Salt Lick Bistro",
                "start": "2019-01",
                "end": "Present",
                "is_current": True,
            },
            {
                "title": "Sous Chef",
                "company": "Riverbend Grille",
                "start": "2016-03",
                "end": "2018-12",
                "is_current": False,
            },
        ],
        "total_years_experience": 10.6,
        "layout": {
            "page_count": 1,
            "is_multi_column": False,
            "has_tables": False,
            "has_images": False,
            "needs_ocr": False,
        },
        "traps": ["R&D", "Microsoft Word"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Generated chef_plain.docx and chef_plain.expected.json")


# -- Fixture 5: nurse_scanned.pdf (Image-Only, No Text Layer) ----------------

def make_nurse_scanned(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate nurse_scanned.pdf (raster image-only PDF) and sidecar."""
    pdf_path = FIXTURES_DIR / "nurse_scanned.pdf"
    json_path = FIXTURES_DIR / "nurse_scanned.expected.json"

    expected_skills = validate_expected_skills(
        ["patient care", "vital signs", "bls", "triage", "wound care"],
        canonical_skills,
        alias_map,
        "nurse_scanned",
    )

    # 1. Render layout to an in-memory PDF canvas
    mem_pdf = io.BytesIO()
    doc = SimpleDocTemplate(
        mem_pdf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NurseTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.black,
    )
    heading_style = ParagraphStyle(
        "NurseHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "NurseBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.black,
    )

    story = [
        Paragraph("Elena Rostova, BSN, RN", title_style),
        Paragraph("elena.rostova@example.com | 555-0176 | Denver, CO", body_style),
        Spacer(1, 6),
        Paragraph("Clinical Summary", heading_style),
        Paragraph(
            "Registered Nurse with 6+ years of acute emergency triage and patient care experience. Dedicated to "
            "delivering compassionate, evidence-based nursing care while participating in hospital clinical R&D.",
            body_style,
        ),
        Paragraph("Clinical Experience", heading_style),
        Paragraph("<b>Staff Nurse – Emergency Department</b> — Saint Luke Medical Center, Denver, CO", body_style),
        Paragraph("<i>August 2020 – Present</i>", body_style),
        Paragraph(
            "• Managed patient care for high-acuity trauma and cardiac admissions in Level II Emergency Center.<br/>"
            "• Conducted rapid triage assessment, monitored vital signs, and initiated BLS emergency resuscitation.<br/>"
            "• Performed specialized wound care, intravenous line placements, and clinical documentation.",
            body_style,
        ),
        Paragraph("<b>Staff Nurse – Medical-Surgical</b> — Rocky Mountain Health, Boulder, CO", body_style),
        Paragraph("<i>June 2018 – July 2020</i>", body_style),
        Paragraph(
            "• Administered scheduled medications and documented vital signs for 24-bed acute unit.<br/>"
            "• Maintained Grade C compliance documentation audits with zero regulatory violations.",
            body_style,
        ),
        Paragraph("Education & Credentials", heading_style),
        Paragraph("<b>Bachelor of Science in Nursing (BSN)</b> — University of Colorado, 2018", body_style),
        Paragraph("<b>Registered Nurse (RN)</b> — Colorado State Board of Nursing License", body_style),
        Paragraph("<b>Basic Life Support (BLS)</b> — American Heart Association", body_style),
    ]

    doc.build(story)
    mem_pdf.seek(0)

    # 2. Rasterize via PyMuPDF at 200 DPI
    pdf_doc = pymupdf.open(stream=mem_pdf.read(), filetype="pdf")
    pix = pdf_doc[0].get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 3. Add slight rotation (0.6 deg) and subtle noise to emulate physical scanning
    rotated = img.rotate(0.6, expand=True, fillcolor=(255, 255, 255), resample=Image.Resampling.BILINEAR)

    # Add light noise
    np_img = np.array(rotated, dtype=np.int16)
    np.random.seed(42)
    noise = np.random.normal(0, 3, np_img.shape).astype(np.int16)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(noisy_img)

    # 4. Save as image-only PDF (no text layer)
    final_img.save(str(pdf_path), "PDF", resolution=200.0)

    sidecar = {
        "as_of": AS_OF_DATE,
        "contact": {
            "name": "Elena Rostova",
            "email": "elena.rostova@example.com",
            "phone": "555-0176",
            "location": "Denver, CO",
        },
        "expected_sections": ["clinical summary", "clinical experience", "education & credentials"],
        "skills": expected_skills,
        "education": [
            {
                "degree": "Bachelor of Science in Nursing",
                "field": "Nursing",
                "institution": "University of Colorado",
                "graduation_year": 2018,
            }
        ],
        "experience": [
            {
                "title": "Staff Nurse – Emergency Department",
                "company": "Saint Luke Medical Center",
                "start": "2020-08",
                "end": "Present",
                "is_current": True,
            },
            {
                "title": "Staff Nurse – Medical-Surgical",
                "company": "Rocky Mountain Health",
                "start": "2018-06",
                "end": "2020-07",
                "is_current": False,
            },
        ],
        "total_years_experience": 8.4,
        "layout": {
            "page_count": 1,
            "is_multi_column": False,
            "has_tables": False,
            "has_images": True,
            "needs_ocr": True,
        },
        "traps": ["Grade C", "R&D"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Generated nurse_scanned.pdf (raster image-only) and nurse_scanned.expected.json")


# -- 3 JDs as .txt ----------------------------------------------------------

def make_jds(canonical_skills: Set[str], alias_map: Dict[str, str]) -> None:
    """Generate 3 Job Descriptions (250-400 words each) and their sidecars."""

    # 1. Data Scientist JD
    ds_skills_req = validate_expected_skills(["python", "sql", "machine learning", "scikit-learn"], canonical_skills, alias_map, "jd_ds_req")
    ds_skills_pref = validate_expected_skills(["pytorch", "spark", "power bi"], canonical_skills, alias_map, "jd_ds_pref")

    ds_text = (
        "Job Title: Senior Data Scientist\n"
        "Company: Apex Technologies Inc.\n"
        "Location: San Francisco, CA (Hybrid)\n\n"
        "About the Role:\n"
        "Apex Technologies is seeking an experienced Senior Data Scientist to lead our core algorithmic modeling "
        "and predictive intelligence platform. In this position, you will partner closely with engineering and product "
        "teams to architect, prototype, and operationalize robust machine learning solutions at scale. The ideal "
        "candidate has excelled at applying statistical rigor to solve complex commercial challenges in high-velocity "
        "R&D environments.\n\n"
        "Minimum Experience:\n"
        "Minimum 4+ years of professional full-time experience in applied data science or quantitative modeling.\n\n"
        "Key Responsibilities:\n"
        "• Design, train, and validate production-ready predictive models using Python and modern data science ecosystems.\n"
        "• Formulate complex queries and data transformations in SQL across enterprise distributed data warehouses.\n"
        "• Implement feature extraction and statistical benchmarking pipelines utilizing scikit-learn.\n"
        "• Explore deep neural network architectures and natural language representations using PyTorch.\n"
        "• Process large-scale streaming and batch data workloads using Apache Spark pipelines.\n"
        "• Communicate analytical findings and model performance metrics to stakeholders using Power BI visualizations.\n\n"
        "Required Qualifications:\n"
        "• Proficiency in Python programming with demonstrated expertise in statistical computing.\n"
        "• Deep competency writing high-performance SQL queries for relational databases.\n"
        "• Hands-on mastery of classical machine learning algorithms and scikit-learn framework.\n"
        "• Minimum 4 years of hands-on industry experience building scalable ML models.\n\n"
        "Preferred Qualifications:\n"
        "• Experience developing deep neural network models with PyTorch.\n"
        "• Familiarity with distributed big data processing using Spark.\n"
        "• Strong background creating executive analytical reporting dashboards in Power BI.\n"
        "• Demonstrated track record in enterprise algorithmic innovation and commercial product delivery.\n"
    )

    ds_words = len(ds_text.split())
    assert 250 <= ds_words <= 400, f"jd_data_scientist word count {ds_words} not in [250, 400]"

    with open(FIXTURES_DIR / "jd_data_scientist.txt", "w", encoding="utf-8") as f:
        f.write(ds_text)

    with open(FIXTURES_DIR / "jd_data_scientist.expected.json", "w", encoding="utf-8") as f:
        json.dump({
            "as_of": AS_OF_DATE,
            "title": "Senior Data Scientist",
            "required_skills": ds_skills_req,
            "preferred_skills": ds_skills_pref,
            "min_years": 4,
        }, f, indent=2)

    # 2. Software Engineer JD
    swe_skills_req = validate_expected_skills(["javascript", "react", "node.js", "git"], canonical_skills, alias_map, "jd_swe_req")
    swe_skills_pref = validate_expected_skills(["docker", "kubernetes", "go", "typescript"], canonical_skills, alias_map, "jd_swe_pref")

    swe_text = (
        "Job Title: Full Stack Software Engineer\n"
        "Company: CloudScale Systems\n"
        "Location: Seattle, WA (Remote Eligible)\n\n"
        "About the Role:\n"
        "CloudScale Systems is looking for a versatile Full Stack Software Engineer to build scalable customer-facing "
        "web applications and resilient microservices. You will collaborate across disciplines to deliver responsive "
        "interfaces and high-concurrency cloud backends that empower our global go-to-market enterprise solutions. "
        "Candidates of all academic backgrounds, regardless of whether you held a Grade C or honors distinction, are "
        "encouraged to demonstrate practical craftsmanship.\n\n"
        "Minimum Experience:\n"
        "Minimum 3+ years of professional software engineering experience delivering production web services.\n\n"
        "Key Responsibilities:\n"
        "• Develop modern, accessible frontend web interfaces using JavaScript and component-based React architecture.\n"
        "• Build modular server-side REST APIs and microservice endpoints powered by Node.js.\n"
        "• Maintain clean version control practices, conduct code reviews, and manage pull requests using Git.\n"
        "• Containerize microservices and deploy service components using Docker environments.\n"
        "• Coordinate cluster scheduling and service orchestration across Kubernetes cloud clusters.\n"
        "• Write concurrent backend backend worker services and networking utilities utilizing Go.\n\n"
        "Required Qualifications:\n"
        "• Strong proficiency in modern JavaScript development (ES6+) and web browser fundamentals.\n"
        "• Production experience building dynamic user interfaces with React.\n"
        "• Solid background creating backend API services utilizing Node.js.\n"
        "• Rigorous discipline in collaborative version control and release workflows using Git.\n"
        "• Minimum 3 years of software engineering experience.\n\n"
        "Preferred Qualifications:\n"
        "• Working experience containerizing applications using Docker containers.\n"
        "• Hands-on familiarity with Kubernetes container orchestration.\n"
        "• Practical knowledge of concurrent systems programming in Go.\n"
        "• Experience working with TypeScript in large codebase architectures.\n"
    )

    swe_words = len(swe_text.split())
    assert 250 <= swe_words <= 400, f"jd_software_engineer word count {swe_words} not in [250, 400]"

    with open(FIXTURES_DIR / "jd_software_engineer.txt", "w", encoding="utf-8") as f:
        f.write(swe_text)

    with open(FIXTURES_DIR / "jd_software_engineer.expected.json", "w", encoding="utf-8") as f:
        json.dump({
            "as_of": AS_OF_DATE,
            "title": "Full Stack Software Engineer",
            "required_skills": swe_skills_req,
            "preferred_skills": swe_skills_pref,
            "min_years": 3,
        }, f, indent=2)

    # 3. Chef JD
    chef_skills_req = validate_expected_skills(["culinary arts", "food safety", "haccp", "menu planning"], canonical_skills, alias_map, "jd_chef_req")
    chef_skills_pref = validate_expected_skills(["recipe development", "inventory control", "knife skills", "baking"], canonical_skills, alias_map, "jd_chef_pref")

    chef_text = (
        "Job Title: Executive Chef\n"
        "Company: The Salt Lick Hospitality Group\n"
        "Location: Austin, TX\n\n"
        "About the Role:\n"
        "The Salt Lick Hospitality Group is seeking a dynamic Executive Chef to oversee culinary production, kitchen "
        "operations, and staff leadership at our flagship dining venue. In this role, you will champion culinary "
        "creativity, operational discipline, and uncompromising kitchen cleanliness. The Executive Chef oversees all "
        "kitchen documentation and supplier agreements, which are maintained using Microsoft Word and cloud spreadsheets, "
        "while leading culinary R&D initiatives for seasonal offerings.\n\n"
        "Minimum Experience:\n"
        "Minimum 5+ years of culinary leadership experience in high-volume scratch kitchens or hospitality venues.\n\n"
        "Key Responsibilities:\n"
        "• Lead daily kitchen brigade operations and uphold classical culinary arts standards across all banquet and service lines.\n"
        "• Design seasonal dining programs and execute cost-effective menu planning based on market sourcing.\n"
        "• Enforce comprehensive kitchen sanitation and regulatory food safety guidelines throughout food storage and preparation.\n"
        "• Maintain active HACCP plans, temperature logs, and regulatory compliance records.\n"
        "• Direct kitchen inventory control, portion standardization, and vendor food waste reduction.\n"
        "• Supervise station cooks on proper knife skills, prep execution, and artisan baking procedures.\n\n"
        "Required Qualifications:\n"
        "• Deep command of professional culinary arts and commercial scratch cooking techniques.\n"
        "• Rigorous commitment to kitchen sanitation and certified food safety standards.\n"
        "• Demonstrated competency in regulatory HACCP documentation and kitchen protocols.\n"
        "• Proven track record in profitable menu planning and food cost management.\n"
        "• Minimum 5 years of progressive culinary experience.\n\n"
        "Preferred Qualifications:\n"
        "• Experience in seasonal recipe development and kitchen costing.\n"
        "• Disciplined background in inventory control and stock management.\n"
        "• Precision mastery of butchery and knife skills.\n"
        "• Practical knowledge of artisan baking and pastry preparation.\n"
    )

    chef_words = len(chef_text.split())
    assert 250 <= chef_words <= 400, f"jd_chef word count {chef_words} not in [250, 400]"

    with open(FIXTURES_DIR / "jd_chef.txt", "w", encoding="utf-8") as f:
        f.write(chef_text)

    with open(FIXTURES_DIR / "jd_chef.expected.json", "w", encoding="utf-8") as f:
        json.dump({
            "as_of": AS_OF_DATE,
            "title": "Executive Chef",
            "required_skills": chef_skills_req,
            "preferred_skills": chef_skills_pref,
            "min_years": 5,
        }, f, indent=2)

    logger.info("Generated 3 JDs (.txt) and sidecars (.expected.json)")


# -- Main Runner ------------------------------------------------------------

def main() -> None:
    """Generate all synthetic fixtures deterministically."""
    logger.info("=== make_fixtures.py start ===")
    canonical_skills, alias_map = load_taxonomy_skills()
    logger.info(f"Loaded taxonomy: {len(canonical_skills)} canonical skills, {len(alias_map)} aliases")

    make_ds_clean(canonical_skills, alias_map)
    make_swe_two_column(canonical_skills, alias_map)
    make_accountant_table(canonical_skills, alias_map)
    make_chef_plain(canonical_skills, alias_map)
    make_nurse_scanned(canonical_skills, alias_map)
    make_jds(canonical_skills, alias_map)

    logger.info("=== make_fixtures.py done ===")


if __name__ == "__main__":
    main()
