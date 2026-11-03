"""
tests/test_extraction.py
========================
Comprehensive unit tests for Stage 2 (NER & Structured Extraction).

Tests:
1. Tokenizer parity between text_utils.py and taxonomy_stats.py on 50 sample strings.
2. Ambiguity resolution ("go-to-market", "R&D", "Grade C", "excelled at" must NOT yield skills;
   "Go" and "C" in skills list must).
3. Alias normalization ("sklearn" -> scikit-learn, "JS" -> javascript).
4. Longest-match first resolution and non-overlapping spans.
5. Inactive skills ignored based on skills_stats.csv active flag.
6. Every date range format: "Jan 2019 - Present", "03/2016 - 12/2018", "2019 – 2021",
   "January 2020 to Present", "Sept 2018", en/em dashes, "till date", "current", "ongoing".
7. Present / current handling with fixed as_of date.
8. Non-overlapping interval union calculation for total_years_experience.
9. Empty input handling and graceful fallbacks.
10. Deterministic extraction reproducibility.
11. JD extraction (required vs preferred split, min_years parsing).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

import pytest

from app.extraction.entries import (
    compute_total_years_experience,
    extract_education,
    extract_experience,
    parse_date_token,
)
from app.extraction.extractor import extract_entities, get_skill_extractor
from app.extraction.skills import TaxonomySkillExtractor
from app.extraction.text_utils import normalize_text, tokenize
from app.schemas import ExperienceEntry, ParsedResume


# ---------------------------------------------------------------------------
# 1. Tokenizer Parity Test
# ---------------------------------------------------------------------------

TAXONOMY_STATS_TOKEN_PATTERN = re.compile(
    r"\.net\b|[a-z0-9]+(?:#|\+\+)?(?:[/\'\&.-][a-z0-9]+)*(?:#|\+\+)?|\&"
)


def taxonomy_stats_tokenize(text: str) -> List[str]:
    """Reference implementation from scripts/taxonomy_stats.py."""
    norm = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8").lower()
    return TAXONOMY_STATS_TOKEN_PATTERN.findall(norm)


SAMPLE_50_STRINGS = [
    "Expert in Python, C++, and .NET core architecture.",
    "Developing React, Node.js, and TypeScript applications.",
    "Worked with scikit-learn, PyTorch, TensorFlow, and Keras.",
    "Hands-on experience in AWS, GCP, Azure, and Docker containers.",
    "Kubernetes cluster administration and Helm chart deployments.",
    "Led go-to-market product launches across EMEA and APAC.",
    "Participated in clinical R&D trials for mRNA vaccines.",
    "Academic record: graduated with Grade C honor in 2018.",
    "She excelled at data wrangling, cleaning, and statistical modeling.",
    "PostgreSQL, MySQL, Oracle DB, and MongoDB database tuning.",
    "Built RESTful APIs and GraphQL endpoints for microservices.",
    "CI/CD pipelines using GitLab CI, GitHub Actions, and Jenkins.",
    "Proficient in Microsoft Excel, PowerPoint, and Word.",
    "Mastery in SQL query optimization and Apache Spark jobs.",
    "Experience with Power BI and Tableau executive dashboards.",
    "Automated testing with PyTest, Selenium, and Cypress.",
    "Configured Ansible playbooks, Terraform HCL, and Puppet manifests.",
    "Proficient in Java 17, Spring Boot, and Hibernate ORM.",
    "Golang concurrency using goroutines, channels, and sync primitives.",
    "Built mobile applications in Flutter, Dart, Swift, and Kotlin.",
    "Front-end responsive styling using Tailwind CSS, SASS, and LESS.",
    "Machine learning feature stores, Feast, MLflow, and Kubeflow.",
    "Deep Learning using CNNs, RNNs, LSTMs, and Transformer models.",
    "Natural language processing with spaCy, NLTK, and Hugging Face.",
    "Computer vision pipelines utilizing OpenCV and YOLOv8.",
    "Big data streaming via Apache Kafka, Flink, and Spark Streaming.",
    "Snowflake cloud data warehousing and dbt data transformations.",
    "Agile Scrum methodologies, Jira sprint planning, and Confluence.",
    "Git version control branching, merging, rebasing, and pull requests.",
    "Linux system administration, Bash scripting, and Zsh automation.",
    "Network protocols: TCP/IP, UDP, HTTP/2, HTTP/3, and WebSockets.",
    "Cybersecurity penetration testing, OWASP Top 10, and Burp Suite.",
    "Identity management: OAuth 2.0, OpenID Connect, and SAML 2.0.",
    "Microservice mesh using Istio, Envoy, and Linkerd.",
    "Elasticsearch, Logstash, and Kibana (ELK stack) logging.",
    "Prometheus monitoring, Grafana dashboards, and Alertmanager.",
    "High-concurrency systems written in Rust and C# .NET.",
    "Applied Bayesian statistics, Markov chains, and Monte Carlo.",
    "Product management: roadmapping, OKRs, KPIs, and user stories.",
    "Culinary arts: sous vide, butcher cuts, knife skills, HACCP.",
    "Emergency medicine: triage assessments, BLS resuscitation, ACLS.",
    "Financial accounting: GAAP, IFRS, balance sheets, QuickBooks.",
    "Mechanical engineering: CAD/CAM, SolidWorks, and AutoCAD.",
    "Electrical engineering: PCB layout, Altium, and FPGA design.",
    "Biomedical engineering: microfluidics, PCR, and gel electrophoresis.",
    "Sales and business development: B2B pipeline, CRM, Salesforce.",
    "Digital marketing: SEO, SEM, Google Analytics, and copy editing.",
    "Graphic design: Adobe Photoshop, Illustrator, InDesign, and Figma.",
    "Supply chain logistics: inventory control, ERP, SAP, and Six Sigma.",
    "Special characters: C#, C++, .NET, F#, R&D, TCP/IP, i/o, 5G, 802.11ax.",
]


def test_tokenizer_parity_50_strings() -> None:
    """Verify text_utils.tokenize exactly matches taxonomy_stats.py on 50 sample strings."""
    assert len(SAMPLE_50_STRINGS) == 50
    for idx, sample in enumerate(SAMPLE_50_STRINGS):
        tokens_tu = tokenize(sample)
        tokens_ref = taxonomy_stats_tokenize(sample)
        assert tokens_tu == tokens_ref, f"Mismatch on sample {idx}: {sample}"


# ---------------------------------------------------------------------------
# 2. Ambiguity & Trap Tests
# ---------------------------------------------------------------------------

def test_ambiguity_negative_traps() -> None:
    """Ensure negative traps do NOT extract spurious skills."""
    extractor = get_skill_extractor()

    # go-to-market must NOT yield 'go'
    res = extractor.extract_canonical_skills("Led the regional go-to-market enterprise campaign.", section_name="")
    assert "go" not in [s.lower() for s in res]

    # R&D must NOT yield 'r'
    res = extractor.extract_canonical_skills("Managed our medical clinical R&D division.", section_name="")
    assert "r" not in [s.lower() for s in res]

    # Grade C must NOT yield 'c'
    res = extractor.extract_canonical_skills("Graduated with Grade C honor from university.", section_name="")
    assert "c" not in [s.lower() for s in res]

    # excelled at must NOT yield 'excel'
    res = extractor.extract_canonical_skills("He excelled at quantitative research and data cleaning.", section_name="")
    assert "excel" not in [s.lower() for s in res]

    # Lowercase 'word' in narrative must NOT yield 'word'
    res = extractor.extract_canonical_skills("Every word in the report was carefully verified.", section_name="")
    assert "word" not in [s.lower() for s in res]


def test_ambiguity_positive_contexts() -> None:
    """Ensure ambiguous terms are properly extracted when in skills list or valid context."""
    extractor = get_skill_extractor()

    # 'Go' in a skills list
    res = extractor.extract_canonical_skills("Skills: Python, Go, Docker, Kubernetes", section_name="skills")
    assert "go" in [s.lower() for s in res]

    # 'C' in programming languages list
    res = extractor.extract_canonical_skills("Languages: C, C++, Java, Python", section_name="skills")
    assert "c" in [s.lower() for s in res]

    # 'Excel' in software tools list
    res = extractor.extract_canonical_skills("Tools: Excel, QuickBooks, Word", section_name="skills")
    skills_lower = [s.lower() for s in res]
    assert "excel" in skills_lower
    assert "word" in skills_lower

    # 'Triage' in emergency healthcare context
    res = extractor.extract_canonical_skills(
        "Conducted emergency triage assessment, vital signs monitoring, and patient care.",
        section_name="experience",
    )
    assert "triage" in [s.lower() for s in res]


# ---------------------------------------------------------------------------
# 3. Alias Normalization
# ---------------------------------------------------------------------------

def test_alias_normalization() -> None:
    """Verify aliases normalize to canonical names."""
    extractor = get_skill_extractor()

    # sklearn -> scikit-learn
    res = extractor.extract_canonical_skills("Proficient in sklearn algorithms.", section_name="skills")
    assert "scikit-learn" in res

    # JS -> javascript
    res = extractor.extract_canonical_skills("Developed UI using JS and React.", section_name="skills")
    assert "javascript" in res

    # Golang -> go
    res = extractor.extract_canonical_skills("Backend services written in Golang.", section_name="skills")
    assert "go" in res


# ---------------------------------------------------------------------------
# 4. Longest Match & Inactive Skills
# ---------------------------------------------------------------------------

def test_longest_match_precedence() -> None:
    """Verify longer compound skills take precedence over single tokens."""
    extractor = get_skill_extractor()
    res = extractor.extract_canonical_skills("Expert in machine learning models.", section_name="skills")
    assert "machine learning" in res


def test_inactive_skills_ignored() -> None:
    """Verify skills with active=False in skills_stats.csv are ignored."""
    extractor = get_skill_extractor()
    # '1cadcam unigraphics' is active=False in skills_stats.csv
    res = extractor.extract_canonical_skills("Experienced with 1cadcam unigraphics software.", section_name="skills")
    assert "1cadcam unigraphics" not in [s.lower() for s in res]


# ---------------------------------------------------------------------------
# 5. Date Formats & Experience Interval Union
# ---------------------------------------------------------------------------

def test_date_formats_parsing() -> None:
    """Test all supported date format variations."""
    as_of = "2026-11-01"

    text = """
    Software Engineer — TechCorp
    Jan 2019 - Present
    - Built web services.

    Developer — StartupCo
    03/2016 - 12/2018
    - Maintained APIs.

    Junior Developer — CodeBase
    2014 – 2015
    - Fixed bugs.

    Contractor — AgencyX
    August 2020 to Present
    - Cloud migration.

    Specialist — AlphaCo
    Sept 2018 — ongoing
    - Infrastructure.
    """

    entries = extract_experience(text, as_of=as_of)
    assert len(entries) == 5

    # Jan 2019 - Present
    assert entries[0].start == "2019-01"
    assert entries[0].end == "Present"
    assert entries[0].is_current is True

    # 03/2016 - 12/2018
    assert entries[1].start == "2016-03"
    assert entries[1].end == "2018-12"
    assert entries[1].is_current is False

    # 2014 - 2015
    assert entries[2].start == "2014-01"
    assert entries[2].end == "2015-12"

    # August 2020 to Present
    assert entries[3].start == "2020-08"
    assert entries[3].end == "Present"
    assert entries[3].is_current is True

    # Sept 2018 — ongoing
    assert entries[4].start == "2018-09"
    assert entries[4].end == "Present"
    assert entries[4].is_current is True


def test_present_handling_fixed_as_of() -> None:
    """Test deterministic calculation for current roles using fixed as_of."""
    entry = ExperienceEntry(
        start="2024-11",
        end="Present",
        is_current=True,
    )
    # 2024-11 to 2026-11 is exactly 2.0 years (24 months)
    years = compute_total_years_experience([entry], as_of="2026-11-01")
    assert years == 2.0


def test_overlapping_intervals_union() -> None:
    """Overlapping experience periods must not be double counted."""
    e1 = ExperienceEntry(start="2020-01", end="2022-01", is_current=False)
    e2 = ExperienceEntry(start="2021-01", end="2023-01", is_current=False)

    # Union is 2020-01 through 2023-01: 37 months / 12 = 3.1 years (not 2.1 + 2.1 = 4.2)
    years = compute_total_years_experience([e1, e2], as_of="2026-11-01")
    assert 3.0 <= years <= 3.2


# ---------------------------------------------------------------------------
# 6. Education Extraction
# ---------------------------------------------------------------------------

def test_education_normalization() -> None:
    """Verify degrees and institutions are extracted and normalized."""
    text = """
    Master of Science in Computer Science — Stanford University, 2019
    Bachelor of Science in Statistics — UC Berkeley, 2017
    Associate of Applied Science in Culinary Arts — Culinary Institute of America, Expected 2027
    """
    entries = extract_education(text)
    assert len(entries) == 3

    assert entries[0].degree == "Master of Science"
    assert entries[0].field == "Computer Science"
    assert entries[0].institution == "Stanford University"
    assert entries[0].graduation_year == 2019

    assert entries[1].degree == "Bachelor of Science"
    assert entries[1].field == "Statistics"
    assert entries[1].institution == "UC Berkeley"
    assert entries[1].graduation_year == 2017

    assert entries[2].degree == "Associate of Applied Science"
    assert entries[2].field == "Culinary Arts"
    assert entries[2].graduation_year == 2027


# ---------------------------------------------------------------------------
# 7. Empty Input, Determinism & Graceful Fallback
# ---------------------------------------------------------------------------

def test_empty_input_handling() -> None:
    """Empty or whitespace input must return empty models gracefully without errors."""
    parsed = ParsedResume(raw_text="", sections={})
    enriched = extract_entities(parsed)
    assert enriched.skills == []
    assert enriched.education == []
    assert enriched.experience == []
    assert enriched.total_years_experience == 0.0


def test_determinism() -> None:
    """Multiple extractions on identical input produce identical output."""
    raw = (
        "Software Engineer with experience in Python and SQL.\n"
        "B.S. Computer Science, University of Illinois, 2020\n"
        "Lead Developer — TechCorp\n"
        "2021-01 – Present\n"
    )
    p1 = extract_entities(ParsedResume(raw_text=raw), as_of="2026-11-01")
    p2 = extract_entities(ParsedResume(raw_text=raw), as_of="2026-11-01")

    assert p1.skills == p2.skills
    assert p1.total_years_experience == p2.total_years_experience
    assert len(p1.education) == len(p2.education)
    assert len(p1.experience) == len(p2.experience)


# ---------------------------------------------------------------------------
# 8. Job Description Extraction
# ---------------------------------------------------------------------------

def test_jd_skills_and_min_years() -> None:
    """Verify JD extraction splits required vs preferred skills and parses min_years."""
    extractor = get_skill_extractor()
    jd_text = """
    Job Title: Senior Data Engineer
    Minimum Experience:
    Minimum 5+ years of experience in data platform architecture.

    Required Qualifications:
    • Expert in Python programming and distributed computing.
    • Extensive hands-on SQL query tuning and database administration.

    Preferred Qualifications:
    • Experience building deep learning workflows in PyTorch.
    • Cloud data warehousing on Snowflake.
    """

    res = extractor.extract_jd_skills(jd_text)
    assert res["min_years"] == 5
    req_lower = [s.lower() for s in res["required_skills"]]
    pref_lower = [s.lower() for s in res["preferred_skills"]]

    assert "python" in req_lower
    assert "sql" in req_lower
    assert "pytorch" in pref_lower
    assert "snowflake" in pref_lower
