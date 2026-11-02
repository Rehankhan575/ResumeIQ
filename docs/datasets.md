# Dataset Research & Evaluation: AI Resume Analyzer

This document records the research findings for candidate datasets to support the four data needs of the project:
- **A. Resume Classification**: Multi-class job category classification.
- **B. Resume NER**: Named entity recognition for skills, degrees, companies, designations, locations, names.
- **C. Job Descriptions**: Job postings with titles, full descriptions, and category/industry fields for JD matching.
- **D. Skills Taxonomy**: Standardized skill ontologies (ESCO / O*NET) for skill extraction and ATS scoring.

---

## 1. Environment & Authentication Audit

The local environment was checked prior to dataset evaluation (without inspecting secret file contents):

| Check | Status | Details |
|---|---|---|
| `~/.kaggle/kaggle.json` | **Missing** (`False`) | Kaggle API credentials are not configured on this machine. |
| `kaggle` Python package | **Not Installed** | Package not found in system python or workspace virtual environment. |
| `datasets` (HuggingFace) package | **Installed in `venv/`** | Hugging Face `datasets` library is already installed in `./venv/bin/python`. |

> **Key Takeaway:** Datasets hosted on Hugging Face can be loaded immediately and programmatically via `datasets.load_dataset()` without API credentials. Datasets hosted exclusively on Kaggle require either configuring `kaggle.json` or manual browser download.

---

## 2. Candidate Evaluations

### A. Resume Classification

Target: Resume text with a job-category label (LiveCareer/Kaggle-style, 20+ categories, thousands of rows).

| Attribute | Candidate A1: LiveCareer Resume Dataset | Candidate A2: opensporks/resumes | Candidate A3: ResumeAtlas |
|---|---|---|---|
| **Exact Name** | `Resume Dataset` | `opensporks/resumes` | `ahmedheakl/resume-atlas` |
| **Exact URL** | [kaggle.com/datasets/snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | [huggingface.co/datasets/opensporks/resumes](https://huggingface.co/datasets/opensporks/resumes) | [huggingface.co/datasets/ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) |
| **Host** | Kaggle | HuggingFace | HuggingFace |
| **Rows** | 2,484 | 2,484 | 13,389 |
| **Columns** | 4 (`ID`, `Resume_str`, `Resume_html`, `Category`) | 4 (`ID`, `Resume_str`, `Resume_html`, `Category`) | 2 (`Category`, `Text`) |
| **File Size** | 65.5 MB zip (~54.8 MB CSV uncompressed) | ~54.8 MB uncompressed (Parquet shards: ~15 MB) | 53.6 MB dataset size (Parquet: 23.6 MB) |
| **Label / Category List (24 or 43)** | **24 Categories:**<br>Accountant, Advocate, Agriculture, Apparel, Arts, Automobile, Aviation, Banking, BPO, Business-Development, Chef, Construction, Consultant, Designer, Digital-Media, Engineering, Finance, Fitness, Healthcare, HR, Information-Technology, Public-Relations, Sales, Teacher | **24 Categories:**<br>Identical to Candidate A1 (Accountant, Advocate, Agriculture, Apparel, Arts, Automobile, Aviation, Banking, BPO, Business-Development, Chef, Construction, Consultant, Designer, Digital-Media, Engineering, Finance, Fitness, Healthcare, HR, Information-Technology, Public-Relations, Sales, Teacher) | **43 Categories:**<br>Expanded multi-class ontology including Accountant, Advocate, Agriculture, Apparel, Arts, Automobile, Aviation, Banking, BPO, Business-Development, Chef, Construction, Consultant, Designer, Digital-Media, Engineering, Finance, Fitness, Healthcare, HR, Information-Technology, Public-Relations, Sales, Teacher, and 19 additional domains |
| **License** | CC0: Public Domain | CC0-1.0 | MIT License |
| **Needs Login / Terms** | Yes (Kaggle account required for API/download) | No (open access on Hugging Face) | No (open access on Hugging Face) |
| **Contains Personal Data** | Yes (scraped LiveCareer profiles contain real candidate names, emails, phones, addresses) | Yes (identical underlying LiveCareer scraped text) | Yes (scraped web images/text contain candidate contact details) |
| **Pipeline Plug-in** | Stage 3 (`app/models/classifier.py`): text and label directly train baseline TF-IDF+MLP, BiLSTM, and DistilBERT. | Stage 3 (`app/models/classifier.py`): drop-in unauthenticated HF replacement for Candidate A1. | Stage 3 (`app/models/classifier.py`): large-scale benchmark for fine-tuning transformer classifiers across 43 job domains. |

*Note on dead links:* The legacy Kaggle slug `gauravduttakiit/resume-dataset` ("Updated Resume Dataset") was audited and returned HTTP 404 (removed from Kaggle).

---

### B. Resume Named Entity Recognition (NER)

Target: Resumes annotated with entities (skills, degrees, companies, designations, locations, names).

| Attribute | Candidate B1: Resume Entities for NER | Candidate B2: AuraATS Synthetic Resume NER 12K v2 | Candidate B3: IT Skills NER Resumes |
|---|---|---|---|
| **Exact Name** | `Resume Entities for NER` | `AuraATS Synthetic Resume NER 12K — Version 2` | `IT Skills Named Entity Recognition (NER) Dataset` (`Annotated_NER_PDF_Resumes`) |
| **Exact URL** | [kaggle.com/datasets/dataturks/resume-entities-for-ner](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) | [huggingface.co/datasets/sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2](https://huggingface.co/datasets/sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2) | [huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes) |
| **Host** | Kaggle | HuggingFace | HuggingFace |
| **Rows** | 220 items (200 in original DataTurks export) | 12,000 resumes (9,600 train / 1,200 val / 1,200 test) | 5,029 CV samples |
| **Columns** | 3 JSON fields (`content`, `annotation`, `extras`) | 12 fields (`id`, `candidate_id`, `template_family`, `split`, `industry`, `country`, `experience_count`, `text`, `entities`, `tokens`, `bio_labels`, `token_offsets`) | 2 JSON fields (`text`, `annotations`) |
| **File Size** | 330 KB zip | 128.8 MB dataset size (JSONL / Parquet) | `ResumesJsonAnnotated.zip` (17.1 MB), `ResumesPDF.zip` (585.8 MB) |
| **Entity Categories** | **10 Categories:**<br>Name, College Name, Degree, Graduation Year, Years of Experience, Companies worked at, Designation, Skills, Location, Email Address | **13 Categories (27 BIO labels):**<br>`CERT`, `COMPANY`, `DATE`, `DEGREE`, `EMAIL`, `FIELD`, `INSTITUTION`, `LANGUAGE`, `LOCATION`, `NAME`, `PHONE`, `SKILL`, `TITLE` | **1 Category:**<br>`SKILL: <skill_name>` (dense technical/IT skill annotations) |
| **Annotation Format** | JSON Lines with character offset spans: `annotation: [{"label": [cat], "points": [{"start": int, "end": int, "text": str}]}]` | Pre-tokenized `tokens` + aligned `bio_labels` (`B-SKILL`, `I-SKILL`, `B-DEGREE`, etc.) + span offset dicts (`token_offsets`, `entities`) | JSON files with character span offset tuples: `[start_char, end_char, "SKILL: label"]` |
| **License** | Unknown / Open DataTurks Community | Custom Research / Open on Hugging Face | MIT License |
| **Needs Login / Terms** | Yes (Kaggle account required) | No (open access on Hugging Face) | No (open access on Hugging Face) |
| **Contains Personal Data** | Yes (real candidate PII from Indeed.com) | **No** (100% synthetically generated resumes with fictitious names/emails/phones) | Yes (raw text from real candidate CVs) |
| **Pipeline Plug-in** | Stage 2 (`app/extraction/`): character spans converted to BIO tags for DistilBERT token classification. | Stage 2 (`app/extraction/`): direct drop-in token classification training with ready-made BIO tags and zero PII risk. | Stage 2 (`app/extraction/`): specialized supervision for dense technical and IT skill entity extraction. |

---

### C. Job Descriptions

Target: Job postings with title and full description text, ideally also a category or industry field.

| Attribute | Candidate C1: LinkedIn Job Postings (2023-2024) | Candidate C2: DataStax LinkedIn Job Listings | Candidate C3: Resume-Job Fit Dataset |
|---|---|---|---|
| **Exact Name** | `LinkedIn Job Postings (2023 - 2024)` | `datastax/linkedin_job_listings` | `batuhanmtl/job_resume_fit` |
| **Exact URL** | [kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | [huggingface.co/datasets/datastax/linkedin_job_listings](https://huggingface.co/datasets/datastax/linkedin_job_listings) | [huggingface.co/datasets/batuhanmtl/job_resume_fit](https://huggingface.co/datasets/batuhanmtl/job_resume_fit) |
| **Host** | Kaggle | HuggingFace | HuggingFace |
| **Rows** | 124,000+ job postings | 123,849 job postings | 2,385 resume-to-JD matched pairs |
| **Columns** | 27 columns in `job_postings.csv` (`job_id`, `company_id`, `title`, `description`, `max_salary`, `med_salary`, `min_salary`, `pay_period`, `formatted_work_type`, `location`, `applies`, `original_listed_time`, `remote_allowed`, `views`, `job_posting_url`, `application_url`, `application_type`, `expiry`, `closed_time`, `formatted_experience_level`, `skills_desc`, `listed_time`, `posting_domain`, `sponsored`, `work_type`, `currency`, `compensation_type`) + companion files for industries, skills, companies | 31 columns (`job_id`, `company_name`, `title`, `description`, `max_salary`, `pay_period`, `location`, `company_id`, `views`, `med_salary`, `min_salary`, `formatted_work_type`, `applies`, `original_listed_time`, `remote_allowed`, `job_posting_url`, `application_url`, `application_type`, `expiry`, `closed_time`, `formatted_experience_level`, `skills_desc`, `listed_time`, `posting_domain`, `sponsored`, `work_type`, `currency`, `compensation_type`, `normalized_salary`, `zip_code`, `fips`) | 10 columns (`ID`, `resume_text`, `job_text`, `category`, `job_required_skills`, `resume_skill_list`, `ai_matched_skills`, `ai_match_score`, `skill_string_match_score`, `fuzzy_match_score`) |
| **File Size** | 166.5 MB zip download | 516.8 MB download (Parquet/CSV) | 30.5 MB download |
| **Category / Industry Field** | Yes: `industries.csv` and `skills_desc` linkable via `job_id` | Yes: `skills_desc` and `formatted_experience_level` included | Yes: explicit `category` field across 23 professional domains |
| **License** | CC BY-SA 4.0 | Open Access / Community mirror | Other / Open Research |
| **Needs Login / Terms** | Yes (Kaggle account required) | No (open access on Hugging Face) | No (open access on Hugging Face) |
| **Contains Personal Data** | No (corporate job postings) | No (corporate job postings) | Yes (candidate resume text included in the pairs) |
| **Pipeline Plug-in** | Stage 4 (`app/models/matcher.py`): provides real-world corporate JD texts and titles for candidate retrieval evaluation (MRR, nDCG@10). | Stage 4 (`app/models/matcher.py`): unauthenticated HF alternative to C1 for building candidate JD corpus. | Stage 4 (`app/models/matcher.py`): ready-made resume-JD training pairs with ground-truth match scores for training Sentence-BERT. |

---

### D. Skills Taxonomy

Target: Standardized skill ontologies (ESCO and/or O*NET) with exact file names, sizes, rows, and columns.

| Attribute | Candidate D1: ESCO Skills Taxonomy | Candidate D2: O*NET 31.0 Database (Software & Essential Skills) |
|---|---|---|
| **Exact Name** | `ESCO Classification (European Skills, Competences, Qualifications and Occupations)` | `O*NET 31.0 Database` (Software Skills & Essential Skills) |
| **Exact URL** | Official Portal: [esco.ec.europa.eu/en/use-esco/download](https://esco.ec.europa.eu/en/use-esco/download)<br>EU Open Data: [data.europa.eu/data/datasets/european-skills-competences-qualifications-and-occupations](https://data.europa.eu/data/datasets/european-skills-competences-qualifications-and-occupations)<br>Direct Repository Mirror: [raw.githubusercontent.com/maudgrol/Data_driven_skills_taxonomy/master/skills_en.csv](https://raw.githubusercontent.com/maudgrol/Data_driven_skills_taxonomy/master/skills_en.csv) | Main Portal: [onetcenter.org/database.html](https://www.onetcenter.org/database.html)<br>Full CSV Archive: [onetcenter.org/dl_files/database/db_31_0_csv.zip](https://www.onetcenter.org/dl_files/database/db_31_0_csv.zip)<br>Direct Software Skills: [onetcenter.org/dl_files/database/db_31_0_csv/software_skills.csv](https://www.onetcenter.org/dl_files/database/db_31_0_csv/software_skills.csv)<br>Direct Essential Skills: [onetcenter.org/dl_files/database/db_31_0_csv/essential_skills.csv](https://www.onetcenter.org/dl_files/database/db_31_0_csv/essential_skills.csv) |
| **Host** | European Commission / EU Publications Office | National Center for O*NET Development (U.S. Department of Labor) |
| **Files & Sizes** | - `skills_en.csv`: 6,772,819 bytes (~6.77 MB uncompressed)<br>- `occupations_en.csv`: 2,047,124 bytes (~2.05 MB uncompressed)<br>- Full ESCO zip package: ~45 MB zip | - `db_31_0_csv.zip`: 16,237,378 bytes (~16.24 MB)<br>- `software_skills.csv`: 3,557,545 bytes (~3.56 MB)<br>- `essential_skills.csv`: 2,328,135 bytes (~2.33 MB)<br>- `occupation_data.csv`: 129,569 bytes (~130 KB) |
| **Rows** | ~13,890 skill and competence concepts | - `software_skills.csv`: 32,773 rows<br>- `essential_skills.csv`: ~67,000+ ratings across 35 core skill elements<br>- `occupation_data.csv`: 1,016 O*NET-SOC occupations |
| **Columns** | 7 core columns in standard CSV (up to 13 in full metadata export):<br>`conceptType`, `conceptUri`, `skillType`, `reuseLevel`, `preferredLabel`, `altLabels`, `description` | - `software_skills.csv` (7 columns):<br>`O*NET-SOC Code`, `Title`, `Workplace Example`, `Element ID`, `Element Name`, `Hot Technology`, `In Demand`<br>- `essential_skills.csv` (15 columns):<br>`O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source` |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Needs Login / Terms** | Official portal requires email address submission for dynamic link generation (no password or fee). Open mirrors require no login. | **No** (Direct HTTP download, unauthenticated, no terms clickwrap) |
| **Contains Personal Data** | No (pure taxonomy of skills, knowledge, and occupations) | No (governmental occupational taxonomy) |
| **Pipeline Plug-in** | `data/taxonomies/skills.csv`: supplies canonical skill names (`preferredLabel`) and synonym variants (`altLabels`) for regex/token matching and skill-gap identification. | `data/taxonomies/skills.csv`: supplies software tools (`Workplace Example`) with `Hot Technology` and `In Demand` priority tags for technical ATS scoring. |

---

## 3. Recommended Picks & Rationales

### Pick for A (Resume Classification): `opensporks/resumes` (or `snehaanbhawal/resume-dataset`)
* **Primary Recommendation:** [`opensporks/resumes`](https://huggingface.co/datasets/opensporks/resumes) on Hugging Face.
* **Fallback / Origin:** [`snehaanbhawal/resume-dataset`](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) on Kaggle.
* **Reasons:**
  1. **Exact 24-Category LiveCareer Benchmark:** Contains the standard 2,484 resumes across 24 job categories specified in the project context.
  2. **Zero-Configuration Download:** Because `~/.kaggle/kaggle.json` is missing on the machine, `opensporks/resumes` can be loaded directly in Python using `datasets.load_dataset("opensporks/resumes")` with zero authentication friction.
  3. **Identical Schema & License:** Contains identical text strings and labels under CC0 Public Domain.

---

### Pick for B (Resume NER): `sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2`
* **Primary Recommendation:** [`sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2`](https://huggingface.co/datasets/sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2) on Hugging Face.
* **Secondary Resource:** [`dataturks/resume-entities-for-ner`](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) on Kaggle (220 rows, for small real-resume evaluation).
* **Reasons:**
  1. **Scale & Balance:** Contains 12,000 resumes (9,600 train / 1,200 val / 1,200 test), compared to only 220 samples in DataTurks, preventing overfitting on small college project models.
  2. **Ready-to-Train BIO Labels:** Comes pre-tokenized with aligned `tokens` and 27 `bio_labels` across 13 entity types (`SKILL`, `DEGREE`, `INSTITUTION`, `COMPANY`, `TITLE`, `LOCATION`, `NAME`, `EMAIL`, `PHONE`, etc.), eliminating brittle character-to-token offset alignment code.
  3. **Privacy Compliance:** 100% synthetically generated resumes ensure zero risk of personal data leakage, fully respecting the privacy principles outlined in `PROJECT_CONTEXT.md`.
  4. **Unauthenticated Download:** Accessible directly via Hugging Face `datasets` without Kaggle API credentials.

---

### Pick for C (Job Descriptions): `batuhanmtl/job_resume_fit` + `datastax/linkedin_job_listings`
* **Primary Recommendation (for Model Training):** [`batuhanmtl/job_resume_fit`](https://huggingface.co/datasets/batuhanmtl/job_resume_fit) on Hugging Face.
* **Primary Recommendation (for Retrieval Corpus):** [`datastax/linkedin_job_listings`](https://huggingface.co/datasets/datastax/linkedin_job_listings) on Hugging Face (or [`arshkon/linkedin-job-postings`](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) on Kaggle).
* **Reasons:**
  1. **Pre-Annotated Training Pairs:** `batuhanmtl/job_resume_fit` includes 2,385 resume-to-job-description pairs with explicit `category` labels, required skills, and numeric similarity scores (`ai_match_score`), providing ground truth for training Siamese Sentence-BERT and validating MRR/nDCG metrics.
  2. **Corpus Scale for Search:** `datastax/linkedin_job_listings` provides 123,849 full LinkedIn postings directly via Hugging Face without needing Kaggle credentials, enabling large-scale candidate-job retrieval testing.

---

### Pick for D (Skills Taxonomy): Combined O*NET 31.0 (`software_skills.csv`) + ESCO (`skills_en.csv`)
* **Primary Recommendation:** Dual combination:
  1. **O*NET 31.0 `software_skills.csv`** via [onetcenter.org direct link](https://www.onetcenter.org/dl_files/database/db_31_0_csv/software_skills.csv).
  2. **ESCO `skills_en.csv`** via [ESCO download portal](https://esco.ec.europa.eu/en/use-esco/download) / [Open Source mirror](https://raw.githubusercontent.com/maudgrol/Data_driven_skills_taxonomy/master/skills_en.csv).
* **Reasons:**
  1. **Direct Download Access:** O*NET 31.0 provides direct, unauthenticated HTTP downloads for `software_skills.csv` (3.56 MB, 32,773 rows) with no login or email required.
  2. **Complementary Coverage:**
     - O*NET provides modern industry software products, libraries, and tools (`Workplace Example`) with explicit `Hot Technology` and `In Demand` flags.
     - ESCO provides 13,890 standardized functional competencies and comprehensive alternative synonym phrases (`altLabels`).
  3. **Clean Fusion into `data/taxonomies/skills.csv`:** Merging O*NET technology tools with ESCO preferred/alt labels creates an authoritative, deduplicated dictionary covering both technical tools (Python, Docker, PyTorch) and transversal competencies (project management, data analysis).
  4. **Open Permissive Licensing:** Both frameworks are licensed under **CC BY 4.0**, permitting free academic and commercial adaptation with attribution.

---

## 4. Final Choices Summary Table

The approved datasets for ingestion and raw storage under `data/raw/`:

| Dataset Identifier | Destination Directory | License | PII Risk | Role in Project | Known Limitations |
|---|---|---|---|---|---|
| **`opensporks/resumes`** (LiveCareer) | `data/raw/livecareer/` | `cc0-1.0` | **High** (real candidate profiles with names, emails, phones, addresses) | Stage 3: Multi-class resume job category classification (2,484 resumes, 24 job categories) | Imbalanced category distributions; OCR/formatting noise; requires PII anonymization in Stage 1 before storage or inference. |
| **`sanaullah7964/AuraATS-Synthetic-Resume-NER-12K-v2`** | `data/raw/ner_synthetic/` | `NOT STATED` | **None** (100% synthetically generated resumes and candidate identities) | Stage 2: Token classification / NER model training (9,600 train / 1,200 val / 1,200 test) across 13 entity types with pre-aligned BIO labels | Synthetic text distribution may underrepresent idiosyncratic human resume typos, diverse tabular column structures, or domain-specific jargon. |
| **`Mehyaar/Annotated_NER_PDF_Resumes`** | `data/raw/ner_real_skills/` | `mit` | **High** (scraped real candidate CV texts) | Stage 2: Specialized technical and IT skill extraction benchmarking on real-world CVs (5,029 annotated samples) | Dense technical annotations cover only the `SKILL` entity type (no degree, institution, or company tags); contains PDF unicode artifacts (e.g. lone surrogate emojis) requiring sanitization. |
| **`batuhanmtl/job_resume_fit`** | `data/raw/job_resume_fit/` | `other` | **Moderate / High** (contains candidate resume text) | Stage 4: Resume-to-Job matching evaluation and Siamese/Cross-Encoder training (2,385 pairs across 23 categories) | Match scores generated via proprietary heuristic/AI models rather than human recruiter hiring outcomes; restricted to 23 job domains. |
| **`datastax/linkedin_job_listings`** (5K Sample) | `data/raw/linkedin_sample/` | `NOT STATED` | **None** (public corporate job postings) | Stage 4: Corpus retrieval pool for job matching evaluation (MRR, nDCG@10) and dense vector search indexing | Scraped postings contain HTML markup and corporate boilerplate; sample restricted to postings with non-empty titles and descriptions ≥ 200 characters. |
| **`O*NET 31.0 Database`** (`software_skills.csv`, `occupation_data.csv`) | `data/raw/onet/` | `CC BY 4.0` | **None** (governmental occupational taxonomy) | Taxonomy: Software tools and technology dictionary with `Hot Technology` and `In Demand` priority weighting for ATS scoring | US-centric occupational taxonomy; emphasis on software products rather than transversal interpersonal competences. |
| **`ESCO Skills Taxonomy`** (`skills_en.csv`) | `data/raw/esco/` | `CC BY 4.0` | **None** (official European multilingual classification) | Taxonomy: Canonical transversal skills and comprehensive synonym mapping (`altLabels`) | Requires manual export from the official EU ESCO portal form; bureaucratic phrasing occasionally diverges from colloquial tech job descriptions. |

