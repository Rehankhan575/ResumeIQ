# Data Audit Report — Raw Datasets Quality, Leakage & Integrity

> **Audit Date**: 2026-11-02  
> **Scope**: `ner_real_skills`, `ner_synthetic`, `job_resume_fit`, `livecareer`, `linkedin_sample`, `onet`.  
> **Privacy Compliance**: No individual resume, job description, or job posting text is printed or saved. Only counts, aggregates, distributions, and frequent corpus tokens (counts >= 50) are reported.

---

## 1. NER Real Skills (`ner_real_skills`)

### 1.1 Annotation Alignment
| Metric | Count | Percentage |
|---|---|---|
| Total Annotations Scanned | 538,482 | 100.00% |
| Full Corpus Aligned (`text[start:end]` matches `label`) | 538,482 | **100.00%** |
| Sample 500 Resumes Scanned (seed 42) | 51,976 | 100.00% |
| Sample 500 Aligned Annotations | 51,976 | **100.00%** |

> [!NOTE]
> Offset alignment is 100.00% consistent across both the full corpus (538,482 annotations) and the random 500-resume test slice. Every labeled slice character matches the expected entity string.

### 1.2 Annotations Per Resume & String Dispersion
| Metric | Value |
|---|---|
| Mean Annotations per Resume | 107.08 |
| Median Annotations per Resume | 92.0 |
| 95th Percentile (p95) Annotations | 239.0 |
| Distinct Annotated Strings | 12,695 |
| Strings Occurring in Only 1 Resume (Hapax Legomena) | 3,298 (25.98%) |

### 1.3 Top 60 Annotated Strings (All Counts >= 50)
| Rank | Annotated String | Frequency Count | In O*NET? |
|---|---|---|---|
| 1 | `com` | 3,874 | No |
| 2 | `skills` | 3,805 | No |
| 3 | `work` | 3,679 | No |
| 4 | `personal` | 3,458 | No |
| 5 | `knowledge` | 3,417 | No |
| 6 | `english` | 3,399 | No |
| 7 | `gmail` | 3,238 | No |
| 8 | `project` | 2,683 | No |
| 9 | `professional` | 2,661 | No |
| 10 | `information` | 2,643 | No |
| 11 | `hindi` | 2,636 | No |
| 12 | `company` | 2,617 | No |
| 13 | `nationality` | 2,588 | No |
| 14 | `management` | 2,562 | No |
| 15 | `team` | 2,499 | No |
| 16 | `office` | 2,465 | No |
| 17 | `is` | 2,406 | No |
| 18 | `technical` | 2,401 | No |
| 19 | `engineering` | 2,358 | No |
| 20 | `organization` | 2,344 | No |
| 21 | `education` | 2,327 | No |
| 22 | `system` | 2,294 | No |
| 23 | `marital status` | 2,266 | No |
| 24 | `c` | 2,179 | Yes |
| 25 | `computer` | 2,164 | No |
| 26 | `ltd` | 2,158 | No |
| 27 | `training` | 2,155 | No |
| 28 | `languages` | 2,136 | No |
| 29 | `college` | 2,130 | No |
| 30 | `board` | 2,000 | No |
| 31 | `email` | 1,984 | No |
| 32 | `qualification` | 1,974 | No |
| 33 | `m` | 1,963 | No |
| 34 | `activities` | 1,956 | No |
| 35 | `ms` | 1,909 | No |
| 36 | `engineer` | 1,766 | No |
| 37 | `projects` | 1,758 | No |
| 38 | `communication` | 1,722 | No |
| 39 | `technology` | 1,718 | No |
| 40 | `power` | 1,664 | No |
| 41 | `mobile` | 1,633 | No |
| 42 | `development` | 1,621 | No |
| 43 | `passport` | 1,620 | No |
| 44 | `time` | 1,601 | No |
| 45 | `data` | 1,588 | No |
| 46 | `academic` | 1,577 | No |
| 47 | `maintenance` | 1,558 | No |
| 48 | `diploma` | 1,545 | No |
| 49 | `quality` | 1,544 | No |
| 50 | `software` | 1,504 | No |
| 51 | `control` | 1,504 | No |
| 52 | `language` | 1,474 | No |
| 53 | `well` | 1,457 | No |
| 54 | `systems` | 1,443 | No |
| 55 | `service` | 1,440 | No |
| 56 | `excel` | 1,436 | No |
| 57 | `can` | 1,401 | No |
| 58 | `ability` | 1,382 | No |
| 59 | `responsibilities` | 1,380 | No |
| 60 | `analysis` | 1,374 | No |

### 1.4 O*NET Software Skills Overlap
| Comparison Set | Matching in O*NET Software Skills | Total In Set | Fraction |
|---|---|---|---|
| All Distinct Annotated Strings | 191 | 12,695 | **1.50%** |
| Top 60 Most Frequent Strings | 1 | 60 | **1.67%** |

> [!WARNING]
> Severe noise detected: Only 1.50% of distinct annotated strings appear in the O*NET software skills taxonomy. Top frequent terms include dictionary words like `com`, `skills`, `work`, `personal`, `knowledge`, and `development`. Downstream skill extractors must filter out generic dictionary stop-words before model ingestion.

---

## 2. Synthetic Resume NER (`ner_synthetic`)

### 2.1 Split Overlap & Partitioning Integrity
| Metric | Train Split | Validation Split | Test Split | Overlap (Train ∩ Val) | Overlap (Train ∩ Test) | Overlap (Val ∩ Test) |
|---|---|---|---|---|---|---|
| Row Count | 9,600 | 1,200 | 1,200 | — | — | — |
| Distinct Template Families | 24 | 3 | 3 | 0 | 0 | 0 |
| Distinct Candidate IDs | 9,600 | 1,200 | 1,200 | 0 | 0 | 0 |
| Exact Text Duplicates (within) | 0 | 0 | 0 | — | — | — |
| Exact Text Overlap (cross) | — | — | — | 0 | 0 | 0 |

> [!TIP]
> Zero candidate_id or template_family overlap exists across train, validation, and test splits. The dataset is cleanly partitioned by template family (24 in train, 3 in val, 3 in test).

### 2.2 SKILL Entity Vocabulary Dispersion Across Splits
| Split / Metric | Count |
|---|---|
| Distinct SKILL Strings (Train) | 301 |
| Distinct SKILL Strings (Validation) | 301 |
| Distinct SKILL Strings (Test) | 301 |
| Distinct SKILL Strings Overall (All Splits Combined) | 301 |
| Test SKILL Mentions Occurring in Train Set | 22,557 / 22,557 (**100.00%**) |

> [!NOTE]
> Exactly 301 distinct skills are reused throughout all splits; 100.00% of test skill mentions are seen in train. While entity boundaries test model generalization across template structures, the vocabulary itself is closed and static.

### 2.3 Per-Resume Entity-Count Statistics (Templating Confirmation)
| Entity Label | Min Count | Median Count | Max Count | Templating Pattern |
|---|---|---|---|---|
| `NAME` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `EMAIL` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `PHONE` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `LOCATION` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `DEGREE` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `FIELD` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `INSTITUTION` | 1 | 1.0 | 1 | Strictly 1 per resume |
| `COMPANY` | 0 | 2.0 | 5 | Variable per template |
| `TITLE` | 1 | 3.0 | 6 | Variable per template |
| `DATE` | 1 | 4.0 | 8 | Variable per template |
| `SKILL` | 8 | 19.0 | 36 | Variable per template |
| `CERT` | 0 | 1.0 | 2 | Variable per template |
| `LANGUAGE` | 1 | 2.0 | 3 | Variable per template |

---

## 3. Resume-Job Fit Dataset (`job_resume_fit`)

### 3.1 ID Alignment with LiveCareer & Uniqueness
| Metric | Count | Note |
|---|---|---|
| Total JRF Rows | 2,385 | |
| Unique Candidate IDs | 2,385 | All unique (no duplicate rows) |
| JRF IDs Present in LiveCareer | 2,385 | **100.00% match** (Direct subset of LiveCareer) |
| Unique `resume_text` Values | 2,383 | 2 duplicate resume texts |
| Unique `job_text` Values | 23 | **Exactly 23 distinct job postings** (1 per category) |
| Max Resumes Sharing One Job Description | 120 | Largest category pairs 120 resumes with identical JD |

### 3.2 `ai_match_score` Distribution & Correlations
| Metric / Quantile | Score Value |
|---|---|
| Minimum Score | 0.00 |
| 5th Percentile (p5) | 11.70 |
| 25th Percentile (p25) | 22.23 |
| Median (50th Percentile) | 40.65 |
| 75th Percentile (p75) | 63.95 |
| 95th Percentile (p95) | 86.52 |
| Maximum Score | 100.00 |

#### 10-Bin Histogram (`ai_match_score`)
| Score Range Bin | Row Count | Percentage | Distribution Bar |
|---|---|---|---|
| [0, 10) | 79 | 3.31% | `█` |
| [10, 20) | 403 | 16.90% | `████████` |
| [20, 30) | 408 | 17.11% | `████████` |
| [30, 40) | 285 | 11.95% | `█████` |
| [40, 50) | 264 | 11.07% | `█████` |
| [50, 60) | 254 | 10.65% | `█████` |
| [60, 70) | 227 | 9.52% | `████` |
| [70, 80) | 213 | 8.93% | `████` |
| [80, 90) | 171 | 7.17% | `███` |
| [90, 100] | 81 | 3.40% | `█` |

#### Spearman Rank Correlation with Heuristic Scores
| Comparison | Spearman Correlation (r) | p-value | Significance |
|---|---|---|---|
| `ai_match_score` vs `skill_string_match_score` | **0.5966** | 5.43e-230 | p < 0.001 |
| `ai_match_score` vs `fuzzy_match_score` | **0.6531** | 4.32e-290 | p < 0.001 |

### 3.3 Skill Fields & Job Text Length Statistics
| Metric | Value |
|---|---|
| Rows with Empty `job_required_skills` | 0 |
| Rows with Empty `resume_skill_list` | 0 |
| `job_text` Length (Min) | 3,677 characters |
| `job_text` Length (Median) | 4,802.0 characters |
| `job_text` Length (Mean) | 4,806.3 characters |
| `job_text` Length (Max) | 5,113 characters |

---

## 4. LiveCareer Resume Dataset (`livecareer`)

### 4.1 Duplicates and Text Length Profile
| Metric | Count | Details |
|---|---|---|
| Duplicate IDs | 0 | Zero ID collisions |
| Exact-Duplicate `Resume_str` Groups | 2 | 2 pairs (4 rows total) |
| Duplicate Groups with Conflicting Categories | **0** | All copies share the exact same Category label |
| `Resume_str` Length < 200 characters | 1 | Near-empty resume |
| `Resume_str` Length < 500 characters | 1 | Stub / truncated resume |
| `Resume_str` Length < 1000 characters | 4 | Truncated resumes |

### 4.2 Near-Duplicate Analysis (TF-IDF Cosine Similarity)
| Cosine Similarity Threshold | Near-Duplicate Pairs | Cross-Category Pairs |
|---|---|---|
| Cosine Similarity >= 0.95 | 14 pairs | **0** (0.00%) |
| Cosine Similarity >= 0.90 | 22 pairs | **0** (0.00%) |

> [!NOTE]
> All near-duplicate resumes share the identical category; zero near-duplicate pairs cross categories.

### 4.3 Category Label Leakage in First 100 Normalised Characters
- **Overall Corpus Leakage Rate**: **68.88%** (1,711 / 2,484 resumes contain their category name or main word in the first 100 characters).

#### Top 5 Worst Leakage Categories
| Rank | Category | Leaked Resumes | Category Total | Leak Percentage |
|---|---|---|---|---|
| 1 | `SALES` | 116 | 116 | **100.00%** |
| 2 | `HR` | 110 | 110 | **100.00%** |
| 3 | `FINANCE` | 118 | 118 | **100.00%** |
| 4 | `ENGINEERING` | 118 | 118 | **100.00%** |
| 5 | `BUSINESS-DEVELOPMENT` | 119 | 120 | **99.17%** |

#### Full Category Leakage Table
| Category | Leaked Count | Total | Leak Percentage |
|---|---|---|---|
| `SALES` | 116 | 116 | 100.00% |
| `HR` | 110 | 110 | 100.00% |
| `FINANCE` | 118 | 118 | 100.00% |
| `ENGINEERING` | 118 | 118 | 100.00% |
| `BUSINESS-DEVELOPMENT` | 119 | 120 | 99.17% |
| `ACCOUNTANT` | 117 | 118 | 99.15% |
| `CONSULTANT` | 114 | 115 | 99.13% |
| `DESIGNER` | 106 | 107 | 99.07% |
| `TEACHER` | 100 | 102 | 98.04% |
| `CHEF` | 98 | 118 | 83.05% |
| `CONSTRUCTION` | 92 | 112 | 82.14% |
| `DIGITAL-MEDIA` | 77 | 96 | 80.21% |
| `INFORMATION-TECHNOLOGY` | 95 | 120 | 79.17% |
| `ADVOCATE` | 81 | 118 | 68.64% |
| `PUBLIC-RELATIONS` | 53 | 111 | 47.75% |
| `HEALTHCARE` | 52 | 115 | 45.22% |
| `BANKING` | 38 | 115 | 33.04% |
| `FITNESS` | 34 | 117 | 29.06% |
| `AVIATION` | 30 | 117 | 25.64% |
| `ARTS` | 25 | 103 | 24.27% |
| `BPO` | 3 | 22 | 13.64% |
| `APPAREL` | 8 | 97 | 8.25% |
| `AGRICULTURE` | 5 | 63 | 7.94% |
| `AUTOMOBILE` | 2 | 36 | 5.56% |

> [!CAUTION]
> **Critical for Stage 3 Classifier**: 68.88% of resumes have their category label verbatim as the header/title in the first 100 characters. If models ingest headers without sanitization, classifier evaluation will artificially inflate. Preprocessing must strip leading job title / category headers before training and evaluation.

### 4.4 PII Presence in LiveCareer (Counts & Rates Only)
| PII Entity Type | Matching Resumes | Corpus Rate | Privacy Action |
|---|---|---|---|
| Email Address Detected | 19 | 0.76% | Redacted; no text serialized |
| Phone Pattern Detected | 77 | 3.10% | Redacted; no text serialized |
| Both Email & Phone Detected | 15 | 0.60% | Redacted; no text serialized |

---

## 5. LinkedIn Job Postings Sample (`linkedin_sample`)

### 5.1 Deduplication Profile
| Deduplication Stage | Row Count | Difference / Removed |
|---|---|---|
| Raw Sample Count | 5,000 | Baseline |
| Exact Deduplicated Count | 4,991 | 9 exact duplicate rows removed |
| Near-Duplicate Rows by `(title, description[:200])` | 235 | Spanning 78 distinct duplicate groups |
| Excess Near-Duplicate Postings | 157 | Redundant identical postings |

### 5.2 Description Length Statistics
| Metric | Character Count |
|---|---|
| Minimum Description Length | 204 |
| Median Description Length | 3,489.0 |
| Mean Description Length | 3,806.4 |
| Maximum Description Length | 17,755 |

### 5.3 Top 30 Job Titles
| Rank | Job Title | Posting Count |
|---|---|---|
| 1 | Sales Manager | 24 |
| 2 | Project Manager | 20 |
| 3 | Customer Service Representative | 17 |
| 4 | Senior Accountant | 14 |
| 5 | Executive Assistant | 13 |
| 6 | Staff Accountant | 12 |
| 7 | Accounting Manager | 12 |
| 8 | Account Executive | 12 |
| 9 | Administrative Assistant | 12 |
| 10 | General Manager | 11 |
| 11 | Account Manager | 10 |
| 12 | OPERATIONS ASSISTANT MANAGER | 9 |
| 13 | Project Engineer | 9 |
| 14 | Business Analyst | 9 |
| 15 | Data Engineer | 9 |
| 16 | Service Technician | 8 |
| 17 | Assistant Manager | 8 |
| 18 | Registered Nurse | 8 |
| 19 | Package Handler - Part Time (Warehouse like) | 8 |
| 20 | Retail Sales Associate | 8 |
| 21 | Quality Engineer | 7 |
| 22 | Salesperson | 7 |
| 23 | Registered Nurse - RN - LTAC | 7 |
| 24 | Data Analyst | 7 |
| 25 | Receptionist | 7 |
| 26 | Business Development Manager | 7 |
| 27 | Sales Specialist | 6 |
| 28 | Field Service Technician | 6 |
| 29 | Mechanical Engineer | 6 |
| 30 | Sales Executive | 6 |

---

*Report generated automatically by `scripts/audit_data.py`.*