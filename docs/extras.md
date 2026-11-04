# Stage 7b: Modular Extras & Value-Add Capabilities

## 1. Multi-Job Description Ranking (`rank_jds`)

`rank_jds(resume, jds)` evaluates a single candidate résumé against an arbitrary collection of job descriptions.
- **Scoring Pipeline**: Evaluates each pair using the Stage 5 ATS composite scorer (`app.scoring.ats.score_resume`), falling back gracefully to the Stage 4 neural matcher (`app.models.matcher.match`) or TF-IDF cosine similarity.
- **Output Structure**: Returns a list sorted strictly descending by composite score, providing:
  - `jd_title`: Target job title.
  - `score`: ATS composite score (0–100).
  - `grade`: Letter grade (A through F).
  - `missing_skills`: List of required skills from that JD absent from the candidate's profile.
  - `rank`: 1-based integer rank.

---

## 2. Skill Gap Analysis & Learning Recommendations (`skill_gap`)

`skill_gap(resume_skills, jd_required)` identifies specific skill deficits between a candidate's verified skills and target job requirements, mapping each gap to curated courseware from `data/courses.json`.

### Curated Course Catalog (`data/courses.json`)
- Covers **70 in-demand skills** across Software Engineering, Machine Learning, Data Science, Cloud/DevOps, and Business Management.
- Each skill features 1–2 verified courses detailing:
  - `title`: Course or specialization title.
  - `provider`: Academic institution or platform (e.g., Coursera, DeepLearning.AI, edX, Linux Foundation).
  - `level`: Difficulty level (`Beginner`, `Intermediate`, `Advanced`).
  - `estimated_hours`: Approximate time commitment.
  - **Zero URLs**: Consistent with privacy and platform policies, no external URLs or web addresses are stored.

> [!NOTE]
> **Learning Resource Verification**: All courses and hours listed in `data/courses.json` and suggested by `skill_gap()` are recommendations intended to guide self-directed professional development. Candidates should independently verify course availability, prerequisites, and syllabi with the respective educational providers.

---

## 3. Unconscious Bias Auditing (`bias_check`)

`bias_check(parsed_resume)` audits resumes for demographic signals that could introduce unconscious human bias during hiring reviews:

1. **Gendered Wording**:
   - Identifies third-person pronouns (`he`, `she`, `her`, `him`) in summary or profile sections.
   - Detects strongly gender-coded adjectives (masculine: *aggressive*, *dominant*; feminine: *nurturing*, *supportive*).
2. **Age Cues**:
   - Flags graduation years older than 15 years (e.g., $\le 2010$).
   - Identifies age-correlated descriptors (*young*, *energetic*, *seasoned veteran*).
3. **Personal Details**:
   - Detects disclosures of Date of Birth (DOB), marital status, nationality/citizenship, religious affiliations, and photographs/headshots.

### Privacy-Preserving Output
To prevent secondary bias and protect privacy:
- Findings report only the **category type** and an **abstract evidence label** (e.g., `Date of birth disclosed`, `Marital status disclosed`, `Graduation year older than 15 years`).
- **Never outputs matched personal details** (e.g., the actual birth date, marital status, or nationality are never returned or logged).
- All feedback is framed constructively as suggestions to optimize compliance with US/UK/EU blind screening standards.

---

## 4. Local SQLite History Logging (`record_history`)

`record_history(score_result, jd_title, db_path, opt_in)` provides opt-in local tracking of scoring performance over time.

### Security & Privacy Protections
- **Strict Opt-In**: The `opt_in` parameter defaults to `False`. When `opt_in=False`, no database file is opened and zero bytes are written.
- **Zero Text Storage**: The SQLite database (`data/history.db`) contains **no text fields for resume content, candidate names, emails, phone numbers, or bullet points**.
- **Table Schema (`score_history`)**:
  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  - `timestamp`: TEXT (ISO 8601 UTC)
  - `jd_title`: TEXT (job title only)
  - `total_score`: REAL (0.0–100.0)
  - `grade`: TEXT (letter grade)
  - Numeric component scores: `semantic_match`, `keyword_coverage`, `section_completeness`, `formatting`, `experience_fit`, `quantified_achievements`, `action_verbs`, `readability`, `contact_completeness` (all REAL numbers).
