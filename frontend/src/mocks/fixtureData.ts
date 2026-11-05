/**
 * frontend/src/mocks/fixtureData.ts
 * =================================
 * Authentic typed mock data generated directly from evaluating
 * tests/fixtures/ds_clean.pdf with tests/fixtures/jd_data_scientist.txt.
 *
 * Privacy: ZERO real candidate personal text or contact info is stored.
 */

import { ScoreResult, SamplesResponse } from "../types/api";

export const MOCK_FIXTURE_RESULT: ScoreResult = {
  total: 84.88,
  grade: "B",
  components: {
    semantic_match: 0.9091,
    keyword_coverage: 1.0,
    section_completeness: 1.0,
    formatting: 0.9,
    experience_fit: 1.0,
    quantified_achievements: 0.0,
    action_verbs: 0.8333,
    readability: 0.9472,
    contact_completeness: 0.85,
  },
  missing_skills: [],
  issues: [
    {
      component: "quantified_achievements",
      message: "Only 0 of 6 bullets (0%) contain quantified metrics. Add metrics, %, or revenue figures to demonstrate impact.",
      severity: "medium",
      estimated_gain: 10.0,
    },
    {
      component: "formatting",
      message: "Resume is relatively brief (167 words); recommended range is 350-900 words.",
      severity: "low",
      estimated_gain: 1.0,
    },
    {
      component: "contact_completeness",
      message: "No professional links (LinkedIn/GitHub/portfolio) provided.",
      severity: "low",
      estimated_gain: 0.75,
    },
  ],
  highlights: [
    {
      start: 46,
      end: 65,
      text: "predictive modeling",
      weight: 0.72,
      type: "positive",
    },
    {
      start: 79,
      end: 95,
      text: "Machine Learning",
      weight: 0.85,
      type: "keyword_match",
    },
    {
      start: 173,
      end: 179,
      text: "Python",
      weight: 0.9,
      type: "keyword_match",
    },
    {
      start: 181,
      end: 193,
      text: "Scikit-Learn",
      weight: 0.82,
      type: "keyword_match",
    },
    {
      start: 195,
      end: 198,
      text: "SQL",
      weight: 0.78,
      type: "keyword_match",
    },
    {
      start: 396,
      end: 412,
      text: "Computer Science",
      weight: 0.65,
      type: "positive",
    },
    {
      start: 454,
      end: 468,
      text: "Neural Network",
      weight: 0.68,
      type: "positive",
    },
  ],
  predicted_categories: [
    { category: "INFORMATION-TECHNOLOGY", probability: 0.395 },
    { category: "BANKING", probability: 0.334 },
    { category: "ENGINEERING", probability: 0.271 },
  ],
  match: {
    score: 0.9091,
    section_scores: {
      skills: 1.0,
      experience: 0.8135,
    },
    matched_skills: ["machine learning", "python", "scikit-learn", "sql"],
    missing_skills: [],
    method: "sbert_fine_tuned",
  },
  suggestions: [
    {
      kind: "bullet_rewrite",
      original: "Built machine learning pipelines and models for data processing.",
      suggested: "Engineered scalable ML pipelines using Scikit-Learn and SQL, accelerating data preprocessing efficiency by 34%.",
      notes: "Adds quantified impact (+34% efficiency) and concrete tool attribution.",
    },
    {
      kind: "improvement_plan",
      original: "Resume summary and achievement section",
      suggested: "Add 2 quantifiable business outcomes (revenue impact, latency reduction, or scale) to work experience bullets.",
      notes: "Directly addresses quantified_achievements component to gain up to +10.0 score points.",
    },
  ],
  timings_ms: {
    parsing_ms: 118.1,
    extraction_ms: 133.1,
    classification_ms: 294.8,
    matching_ms: 5999.3,
    scoring_ms: 13.1,
    explain_ms: 352.3,
    genai_ms: 0.0,
  },
  warnings: [],
  bias: [],
  skill_gap: {
    matched: ["machine learning", "python", "scikit-learn", "sql"],
    missing: [],
    recommendations: {},
  },
};

export const MOCK_SAMPLES: SamplesResponse = {
  resumes: [
    { filename: "ds_clean.pdf", title: "Data Scientist (Clean)", extension: ".pdf", size_bytes: 42120 },
    { filename: "swe_two_column.pdf", title: "Software Engineer (Two-Column)", extension: ".pdf", size_bytes: 58940 },
    { filename: "accountant_table.pdf", title: "Accountant (Tables)", extension: ".pdf", size_bytes: 51230 },
    { filename: "nurse_scanned.pdf", title: "Nurse (Scanned/OCR)", extension: ".pdf", size_bytes: 142850 },
    { filename: "chef_plain.pdf", title: "Executive Chef (Plain)", extension: ".pdf", size_bytes: 38400 },
  ],
  jds: [
    {
      filename: "jd_data_scientist.txt",
      title: "Senior Data Scientist",
      preview: "We are seeking a Senior Data Scientist to design and deploy robust machine learning models, ETL pipelines, and scalable predictive systems using Python, SQL, and Scikit-Learn.",
      full_text: "We are seeking a Senior Data Scientist to design and deploy robust machine learning models, ETL pipelines, and scalable predictive systems using Python, SQL, and Scikit-Learn. Requirements: 3+ years in ML engineering, proficiency in data modeling, statistical analysis, and production deployment.",
    },
    {
      filename: "jd_software_engineer.txt",
      title: "Full Stack Software Engineer",
      preview: "Looking for an experienced software engineer skilled in React, TypeScript, Node.js, and distributed backend systems with strong algorithmic problem-solving abilities.",
      full_text: "Looking for an experienced software engineer skilled in React, TypeScript, Node.js, and distributed backend systems with strong algorithmic problem-solving abilities.",
    },
    {
      filename: "jd_chef.txt",
      title: "Executive Chef",
      preview: "High-end dining establishment looking for an Executive Chef to manage kitchen staff, culinary design, menu development, inventory control, and food safety standards.",
      full_text: "High-end dining establishment looking for an Executive Chef to manage kitchen staff, culinary design, menu development, inventory control, and food safety standards.",
    },
  ],
};

export const REAL_PROJECT_STATS = [
  { label: "Scoring Dimensions", value: "9 Rubrics", detail: "Weighted ATS composite" },
  { label: "Neural Model", value: "Siamese SBERT", detail: "Fine-tuned bi-encoder" },
  { label: "Data Privacy", value: "Zero Storage", detail: "In-memory processing only" },
];

