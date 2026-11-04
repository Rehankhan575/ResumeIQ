/**
 * frontend/src/types/api.ts
 * =========================
 * TypeScript definitions generated directly from app/schemas.py (SCHEMA_VERSION = 1.3.0).
 * Preserves strict type safety across all pipeline outputs and API responses.
 */

export type IssueSeverity = "low" | "medium" | "high";

export interface Issue {
  component: string;
  message: string;
  severity: IssueSeverity;
  estimated_gain: number;
}

export type HighlightType = "positive" | "negative" | "keyword_match" | "keyword_missing";

export interface Highlight {
  start: number;
  end: number;
  text: string;
  weight: number;
  type: HighlightType;
}

export interface CategoryPrediction {
  category: string;
  probability: number;
}

export interface MatchResult {
  score: number;
  section_scores: Record<string, number>;
  matched_skills: string[];
  missing_skills: string[];
  method: string;
}

export type SuggestionKind = "bullet_rewrite" | "cover_letter" | "improvement_plan";

export interface Suggestion {
  kind: SuggestionKind;
  original: string;
  suggested: string;
  notes: string;
}

export interface ContactInfo {
  name: string;
  email: string;
  phone: string;
  linkedin: string;
  github: string;
  other_links: string[];
}

export interface LayoutMeta {
  file_type: string;
  page_count: number;
  word_count: number;
  has_tables: boolean;
  has_images: boolean;
  is_multi_column: boolean;
  font_count: number;
  ocr_used: boolean;
  needs_ocr: boolean;
  extraction_method: string;
}

export interface EducationEntry {
  degree: string;
  field: string;
  institution: string;
  start_year?: number | null;
  end_year?: number | null;
  graduation_year?: number | null;
  gpa?: number | null;
}

export interface ExperienceEntry {
  title: string;
  company: string;
  start_date: string;
  end_date: string;
  start: string;
  end: string;
  is_current: boolean;
  duration_months: number;
  bullets: string[];
}

export interface ProjectEntry {
  name: string;
  description: string;
  technologies: string[];
}

export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  score: number;
  source: "rules" | "ner";
}

export interface ParsedResume {
  raw_text: string;
  sections: Record<string, string>;
  contact: ContactInfo;
  layout: LayoutMeta;
  skills: string[];
  entities: Entity[];
  education: EducationEntry[];
  experience: ExperienceEntry[];
  projects: ProjectEntry[];
  certifications: string[];
  total_years_experience: number;
  warnings: string[];
}

export interface ScoreResult {
  total: number;
  grade: string;
  components: Record<string, number>;
  missing_skills: string[];
  issues: Issue[];
  highlights: Highlight[];
  predicted_categories: CategoryPrediction[];
  match?: MatchResult | null;
  suggestions: Suggestion[];
  timings_ms: Record<string, number>;
  warnings: string[];
}

export interface SampleResume {
  filename: string;
  title: string;
  extension: string;
  size_bytes: number;
}

export interface SampleJD {
  filename: string;
  title: string;
  preview: string;
  full_text: string;
}

export interface SamplesResponse {
  resumes: SampleResume[];
  jds: SampleJD[];
}

export interface RankedJob {
  jd_title: string;
  score: number;
  grade: string;
  matched_skills: string[];
  missing_skills: string[];
}

export interface RankResponse {
  ranked: RankedJob[];
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  jd_title: string;
  score: number;
  grade: string;
}

