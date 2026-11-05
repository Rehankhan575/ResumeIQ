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

export interface BiasFinding {
  type: string;
  evidence_label: string;
  suggestion: string;
}

export interface CourseRecommendation {
  title: string;
  provider: string;
  level: string;
  estimated_hours: number;
}

export interface SkillGapSummary {
  matched: string[];
  missing: string[];
  recommendations: Record<string, CourseRecommendation[]>;
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
  bias?: BiasFinding[];
  skill_gap?: SkillGapSummary;
  resume_text?: string;
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

export interface ClassifierModelRow {
  key: string;
  model: string;
  val_macro_f1: string;
  test_macro_f1_ci: string;
  test_accuracy: string;
  train_val_gap: string;
  served: boolean;
}

export interface PairedTestRow {
  comparison: string;
  metric: string;
  diff: string;
  ci_95: string;
  p_value: string;
}

export interface MatcherModelRow {
  key: string;
  model: string;
  val_mrr: string;
  val_ndcg10: string;
  val_recall5: string;
  val_recall10: string;
  test_mrr: string;
  test_ndcg10: string;
  test_recall5: string;
  test_recall10: string;
  served: boolean;
}

export interface MatcherCategoryRow {
  category: string;
  val_count: number;
  test_count: number;
  val_tfidf_mrr: string;
  val_zs_mrr: string;
  val_ft_mrr: string;
  test_tfidf_mrr: string;
  test_zs_mrr: string;
  test_ft_mrr: string;
}

export interface AblationRow {
  component: string;
  original_weight: string;
  mean_original: string;
  mean_ablated: string;
  mean_delta: string;
}

export interface XAiRow {
  attribution_level: string;
  method: string;
  observed_drops: string;
  verdict: string;
}

export interface ModelResultsResponse {
  classifier: {
    models: ClassifierModelRow[];
    paired_tests: PairedTestRow[];
    note: string;
  };
  matcher: {
    models: MatcherModelRow[];
    categories: MatcherCategoryRow[];
    note: string;
  };
  ablation: {
    components: AblationRow[];
    note: string;
  };
  xai: {
    rows: XAiRow[];
    note: string;
  };
}
