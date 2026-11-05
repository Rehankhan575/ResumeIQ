/**
 * frontend/src/services/api.ts
 * ============================
 * Production API client connecting to FastAPI backend.
 *
 * Strict Error Policy (Item 3):
 * - NO silent mock fallbacks on network or server errors.
 * - Offline mock data is ONLY loaded when VITE_USE_MOCKS === "1".
 * - If the API is unreachable or fails, throw so the UI displays an ErrorBanner.
 */

import {
  ScoreResult,
  SamplesResponse,
  RankResponse,
  ModelResultsResponse,
} from "../types/api";
import { MOCK_FIXTURE_RESULT, MOCK_SAMPLES } from "../mocks/fixtureData";

const API_BASE = "";

export function isMockMode(): boolean {
  return (import.meta as any).env?.VITE_USE_MOCKS === "1";
}

export async function fetchSamples(): Promise<SamplesResponse> {
  if (isMockMode()) {
    return MOCK_SAMPLES;
  }
  const res = await fetch(`${API_BASE}/api/samples`, { method: "GET" });
  if (!res.ok) {
    throw new Error(`Failed to load samples (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function analyzeResume(
  file: File | null,
  resumeChoice: string,
  jdText: string,
  jdChoice: string,
  useGenai: boolean = true
): Promise<ScoreResult> {
  if (isMockMode()) {
    return MOCK_FIXTURE_RESULT;
  }

  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  } else if (resumeChoice) {
    formData.append("resume_choice", resumeChoice);
  }

  if (jdText && jdText.trim()) {
    formData.append("jd_text", jdText.trim());
  } else if (jdChoice) {
    formData.append("jd_choice", jdChoice);
  }

  formData.append("use_genai", String(useGenai));

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const detail = errorData.detail || `Server returned error HTTP ${res.status}`;
    throw new Error(detail);
  }

  return await res.json();
}

export async function rankJobs(
  file: File | null,
  resumeChoice: string,
  jdFilenames: string[]
): Promise<RankResponse> {
  if (isMockMode()) {
    return {
      ranked: [
        {
          jd_title: "Senior Data Scientist",
          score: 84.88,
          grade: "B",
          matched_skills: ["machine learning", "python", "scikit-learn", "sql"],
          missing_skills: [],
        },
        {
          jd_title: "Full Stack Software Engineer",
          score: 60.5,
          grade: "D",
          matched_skills: ["python", "sql"],
          missing_skills: ["javascript", "react", "node.js", "git"],
        },
        {
          jd_title: "Executive Chef",
          score: 55.5,
          grade: "F",
          matched_skills: [],
          missing_skills: [
            "culinary arts",
            "kitchen sanitation",
            "food safety",
            "menu planning",
            "haccp",
            "management",
          ],
        },
      ],
    };
  }

  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  } else if (resumeChoice) {
    formData.append("resume_choice", resumeChoice);
  }

  jdFilenames.forEach((name) => formData.append("jd_filenames", name));

  const res = await fetch(`${API_BASE}/api/rank`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const detail = errorData.detail || `Ranking failed with HTTP ${res.status}`;
    throw new Error(detail);
  }
  return await res.json();
}

export async function fetchModelResults(): Promise<ModelResultsResponse> {
  if (isMockMode()) {
    // Return sample mock structure if offline
    return {
      classifier: {
        models: [
          {
            key: "tfidf_lr",
            model: "TF-IDF + LR",
            val_macro_f1: "0.5913",
            test_macro_f1_ci: "0.5959 [0.5465, 0.6352]",
            test_accuracy: "66.31%",
            train_val_gap: "+0.3969",
            served: false,
          },
          {
            key: "tfidf_mlp",
            model: "TF-IDF + MLP (Served)",
            val_macro_f1: "0.6001",
            test_macro_f1_ci: "0.6181 [0.5606, 0.6587]",
            test_accuracy: "65.24%",
            train_val_gap: "+0.3999",
            served: true,
          },
          {
            key: "bilstm",
            model: "BiLSTM",
            val_macro_f1: "0.1860",
            test_macro_f1_ci: "0.1966 [0.1610, 0.2290]",
            test_accuracy: "25.13%",
            train_val_gap: "+0.5231",
            served: false,
          },
          {
            key: "distilbert",
            model: "DistilBERT (3 epochs)",
            val_macro_f1: "0.5675",
            test_macro_f1_ci: "0.6454 [0.5970, 0.6874]",
            test_accuracy: "72.73%",
            train_val_gap: "+0.1073",
            served: false,
          },
        ],
        paired_tests: [],
        note: "The served classifier is the MLP because the selection rule (val macro-F1) was fixed on validation before the test set was opened.",
      },
      matcher: {
        models: [],
        categories: [],
        note: "Matcher note: Overall metrics report query-level pooled (micro) averages across all queries; per-category metrics report within-category query means.",
      },
      ablation: {
        components: [],
        note: "Scoring note: No ground-truth ATS score exists; weights are heuristics.",
      },
      xai: {
        rows: [],
        note: "XAI note: Sentence deletion is inconclusive and unvalidated. Word deletion shows faithfulness to the linear model and transfer to the served model.",
      },
    };
  }

  const res = await fetch(`${API_BASE}/api/results`, { method: "GET" });
  if (!res.ok) {
    throw new Error(`Failed to load model reports (HTTP ${res.status})`);
  }
  return await res.json();
}
