import React, { useState } from "react";
import { ScoreResult } from "../../types/api";
import { Card } from "../ui/Card";
import { ScoreRing } from "./ScoreRing";
import { ComponentBreakdown } from "./ComponentBreakdown";
import { IssuesList } from "./IssuesList";
import { KeywordChips } from "./KeywordChips";
import { HighlightedResume } from "./HighlightedResume";
import { CategoryPredictions } from "./CategoryPredictions";
import { TimingsCard } from "./TimingsCard";
import { AiSuggestions } from "./AiSuggestions";
import { ReportExport } from "./ReportExport";
import { Sparkles, Layers, FileText } from "lucide-react";

interface ResultsViewProps {
  result: ScoreResult | null;
  onGoToAnalyze: () => void;
  rawResumeText?: string;
}

export const ResultsView: React.FC<ResultsViewProps> = ({
  result,
  onGoToAnalyze,
  rawResumeText,
}) => {
  const [activeTab, setActiveTab] = useState<"overview" | "suggestions">("overview");

  if (!result) {
    return (
      <div className="py-20 text-center max-w-md mx-auto space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-surface-card border border-border flex items-center justify-center text-content-muted mx-auto">
          <FileText strokeWidth={1.5} className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-semibold tracking-heading text-content-primary">
          No Evaluation Results Yet
        </h3>
        <p className="text-xs text-content-secondary leading-relaxed">
          Upload a resume and job description or select an authentic sample fixture on the Analyze screen to generate an ATS scorecard.
        </p>
        <button
          onClick={onGoToAnalyze}
          className="mt-2 px-5 py-2 text-xs font-mono text-sand border border-sand/20 rounded-full bg-sand/10 hover:bg-sand/20 transition-colors"
        >
          Go to Analyze →
        </button>
      </div>
    );
  }

  // Calculate job match and quality sub-scores from weights
  const fitKeys = ["semantic_match", "keyword_coverage", "experience_fit"];
  const qualKeys = [
    "section_completeness",
    "formatting",
    "quantified_achievements",
    "action_verbs",
    "readability",
    "contact_completeness",
  ];

  const weights: Record<string, number> = {
    semantic_match: 0.25,
    keyword_coverage: 0.20,
    section_completeness: 0.10,
    formatting: 0.10,
    experience_fit: 0.10,
    quantified_achievements: 0.10,
    action_verbs: 0.05,
    readability: 0.05,
    contact_completeness: 0.05,
  };

  const wFit = 0.55;
  const wQual = 0.45;

  const fitWeighted = fitKeys.reduce((acc, k) => acc + (weights[k] || 0) * (result.components[k] || 0), 0);
  const qualWeighted = qualKeys.reduce((acc, k) => acc + (weights[k] || 0) * (result.components[k] || 0), 0);

  const jobMatchScore = wFit > 0 ? (fitWeighted / wFit) * 100 : 0;
  const resumeQualityScore = wQual > 0 ? (qualWeighted / wQual) * 100 : 0;

  const topIssue = result.issues && result.issues.length > 0 ? result.issues[0] : undefined;

  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      {/* Top Header & Actions Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-2 border-b border-border">
        {/* Sub-tabs: Overview vs AI Suggestions */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-surface border border-border">
          <button
            onClick={() => setActiveTab("overview")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5 ${
              activeTab === "overview"
                ? "bg-surface-subtle text-content-primary shadow-sm"
                : "text-content-muted hover:text-content-secondary"
            }`}
          >
            <Layers strokeWidth={1.5} className="w-3.5 h-3.5" />
            <span>Scorecard Overview</span>
          </button>
          <button
            onClick={() => setActiveTab("suggestions")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5 ${
              activeTab === "suggestions"
                ? "bg-surface-subtle text-sand shadow-sm"
                : "text-content-muted hover:text-content-secondary"
            }`}
          >
            <Sparkles strokeWidth={1.5} className="w-3.5 h-3.5" />
            <span>AI Rewrites & Suggestions ({result.suggestions?.length || 0})</span>
          </button>
        </div>

        {/* Report Download Button */}
        <div className="flex items-center gap-3">
          <ReportExport result={result} />
        </div>
      </div>

      {activeTab === "suggestions" ? (
        <Card glass={true}>
          <AiSuggestions suggestions={result.suggestions || []} />
        </Card>
      ) : (
        /* 12-Column Bento Grid */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Row 1, Left: Score Card (5 cols) */}
          <div className="lg:col-span-5">
            <Card glass={true} className="h-full">
              <ScoreRing
                score={result.total}
                grade={result.grade}
                topFixMessage={topIssue?.message}
                topFixGain={topIssue?.estimated_gain}
                jobMatchScore={jobMatchScore}
                resumeQualityScore={resumeQualityScore}
              />
            </Card>
          </div>

          {/* Row 1, Right: Component Breakdown (7 cols) */}
          <div className="lg:col-span-7">
            <Card glass={true} className="h-full">
              <ComponentBreakdown components={result.components} />
            </Card>
          </div>

          {/* Row 2, Left: Issues Action Items (7 cols) */}
          <div className="lg:col-span-7">
            <Card surface="card">
              <IssuesList issues={result.issues} />
            </Card>
          </div>

          {/* Row 2, Right: Keywords Matched vs Missing (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <Card surface="card">
              <KeywordChips
                matchedSkills={result.match?.matched_skills || []}
                missingSkills={result.match?.missing_skills || result.missing_skills || []}
              />
            </Card>

            <Card surface="card">
              <CategoryPredictions categories={result.predicted_categories || []} />
            </Card>

            {/* Execution Latency Card */}
            <TimingsCard timings={result.timings_ms || {}} />
          </div>

          {/* Row 3: Highlighted Resume Reader Panel (12 cols) */}
          <div className="lg:col-span-12">
            <Card surface="card">
              <HighlightedResume
                rawText={rawResumeText}
                highlights={result.highlights || []}
              />
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

