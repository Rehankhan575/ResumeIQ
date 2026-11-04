import React from "react";

interface ComponentBreakdownProps {
  components: Record<string, number>;
}

const RUBRIC_META: Record<string, { label: string; maxPts: number }> = {
  semantic_match: { label: "Semantic Match", maxPts: 25 },
  keyword_coverage: { label: "Keyword Coverage", maxPts: 20 },
  section_completeness: { label: "Section Completeness", maxPts: 10 },
  formatting: { label: "Formatting & Structure", maxPts: 10 },
  experience_fit: { label: "Experience Fit", maxPts: 10 },
  quantified_achievements: { label: "Quantified Metrics", maxPts: 10 },
  action_verbs: { label: "Action Verbs", maxPts: 5 },
  readability: { label: "Readability & Flow", maxPts: 5 },
  contact_completeness: { label: "Contact Details", maxPts: 5 },
};

export const ComponentBreakdown: React.FC<ComponentBreakdownProps> = ({ components }) => {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between pb-1 border-b border-border">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
          9-Rubric ATS Decomposition
        </span>
        <span className="text-[11px] font-mono text-content-muted">
          Weight Distribution
        </span>
      </div>

      <div className="space-y-3.5">
        {Object.entries(RUBRIC_META).map(([key, meta]) => {
          const ratio = Math.max(0, Math.min(1, components[key] ?? 0));
          const pct = Math.round(ratio * 100);
          const earnedPts = (ratio * meta.maxPts).toFixed(1);

          // Thin 4px bar: white fill, band-coloured (warn/bad) only for weak scores (<50% or 0%)
          const barFill =
            pct === 0
              ? "bg-semantic-bad"
              : pct < 50
              ? "bg-semantic-warn"
              : "bg-content-primary";

          return (
            <div key={key} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-content-secondary font-medium truncate max-w-[200px]">
                  {meta.label}
                </span>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="font-mono text-content-primary">{pct}%</span>
                  <span className="text-[11px] font-mono text-content-muted px-2 py-0.5 rounded-full bg-surface border border-border">
                    {earnedPts} / {meta.maxPts} pts
                  </span>
                </div>
              </div>

              {/* Thin 4px Bar */}
              <div className="w-full h-1 bg-surface-subtle rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-out ${barFill}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
