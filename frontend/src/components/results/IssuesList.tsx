import React, { useState } from "react";
import { Issue } from "../../types/api";
import { ChevronDown, AlertCircle } from "lucide-react";

interface IssuesListProps {
  issues: Issue[];
}

export const IssuesList: React.FC<IssuesListProps> = ({ issues }) => {
  const [expandedIndices, setExpandedIndices] = useState<Record<number, boolean>>({ 0: true });

  const toggleExpand = (idx: number) => {
    setExpandedIndices((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  // Sort by estimated gain descending
  const sorted = [...issues].sort((a, b) => b.estimated_gain - a.estimated_gain);

  const getSeverityDot = (sev: string) => {
    switch (sev) {
      case "high":
        return "bg-semantic-bad";
      case "medium":
        return "bg-semantic-warn";
      default:
        return "bg-content-muted";
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between pb-1 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
            Prioritized Action Items
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-surface border border-border text-content-secondary">
            {issues.length} detected
          </span>
        </div>
        <span className="text-[11px] font-mono text-content-muted">
          Sorted by impact
        </span>
      </div>

      {sorted.length === 0 ? (
        <div className="p-6 rounded-xl bg-surface border border-border text-center text-xs text-content-muted">
          No critical ATS formatting or keyword deficiencies detected.
        </div>
      ) : (
        <div className="space-y-2.5">
          {sorted.map((issue, idx) => {
            const isExpanded = Boolean(expandedIndices[idx]);
            return (
              <div
                key={idx}
                className="rounded-xl bg-surface border border-border overflow-hidden transition-colors hover:border-border-strong"
              >
                <button
                  type="button"
                  onClick={() => toggleExpand(idx)}
                  className="w-full p-3.5 flex items-start justify-between text-left gap-3 focus:outline-none"
                >
                  <div className="flex items-start gap-2.5 min-w-0">
                    <span
                      className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${getSeverityDot(
                        issue.severity
                      )}`}
                    />
                    <div>
                      <p className="text-xs font-medium text-content-primary leading-snug">
                        {issue.message}
                      </p>
                      <span className="text-[11px] font-mono text-content-muted capitalize mt-0.5 block">
                        Category: {issue.component.replace("_", " ")}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-xs font-mono text-sand px-2 py-0.5 rounded-full bg-sand/10 border border-sand/20 font-medium">
                      +{issue.estimated_gain.toFixed(1)} pts
                    </span>
                    <ChevronDown
                      strokeWidth={1.5}
                      className={`w-4 h-4 text-content-muted transition-transform duration-200 ${
                        isExpanded ? "transform rotate-180" : ""
                      }`}
                    />
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-4 pb-3.5 pt-1 text-xs text-content-secondary border-t border-border/50 bg-surface-card/40 flex items-start gap-2">
                    <AlertCircle strokeWidth={1.5} className="w-3.5 h-3.5 text-sand shrink-0 mt-0.5" />
                    <p className="text-[12px] leading-relaxed">
                      Remediating this finding directly optimizes the{" "}
                      <span className="font-mono text-content-primary">{issue.component}</span>{" "}
                      rubric weight. Follow the suggested action plan to recover up to{" "}
                      <span className="font-mono text-sand">+{issue.estimated_gain.toFixed(1)} score points</span>.
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
