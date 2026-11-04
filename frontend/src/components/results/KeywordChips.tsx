import React from "react";
import { Check, AlertTriangle } from "lucide-react";

interface KeywordChipsProps {
  matchedSkills: string[];
  missingSkills: string[];
}

export const KeywordChips: React.FC<KeywordChipsProps> = ({
  matchedSkills = [],
  missingSkills = [],
}) => {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between pb-1 border-b border-border">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
          Skill Extraction & Match
        </span>
        <span className="text-[11px] font-mono text-content-muted">
          {matchedSkills.length} Matched / {missingSkills.length} Missing
        </span>
      </div>

      {/* Matched Skills */}
      <div className="space-y-2">
        <span className="text-[11px] font-mono uppercase tracking-wider text-content-muted flex items-center gap-1.5">
          <Check strokeWidth={2} className="w-3 h-3 text-semantic-good" />
          <span>Matched In Document ({matchedSkills.length})</span>
        </span>
        {matchedSkills.length === 0 ? (
          <p className="text-xs text-content-muted italic">No matched skills detected.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {matchedSkills.map((skill) => (
              <span
                key={skill}
                className="px-2.5 py-1 rounded-full text-xs font-mono bg-surface border border-border text-content-primary hover:border-border-strong transition-colors"
              >
                {skill}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Missing Skills */}
      <div className="space-y-2 pt-2">
        <span className="text-[11px] font-mono uppercase tracking-wider text-content-muted flex items-center gap-1.5">
          <AlertTriangle strokeWidth={1.5} className="w-3 h-3 text-semantic-warn" />
          <span>Missing From Resume ({missingSkills.length})</span>
        </span>
        {missingSkills.length === 0 ? (
          <p className="text-xs text-content-secondary italic">
            Zero required skills missing. Full coverage across JD requirements.
          </p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {missingSkills.map((skill) => (
              <span
                key={skill}
                className="px-2.5 py-1 rounded-full text-xs font-mono bg-semantic-warn-dim text-semantic-warn border border-semantic-warn/20"
              >
                {skill}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

