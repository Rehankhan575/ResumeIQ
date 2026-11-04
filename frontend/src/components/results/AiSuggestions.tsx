import React from "react";
import { Suggestion } from "../../types/api";
import { ArrowRight, Sparkles } from "lucide-react";

interface AiSuggestionsProps {
  suggestions: Suggestion[];
}

export const AiSuggestions: React.FC<AiSuggestionsProps> = ({ suggestions }) => {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between pb-1 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
          <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
            Grounding-Guarded Suggestions
          </span>
        </div>
        <span className="text-[11px] font-mono text-content-muted">
          Rule-Based & GenAI Fixes
        </span>
      </div>

      {suggestions.length === 0 ? (
        <div className="p-6 rounded-xl bg-surface border border-border text-center text-xs text-content-muted">
          No generative rewrites available. Enable AI suggestions in pipeline settings or review rubric action items.
        </div>
      ) : (
        <div className="space-y-4">
          {suggestions.map((s, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-surface border border-border flex flex-col gap-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono uppercase tracking-wider text-sand">
                  {s.kind.replace("_", " ")}
                </span>
                <span className="text-[11px] font-mono text-content-muted">
                  Grounding Verified
                </span>
              </div>

              {/* Before / After Diff */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Original */}
                <div className="p-3 rounded-lg bg-surface-subtle/80 border border-border/60">
                  <span className="text-[10px] font-mono text-content-muted uppercase block mb-1">
                    Original Bullet
                  </span>
                  <p className="text-xs text-content-secondary line-through opacity-75">
                    {s.original}
                  </p>
                </div>

                {/* Suggested */}
                <div className="p-3 rounded-lg bg-surface-card border border-sand/30 shadow-sm">
                  <span className="text-[10px] font-mono text-sand uppercase flex items-center gap-1 mb-1 font-semibold">
                    <span>Optimized Bullet</span>
                    <ArrowRight strokeWidth={1.5} className="w-3 h-3 text-sand" />
                  </span>
                  <p className="text-xs font-medium text-content-primary">
                    {s.suggested}
                  </p>
                </div>
              </div>

              {/* Notes */}
              {s.notes && (
                <p className="text-[11px] text-content-muted leading-relaxed border-t border-border/40 pt-2">
                  <span className="font-semibold text-content-secondary">Rationale:</span> {s.notes}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
