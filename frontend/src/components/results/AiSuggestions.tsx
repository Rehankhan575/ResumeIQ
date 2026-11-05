import React from "react";
import { Suggestion } from "../../types/api";
import { ArrowRight, Sparkles, CheckCircle2, TrendingUp, Lightbulb } from "lucide-react";

interface AiSuggestionsProps {
  suggestions: Suggestion[];
}

export const AiSuggestions: React.FC<AiSuggestionsProps> = ({ suggestions }) => {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between pb-3 border-b border-border/80">
        <div className="flex items-center gap-2">
          <Sparkles strokeWidth={1.5} className="w-4 h-4 text-sand" />
          <span className="text-xs font-mono uppercase tracking-wider text-content-primary font-semibold">
            Actionable Rewrites & Strategic Recommendations
          </span>
        </div>
        <span className="text-[11px] font-mono text-content-muted bg-surface-subtle px-2.5 py-0.5 rounded-full border border-border/50">
          Deterministic & Grounding-Guarded
        </span>
      </div>

      {suggestions.length === 0 ? (
        <div className="p-8 rounded-xl bg-surface border border-border text-center space-y-2">
          <p className="text-xs text-content-secondary font-medium">
            No generative rewrites available.
          </p>
          <p className="text-[11px] text-content-muted">
            The candidate bullets satisfy standard ATS criteria, or no extractable bullet statements were detected.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {suggestions.map((s, idx) => {
            const isRewrite = s.kind === "bullet_rewrite" && Boolean(s.original);

            return (
              <div
                key={idx}
                className="p-4 rounded-xl bg-surface border border-border/70 shadow-sm flex flex-col gap-3 transition-colors hover:border-border"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {isRewrite ? (
                      <TrendingUp strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
                    ) : (
                      <Lightbulb strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
                    )}
                    <span className="text-[11px] font-mono uppercase tracking-wider text-sand font-medium">
                      {isRewrite ? "Experience Bullet Rewrite" : "ATS Improvement Action"}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-emerald-400/90 bg-emerald-500/10 px-2 py-0.5 rounded flex items-center gap-1 border border-emerald-500/20">
                    <CheckCircle2 strokeWidth={1.5} className="w-3 h-3" />
                    Grounded Fix
                  </span>
                </div>

                {/* Before / After Diff for Bullet Rewrites */}
                {isRewrite ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {/* Original */}
                    <div className="p-3.5 rounded-lg bg-surface-subtle/80 border border-border/60">
                      <span className="text-[10px] font-mono text-content-muted uppercase block mb-1.5 font-semibold">
                        Original Statement
                      </span>
                      <p className="text-xs text-content-secondary line-through opacity-70 leading-relaxed">
                        {s.original}
                      </p>
                    </div>

                    {/* Suggested */}
                    <div className="p-3.5 rounded-lg bg-surface-card border border-sand/30 shadow-sm">
                      <span className="text-[10px] font-mono text-sand uppercase flex items-center gap-1 mb-1.5 font-semibold">
                        <span>Optimized Power Statement</span>
                        <ArrowRight strokeWidth={1.5} className="w-3 h-3 text-sand" />
                      </span>
                      <p className="text-xs font-medium text-content-primary leading-relaxed">
                        {s.suggested}
                      </p>
                    </div>
                  </div>
                ) : (
                  /* Improvement Plan Recommendation */
                  <div className="p-3.5 rounded-lg bg-surface-card border border-border/80">
                    <span className="text-[10px] font-mono text-sand uppercase block mb-1.5 font-semibold">
                      Recommendation
                    </span>
                    <p className="text-xs font-medium text-content-primary leading-relaxed">
                      {s.suggested}
                    </p>
                  </div>
                )}

                {/* Rationale & Diagnostics */}
                {s.notes && (
                  <div className="text-[11px] text-content-muted leading-relaxed border-t border-border/40 pt-2.5">
                    <span className="font-semibold text-content-secondary">Rationale:</span> {s.notes}
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
