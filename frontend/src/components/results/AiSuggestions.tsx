import React, { useState } from "react";
import { Suggestion } from "../../types/api";
import {
  ArrowRight,
  Sparkles,
  CheckCircle2,
  TrendingUp,
  Lightbulb,
  Copy,
  Check,
  Cpu,
  Info,
} from "lucide-react";

interface AiSuggestionsProps {
  suggestions: Suggestion[];
}

export const AiSuggestions: React.FC<AiSuggestionsProps> = ({ suggestions }) => {
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const handleCopy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  // Detect whether suggestions used LLM or rule-based fallback
  const isLlmActive = suggestions.some((s) => s.notes?.includes("llm"));

  // Helper to highlight "[add metric]" placeholder in suggested text
  const renderEnhancedText = (text: string) => {
    if (!text.includes("[add metric]")) {
      return text;
    }
    const parts = text.split("[add metric]");
    return (
      <>
        {parts[0]}
        <span className="inline-flex items-center px-2 py-0.5 mx-1 rounded text-xs font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm animate-pulse">
          [add metric, e.g. +25% or $50k]
        </span>
        {parts.slice(1).join("[add metric]")}
      </>
    );
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Top Banner & Mode Explainer */}
      <div className="p-4 rounded-xl bg-surface-card border border-border flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-glass">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-sand/10 border border-sand/20 flex items-center justify-center text-sand shrink-0">
            <Sparkles strokeWidth={1.5} className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-content-primary tracking-heading">
              Grounded Resume Rewrites & Recommendations
            </h4>
            <p className="text-xs text-content-secondary mt-0.5">
              Targeted improvements generated to maximize your ATS composite score and bullet impact.
            </p>
          </div>
        </div>

        {/* Engine Badge */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-surface-subtle border border-border text-content-secondary">
            <Cpu strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
            {isLlmActive ? "LLM Powered" : "Rule-Based Engine (Offline)"}
          </span>
        </div>
      </div>

      {/* LLM Status Notice */}
      {!isLlmActive && (
        <div className="px-4 py-3 rounded-lg bg-sand-dim/60 border border-sand/20 flex items-start gap-2.5 text-xs text-content-secondary leading-relaxed">
          <Info strokeWidth={1.5} className="w-4 h-4 text-sand shrink-0 mt-0.5" />
          <span>
            <strong className="text-sand font-medium">Running 100% locally and offline:</strong> Active rewrites use our deterministic linguistic engine (verb strengthening & metric scaffolding). To enable generative AI rewrites, add <code className="px-1.5 py-0.5 rounded bg-black/40 text-sand font-mono text-[11px]">GEMINI_API_KEY</code> or <code className="px-1.5 py-0.5 rounded bg-black/40 text-sand font-mono text-[11px]">GROQ_API_KEY</code> to your <code className="px-1.5 py-0.5 rounded bg-black/40 text-sand font-mono text-[11px]">.env</code> file.
          </span>
        </div>
      )}

      {/* Empty State */}
      {suggestions.length === 0 ? (
        <div className="py-12 px-6 rounded-xl bg-surface border border-border text-center space-y-3">
          <p className="text-sm font-medium text-content-primary">
            No suggestions generated for this document.
          </p>
          <p className="text-xs text-content-secondary max-w-md mx-auto">
            The candidate bullets satisfy standard ATS criteria, or no extractable bullet statements were detected.
          </p>
        </div>
      ) : (
        /* Suggestions List */
        <div className="space-y-6">
          {suggestions.map((s, idx) => {
            const isRewrite = s.kind === "bullet_rewrite" && Boolean(s.original);

            return (
              <div
                key={idx}
                className="p-5 sm:p-6 rounded-xl bg-surface border border-border/80 shadow-glass flex flex-col gap-4 transition-all hover:border-border-strong"
              >
                {/* Card Header */}
                <div className="flex items-center justify-between pb-3 border-b border-border/60">
                  <div className="flex items-center gap-2.5">
                    <span className="w-6 h-6 rounded-md bg-surface-subtle border border-border flex items-center justify-center text-xs font-mono font-semibold text-content-secondary">
                      {idx + 1}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {isRewrite ? (
                        <TrendingUp strokeWidth={1.5} className="w-4 h-4 text-sand" />
                      ) : (
                        <Lightbulb strokeWidth={1.5} className="w-4 h-4 text-sand" />
                      )}
                      <span className="text-xs font-mono uppercase tracking-wider text-sand font-semibold">
                        {isRewrite ? "Experience Bullet Rewrite" : "ATS Improvement Action"}
                      </span>
                    </div>
                  </div>

                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20">
                    <CheckCircle2 strokeWidth={1.5} className="w-3.5 h-3.5" />
                    Grounded & Verified
                  </span>
                </div>

                {/* Main Content Area */}
                {isRewrite ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
                    {/* Before: Original Bullet */}
                    <div className="p-4 rounded-xl bg-zinc-950/70 border border-rose-500/20 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider text-rose-400/90 font-semibold">
                            <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
                            Before (Current Draft)
                          </span>
                        </div>
                        <p className="text-sm sm:text-[15px] text-content-secondary leading-relaxed font-sans">
                          {s.original}
                        </p>
                      </div>
                      <span className="text-[11px] font-mono text-content-muted mt-3 pt-2 border-t border-border/40">
                        Needs active verb & measurable outcome
                      </span>
                    </div>

                    {/* After: Recommended Rewrite */}
                    <div className="p-4 rounded-xl bg-surface-card border-2 border-sand/40 shadow-sm flex flex-col justify-between relative group">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider text-emerald-400 font-semibold">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                            After (Optimized Power Statement)
                          </span>
                          <button
                            onClick={() => handleCopy(s.suggested, idx)}
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-mono bg-surface-subtle hover:bg-surface text-content-secondary hover:text-content-primary border border-border transition-colors"
                            title="Copy rewrite to clipboard"
                          >
                            {copiedIdx === idx ? (
                              <>
                                <Check strokeWidth={2} className="w-3 h-3 text-emerald-400" />
                                <span className="text-emerald-400">Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy strokeWidth={1.5} className="w-3 h-3" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        </div>
                        <p className="text-sm sm:text-[15px] text-content-primary font-medium leading-relaxed font-sans">
                          {renderEnhancedText(s.suggested)}
                        </p>
                      </div>

                      <div className="flex items-center gap-1 text-xs text-sand font-mono mt-3 pt-2 border-t border-border/40 font-medium">
                        <ArrowRight strokeWidth={1.5} className="w-3.5 h-3.5" />
                        <span>Ready to paste into your resume</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Improvement Plan Card */
                  <div className="p-4 sm:p-5 rounded-xl bg-surface-card border border-border flex flex-col gap-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-sand font-semibold">
                      Action Recommendation
                    </span>
                    <p className="text-sm sm:text-[15px] text-content-primary font-medium leading-relaxed">
                      {s.suggested}
                    </p>
                  </div>
                )}

                {/* Rationale & Diagnostics Footer */}
                {s.notes && (
                  <div className="px-3.5 py-2 rounded-lg bg-surface-subtle/80 border border-border/50 text-xs text-content-secondary flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <span className="font-semibold text-content-primary">Rationale:</span>
                      <span>{s.notes}</span>
                    </div>
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
