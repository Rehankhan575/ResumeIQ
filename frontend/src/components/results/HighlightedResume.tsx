import React, { useState } from "react";
import { Highlight } from "../../types/api";
import { Info } from "lucide-react";

interface HighlightedResumeProps {
  rawText?: string;
  highlights: Highlight[];
}

// Fallback sample resume text snippet for fixture display
const DEFAULT_RESUME_SNIPPET = `SENIOR DATA SCIENTIST & ML ENGINEER
Expert in predictive modeling, distributed Machine Learning pipelines, and scalable statistical architectures.
Core competencies include Python, Scikit-Learn, SQL, model deployment, and high-throughput data processing.
Led predictive modeling and cross-functional ML workflows across enterprise cloud systems, delivering high reliability.
Master of Science in Computer Science, specialized in Applied Intelligence and Neural Network Architectures.`;

export const HighlightedResume: React.FC<HighlightedResumeProps> = ({
  rawText = DEFAULT_RESUME_SNIPPET,
  highlights = [],
}) => {
  const [activeTooltip, setActiveTooltip] = useState<{
    text: string;
    weight: number;
    type: string;
    x: number;
    y: number;
  } | null>(null);

  // Parse text into React text nodes and highlighted span nodes safely without innerHTML
  const renderSafeSpans = () => {
    if (!highlights || highlights.length === 0) {
      return <span>{rawText}</span>;
    }

    // Sort highlights by start offset and filter out overlaps
    const sorted = [...highlights]
      .filter((h) => h.start >= 0 && h.end <= rawText.length && h.start < h.end)
      .sort((a, b) => a.start - b.start);

    const elements: React.ReactNode[] = [];
    let cursor = 0;

    sorted.forEach((h, idx) => {
      // If there's an overlap with previous highlight, skip
      if (h.start < cursor) return;

      // Plain text before highlight (React automatically safely escapes strings)
      if (h.start > cursor) {
        elements.push(
          <span key={`plain-${idx}`}>{rawText.slice(cursor, h.start)}</span>
        );
      }

      // Highlighted span
      const highlightText = rawText.slice(h.start, h.end);
      const isKeyword = h.type === "keyword_match";
      const isNegative = h.type === "negative";

      const underlineClass = isNegative
        ? "underline decoration-semantic-bad decoration-2 underline-offset-4 bg-semantic-bad-dim text-content-primary rounded-sm px-1 py-0.5"
        : isKeyword
        ? "underline decoration-sand decoration-2 underline-offset-4 bg-sand-dim text-content-primary rounded-sm px-1 py-0.5 font-medium"
        : "underline decoration-semantic-good decoration-2 underline-offset-4 bg-semantic-good-dim text-content-primary rounded-sm px-1 py-0.5";

      elements.push(
        <span
          key={`hl-${idx}`}
          className={`cursor-help transition-colors ${underlineClass}`}
          onMouseEnter={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            setActiveTooltip({
              text: h.text || highlightText,
              weight: h.weight,
              type: h.type,
              x: rect.left + rect.width / 2,
              y: rect.top - 8,
            });
          }}
          onMouseLeave={() => setActiveTooltip(null)}
        >
          {highlightText}
        </span>
      );

      cursor = h.end;
    });

    // Remaining trailing plain text
    if (cursor < rawText.length) {
      elements.push(
        <span key="plain-end">{rawText.slice(cursor)}</span>
      );
    }

    return elements;
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Legend & Title Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
            Explainability Overlay
          </span>
          <span className="text-[11px] font-mono text-content-muted">
            Reader-Style Token Inspection
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-1 bg-sand rounded-full" />
            <span className="text-content-secondary">Keyword Match</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-1 bg-semantic-good rounded-full" />
            <span className="text-content-secondary">High Weight</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-1 bg-semantic-bad rounded-full" />
            <span className="text-content-secondary">Penalty / Missing</span>
          </div>
        </div>
      </div>

      {/* Reader Panel */}
      <div className="p-6 rounded-xl bg-surface border border-border leading-relaxed text-xs sm:text-sm font-mono text-content-secondary whitespace-pre-wrap selection:bg-sand/30 relative">
        {renderSafeSpans()}
      </div>

      {/* Floating Tooltip */}
      {activeTooltip && (
        <div
          className="fixed z-50 transform -translate-x-1/2 -translate-y-full px-2.5 py-1.5 rounded-lg bg-surface-card border border-border shadow-xl text-xs font-mono pointer-events-none"
          style={{ left: `${activeTooltip.x}px`, top: `${activeTooltip.y}px` }}
        >
          <div className="flex items-center gap-2">
            <span className="text-content-primary font-medium">{activeTooltip.text}</span>
            <span className="text-[11px] text-sand font-mono">
              w: {activeTooltip.weight.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Security Guarantee Note */}
      <div className="flex items-center gap-2 text-[11px] text-content-muted">
        <Info strokeWidth={1.5} className="w-3.5 h-3.5 text-content-muted shrink-0" />
        <span>
          Rendered through typed React virtual DOM nodes with strict zero-injection sanitation.
        </span>
      </div>
    </div>
  );
};

