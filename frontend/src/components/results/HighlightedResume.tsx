import React, { useState } from "react";
import { Highlight } from "../../types/api";
import { Info } from "lucide-react";

interface HighlightedResumeProps {
  rawText?: string;
  highlights: Highlight[];
}

// Fallback sample resume text snippet for fixture display
export const DEFAULT_RESUME_SNIPPET = `SENIOR DATA SCIENTIST & ML ENGINEER
Expert in predictive modeling, distributed Machine Learning pipelines, and scalable statistical architectures.
Core competencies include Python, Scikit-Learn, SQL, model deployment, and high-throughput data processing.
Led predictive modeling and cross-functional ML workflows across enterprise cloud systems, delivering high reliability.
Master of Science in Computer Science, specialized in Applied Intelligence and Neural Network Architectures.`;

/**
 * Snap arbitrary character offsets (from subword tokens or misaligned spans)
 * to clean whole-word boundaries so words are never cut mid-token.
 */
function snapToWordBoundaries(text: string, start: number, end: number): [number, number] {
  let s = Math.max(0, Math.min(start, text.length));
  let e = Math.max(s, Math.min(end, text.length));
  if (s === e) return [s, e];

  const isWordChar = (char: string) => /[\w]/.test(char);

  // Expand start backwards if inside a word
  while (s > 0 && isWordChar(text[s - 1]) && isWordChar(text[s])) {
    s--;
  }

  // Expand end forwards if inside a word
  while (e < text.length && isWordChar(text[e - 1]) && isWordChar(text[e])) {
    e++;
  }

  return [s, e];
}

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

  const textToRender = rawText && rawText.trim() ? rawText : DEFAULT_RESUME_SNIPPET;

  // Parse text into React text nodes and highlighted span nodes safely without innerHTML
  const renderSafeSpans = () => {
    if (!highlights || highlights.length === 0) {
      return <span>{textToRender}</span>;
    }

    // Resolve and align spans to whole-word boundaries in textToRender
    const preparedSpans: Array<{
      start: number;
      end: number;
      weight: number;
      type: string;
      sourceText: string;
    }> = [];

    highlights.forEach((h) => {
      if (h.type === "keyword_missing") return;

      let s = h.start;
      let e = h.end;

      // If the slice does not match the intended highlight text, attempt whole-word lookup
      if (h.text && h.text.trim()) {
        const slice = textToRender.slice(s, e);
        if (slice.toLowerCase() !== h.text.toLowerCase()) {
          const escaped = h.text.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          const wordRegex = new RegExp(`\\b${escaped}\\b`, "i");
          const match = wordRegex.exec(textToRender);
          if (match && match.index !== undefined) {
            s = match.index;
            e = match.index + match[0].length;
          }
        }
      }

      // Snap to whole-word boundaries
      const [snappedStart, snappedEnd] = snapToWordBoundaries(textToRender, s, e);
      if (snappedStart < snappedEnd && snappedStart >= 0 && snappedEnd <= textToRender.length) {
        preparedSpans.push({
          start: snappedStart,
          end: snappedEnd,
          weight: h.weight,
          type: h.type,
          sourceText: h.text || textToRender.slice(snappedStart, snappedEnd),
        });
      }
    });

    // Sort by start offset and filter out overlaps
    const sorted = preparedSpans.sort((a, b) => a.start - b.start || b.end - a.end);

    const elements: React.ReactNode[] = [];
    let cursor = 0;

    sorted.forEach((span, idx) => {
      // If there's an overlap with previous highlight, skip
      if (span.start < cursor) return;

      // Plain text before highlight
      if (span.start > cursor) {
        elements.push(
          <span key={`plain-${idx}`}>{textToRender.slice(cursor, span.start)}</span>
        );
      }

      // Highlighted whole-word span
      const highlightText = textToRender.slice(span.start, span.end);
      const isKeyword = span.type === "keyword_match";
      const isNegative = span.type === "negative";

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
              text: span.sourceText || highlightText,
              weight: span.weight,
              type: span.type,
              x: rect.left + rect.width / 2,
              y: rect.top - 8,
            });
          }}
          onMouseLeave={() => setActiveTooltip(null)}
        >
          {highlightText}
        </span>
      );

      cursor = span.end;
    });

    // Remaining trailing plain text
    if (cursor < textToRender.length) {
      elements.push(
        <span key="plain-end">{textToRender.slice(cursor)}</span>
      );
    }

    return elements;
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Legend & Title Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-2 border-b border-border/80 gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono uppercase tracking-wider text-content-primary font-semibold">
            Explainability Overlay
          </span>
          <span className="text-[11px] text-content-muted font-mono">Reader-Style Token Inspection</span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1.5 text-sand">
            <span className="w-2.5 h-0.5 bg-sand rounded-full"></span>
            Keyword Match
          </span>
          <span className="flex items-center gap-1.5 text-semantic-good">
            <span className="w-2.5 h-0.5 bg-semantic-good rounded-full"></span>
            High Weight
          </span>
          <span className="flex items-center gap-1.5 text-semantic-bad">
            <span className="w-2.5 h-0.5 bg-semantic-bad rounded-full"></span>
            Penalty / Missing
          </span>
        </div>
      </div>

      {/* Reader Container */}
      <div className="p-5 rounded-xl bg-canvas border border-border/60 text-xs sm:text-[13px] leading-relaxed font-mono whitespace-pre-wrap break-words max-h-96 overflow-y-auto selection:bg-sand/20">
        {renderSafeSpans()}
      </div>

      {/* Security & Token Guarantee */}
      <div className="flex items-center gap-1.5 text-[11px] text-content-muted">
        <Info strokeWidth={1.5} className="w-3.5 h-3.5 text-content-muted" />
        <span>Rendered through whole-word snapped tokens with strict zero-injection sanitization.</span>
      </div>

      {/* Floating Tooltip */}
      {activeTooltip && (
        <div
          className="fixed z-50 transform -translate-x-1/2 -translate-y-full px-2.5 py-1.5 rounded-lg bg-surface border border-border text-[11px] font-mono shadow-glass pointer-events-none flex flex-col gap-0.5"
          style={{ left: `${activeTooltip.x}px`, top: `${activeTooltip.y}px` }}
        >
          <div className="flex items-center gap-2">
            <span className="text-sand font-semibold">{activeTooltip.text}</span>
            <span
              className={
                activeTooltip.weight >= 0 ? "text-semantic-good" : "text-semantic-bad"
              }
            >
              {activeTooltip.weight >= 0 ? "+" : ""}
              {activeTooltip.weight.toFixed(4)}
            </span>
          </div>
          <span className="text-[10px] text-content-muted uppercase">
            {activeTooltip.type.replace("_", " ")}
          </span>
        </div>
      )}
    </div>
  );
};
