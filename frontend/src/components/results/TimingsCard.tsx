import React, { useState } from "react";
import { ChevronDown, Clock } from "lucide-react";

interface TimingsCardProps {
  timings: Record<string, number>;
}

export const TimingsCard: React.FC<TimingsCardProps> = ({ timings }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const totalMs = Object.values(timings).reduce((acc, v) => acc + (isNaN(v) ? 0 : v), 0);

  return (
    <div className="rounded-xl bg-surface border border-border overflow-hidden">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3.5 flex items-center justify-between text-left focus:outline-none hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-2">
          <Clock strokeWidth={1.5} className="w-4 h-4 text-sand" />
          <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
            Execution Latency
          </span>
          <span className="text-[11px] font-mono text-content-muted">
            ({totalMs.toFixed(0)} ms total)
          </span>
        </div>
        <ChevronDown
          strokeWidth={1.5}
          className={`w-4 h-4 text-content-muted transition-transform duration-200 ${
            isExpanded ? "transform rotate-180" : ""
          }`}
        />
      </button>

      {isExpanded && (
        <div className="p-4 pt-1 border-t border-border/60 bg-surface-card/40 space-y-2">
          {Object.entries(timings).map(([stage, ms]) => (
            <div key={stage} className="flex items-center justify-between text-xs font-mono">
              <span className="text-content-muted capitalize">
                {stage.replace("_ms", "").replace("_", " ")}
              </span>
              <span className="text-content-primary">{ms.toFixed(1)} ms</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
