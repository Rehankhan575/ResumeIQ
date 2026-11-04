import React from "react";
import { REAL_PROJECT_STATS } from "../../mocks/fixtureData";

export const StatStrip: React.FC = () => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {REAL_PROJECT_STATS.map((stat) => (
        <div
          key={stat.label}
          className="p-4 rounded-xl bg-surface border border-border flex flex-col gap-1"
        >
          <span className="text-[11px] font-mono text-content-muted uppercase tracking-wider">
            {stat.label}
          </span>
          <span className="text-lg font-mono font-semibold tracking-heading text-content-primary">
            {stat.value}
          </span>
          <span className="text-[11px] text-content-secondary">
            {stat.detail}
          </span>
        </div>
      ))}
    </div>
  );
};

