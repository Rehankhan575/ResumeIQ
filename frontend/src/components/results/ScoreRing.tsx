import React from "react";
import { Badge } from "../ui/Badge";
import { ArrowUpRight } from "lucide-react";

interface ScoreRingProps {
  score: number;
  grade: string;
  topFixMessage?: string;
  topFixGain?: number;
  jobMatchScore?: number;
  resumeQualityScore?: number;
}

export const ScoreRing: React.FC<ScoreRingProps> = ({
  score,
  grade,
  topFixMessage = "Add metrics, %, or revenue figures to bullets to demonstrate business impact.",
  topFixGain = 10.0,
  jobMatchScore = 95.9,
  resumeQualityScore = 71.4,
}) => {
  // Safe clamping for out-of-range, NaN, negative, or excessive values
  const clamped = Math.max(0, Math.min(100, isNaN(score) ? 0 : Number(score)));
  const radius = 54;
  const strokeWidth = 6;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (clamped / 100) * circumference;

  // Grade semantic style
  const getVerdict = (g: string) => {
    switch (g.toUpperCase()) {
      case "A":
      case "A-":
      case "A+":
        return "Excellent";
      case "B":
      case "B+":
      case "B-":
        return "Good";
      case "C":
      case "C+":
      case "C-":
        return "Fair";
      default:
        return "Needs work";
    }
  };

  const getGradeVariant = (g: string): "good" | "sand" | "warn" | "bad" => {
    const upper = g.toUpperCase();
    if (upper.startsWith("A") || upper.startsWith("B")) return "good";
    if (upper.startsWith("C")) return "warn";
    return "bad";
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Header Label */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
          ATS Evaluation
        </span>
        <Badge variant={getGradeVariant(grade)} size="sm">
          Grade {grade}
        </Badge>
      </div>

      {/* Center Circular Ring */}
      <div className="flex flex-col items-center justify-center py-2">
        <div className="relative w-36 h-36 flex items-center justify-center">
          <svg
            className="w-full h-full transform -rotate-90"
            viewBox="0 0 136 136"
            aria-label={`Score Ring: ${clamped.toFixed(0)} out of 100`}
          >
            {/* Background Dark Track */}
            <circle
              cx="68"
              cy="68"
              r={radius}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeWidth={strokeWidth}
              fill="transparent"
            />
            {/* Animated White Arc */}
            <circle
              cx="68"
              cy="68"
              r={radius}
              stroke="#F5F5F4"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              className="transition-all duration-700 ease-out"
            />
          </svg>

          {/* Centered Number in Geist Mono */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span
              data-testid="score-value"
              className="font-mono text-4xl font-bold tracking-tight text-content-primary"
            >
              {clamped.toFixed(0)}
            </span>
            <span className="text-xs font-mono text-content-muted mt-0.5">
              out of 100
            </span>
          </div>
        </div>

        {/* Verdict */}
        <span className="text-sm font-medium text-content-secondary mt-3">
          {getVerdict(grade)} Fit
        </span>
      </div>

      {/* Top Fix Callout */}
      <div className="p-3.5 rounded-xl bg-surface border border-border flex items-start gap-3">
        <div className="w-6 h-6 rounded-lg bg-sand/10 border border-sand/20 flex items-center justify-center text-sand shrink-0 mt-0.5">
          <ArrowUpRight strokeWidth={1.5} className="w-3.5 h-3.5" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-content-primary">Top fix</span>
            <span className="text-[11px] font-mono text-sand font-medium shrink-0">
              +{topFixGain.toFixed(1)} pts
            </span>
          </div>
          <p className="text-xs text-content-muted leading-relaxed mt-1 line-clamp-2">
            {topFixMessage}
          </p>
        </div>
      </div>

      {/* Sub-score Stat Tiles */}
      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="p-3 rounded-xl bg-surface border border-border flex flex-col">
          <span className="text-[11px] font-mono text-content-muted">Job Match</span>
          <span className="text-lg font-mono font-semibold text-content-primary mt-1">
            {jobMatchScore.toFixed(1)}%
          </span>
          <span className="text-[10px] text-content-muted mt-0.5">55% weight</span>
        </div>
        <div className="p-3 rounded-xl bg-surface border border-border flex flex-col">
          <span className="text-[11px] font-mono text-content-muted">Resume Quality</span>
          <span className="text-lg font-mono font-semibold text-content-primary mt-1">
            {resumeQualityScore.toFixed(1)}%
          </span>
          <span className="text-[10px] text-content-muted mt-0.5">45% weight</span>
        </div>
      </div>
    </div>
  );
};
