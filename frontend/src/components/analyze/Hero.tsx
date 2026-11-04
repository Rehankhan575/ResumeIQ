import React from "react";
import { Sparkles, ArrowRight } from "lucide-react";
import { Button } from "../ui/Button";

interface HeroProps {
  onAnalyzeSample: () => void;
  onExploreRubric?: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onAnalyzeSample, onExploreRubric }) => {
  return (
    <div className="flex flex-col justify-center py-4">
      {/* Category Pill */}
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface border border-border w-fit mb-4">
        <Sparkles strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
        <span className="text-xs font-mono text-content-secondary uppercase tracking-wider">
          Neural ATS Optimization
        </span>
      </div>

      {/* Main Heading with Instrument Serif Italic Accent */}
      <h1 className="text-4xl sm:text-5xl lg:text-[54px] font-semibold tracking-heading text-content-primary leading-[1.08] mb-5">
        See your resume the way an ATS{" "}
        <span className="font-serif italic font-normal text-content-primary">
          does
        </span>
        .
      </h1>

      {/* Subcopy */}
      <p className="text-base sm:text-lg text-content-secondary leading-relaxed max-w-xl mb-8">
        Evaluate your candidate profile against automated applicant tracking systems.
        Neural matching, deterministic rubrics, and bullet-level impact attribution—run
        entirely locally in-memory.
      </p>

      {/* CTAs */}
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" size="md" onClick={onAnalyzeSample}>
          <span>Evaluate Sample Profile</span>
          <ArrowRight strokeWidth={1.5} className="w-4 h-4 ml-1" />
        </Button>
        {onExploreRubric && (
          <Button variant="secondary" size="md" onClick={onExploreRubric}>
            Explore 9 Rubrics
          </Button>
        )}
      </div>
    </div>
  );
};

