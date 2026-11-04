import React from "react";

export const WorkflowSteps: React.FC = () => {
  const steps = [
    {
      step: "01",
      title: "Choose",
      desc: "Upload a candidate resume and paste a target job description, or load an authentic fixture.",
    },
    {
      step: "02",
      title: "Analyze",
      desc: "Siamese bi-encoders and 9 rubric algorithms calculate match score, keyword gaps, and impact.",
    },
    {
      step: "03",
      title: "Improve",
      desc: "Review prioritized bullet fixes and recommended edits.",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {steps.map((s) => (
        <div
          key={s.step}
          className="p-4 rounded-xl bg-surface border border-border flex flex-col gap-1.5 transition-colors hover:border-border-strong"
        >
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono font-medium text-sand px-1.5 py-0.5 rounded bg-sand/10 border border-sand/20">
              {s.step}
            </span>
            <span className="text-xs font-semibold tracking-heading text-content-primary">
              {s.title}
            </span>
          </div>
          <p className="text-[12px] text-content-muted leading-relaxed">
            {s.desc}
          </p>
        </div>
      ))}
    </div>
  );
};

