import React from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { BookOpen, Check, AlertTriangle, Clock } from "lucide-react";

export const SkillGapView: React.FC = () => {
  const matched = [
    { name: "Python", category: "Programming", level: "Expert" },
    { name: "SQL", category: "Database", level: "Proficient" },
    { name: "Scikit-Learn", category: "ML Framework", level: "Advanced" },
    { name: "Machine Learning", category: "Domain", level: "Advanced" },
  ];

  const missing = [
    { name: "Docker", category: "DevOps", importance: "Recommended" },
    { name: "Kubernetes", category: "Cloud Orchestration", importance: "Preferred" },
    { name: "MLflow", category: "MLOps", importance: "Preferred" },
  ];

  const learningPaths = [
    {
      skill: "Docker",
      title: "Docker for Data Science & ML Engineering",
      provider: "Internal Curriculum / O'Reilly",
      duration: "6 hours",
      level: "Intermediate",
      description: "Containerize Python & Scikit-Learn training pipelines and deploy deterministic REST inference services.",
    },
    {
      skill: "MLflow",
      title: "Production ML Lifecycle with MLflow & DVC",
      provider: "Open Source Courseware",
      duration: "8 hours",
      level: "Advanced",
      description: "Experiment tracking, model registry governance, artifact versioning, and continuous model evaluation.",
    },
    {
      skill: "Kubernetes",
      title: "Kubernetes Fundamentals for ML Workloads",
      provider: "Cloud Native Computing Foundation",
      duration: "10 hours",
      level: "Intermediate",
      description: "Deploy scalable distributed inference pods with GPU resource management and autoscaling policies.",
    },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Skill Gap Analysis & Learning Paths
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Detailed taxonomy cross-reference between your parsed experience tokens and industry skill benchmarks.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Matched Competencies */}
        <Card surface="card" className="space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-border">
            <div className="flex items-center gap-2">
              <Check strokeWidth={2} className="w-4 h-4 text-semantic-good" />
              <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                Matched Competencies ({matched.length})
              </span>
            </div>
            <Badge variant="good" size="sm">
              Verified
            </Badge>
          </div>

          <div className="space-y-2">
            {matched.map((m) => (
              <div
                key={m.name}
                className="p-3 rounded-xl bg-surface border border-border flex items-center justify-between text-xs"
              >
                <div>
                  <span className="font-medium text-content-primary">{m.name}</span>
                  <span className="text-[11px] font-mono text-content-muted block">
                    {m.category}
                  </span>
                </div>
                <span className="font-mono text-content-secondary px-2 py-0.5 rounded-full bg-surface-card border border-border">
                  {m.level}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Missing / Target Opportunities */}
        <Card surface="card" className="space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-border">
            <div className="flex items-center gap-2">
              <AlertTriangle strokeWidth={1.5} className="w-4 h-4 text-semantic-warn" />
              <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                High-Impact Additions ({missing.length})
              </span>
            </div>
            <Badge variant="warn" size="sm">
              Gap Identified
            </Badge>
          </div>

          <div className="space-y-2">
            {missing.map((m) => (
              <div
                key={m.name}
                className="p-3 rounded-xl bg-surface border border-border flex items-center justify-between text-xs"
              >
                <div>
                  <span className="font-medium text-content-primary">{m.name}</span>
                  <span className="text-[11px] font-mono text-content-muted block">
                    {m.category}
                  </span>
                </div>
                <span className="font-mono text-semantic-warn px-2 py-0.5 rounded-full bg-semantic-warn-dim border border-semantic-warn/20">
                  {m.importance}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Curated URL-Free Learning Paths */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-border">
          <div className="flex items-center gap-2">
            <BookOpen strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
              Recommended Learning Curriculum
            </span>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            Curated Courseware Catalog
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {learningPaths.map((lp) => (
            <div
              key={lp.skill}
              className="p-4 rounded-xl bg-surface border border-border flex flex-col justify-between gap-3 transition-colors hover:border-border-strong"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-sand px-2 py-0.5 rounded bg-sand/10 border border-sand/20">
                    {lp.skill}
                  </span>
                  <span className="text-[11px] font-mono text-content-muted flex items-center gap-1">
                    <Clock strokeWidth={1.5} className="w-3 h-3" />
                    {lp.duration}
                  </span>
                </div>
                <h4 className="text-xs font-semibold text-content-primary leading-snug">
                  {lp.title}
                </h4>
                <p className="text-[11px] text-content-muted leading-relaxed">
                  {lp.description}
                </p>
              </div>

              <div className="pt-2 border-t border-border/60 text-[10px] font-mono text-content-secondary flex items-center justify-between">
                <span>{lp.provider}</span>
                <span>{lp.level}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
