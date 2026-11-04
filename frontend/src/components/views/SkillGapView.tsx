import React from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { BookOpen, Check, AlertTriangle, Clock } from "lucide-react";
import { CourseRecommendation } from "../../types/api";

interface SkillGapViewProps {
  matchedSkills?: string[];
  missingSkills?: string[];
  recommendations?: Record<string, CourseRecommendation[]>;
}

export const SkillGapView: React.FC<SkillGapViewProps> = ({
  matchedSkills = ["machine learning", "python", "scikit-learn", "sql"],
  missingSkills = [],
  recommendations = {},
}) => {
  // Collect all course recommendations for missing skills
  const allCourses: { skill: string; course: CourseRecommendation }[] = [];
  Object.entries(recommendations).forEach(([skill, courses]) => {
    courses.forEach((c) => allCourses.push({ skill, course: c }));
  });

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Skill Gap Analysis & Recommendations
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Exact taxonomy cross-reference between your parsed candidate skills and target job requirements.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Matched Competencies */}
        <Card surface="card" className="space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-border">
            <div className="flex items-center gap-2">
              <Check strokeWidth={2} className="w-4 h-4 text-semantic-good" />
              <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                Matched Requirements ({matchedSkills.length})
              </span>
            </div>
            <Badge variant="good" size="sm">
              Verified
            </Badge>
          </div>

          {matchedSkills.length === 0 ? (
            <p className="text-xs text-content-muted py-4 text-center">
              No overlapping requirements detected.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {matchedSkills.map((m) => (
                <div
                  key={m}
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border flex items-center gap-2 text-xs font-mono"
                >
                  <Check strokeWidth={2} className="w-3.5 h-3.5 text-semantic-good" />
                  <span className="text-content-primary font-medium">{m}</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Missing Required Skills */}
        <Card surface="card" className="space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-border">
            <div className="flex items-center gap-2">
              <AlertTriangle strokeWidth={1.5} className="w-4 h-4 text-semantic-warn" />
              <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                Missing Requirements ({missingSkills.length})
              </span>
            </div>
            <Badge variant={missingSkills.length === 0 ? "good" : "warn"} size="sm">
              {missingSkills.length === 0 ? "Complete Coverage" : "Gaps Identified"}
            </Badge>
          </div>

          {missingSkills.length === 0 ? (
            <div className="py-6 text-center text-xs text-content-muted space-y-1">
              <p className="text-semantic-good font-medium">100% Keyword Coverage</p>
              <p className="text-[11px]">All required target skills are verified in your profile.</p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {missingSkills.map((m) => (
                <div
                  key={m}
                  className="px-3 py-1.5 rounded-lg bg-semantic-warn-dim/50 border border-semantic-warn/30 flex items-center gap-2 text-xs font-mono"
                >
                  <AlertTriangle strokeWidth={1.5} className="w-3.5 h-3.5 text-semantic-warn" />
                  <span className="text-content-primary font-medium">{m}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Curated URL-Free Learning Paths */}
      {allCourses.length > 0 && (
        <Card glass={true} className="space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-border">
            <div className="flex items-center gap-2">
              <BookOpen strokeWidth={1.5} className="w-4 h-4 text-sand" />
              <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                Curated Learning Curriculum ({allCourses.length})
              </span>
            </div>
            <span className="text-[11px] font-mono text-content-muted">
              URL-Free Educational Directory
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {allCourses.map(({ skill, course }, idx) => (
              <div
                key={idx}
                className="p-4 rounded-xl bg-surface border border-border flex flex-col justify-between gap-3 transition-colors hover:border-border-strong"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold text-sand px-2 py-0.5 rounded bg-sand/10 border border-sand/20">
                      {skill}
                    </span>
                    <span className="text-[11px] font-mono text-content-muted flex items-center gap-1">
                      <Clock strokeWidth={1.5} className="w-3 h-3" />
                      {course.estimated_hours}h
                    </span>
                  </div>
                  <h4 className="text-xs font-semibold text-content-primary leading-snug">
                    {course.title}
                  </h4>
                </div>

                <div className="pt-2 border-t border-border/60 text-[10px] font-mono text-content-secondary flex items-center justify-between">
                  <span>{course.provider}</span>
                  <span className="px-1.5 py-0.5 rounded bg-surface-card border border-border">
                    {course.level}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
