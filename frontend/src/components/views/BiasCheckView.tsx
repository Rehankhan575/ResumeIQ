import React from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { ShieldAlert, ArrowRight } from "lucide-react";

export const BiasCheckView: React.FC = () => {
  const flags = [
    {
      category: "Ageist Proxy Term",
      phrase: "Digital Native",
      context: "Digital native with extensive cloud-native architecture experience",
      recommendation: "Tech-savvy or Cloud-focused",
      explanation: "Terms like 'digital native' can be construed as ageist screening proxies under EEOC guidelines.",
      severity: "warn" as const,
    },
    {
      category: "Gender-Coded Phrasing",
      phrase: "Ninja / Rock star developer",
      context: "Recognized as a rock star developer across cross-functional sprints",
      recommendation: "High-performing engineer or Accomplished software developer",
      explanation: "Heavily masculine-coded jargon reduces inclusivity and can trigger automated ATS screening penalties.",
      severity: "warn" as const,
    },
    {
      category: "Demographic Indicator",
      phrase: "Graduation year: 2008",
      context: "B.S. in Computer Science, 2008",
      recommendation: "Omit graduation year (list degree and institution only)",
      explanation: "Listing graduation years older than 10-15 years introduces implicit age bias risks during initial recruitment screenings.",
      severity: "neutral" as const,
    },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Fairness, Neutrality & Bias Audit
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Automated scan for demographic proxies, gendered terminology, and bias-prone keywords that harm recruitment fairness.
        </p>
      </div>

      {/* Summary Banner */}
      <div className="p-4 rounded-xl bg-surface border border-border flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-sand/10 border border-sand/20 flex items-center justify-center text-sand">
            <ShieldAlert strokeWidth={1.5} className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-content-primary">
              Neutrality Audit Completed
            </h4>
            <p className="text-[11px] text-content-muted">
              {flags.length} potential phrasing improvements flagged. Zero candidate identity text stored.
            </p>
          </div>
        </div>
        <Badge variant="sand" size="sm">
          Audit Score: 94/100
        </Badge>
      </div>

      {/* Flagged Phrase Cards */}
      <div className="space-y-3">
        {flags.map((flag, idx) => (
          <Card key={idx} surface="card" className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-medium text-sand px-2 py-0.5 rounded bg-sand/10 border border-sand/20">
                  {flag.category}
                </span>
                <span className="text-xs font-medium text-content-primary line-through opacity-75">
                  "{flag.phrase}"
                </span>
              </div>
              <Badge variant={flag.severity === "warn" ? "warn" : "neutral"} size="sm">
                {flag.severity === "warn" ? "Action Recommended" : "Optional Polish"}
              </Badge>
            </div>

            <div className="p-3 rounded-lg bg-surface border border-border/70 text-xs flex flex-col gap-2">
              <div className="text-[11px] text-content-muted">
                <span className="font-mono text-content-secondary">Detected In:</span> "{flag.context}"
              </div>
              <div className="flex items-center gap-2 font-medium text-content-primary">
                <ArrowRight strokeWidth={1.5} className="w-3.5 h-3.5 text-sand shrink-0" />
                <span className="text-sand font-mono text-[11px]">Recommended Replacement:</span>
                <span className="text-xs">{flag.recommendation}</span>
              </div>
            </div>

            <p className="text-[11px] text-content-muted leading-relaxed">
              <span className="font-semibold text-content-secondary">Guidance:</span> {flag.explanation}
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
};
