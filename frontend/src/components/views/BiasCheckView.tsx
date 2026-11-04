import React from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { ShieldCheck, ShieldAlert, ArrowRight } from "lucide-react";
import { BiasFinding } from "../../types/api";

interface BiasCheckViewProps {
  findings?: BiasFinding[];
}

export const BiasCheckView: React.FC<BiasCheckViewProps> = ({
  findings = [],
}) => {
  const isClean = findings.length === 0;

  const formatCategory = (cat: string) => {
    switch (cat.toLowerCase()) {
      case "personal_info":
        return "Demographic Disclosure";
      case "age_bias":
        return "Age Bias Indicator";
      case "gender_bias":
        return "Gendered Terminology";
      default:
        return cat.replace("_", " ").toUpperCase();
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Fairness, Neutrality & Bias Audit
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Automated scan for demographic proxies, gendered terminology, and bias-prone indicators that harm recruitment neutrality.
        </p>
      </div>

      {/* Summary Banner */}
      <div className="p-4 rounded-xl bg-surface border border-border flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={`w-9 h-9 rounded-full flex items-center justify-center ${
              isClean
                ? "bg-semantic-good-dim border border-semantic-good/20 text-semantic-good"
                : "bg-semantic-warn-dim border border-semantic-warn/20 text-semantic-warn"
            }`}
          >
            {isClean ? (
              <ShieldCheck strokeWidth={1.5} className="w-5 h-5" />
            ) : (
              <ShieldAlert strokeWidth={1.5} className="w-5 h-5" />
            )}
          </div>
          <div>
            <h4 className="text-xs font-semibold text-content-primary">
              {isClean ? "Profile Neutrality Verified" : "Phrasing Improvements Identified"}
            </h4>
            <p className="text-[11px] text-content-muted">
              {isClean
                ? "Zero demographic proxies or bias indicators detected in parsed resume sections."
                : `${findings.length} phrasing indicator${findings.length > 1 ? "s" : ""} flagged. Zero candidate text persisted.`}
            </p>
          </div>
        </div>

        <Badge variant={isClean ? "good" : "warn"} size="sm">
          {isClean ? "Clean Profile" : `${findings.length} Finding${findings.length > 1 ? "s" : ""}`}
        </Badge>
      </div>

      {/* Flagged or Clean State */}
      {isClean ? (
        <Card glass={true} className="py-12 text-center space-y-2">
          <div className="w-12 h-12 rounded-full bg-semantic-good-dim border border-semantic-good/20 flex items-center justify-center text-semantic-good mx-auto mb-3">
            <ShieldCheck strokeWidth={1.5} className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-content-primary">
            No Demographic or Bias Risks Detected
          </h3>
          <p className="text-xs text-content-muted max-w-md mx-auto leading-relaxed">
            The analyzed profile avoids non-job-relevant personal information, graduation dates older than 15 years, and gendered phrasing.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {findings.map((flag, idx) => (
            <Card key={idx} surface="card" className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-medium text-sand px-2 py-0.5 rounded bg-sand/10 border border-sand/20">
                  {formatCategory(flag.type)}
                </span>
                <Badge variant="warn" size="sm">
                  Action Recommended
                </Badge>
              </div>

              <div className="p-3 rounded-lg bg-surface border border-border/70 text-xs flex flex-col gap-1.5">
                <span className="font-mono text-content-secondary font-medium">
                  {flag.evidence_label}
                </span>
                <div className="flex items-start gap-2 text-content-primary pt-1">
                  <ArrowRight strokeWidth={1.5} className="w-3.5 h-3.5 text-sand shrink-0 mt-0.5" />
                  <span className="text-xs text-content-secondary leading-relaxed">
                    {flag.suggestion}
                  </span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
