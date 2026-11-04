import React from "react";
import { ChevronRight, User } from "lucide-react";
import { NavItemKey } from "./Sidebar";
import { Badge } from "../ui/Badge";
import { isMockMode } from "../../services/api";

interface TopBarProps {
  currentTab: NavItemKey;
}

const TAB_TITLES: Record<NavItemKey, { title: string; subtitle: string }> = {
  analyze: { title: "Analyze Resume", subtitle: "Diagnostic Ingestion & ATS Matcher" },
  results: { title: "Results & Breakdown", subtitle: "Composite ATS Scorecard & Neural Explanations" },
  models: { title: "Model Results", subtitle: "Empirical Validation & Benchmark Reports" },
  rank: { title: "Multi-Job Ranking", subtitle: "Cross-Job Description Fit Matrix" },
  skills: { title: "Skill Gap Analysis", subtitle: "Matched vs Missing Competency Mapping" },
  bias: { title: "Fairness & Bias Audit", subtitle: "Demographic & Proxy Neutrality Checks" },
  history: { title: "Score History", subtitle: "Local Opt-in Evaluation Records" },
  privacy: { title: "Privacy Architecture", subtitle: "In-Memory Candidate Guarantees" },
};

export const TopBar: React.FC<TopBarProps> = ({ currentTab }) => {
  const current = TAB_TITLES[currentTab] || { title: "ResumeIQ", subtitle: "ATS Platform" };
  const mockActive = isMockMode();

  return (
    <header className="h-16 border-b border-border bg-canvas/80 backdrop-blur-md sticky top-0 z-20 flex items-center justify-between px-6 select-none">
      {/* Breadcrumb & Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs font-mono text-content-muted">
          <span>ResumeIQ</span>
          <ChevronRight strokeWidth={1.5} className="w-3.5 h-3.5" />
          <span className="text-content-secondary capitalize">
            {currentTab === "models" ? "Models" : currentTab}
          </span>
        </div>
        <span className="h-4 w-px bg-border mx-1 hidden sm:block" />
        <span className="text-sm font-semibold tracking-heading text-content-primary hidden sm:block">
          {current.title}
        </span>
      </div>

      {/* Right Session & Demo status */}
      <div className="flex items-center gap-3">
        {mockActive && (
          <Badge variant="warn" size="sm">
            Demo data
          </Badge>
        )}
        <div className="flex items-center gap-2 pl-3">
          <div className="w-8 h-8 rounded-full bg-surface-card border border-border flex items-center justify-center text-content-secondary">
            <User strokeWidth={1.5} className="w-4 h-4" />
          </div>
          <div className="hidden sm:block text-left">
            <span className="block text-xs font-medium text-content-primary leading-none">
              Candidate
            </span>
            <span className="text-[10px] font-mono text-content-muted">
              Local In-Memory Session
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
