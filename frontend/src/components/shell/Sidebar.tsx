import React from "react";
import {
  FileSearch,
  BarChart3,
  Layers,
  Compass,
  ShieldAlert,
  History,
  Lock,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Cpu,
} from "lucide-react";
import { Badge } from "../ui/Badge";

export type NavItemKey =
  | "analyze"
  | "results"
  | "models"
  | "rank"
  | "skills"
  | "bias"
  | "history"
  | "privacy";

interface NavItemDef {
  key: NavItemKey;
  label: string;
  icon: React.ElementType;
  badge?: string | number;
}

interface SidebarProps {
  currentTab: NavItemKey;
  onSelectTab: (tab: NavItemKey) => void;
  expanded: boolean;
  onToggleExpanded: () => void;
  issuesCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  expanded,
  onToggleExpanded,
  issuesCount = 3,
}) => {
  const navItems: NavItemDef[] = [
    { key: "analyze", label: "Analyze", icon: FileSearch },
    { key: "results", label: "Results", icon: BarChart3, badge: issuesCount },
    { key: "models", label: "Model results", icon: Cpu },
    { key: "rank", label: "Rank jobs", icon: Layers },
    { key: "skills", label: "Skill gap", icon: Compass },
    { key: "bias", label: "Bias check", icon: ShieldAlert },
    { key: "history", label: "History", icon: History },
    { key: "privacy", label: "Privacy", icon: Lock },
  ];

  return (
    <aside
      className={`fixed top-0 bottom-0 left-0 z-30 flex bg-canvas border-r border-border transition-all duration-200 select-none ${
        expanded ? "w-[304px]" : "w-16"
      }`}
    >
      {/* Slim 64px Icon Rail */}
      <div className="w-16 h-full flex flex-col items-center justify-between py-5 border-r border-border/50 shrink-0">
        <div className="flex flex-col items-center gap-6">
          {/* Logo Mark */}
          <div className="w-10 h-10 rounded-xl bg-surface-card border border-border flex items-center justify-center text-content-primary shadow-sm">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#D9C7A0"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect width="18" height="18" x="3" y="3" rx="4" />
              <path d="M7 8h10" />
              <path d="M7 12h7" />
              <path d="M7 16h4" />
            </svg>
          </div>

          {/* Rail Icon Buttons */}
          <nav className="flex flex-col items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => onSelectTab(item.key)}
                  title={item.label}
                  className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-150 relative ${
                    isActive
                      ? "bg-surface-subtle text-content-primary"
                      : "text-content-muted hover:text-content-secondary hover:bg-white/[0.03]"
                  }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-2 bottom-2 w-[3px] bg-sand rounded-r" />
                  )}
                  <Icon strokeWidth={1.5} className="w-5 h-5" />
                </button>
              );
            })}
          </nav>
        </div>

        {/* Toggle Collapse Button */}
        <button
          onClick={onToggleExpanded}
          className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-content-muted hover:text-content-primary hover:bg-white/[0.04] transition-colors"
          title={expanded ? "Collapse Sidebar" : "Expand Sidebar"}
        >
          {expanded ? (
            <ChevronLeft strokeWidth={1.5} className="w-4 h-4" />
          ) : (
            <ChevronRight strokeWidth={1.5} className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Expanded 240px Navigation Panel */}
      {expanded && (
        <div className="w-60 h-full flex flex-col justify-between p-4 glass-panel border-y-0 border-r-0">
          <div>
            {/* Header Brand */}
            <div className="flex items-center justify-between px-2 pt-1 pb-6 border-b border-border">
              <div>
                <span className="text-base font-semibold tracking-heading text-content-primary">
                  ResumeIQ
                </span>
                <span className="block text-[11px] font-mono text-content-muted uppercase tracking-wider">
                  ATS Optimization
                </span>
              </div>
              <Badge variant="sand" size="sm">
                v1.3
              </Badge>
            </div>

            {/* Nav List */}
            <div className="mt-5 space-y-1">
              <span className="px-3 text-[11px] font-mono uppercase text-content-muted tracking-wider block mb-2">
                Workspace
              </span>
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentTab === item.key;
                return (
                  <button
                    key={item.key}
                    onClick={() => onSelectTab(item.key)}
                    className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm transition-all duration-150 relative ${
                      isActive
                        ? "bg-surface-subtle text-content-primary font-medium"
                        : "text-content-secondary hover:text-content-primary hover:bg-white/[0.03]"
                    }`}
                  >
                    {isActive && (
                      <span className="absolute left-0 top-2 bottom-2 w-[3px] bg-sand rounded-r" />
                    )}
                    <div className="flex items-center gap-3">
                      <Icon strokeWidth={1.5} className={`w-4 h-4 ${isActive ? "text-sand" : "text-content-muted"}`} />
                      <span>{item.label}</span>
                    </div>
                    {item.badge !== undefined && (
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-surface-card border border-border text-content-muted">
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Privacy Status Chip */}
          <div className="p-3 rounded-xl bg-surface border border-border/80 flex items-center gap-2.5">
            <ShieldCheck strokeWidth={1.5} className="w-4 h-4 text-semantic-good shrink-0" />
            <div className="overflow-hidden">
              <span className="block text-xs font-medium text-content-primary truncate">
                Files are never stored
              </span>
              <span className="block text-[11px] text-content-muted truncate">
                100% In-memory processing
              </span>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

