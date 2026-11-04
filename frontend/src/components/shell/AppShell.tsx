import React, { useState } from "react";
import { Sidebar, NavItemKey } from "./Sidebar";
import { TopBar } from "./TopBar";

interface AppShellProps {
  currentTab: NavItemKey;
  onSelectTab: (tab: NavItemKey) => void;
  children: React.ReactNode;
  issuesCount?: number;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentTab,
  onSelectTab,
  children,
  issuesCount = 3,
}) => {
  const [expanded, setExpanded] = useState<boolean>(true);

  return (
    <div className="min-h-screen bg-canvas text-content-primary flex overflow-hidden relative">
      {/* Dim Static Ambient Beam (Mood matching sidebar reference) */}
      <div className="ambient-beam" aria-hidden="true" />
      {/* 3% Fine Grain Noise Texture */}
      <div className="ambient-noise" aria-hidden="true" />

      {/* Desktop/Tablet Sidebar (Hidden on small mobile < 720px) */}
      <div className="hidden sm:block">
        <Sidebar
          currentTab={currentTab}
          onSelectTab={onSelectTab}
          expanded={expanded}
          onToggleExpanded={() => setExpanded(!expanded)}
          issuesCount={issuesCount}
        />
      </div>

      {/* Main Content Area */}
      <div
        className={`flex-1 flex flex-col min-h-screen transition-all duration-200 ${
          expanded ? "sm:pl-[304px]" : "sm:pl-16"
        }`}
      >
        {/* Top Sticky Bar */}
        <TopBar currentTab={currentTab} />

        {/* Content Pane (Only this pane scrolls) */}
        <main className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 z-10">
          <div className="max-w-[1280px] mx-auto pb-16">{children}</div>
        </main>

        {/* Mobile Bottom Bar (< 720px) */}
        <nav className="sm:hidden fixed bottom-0 left-0 right-0 z-30 h-14 bg-canvas/95 backdrop-blur-md border-t border-border flex items-center justify-around px-2">
          {[
            { key: "analyze" as NavItemKey, label: "Analyze" },
            { key: "results" as NavItemKey, label: "Results" },
            { key: "rank" as NavItemKey, label: "Rank" },
            { key: "skills" as NavItemKey, label: "Skills" },
            { key: "bias" as NavItemKey, label: "Bias" },
            { key: "history" as NavItemKey, label: "History" },
          ].map((item) => (
            <button
              key={item.key}
              onClick={() => onSelectTab(item.key)}
              className={`flex flex-col items-center justify-center text-xs py-1 px-2 rounded-lg transition-colors ${
                currentTab === item.key
                  ? "text-sand font-semibold"
                  : "text-content-muted hover:text-content-secondary"
              }`}
            >
              <span>{item.label}</span>
              {currentTab === item.key && (
                <span className="w-1 h-1 rounded-full bg-sand mt-0.5" />
              )}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
};

