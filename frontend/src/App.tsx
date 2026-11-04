import React, { useState } from "react";
import { AppShell } from "./components/shell/AppShell";
import { NavItemKey } from "./components/shell/Sidebar";
import { AnalyzeView } from "./components/analyze/AnalyzeView";
import { ResultsView } from "./components/results/ResultsView";
import { RankView } from "./components/views/RankView";
import { SkillGapView } from "./components/views/SkillGapView";
import { BiasCheckView } from "./components/views/BiasCheckView";
import { HistoryView } from "./components/views/HistoryView";
import { PrivacyView } from "./components/views/PrivacyView";
import { ErrorBanner } from "./components/ui/ErrorBanner";
import { ResultsSkeleton } from "./components/ui/Skeleton";
import { ScoreResult, HistoryItem } from "./types/api";
import { MOCK_FIXTURE_RESULT } from "./mocks/fixtureData";
import { analyzeResume } from "./services/api";

const STORAGE_KEY_OPT_IN = "resumeiq_history_opt_in";
const STORAGE_KEY_RECORDS = "resumeiq_history";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavItemKey>("analyze");
  const [result, setResult] = useState<ScoreResult | null>(MOCK_FIXTURE_RESULT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const saveToHistoryIfOptedIn = (res: ScoreResult, targetTitle: string) => {
    try {
      const isOptedIn = localStorage.getItem(STORAGE_KEY_OPT_IN) === "true";
      if (!isOptedIn) return;

      const raw = localStorage.getItem(STORAGE_KEY_RECORDS);
      const prev: HistoryItem[] = raw ? JSON.parse(raw) : [];

      // Privacy: Only timestamp, JD title, score, and grade are serialized.
      // ZERO resume text is EVER recorded.
      const now = new Date();
      const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(
        now.getDate()
      ).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(
        now.getMinutes()
      ).padStart(2, "0")}`;

      const newItem: HistoryItem = {
        id: `rec-${Date.now()}`,
        timestamp: dateStr,
        jd_title: targetTitle || "Target Role",
        score: res.total,
        grade: res.grade,
      };

      localStorage.setItem(STORAGE_KEY_RECORDS, JSON.stringify([newItem, ...prev.slice(0, 49)]));
    } catch {
      // Safe fallback
    }
  };

  const handleAnalyze = async (
    resumeFile: File | null,
    resumeChoice: string,
    jdText: string,
    jdChoice: string
  ) => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const scoreResult = await analyzeResume(
        resumeFile,
        resumeChoice,
        jdText,
        jdChoice,
        false
      );
      setResult(scoreResult);
      const title = jdChoice ? jdChoice.replace("jd_", "").replace(".txt", "").replace("_", " ").toUpperCase() : "Target Job";
      saveToHistoryIfOptedIn(scoreResult, title);
      setCurrentTab("results");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to analyze resume. Please check server connectivity or try a sample.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AppShell
      currentTab={currentTab}
      onSelectTab={setCurrentTab}
      issuesCount={result?.issues?.length || 3}
    >
      {/* Top Error Alert Banner if an API or file error occurred */}
      {errorMessage && (
        <div className="max-w-6xl mx-auto mb-6">
          <ErrorBanner
            message={errorMessage}
            onRetry={() => {
              setErrorMessage(null);
              setCurrentTab("analyze");
            }}
          />
        </div>
      )}

      {/* Screen Routing */}
      {isLoading ? (
        <ResultsSkeleton />
      ) : (
        <>
          {currentTab === "analyze" && (
            <AnalyzeView onAnalyze={handleAnalyze} isLoading={isLoading} />
          )}

          {currentTab === "results" && (
            <ResultsView
              result={result}
              onGoToAnalyze={() => setCurrentTab("analyze")}
            />
          )}

          {currentTab === "rank" && <RankView />}

          {currentTab === "skills" && <SkillGapView />}

          {currentTab === "bias" && <BiasCheckView />}

          {currentTab === "history" && <HistoryView />}

          {currentTab === "privacy" && <PrivacyView />}
        </>
      )}
    </AppShell>
  );
};

export default App;
