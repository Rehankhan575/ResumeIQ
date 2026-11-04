import React, { useState } from "react";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { Badge } from "../ui/Badge";
import { Layers, ArrowRight, Check } from "lucide-react";
import { rankJobs } from "../../services/api";
import { RankedJob } from "../../types/api";

const AVAILABLE_FIXTURE_JDS = [
  { id: "jd_data_scientist.txt", title: "Senior Data Scientist", category: "Data Science" },
  { id: "jd_software_engineer.txt", title: "Full Stack Software Engineer", category: "Engineering" },
  { id: "jd_chef.txt", title: "Executive Chef", category: "Hospitality" },
];

export const RankView: React.FC = () => {
  const [selectedJDs, setSelectedJDs] = useState<string[]>([
    "jd_data_scientist.txt",
    "jd_software_engineer.txt",
    "jd_chef.txt",
  ]);
  const [rankedList, setRankedList] = useState<RankedJob[]>([
    {
      jd_title: "Senior Data Scientist",
      score: 84.88,
      grade: "B",
      matched_skills: ["machine learning", "python", "scikit-learn", "sql"],
      missing_skills: [],
    },
    {
      jd_title: "Full Stack Software Engineer",
      score: 61.2,
      grade: "C",
      matched_skills: ["python", "sql"],
      missing_skills: ["react", "typescript"],
    },
    {
      jd_title: "Executive Chef",
      score: 32.5,
      grade: "F",
      matched_skills: [],
      missing_skills: ["culinary design", "menu development", "food safety"],
    },
  ]);
  const [isRanking, setIsRanking] = useState(false);

  const toggleSelectJD = (id: string) => {
    setSelectedJDs((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleRunRanking = async () => {
    if (selectedJDs.length === 0) return;
    setIsRanking(true);
    try {
      const res = await rankJobs(null, "ds_clean.pdf", selectedJDs);
      setRankedList(res.ranked);
    } catch {
      // Keep state
    } finally {
      setIsRanking(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      {/* Title & Explainer */}
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Multi-Job Description Ranking
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Rank multiple candidate positions simultaneously against the same resume fixture to identify best-fit career paths.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Job Selection Card (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <Card surface="card">
            <span className="text-xs font-mono uppercase tracking-wider text-content-muted block mb-3">
              Select Target Positions
            </span>

            <div className="space-y-2">
              {AVAILABLE_FIXTURE_JDS.map((jd) => {
                const isSelected = selectedJDs.includes(jd.id);
                return (
                  <button
                    key={jd.id}
                    onClick={() => toggleSelectJD(jd.id)}
                    className={`w-full p-3 rounded-xl border text-left text-xs transition-all flex items-center justify-between ${
                      isSelected
                        ? "bg-surface-subtle border-sand/40 text-content-primary"
                        : "bg-surface border-border text-content-secondary hover:border-border-strong"
                    }`}
                  >
                    <div>
                      <p className="font-medium truncate">{jd.title}</p>
                      <span className="text-[10px] font-mono text-content-muted">
                        {jd.category}
                      </span>
                    </div>
                    {isSelected && (
                      <div className="w-5 h-5 rounded-full bg-sand/10 border border-sand/30 flex items-center justify-center text-sand">
                        <Check strokeWidth={2} className="w-3 h-3" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            <div className="pt-4 border-t border-border mt-4">
              <Button
                variant="primary"
                size="md"
                disabled={selectedJDs.length === 0 || isRanking}
                loading={isRanking}
                onClick={handleRunRanking}
                className="w-full text-xs font-semibold"
              >
                <span>Re-Rank {selectedJDs.length} Positions</span>
                <ArrowRight strokeWidth={1.5} className="w-3.5 h-3.5" />
              </Button>
            </div>
          </Card>
        </div>

        {/* Right: Comparative Leaderboard & Bar Chart (8 cols) */}
        <div className="lg:col-span-8">
          <Card glass={true} className="space-y-6">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Layers strokeWidth={1.5} className="w-4 h-4 text-sand" />
                <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
                  Comparative Alignment Matrix
                </span>
              </div>
              <span className="text-[11px] font-mono text-content-muted">
                Evaluated against ds_clean.pdf
              </span>
            </div>

            {/* Ranked List */}
            <div className="space-y-4">
              {rankedList.map((item, index) => {
                const pct = Math.round(item.score);
                const isTop = index === 0;

                return (
                  <div
                    key={item.jd_title}
                    className="p-4 rounded-xl bg-surface border border-border flex flex-col gap-2.5 transition-colors hover:border-border-strong"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-sand font-bold">
                          #{index + 1}
                        </span>
                        <span className="text-sm font-semibold text-content-primary">
                          {item.jd_title}
                        </span>
                        {isTop && (
                          <Badge variant="good" size="sm">
                            Optimal Fit
                          </Badge>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="font-mono text-base font-bold text-content-primary">
                          {item.score.toFixed(1)}
                        </span>
                        <Badge
                          variant={
                            item.grade.startsWith("A") || item.grade.startsWith("B")
                              ? "good"
                              : item.grade.startsWith("C")
                              ? "warn"
                              : "bad"
                          }
                          size="sm"
                        >
                          Grade {item.grade}
                        </Badge>
                      </div>
                    </div>

                    {/* Bar visualization */}
                    <div className="w-full h-1.5 bg-surface-subtle rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${
                          isTop ? "bg-sand" : "bg-content-muted"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>

                    {/* Matched skills chips */}
                    <div className="flex flex-wrap items-center gap-1.5 pt-1">
                      <span className="text-[10px] font-mono text-content-muted mr-1">
                        Matched:
                      </span>
                      {item.matched_skills.map((s) => (
                        <span
                          key={s}
                          className="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-card border border-border text-content-secondary"
                        >
                          {s}
                        </span>
                      ))}
                      {item.missing_skills.map((s) => (
                        <span
                          key={s}
                          className="px-2 py-0.5 rounded text-[10px] font-mono bg-semantic-warn-dim text-semantic-warn border border-semantic-warn/20"
                        >
                          missing: {s}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
