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
      jd_title: "Data Scientist",
      score: 84.9,
      grade: "B",
      matched_skills: ["machine learning", "python", "scikit-learn", "sql"],
      missing_skills: [],
    },
    {
      jd_title: "Software Engineer",
      score: 60.5,
      grade: "D",
      matched_skills: ["python", "sql"],
      missing_skills: ["javascript", "react", "node.js", "git"],
    },
    {
      jd_title: "Chef",
      score: 55.5,
      grade: "F",
      matched_skills: [],
      missing_skills: [
        "culinary arts",
        "kitchen sanitation",
        "food safety",
        "menu planning",
        "haccp",
        "management",
      ],
    },
  ]);
  const [isRanking, setIsRanking] = useState(false);
  const [rankError, setRankError] = useState<string | null>(null);

  const toggleSelectJD = (id: string) => {
    setSelectedJDs((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleRunRanking = async () => {
    if (selectedJDs.length === 0) return;
    setIsRanking(true);
    setRankError(null);
    try {
      const res = await rankJobs(null, "ds_clean.pdf", selectedJDs);
      setRankedList(res.ranked);
    } catch (err: any) {
      setRankError(err.message || "Failed to rank job descriptions");
    } finally {
      setIsRanking(false);
    }
  };

  const getGradeVariant = (g: string): "good" | "sand" | "warn" | "bad" => {
    const upper = g.toUpperCase();
    if (upper.startsWith("A") || upper.startsWith("B")) return "good";
    if (upper.startsWith("C")) return "warn";
    return "bad";
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Multi-Job Description Ranking
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Rank your candidate resume across multiple target openings simultaneously. Powered by Siamese neural embeddings.
        </p>
      </div>

      {/* Target Selection Card */}
      <Card surface="card" className="space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-border">
          <div className="flex items-center gap-2">
            <Layers strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
              Target Job Fixtures
            </span>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            {selectedJDs.length} of {AVAILABLE_FIXTURE_JDS.length} selected
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {AVAILABLE_FIXTURE_JDS.map((jd) => {
            const isSelected = selectedJDs.includes(jd.id);
            return (
              <button
                key={jd.id}
                onClick={() => toggleSelectJD(jd.id)}
                className={`p-3 rounded-xl border text-left transition-all flex items-start justify-between ${
                  isSelected
                    ? "bg-surface-subtle border-sand/40 text-content-primary shadow-sm"
                    : "bg-surface border-border text-content-muted hover:border-border-strong hover:text-content-secondary"
                }`}
              >
                <div>
                  <span className="text-xs font-semibold block leading-snug">
                    {jd.title}
                  </span>
                  <span className="text-[11px] font-mono text-content-muted mt-1 block">
                    {jd.category}
                  </span>
                </div>
                <div
                  className={`w-5 h-5 rounded-full border flex items-center justify-center shrink-0 ml-2 ${
                    isSelected
                      ? "bg-sand border-sand text-content-inverse"
                      : "border-border"
                  }`}
                >
                  {isSelected && <Check strokeWidth={2.5} className="w-3 h-3 text-black" />}
                </div>
              </button>
            );
          })}
        </div>

        <div className="pt-2 flex items-center justify-between">
          <span className="text-xs font-mono text-content-muted">
            Active Resume: <strong className="text-content-primary font-medium">ds_clean.pdf</strong>
          </span>
          <Button
            variant="primary"
            size="sm"
            onClick={handleRunRanking}
            disabled={selectedJDs.length === 0 || isRanking}
            className="gap-2"
          >
            <span>{isRanking ? "Computing Neural Match..." : "Calculate Comparative Ranking"}</span>
            <ArrowRight strokeWidth={1.5} className="w-3.5 h-3.5" />
          </Button>
        </div>
      </Card>

      {rankError && (
        <div className="p-3 rounded-xl bg-semantic-bad-dim border border-semantic-bad/30 text-xs text-semantic-bad">
          {rankError}
        </div>
      )}

      {/* Ranked Job Output List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
            Ranked Opportunities (Best Fit to Lowest)
          </span>
          <span className="text-[11px] font-mono text-content-muted">
            Cosine Similarity & Rubric Composite
          </span>
        </div>

        {rankedList.map((job, idx) => (
          <Card
            key={job.jd_title}
            surface="card"
            className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 transition-colors hover:border-border-strong"
          >
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 rounded-full bg-surface-subtle border border-border flex items-center justify-center text-xs font-mono font-bold text-sand shrink-0">
                0{idx + 1}
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-content-primary">
                    {job.jd_title}
                  </h4>
                  <Badge variant={getGradeVariant(job.grade)} size="sm">
                    Grade {job.grade}
                  </Badge>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-content-muted">
                  <span>Matched: {job.matched_skills.length}</span>
                  <span>•</span>
                  <span>Missing: {job.missing_skills.length}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center sm:flex-col sm:items-end justify-between border-t sm:border-t-0 pt-3 sm:pt-0 border-border">
              <span className="text-2xl font-mono font-bold tracking-tight text-content-primary">
                {job.score.toFixed(1)}
              </span>
              <span className="text-[11px] font-mono text-content-muted">
                ATS Composite Score
              </span>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
