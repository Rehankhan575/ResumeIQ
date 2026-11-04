import React, { useState } from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { History, Trash2, ShieldCheck } from "lucide-react";
import { HistoryItem } from "../../types/api";

const STORAGE_KEY_OPT_IN = "resumeiq_history_opt_in";
const STORAGE_KEY_RECORDS = "resumeiq_history";

export const HistoryView: React.FC = () => {
  const [optIn, setOptIn] = useState<boolean>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY_OPT_IN) === "true";
    } catch {
      return false;
    }
  });

  const [records, setRecords] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_RECORDS);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return parsed;
      }
    } catch {
      // Fallback
    }
    // Clean empty default: zero seeded mock records
    return [];
  });

  const handleToggleOptIn = (val: boolean) => {
    setOptIn(val);
    try {
      localStorage.setItem(STORAGE_KEY_OPT_IN, String(val));
      if (!val) {
        localStorage.removeItem(STORAGE_KEY_RECORDS);
        setRecords([]);
      }
    } catch {
      // Ignore storage errors
    }
  };

  const handleClearHistory = () => {
    setRecords([]);
    try {
      localStorage.removeItem(STORAGE_KEY_RECORDS);
    } catch {
      // Ignore
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Evaluation History
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Local, privacy-preserving score tracking. Never sends or retains raw candidate text.
        </p>
      </div>

      {/* Opt-In Control Card */}
      <Card surface="card" className="flex items-center justify-between p-5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-sand/10 border border-sand/20 flex items-center justify-center text-sand">
            <ShieldCheck strokeWidth={1.5} className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-content-primary">
              Local Evaluation History (Opt-In)
            </h4>
            <p className="text-[11px] text-content-muted">
              Stores only numeric scores, timestamps, and job titles in browser localStorage.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={optIn}
              onChange={(e) => handleToggleOptIn(e.target.checked)}
              className="sr-only peer"
              aria-label="Opt-in to local score history"
            />
            <div className="w-11 h-6 bg-surface-subtle peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-content-primary after:border-border after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sand" />
          </label>
          <span className="text-xs font-mono text-content-secondary">
            {optIn ? "Enabled" : "Disabled"}
          </span>
        </div>
      </Card>

      {/* History Table */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-border">
          <div className="flex items-center gap-2">
            <History strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <span className="text-xs font-mono uppercase tracking-wider text-content-primary">
              Saved Evaluation Records ({records.length})
            </span>
          </div>

          {records.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearHistory}
              className="text-[11px] font-mono text-content-muted hover:text-semantic-bad gap-1.5 h-8"
            >
              <Trash2 strokeWidth={1.5} className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </Button>
          )}
        </div>

        {records.length === 0 ? (
          <div className="py-12 text-center text-xs text-content-muted space-y-2">
            <p className="text-content-secondary font-medium">No evaluation records saved.</p>
            <p className="text-[11px]">
              Enable history above and run an evaluation on the Analyze screen to record past diagnostic scores.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border/80 text-[11px] font-mono text-content-muted uppercase">
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Target Job Position</th>
                  <th className="py-2.5 px-3">Grade</th>
                  <th className="py-2.5 px-3 text-right">ATS Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {records.map((item) => (
                  <tr
                    key={item.id}
                    className="hover:bg-white/[0.02] transition-colors font-mono"
                  >
                    <td className="py-3 px-3 text-content-muted text-[11px]">
                      {item.timestamp}
                    </td>
                    <td className="py-3 px-3 font-medium text-content-primary">
                      {item.jd_title}
                    </td>
                    <td className="py-3 px-3">
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
                        {item.grade}
                      </Badge>
                    </td>
                    <td className="py-3 px-3 text-right font-bold text-content-primary">
                      {item.score.toFixed(1)} / 100
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};
