import React, { useState } from "react";
import { ScoreResult } from "../../types/api";
import { Download, CheckCircle2 } from "lucide-react";
import { Button } from "../ui/Button";

interface ReportExportProps {
  result: ScoreResult;
  jdTitle?: string;
}

export const ReportExport: React.FC<ReportExportProps> = ({
  result,
  jdTitle = "Target Position",
}) => {
  const [showToast, setShowToast] = useState(false);

  const generateMarkdown = () => {
    const timestamp = new Date().toISOString();
    return `# ResumeIQ ATS Diagnostics & Score Report
Generated: ${timestamp}
Target Role: ${jdTitle}

## Executive Summary
- Total Score: ${result.total.toFixed(1)} / 100
- Grade: ${result.grade}
- Issues Count: ${result.issues.length}
- Missing Skills: ${result.missing_skills.length === 0 ? "None" : result.missing_skills.join(", ")}

## 9-Rubric Component Decomposition
${Object.entries(result.components)
  .map(([k, v]) => `- ${k.replace("_", " ")}: ${(v * 100).toFixed(1)}%`)
  .join("\n")}

## Prioritized Action Items
${result.issues
  .map(
    (issue, i) =>
      `${i + 1}. [${issue.severity.toUpperCase()}] ${issue.message} (+${issue.estimated_gain.toFixed(1)} pts)`
  )
  .join("\n")}

## Skills Breakdown
- Matched: ${result.match?.matched_skills?.join(", ") || "None"}
- Missing: ${result.match?.missing_skills?.join(", ") || "None"}

---
*Report generated purely in-memory by ResumeIQ. Candidate resume text is never retained on disk.*
`;
  };

  const handleDownload = () => {
    const md = generateMarkdown();
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `resumeiq_report_${Date.now()}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  return (
    <>
      <Button
        variant="secondary"
        size="sm"
        onClick={handleDownload}
        className="text-xs h-9"
      >
        <Download strokeWidth={1.5} className="w-3.5 h-3.5" />
        <span>Download Report (.md)</span>
      </Button>

      {/* Toast Notification */}
      {showToast && (
        <div className="fixed bottom-6 right-6 z-50 p-4 rounded-xl glass-panel border border-border shadow-2xl flex items-center gap-3 animate-fade-in text-xs font-mono">
          <CheckCircle2 strokeWidth={1.5} className="w-4 h-4 text-semantic-good shrink-0" />
          <div>
            <span className="block font-medium text-content-primary">
              Markdown Report Downloaded
            </span>
            <span className="text-[11px] text-content-muted">
              Zero resume text leaked to disk.
            </span>
          </div>
        </div>
      )}
    </>
  );
};

