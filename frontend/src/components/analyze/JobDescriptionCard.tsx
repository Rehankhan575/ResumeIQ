import React from "react";
import { SampleJD } from "../../types/api";
import { FileText } from "lucide-react";

interface JobDescriptionCardProps {
  mode: "sample" | "paste";
  onModeChange: (mode: "sample" | "paste") => void;
  jdText: string;
  onChangeText: (text: string) => void;
  selectedSample: string;
  onSelectSample: (sample: string) => void;
  samples: SampleJD[];
}

export const JobDescriptionCard: React.FC<JobDescriptionCardProps> = ({
  mode,
  onModeChange,
  jdText,
  onChangeText,
  selectedSample,
  onSelectSample,
  samples,
}) => {
  const charCount = jdText.length;
  const wordCount = jdText.trim() ? jdText.trim().split(/\s+/).length : 0;
  const currentSample = samples.find((j) => j.filename === selectedSample);

  return (
    <div className="flex flex-col gap-3">
      {/* Header & Segmented Mode Toggle */}
      <div className="flex items-center justify-between">
        <label className="text-xs font-mono uppercase tracking-wider text-content-muted">
          Job Description
        </label>
        <div className="flex items-center p-0.5 rounded-lg bg-surface border border-border">
          <button
            type="button"
            onClick={() => onModeChange("sample")}
            className={`px-2.5 py-1 text-[11px] font-mono rounded-md transition-all ${
              mode === "sample"
                ? "bg-surface-subtle text-content-primary shadow-sm font-medium"
                : "text-content-muted hover:text-content-secondary"
            }`}
          >
            Sample
          </button>
          <button
            type="button"
            onClick={() => onModeChange("paste")}
            className={`px-2.5 py-1 text-[11px] font-mono rounded-md transition-all ${
              mode === "paste"
                ? "bg-surface-subtle text-content-primary shadow-sm font-medium"
                : "text-content-muted hover:text-content-secondary"
            }`}
          >
            Paste
          </button>
        </div>
      </div>

      {/* Mode = Sample: Dropdown picker */}
      {mode === "sample" && (
        <div className="p-3.5 rounded-xl bg-surface border border-border flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 overflow-hidden">
            <FileText strokeWidth={1.5} className="w-4 h-4 text-sand shrink-0" />
            <span className="text-xs text-content-secondary truncate">
              Select target job fixture:
            </span>
          </div>
          <select
            value={selectedSample}
            onChange={(e) => onSelectSample(e.target.value)}
            className="bg-surface-card border border-border text-xs text-content-primary rounded-lg px-2.5 py-1.5 focus:border-sand focus:outline-none cursor-pointer max-w-[220px] truncate"
          >
            {samples.map((jd) => (
              <option key={jd.filename} value={jd.filename}>
                {jd.title}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Mode = Paste: Textarea */}
      {mode === "paste" && (
        <div className="space-y-2">
          <textarea
            rows={5}
            value={jdText}
            onChange={(e) => onChangeText(e.target.value)}
            placeholder="Paste target job description requirements, responsibilities, or qualification criteria here..."
            className="w-full bg-surface border border-border rounded-xl p-3.5 text-xs text-content-primary placeholder:text-content-muted focus:border-sand focus:outline-none resize-none transition-colors"
          />
          <div className="flex items-center justify-end gap-2 text-[11px] font-mono text-content-muted">
            <span>{charCount} chars</span>
            <span>•</span>
            <span>{wordCount} words</span>
          </div>
        </div>
      )}

      {/* Precedence Indicator: "Using: <name>" */}
      <div className="text-[11px] font-mono text-content-muted px-1 flex items-center justify-between">
        <span>
          Using:{" "}
          <strong className="text-content-primary font-medium">
            {mode === "sample"
              ? currentSample?.title || selectedSample
              : jdText.trim()
              ? `Pasted Job Description (${wordCount} words)`
              : "No text pasted"}
          </strong>
        </span>
        <span className="text-[10px] text-content-muted">
          {mode === "sample" ? "Fixture Mode" : "Paste Mode"}
        </span>
      </div>
    </div>
  );
};
