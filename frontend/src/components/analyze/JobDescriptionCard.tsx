import React from "react";
import { SampleJD } from "../../types/api";

interface JobDescriptionCardProps {
  jdText: string;
  onChangeText: (text: string) => void;
  selectedSample: string;
  onSelectSample: (sample: string) => void;
  samples: SampleJD[];
}

export const JobDescriptionCard: React.FC<JobDescriptionCardProps> = ({
  jdText,
  onChangeText,
  selectedSample,
  onSelectSample,
  samples,
}) => {
  const charCount = jdText.length;
  const wordCount = jdText.trim() ? jdText.trim().split(/\s+/).length : 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-mono uppercase tracking-wider text-content-muted">
          Job Description
        </label>
        <div className="flex items-center gap-2 text-[11px] font-mono text-content-muted">
          <span>{charCount} chars</span>
          <span>•</span>
          <span>{wordCount} words</span>
        </div>
      </div>

      {/* Textarea */}
      <div className="relative">
        <textarea
          rows={6}
          value={jdText}
          onChange={(e) => onChangeText(e.target.value)}
          placeholder="Paste the target job description requirements, responsibilities, or qualification criteria here..."
          className="w-full bg-surface border border-border rounded-xl p-3.5 text-xs text-content-primary placeholder:text-content-muted focus:border-sand focus:outline-none resize-none transition-colors"
        />
      </div>

      {/* Sample JD Selector */}
      <div className="flex items-center justify-between pt-1">
        <span className="text-xs text-content-muted">Or try a sample job description:</span>
        <select
          value={selectedSample}
          onChange={(e) => onSelectSample(e.target.value)}
          className="bg-surface border border-border text-xs text-content-primary rounded-lg px-2.5 py-1.5 focus:border-sand focus:outline-none cursor-pointer max-w-[220px] truncate"
        >
          {samples.map((jd) => (
            <option key={jd.filename} value={jd.filename}>
              {jd.title}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

