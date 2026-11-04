import React, { useRef, useState } from "react";
import { UploadCloud, FileText, X } from "lucide-react";
import { SampleResume } from "../../types/api";

interface DropZoneProps {
  selectedFile: File | null;
  onSelectFile: (file: File | null) => void;
  selectedSample: string;
  onSelectSample: (sample: string) => void;
  samples: SampleResume[];
}

export const DropZone: React.FC<DropZoneProps> = ({
  selectedFile,
  onSelectFile,
  selectedSample,
  onSelectSample,
  samples,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      onSelectFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onSelectFile(e.target.files[0]);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-mono uppercase tracking-wider text-content-muted">
          Your Resume
        </label>
        <span className="text-[11px] font-mono text-content-muted">
          PDF, DOCX, PNG, JPG ≤ 5 MB
        </span>
      </div>

      {/* Drop Zone Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? "border-sand bg-sand/5"
            : selectedFile
            ? "border-border-strong bg-surface"
            : "border-border bg-surface hover:border-border-strong hover:bg-surface-card"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.png,.jpg,.jpeg"
          onChange={handleFileChange}
          className="hidden"
        />

        {selectedFile ? (
          <div className="w-full flex items-center justify-between p-2 rounded-lg bg-surface-card border border-border">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="w-8 h-8 rounded-lg bg-surface-subtle flex items-center justify-center text-sand shrink-0">
                <FileText strokeWidth={1.5} className="w-4 h-4" />
              </div>
              <div className="text-left overflow-hidden">
                <p className="text-xs font-medium text-content-primary truncate">
                  {selectedFile.name}
                </p>
                <p className="text-[11px] font-mono text-content-muted">
                  {formatSize(selectedFile.size)} • Uploaded
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onSelectFile(null);
              }}
              className="p-1 rounded-full text-content-muted hover:text-content-primary hover:bg-white/[0.06] transition-colors"
              title="Remove file"
            >
              <X strokeWidth={1.5} className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 py-3">
            <div className="w-9 h-9 rounded-full bg-surface-subtle flex items-center justify-center text-content-secondary">
              <UploadCloud strokeWidth={1.5} className="w-4 h-4" />
            </div>
            <div>
              <p className="text-xs font-medium text-content-primary">
                Click or drag & drop your resume
              </p>
              <p className="text-[11px] text-content-muted mt-0.5">
                Processed in-memory. Zero text stored on disk.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Sample Resume Selector */}
      <div className="flex items-center justify-between pt-1">
        <span className="text-xs text-content-muted">Or try a sample fixture:</span>
        <select
          value={selectedFile ? "" : selectedSample}
          onChange={(e) => {
            onSelectFile(null);
            onSelectSample(e.target.value);
          }}
          className="bg-surface border border-border text-xs text-content-primary rounded-lg px-2.5 py-1.5 focus:border-sand focus:outline-none cursor-pointer"
        >
          {samples.map((s) => (
            <option key={s.filename} value={s.filename}>
              {s.title} ({s.extension})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
