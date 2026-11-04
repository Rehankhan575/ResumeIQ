import React, { useState } from "react";
import { Hero } from "./Hero";
import { DropZone } from "./DropZone";
import { JobDescriptionCard } from "./JobDescriptionCard";
import { WorkflowSteps } from "./WorkflowSteps";
import { StatStrip } from "./StatStrip";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { MOCK_SAMPLES } from "../../mocks/fixtureData";
import { ArrowRight } from "lucide-react";

interface AnalyzeViewProps {
  onAnalyze: (resumeFile: File | null, resumeChoice: string, jdText: string, jdChoice: string) => void;
  isLoading?: boolean;
}

export const AnalyzeView: React.FC<AnalyzeViewProps> = ({
  onAnalyze,
  isLoading = false,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSampleResume, setSelectedSampleResume] = useState<string>("ds_clean.pdf");
  const [selectedSampleJD, setSelectedSampleJD] = useState<string>("jd_data_scientist.txt");

  const initialJD = MOCK_SAMPLES.jds.find((j) => j.filename === "jd_data_scientist.txt")?.full_text || "";
  const [jdText, setJdText] = useState<string>(initialJD);

  const handleSelectSampleJD = (filename: string) => {
    setSelectedSampleJD(filename);
    const found = MOCK_SAMPLES.jds.find((j) => j.filename === filename);
    if (found) {
      setJdText(found.full_text);
    }
  };

  const handleAnalyzeClick = () => {
    onAnalyze(selectedFile, selectedSampleResume, jdText, selectedSampleJD);
  };

  const hasInput = Boolean(selectedFile || selectedSampleResume) && Boolean(jdText.trim() || selectedSampleJD);

  return (
    <div className="space-y-8 max-w-6xl mx-auto py-2">
      {/* Top 2-Column Split: Hero on Left, Input Workspace Card on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Hero Block (5 cols) */}
        <div className="lg:col-span-5 flex flex-col justify-between pt-2">
          <Hero
            onAnalyzeSample={() => {
              setSelectedFile(null);
              setSelectedSampleResume("ds_clean.pdf");
              handleSelectSampleJD("jd_data_scientist.txt");
              onAnalyze(null, "ds_clean.pdf", initialJD, "jd_data_scientist.txt");
            }}
          />
        </div>

        {/* Right: Workspace Card (7 cols) */}
        <div className="lg:col-span-7">
          <Card glass={true} className="border border-border/90 shadow-2xl space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-border">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-sand" />
                <span className="text-sm font-semibold tracking-heading text-content-primary">
                  Evaluation Workspace
                </span>
              </div>
              <span className="text-[11px] font-mono text-content-muted">
                Target: In-Memory Scoring
              </span>
            </div>

            {/* Resume Upload / DropZone */}
            <DropZone
              selectedFile={selectedFile}
              onSelectFile={setSelectedFile}
              selectedSample={selectedSampleResume}
              onSelectSample={setSelectedSampleResume}
              samples={MOCK_SAMPLES.resumes}
            />

            {/* Job Description Textarea */}
            <JobDescriptionCard
              jdText={jdText}
              onChangeText={setJdText}
              selectedSample={selectedSampleJD}
              onSelectSample={handleSelectSampleJD}
              samples={MOCK_SAMPLES.jds}
            />

            {/* Analyze Full-Width White Pill Button */}
            <div className="pt-2">
              <Button
                variant="primary"
                size="md"
                disabled={!hasInput || isLoading}
                loading={isLoading}
                onClick={handleAnalyzeClick}
                className="w-full text-sm font-semibold tracking-wide"
              >
                <span>Analyze Resume Against Target JD</span>
                <ArrowRight strokeWidth={1.5} className="w-4 h-4" />
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Underneath: 3-Step Row */}
      <div className="space-y-3 pt-2">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted block">
          Evaluation Architecture
        </span>
        <WorkflowSteps />
      </div>

      {/* Underneath: 3-Tile Stat Strip */}
      <div className="space-y-3 pt-2">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted block">
          Project Verification
        </span>
        <StatStrip />
      </div>
    </div>
  );
};
