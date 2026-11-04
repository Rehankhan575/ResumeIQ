import React from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Lock, ShieldCheck, Database, HardDrive, Cpu } from "lucide-react";

export const PrivacyView: React.FC = () => {
  const principles = [
    {
      title: "In-Memory Processing",
      icon: Cpu,
      desc: "Uploaded bytes (.pdf, .docx) are converted into transient byte arrays in RAM, evaluated through model forward passes, and purged immediately after response delivery.",
    },
    {
      title: "Zero Database Retention",
      icon: Database,
      desc: "There is no relational database, document store, or telemetry logging attached to resume content. Candidate identities, phone numbers, and addresses are never saved.",
    },
    {
      title: "No Disk Log Leakage",
      icon: HardDrive,
      desc: "Application logs at the API and pipeline layers enforce strict redaction filters. Error handlers report diagnostic status codes without echoing resume payload buffers.",
    },
    {
      title: "Local Storage Isolation",
      icon: Lock,
      desc: "Score history is purely opt-in and restricted to numeric scores and timestamps. Raw text or extracted resume sections are never serialized into browser storage.",
    },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      <div>
        <h2 className="text-2xl font-semibold tracking-heading text-content-primary">
          Privacy Architecture & Data Handling
        </h2>
        <p className="text-xs text-content-secondary mt-1">
          Designed from the ground up for strict confidentiality and local execution.
        </p>
      </div>

      <div className="p-5 rounded-2xl bg-surface border border-border flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-semantic-good-dim border border-semantic-good/20 flex items-center justify-center text-semantic-good shrink-0">
            <ShieldCheck strokeWidth={1.5} className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-content-primary">
              In-Memory Local Processing
            </h4>
            <p className="text-xs text-content-muted max-w-2xl mt-0.5">
              Files are processed in memory on this machine and not stored by the app. If an optional LLM key is configured in .env, bullet text is sent to that provider with its provider label; otherwise rule-based heuristics generate suggestions.
            </p>
          </div>
        </div>
        <Badge variant="good" size="md">
          In-Memory Only
        </Badge>
      </div>

      {/* Grid of 4 architectural pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {principles.map((p) => {
          const Icon = p.icon;
          return (
            <Card key={p.title} surface="card" className="space-y-2">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-surface-subtle border border-border flex items-center justify-center text-sand">
                  <Icon strokeWidth={1.5} className="w-4 h-4" />
                </div>
                <h4 className="text-xs font-semibold text-content-primary">
                  {p.title}
                </h4>
              </div>
              <p className="text-xs text-content-muted leading-relaxed">
                {p.desc}
              </p>
            </Card>
          );
        })}
      </div>
    </div>
  );
};
