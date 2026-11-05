import React, { useEffect, useState } from "react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Skeleton } from "../ui/Skeleton";
import { Cpu, CheckCircle2, ChevronDown, ChevronRight, BarChart2 } from "lucide-react";
import { fetchModelResults } from "../../services/api";
import { ModelResultsResponse } from "../../types/api";

export const ModelResultsView: React.FC = () => {
  const [data, setData] = useState<ModelResultsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showCategoryMatcher, setShowCategoryMatcher] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    fetchModelResults()
      .then((res) => {
        if (mounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || "Failed to load model reports");
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="space-y-6 max-w-5xl mx-auto py-2">
        <div className="space-y-2">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-4 w-96" />
        </div>
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 rounded-2xl bg-surface border border-border text-center max-w-xl mx-auto my-12 space-y-3">
        <p className="text-sm font-medium text-semantic-bad">
          {error || "Could not retrieve empirical evaluation data."}
        </p>
        <p className="text-xs text-content-muted">
          Ensure artifacts/reports and artifacts/metrics are present in the repository root.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto py-2">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface border border-border w-fit mb-3">
          <Cpu strokeWidth={1.5} className="w-3.5 h-3.5 text-sand" />
          <span className="text-xs font-mono text-content-secondary uppercase tracking-wider">
            Empirical Validation
          </span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-semibold tracking-heading text-content-primary">
          Model Results & Verification
        </h2>
        <p className="text-xs sm:text-sm text-content-secondary mt-1">
          Rigorous held-out testing, paired bootstrap significance, Siamese retrieval metrics, and XAI faithfulness.
        </p>
      </div>

      {/* 1. Classifier Performance */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-border">
          <div className="flex items-center gap-2">
            <BarChart2 strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <h3 className="text-sm font-semibold text-content-primary tracking-heading">
              Classifier Performance on Held-Out Test Data
            </h3>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            4 Model Architectures Benchmarked
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-[11px] text-content-muted uppercase">
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Val Macro F1</th>
                <th className="py-2.5 px-3">Test Macro F1 (95% CI)</th>
                <th className="py-2.5 px-3">Test Accuracy</th>
                <th className="py-2.5 px-3">Train-Val Gap</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {data.classifier.models.map((m) => (
                <tr key={m.key} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-3 font-medium text-content-primary flex items-center gap-2">
                    <span>{m.model}</span>
                    {m.served && (
                      <Badge variant="good" size="sm">
                        Served
                      </Badge>
                    )}
                  </td>
                  <td className="py-3 px-3 text-content-secondary">{m.val_macro_f1}</td>
                  <td className="py-3 px-3 text-content-primary font-semibold">{m.test_macro_f1_ci}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.test_accuracy}</td>
                  <td className="py-3 px-3 text-content-muted">{m.train_val_gap}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Paired Tests */}
        {data.classifier.paired_tests.length > 0 && (
          <div className="pt-4 border-t border-border/70 space-y-3">
            <h4 className="text-xs font-semibold text-content-secondary uppercase tracking-wider">
              Paired Statistical Comparison
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-border text-[11px] text-content-muted uppercase">
                    <th className="py-2 px-3">Comparison</th>
                    <th className="py-2 px-3">Metric</th>
                    <th className="py-2 px-3">Diff (A - B)</th>
                    <th className="py-2 px-3">95% CI</th>
                    <th className="py-2 px-3">P-Value</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {data.classifier.paired_tests.map((p, idx) => (
                    <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-2.5 px-3 text-content-primary">{p.comparison}</td>
                      <td className="py-2.5 px-3 text-content-secondary">{p.metric}</td>
                      <td className="py-2.5 px-3 text-content-primary font-medium">{p.diff}</td>
                      <td className="py-2.5 px-3 text-content-muted">{p.ci_95}</td>
                      <td className="py-2.5 px-3 text-content-secondary">{p.p_value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <p className="text-[11px] text-content-muted leading-relaxed pt-2 border-t border-border/50">
          <span className="font-semibold text-content-secondary">Selection Rationale: </span>
          {data.classifier.note}
        </p>
      </Card>

      {/* 2. Matcher Retrieval Metrics */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-border">
          <div className="flex items-center gap-2">
            <BarChart2 strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <h3 className="text-sm font-semibold text-content-primary tracking-heading">
              Semantic Matcher Retrieval Metrics
            </h3>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            Siamese Bi-Encoder vs Baselines
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-[11px] text-content-muted uppercase">
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Val MRR</th>
                <th className="py-2.5 px-3">Val nDCG@10</th>
                <th className="py-2.5 px-3">Val R@5</th>
                <th className="py-2.5 px-3">Val R@10</th>
                <th className="py-2.5 px-3">Test MRR</th>
                <th className="py-2.5 px-3">Test nDCG@10</th>
                <th className="py-2.5 px-3">Test R@5</th>
                <th className="py-2.5 px-3">Test R@10</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {data.matcher.models.map((m) => (
                <tr key={m.key} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-3 font-medium text-content-primary flex items-center gap-2">
                    <span>{m.model}</span>
                    {m.served && (
                      <Badge variant="good" size="sm">
                        Served
                      </Badge>
                    )}
                  </td>
                  <td className="py-3 px-3 text-content-secondary">{m.val_mrr}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.val_ndcg10}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.val_recall5}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.val_recall10}</td>
                  <td className="py-3 px-3 text-content-primary font-semibold">{m.test_mrr}</td>
                  <td className="py-3 px-3 text-content-primary">{m.test_ndcg10}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.test_recall5}</td>
                  <td className="py-3 px-3 text-content-secondary">{m.test_recall10}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Collapsible Per-Category Matcher Results */}
        {data.matcher.categories.length > 0 && (
          <div className="pt-2">
            <button
              onClick={() => setShowCategoryMatcher(!showCategoryMatcher)}
              className="flex items-center gap-2 text-xs font-mono text-sand hover:text-content-primary transition-colors py-1"
            >
              {showCategoryMatcher ? (
                <ChevronDown strokeWidth={1.5} className="w-4 h-4" />
              ) : (
                <ChevronRight strokeWidth={1.5} className="w-4 h-4" />
              )}
              <span>
                {showCategoryMatcher ? "Hide" : "Show"} Per-Category Breakdown (15 Categories)
              </span>
            </button>

            {showCategoryMatcher && (
              <div className="mt-3 overflow-x-auto border border-border/80 rounded-xl p-3 bg-surface/50">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-border text-[10px] text-content-muted uppercase">
                      <th className="py-2 px-2">Category</th>
                      <th className="py-2 px-2">Val N</th>
                      <th className="py-2 px-2">Test N</th>
                      <th className="py-2 px-2">TF-IDF Val</th>
                      <th className="py-2 px-2">ZS Val</th>
                      <th className="py-2 px-2">FT Val</th>
                      <th className="py-2 px-2">TF-IDF Test</th>
                      <th className="py-2 px-2">ZS Test</th>
                      <th className="py-2 px-2">FT Test</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40 text-[11px]">
                    {data.matcher.categories.map((c) => (
                      <tr key={c.category} className="hover:bg-white/[0.02]">
                        <td className="py-2 px-2 text-content-primary font-medium">{c.category}</td>
                        <td className="py-2 px-2 text-content-muted">{c.val_count}</td>
                        <td className="py-2 px-2 text-content-muted">{c.test_count}</td>
                        <td className="py-2 px-2 text-content-secondary">{c.val_tfidf_mrr}</td>
                        <td className="py-2 px-2 text-content-secondary">{c.val_zs_mrr}</td>
                        <td className="py-2 px-2 text-content-primary font-semibold">{c.val_ft_mrr}</td>
                        <td className="py-2 px-2 text-content-secondary">{c.test_tfidf_mrr}</td>
                        <td className="py-2 px-2 text-content-secondary">{c.test_zs_mrr}</td>
                        <td className="py-2 px-2 text-content-primary font-semibold">{c.test_ft_mrr}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <p className="text-[11px] text-content-muted leading-relaxed pt-2 border-t border-border/50">
          {data.matcher.note}
        </p>
      </Card>

      {/* 3. Scoring Sensitivity & Ablation */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-border">
          <div className="flex items-center gap-2">
            <BarChart2 strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <h3 className="text-sm font-semibold text-content-primary tracking-heading">
              Scoring Component Sensitivity & Ablation Analysis
            </h3>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            9 Rubric Weight Deltas
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-[11px] text-content-muted uppercase">
                <th className="py-2.5 px-3">Component</th>
                <th className="py-2.5 px-3">Weight</th>
                <th className="py-2.5 px-3">Mean Original Score</th>
                <th className="py-2.5 px-3">Mean Ablated Score</th>
                <th className="py-2.5 px-3">Mean Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {data.ablation.components.map((a) => (
                <tr key={a.component} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 px-3 text-content-primary font-medium">{a.component}</td>
                  <td className="py-2.5 px-3 text-content-muted">{a.original_weight}</td>
                  <td className="py-2.5 px-3 text-content-secondary">{a.mean_original}</td>
                  <td className="py-2.5 px-3 text-content-secondary">{a.mean_ablated}</td>
                  <td
                    className={`py-2.5 px-3 font-semibold ${
                      a.mean_delta.startsWith("+") ? "text-semantic-good" : "text-content-primary"
                    }`}
                  >
                    {a.mean_delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-[11px] text-content-muted leading-relaxed pt-2 border-t border-border/50">
          {data.ablation.note}
        </p>
      </Card>

      {/* 4. Explainability (XAI) Faithfulness */}
      <Card glass={true} className="space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-border">
          <div className="flex items-center gap-2">
            <CheckCircle2 strokeWidth={1.5} className="w-4 h-4 text-sand" />
            <h3 className="text-sm font-semibold text-content-primary tracking-heading">
              Explainability (XAI) Faithfulness & Transfer Validation
            </h3>
          </div>
          <span className="text-[11px] font-mono text-content-muted">
            LOSO & Word Deletion Audits
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-[11px] text-content-muted uppercase">
                <th className="py-2.5 px-3">Attribution Level</th>
                <th className="py-2.5 px-3">Method & Sample</th>
                <th className="py-2.5 px-3">Observed Drops (Top vs Random)</th>
                <th className="py-2.5 px-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {data.xai.rows.map((x, idx) => (
                <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-3 text-content-primary font-medium">{x.attribution_level}</td>
                  <td className="py-3 px-3 text-content-secondary text-[11px]">{x.method}</td>
                  <td className="py-3 px-3 text-content-primary font-semibold text-[11px]">{x.observed_drops}</td>
                  <td className="py-3 px-3">
                    <Badge
                      variant={x.verdict.includes("Faithful") ? "good" : "warn"}
                      size="sm"
                    >
                      {x.verdict}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-[11px] text-content-muted leading-relaxed pt-2 border-t border-border/50">
          {data.xai.note}
        </p>
      </Card>
    </div>
  );
};

