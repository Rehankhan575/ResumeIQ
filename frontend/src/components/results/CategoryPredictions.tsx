import React from "react";
import { CategoryPrediction } from "../../types/api";

interface CategoryPredictionsProps {
  categories: CategoryPrediction[];
}

export const CategoryPredictions: React.FC<CategoryPredictionsProps> = ({ categories }) => {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between pb-1 border-b border-border">
        <span className="text-xs font-mono uppercase tracking-wider text-content-muted">
          Classifier Inference
        </span>
        <span className="text-[11px] font-mono text-content-muted">
          Softmax Confidence
        </span>
      </div>

      <div className="space-y-3">
        {categories.map((cat) => {
          const pct = Math.round(cat.probability * 100);
          return (
            <div key={cat.category} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-content-secondary font-medium truncate">
                  {cat.category.replace("-", " ")}
                </span>
                <span className="font-mono text-content-primary">{pct}%</span>
              </div>
              <div className="w-full h-1 bg-surface rounded-full overflow-hidden">
                <div
                  className="h-full bg-content-primary rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
