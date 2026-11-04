import React from "react";
import { AlertCircle, RotateCcw } from "lucide-react";
import { Button } from "./Button";

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  message,
  onRetry,
  className = "",
}) => {
  return (
    <div
      role="alert"
      className={`p-4 rounded-xl bg-semantic-bad-dim border border-semantic-bad/30 flex items-center justify-between gap-4 text-xs ${className}`}
    >
      <div className="flex items-center gap-2.5">
        <AlertCircle strokeWidth={1.5} className="w-4 h-4 text-semantic-bad shrink-0" />
        <span className="text-content-primary font-medium">{message}</span>
      </div>

      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          className="text-xs h-8 shrink-0 gap-1.5"
        >
          <RotateCcw strokeWidth={1.5} className="w-3.5 h-3.5" />
          <span>Retry</span>
        </Button>
      )}
    </div>
  );
};

