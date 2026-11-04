import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "neutral" | "sand" | "good" | "warn" | "bad";
  size?: "sm" | "md";
}

export const Badge: React.FC<BadgeProps> = ({
  variant = "neutral",
  size = "md",
  className,
  children,
  ...props
}) => {
  const base = "inline-flex items-center font-mono font-medium rounded-full border tracking-wide";

  const sizeClasses = {
    sm: "px-2 py-0.5 text-[11px]",
    md: "px-2.5 py-1 text-xs",
  };

  const variantClasses = {
    neutral: "bg-surface-subtle text-content-secondary border-border",
    sand: "bg-sand-dim text-sand border-sand/20",
    good: "bg-semantic-good-dim text-semantic-good border-semantic-good/20",
    warn: "bg-semantic-warn-dim text-semantic-warn border-semantic-warn/20",
    bad: "bg-semantic-bad-dim text-semantic-bad border-semantic-bad/20",
  };

  return (
    <span
      className={twMerge(clsx(base, sizeClasses[size], variantClasses[variant], className))}
      {...props}
    >
      {children}
    </span>
  );
};

