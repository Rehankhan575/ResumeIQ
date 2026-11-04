import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  size = "md",
  loading = false,
  className,
  children,
  disabled,
  ...props
}) => {
  const baseClasses =
    "inline-flex items-center justify-center font-medium transition-all duration-200 outline-none select-none disabled:opacity-40 disabled:cursor-not-allowed";

  const sizeClasses = {
    sm: "h-8 px-3 text-xs rounded-full gap-1.5",
    md: "h-11 px-5 text-sm rounded-full gap-2", // 44px high
    lg: "h-12 px-6 text-base rounded-full gap-2.5",
  };

  const variantClasses = {
    // Solid #F5F5F4 background, #0A0A0B text, fully rounded, no color gradients
    primary:
      "bg-content-primary text-canvas hover:bg-white active:scale-[0.99] shadow-sm",
    // Surface with 1px border
    secondary:
      "bg-surface-card border border-border text-content-primary hover:bg-surface-subtle hover:border-border-strong active:scale-[0.99]",
    // Ghost
    ghost:
      "bg-transparent text-content-secondary hover:text-content-primary hover:bg-white/[0.04]",
  };

  return (
    <button
      className={twMerge(
        clsx(baseClasses, sizeClasses[size], variantClasses[variant], className)
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="inline-flex items-center gap-2">
          <svg
            className="animate-spin -ml-1 mr-1 h-4 w-4 text-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="3"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span>Analyzing...</span>
        </span>
      ) : (
        children
      )}
    </button>
  );
};

