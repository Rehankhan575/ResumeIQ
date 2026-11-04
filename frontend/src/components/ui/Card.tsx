import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glass?: boolean;
  surface?: "surface" | "card" | "subtle";
}

export const Card: React.FC<CardProps> = ({
  glass = false,
  surface = "card",
  className,
  children,
  ...props
}) => {
  const surfaceClasses = {
    surface: "bg-surface border-border",
    card: "bg-surface-card border-border",
    subtle: "bg-surface-subtle border-border",
  };

  const glassClasses =
    "glass-panel rounded-2xl p-6 relative overflow-hidden transition-all duration-200";

  const flatClasses =
    "rounded-2xl border p-6 relative transition-all duration-200";

  return (
    <div
      className={twMerge(
        clsx(glass ? glassClasses : [flatClasses, surfaceClasses[surface]], className)
      )}
      {...props}
    >
      {children}
    </div>
  );
};

