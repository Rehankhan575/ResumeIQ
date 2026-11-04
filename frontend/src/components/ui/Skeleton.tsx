import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className, ...props }) => {
  return (
    <div
      className={twMerge(
        clsx("animate-pulse rounded-lg bg-surface-subtle", className)
      )}
      {...props}
    />
  );
};

export const ResultsSkeleton: React.FC = () => {
  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      <div className="h-10 w-64 bg-surface-subtle rounded-xl animate-pulse" />
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 h-96 bg-surface-card rounded-2xl animate-pulse" />
        <div className="lg:col-span-7 h-96 bg-surface-card rounded-2xl animate-pulse" />
        <div className="lg:col-span-7 h-80 bg-surface-card rounded-2xl animate-pulse" />
        <div className="lg:col-span-5 h-80 bg-surface-card rounded-2xl animate-pulse" />
        <div className="lg:col-span-12 h-64 bg-surface-card rounded-2xl animate-pulse" />
      </div>
    </div>
  );
};
