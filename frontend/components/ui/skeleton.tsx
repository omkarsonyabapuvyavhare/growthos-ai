import * as React from "react";

import { cn } from "@/lib/utils";

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-xl bg-blue-100/80 dark:bg-slate-800",
        className,
      )}
      {...props}
    />
  );
}

export { Skeleton };
