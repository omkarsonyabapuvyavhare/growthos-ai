"use client";

import Link from "next/link";
import { AlertCircle, Inbox, RefreshCw } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function PageSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-4" aria-busy="true" aria-live="polite">
      <Skeleton className="h-10 w-2/3" />
      <Skeleton className="h-5 w-1/2" />
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton key={index} className="h-28 w-full" />
      ))}
    </div>
  );
}

export function ErrorPanel({
  title = "Something went wrong",
  message,
  onRetry,
  recoveryHref,
  recoveryLabel = "Go back",
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
  recoveryHref?: string;
  recoveryLabel?: string;
}) {
  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="mt-2 space-y-3">
        <p>{message}</p>
        <div className="flex flex-wrap gap-2">
          {onRetry ? (
            <Button type="button" variant="outline" size="sm" onClick={onRetry}>
              <RefreshCw className="h-4 w-4" />
              Retry
            </Button>
          ) : null}
          {recoveryHref ? (
            <Button asChild type="button" variant="secondary" size="sm">
              <Link href={recoveryHref}>{recoveryLabel}</Link>
            </Button>
          ) : null}
        </div>
      </AlertDescription>
    </Alert>
  );
}

export function EmptyPanel({
  title,
  description,
  actionHref,
  actionLabel,
}: {
  title: string;
  description: string;
  actionHref?: string;
  actionLabel?: string;
}) {
  return (
    <Card className="border-dashed">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Inbox className="h-5 w-5 text-blue-600" aria-hidden />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-slate-600 dark:text-slate-300">{description}</p>
        {actionHref && actionLabel ? (
          <Button asChild>
            <Link href={actionHref}>{actionLabel}</Link>
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function MissingUserPanel() {
  return (
    <EmptyPanel
      title="No growth profile yet"
      description="Start your journey to create a personalized roadmap and daily plan."
      actionHref="/onboarding"
      actionLabel="Start My Growth Journey"
    />
  );
}
