"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ExternalLink } from "lucide-react";

import {
  EmptyPanel,
  ErrorPanel,
  MissingUserPanel,
  PageSkeleton,
} from "@/components/state-panels";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiClientError, getDashboard, getTodayPlan, updateTask } from "@/lib/api";
import { getUserId, setCurrentPlanId } from "@/lib/user-state";
import { formatLabel } from "@/lib/utils";
import type { DailyPlanResponse, DailyTaskResponse, TaskStatus } from "@/types";

export default function TodayPage() {
  const router = useRouter();
  const [userId, setUserIdState] = useState<number | null>(null);
  const [plan, setPlan] = useState<DailyPlanResponse | null>(null);
  const [mood, setMood] = useState<string | null>(null);
  const [milestoneTitle, setMilestoneTitle] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyTaskId, setBusyTaskId] = useState<number | null>(null);
  const [drafts, setDrafts] = useState<
    Record<number, { percent: string; minutes: string }>
  >({});

  async function load() {
    const id = getUserId();
    setUserIdState(id);
    if (id === null) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [today, dashboard] = await Promise.all([
        getTodayPlan(id),
        getDashboard(id).catch(() => null),
      ]);
      setPlan(today);
      setCurrentPlanId(today.id);
      setMood(dashboard?.today_mood ?? null);
      setMilestoneTitle(dashboard?.current_milestone?.title ?? null);
      const nextDrafts: Record<number, { percent: string; minutes: string }> = {};
      today.tasks.forEach((task) => {
        nextDrafts[task.id] = { percent: "100", minutes: String(task.estimated_minutes) };
      });
      setDrafts(nextDrafts);
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not load today’s plan.");
      setPlan(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const tasks = useMemo(() => {
    if (!plan) return [];
    return plan.tasks
      .slice()
      .sort((a, b) => a.sequence_number - b.sequence_number)
      .slice(0, 5);
  }, [plan]);

  async function markTask(task: DailyTaskResponse, status: TaskStatus) {
    if (!userId) return;
    setBusyTaskId(task.id);
    setError(null);
    try {
      const draft = drafts[task.id];
      const updated = await updateTask(userId, task.id, {
        status,
        completion_percent:
          status === "completed"
            ? Number(draft?.percent || 100)
            : status === "skipped"
              ? 0
              : Number(draft?.percent || 50),
        duration_minutes: Number(draft?.minutes || task.estimated_minutes),
      });
      setPlan((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          tasks: prev.tasks.map((item) => (item.id === updated.id ? updated : item)),
        };
      });
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not update task.");
    } finally {
      setBusyTaskId(null);
    }
  }

  if (loading) return <PageSkeleton />;
  if (userId === null) return <MissingUserPanel />;
  if (error && !plan) {
    return (
      <ErrorPanel
        title="No plan for today"
        message={error}
        onRetry={() => void load()}
        recoveryHref="/check-in"
        recoveryLabel="Go to check-in"
      />
    );
  }
  if (!plan) {
    return (
      <EmptyPanel
        title="No plan for today"
        description="Check in to generate a focused session shaped by how you feel."
        actionHref="/check-in"
        actionLabel="Start check-in"
      />
    );
  }

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="text-sm font-medium uppercase tracking-[0.18em] text-blue-600 dark:text-blue-300">
          Today’s plan
        </p>
        <h1 className="text-3xl font-semibold tracking-tight">{plan.summary}</h1>
        <div className="flex flex-wrap gap-2">
          {mood ? <Badge variant="secondary">Mood: {formatLabel(mood)}</Badge> : null}
          {milestoneTitle ? (
            <Badge variant="outline">Milestone: {milestoneTitle}</Badge>
          ) : null}
          <Badge>{plan.total_estimated_minutes} min total</Badge>
          <Badge variant="secondary">{tasks.length} focused tasks</Badge>
        </div>
        {plan.guidance_tone ? (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium">Guidance tone:</span> {plan.guidance_tone}
          </p>
        ) : null}
        {plan.mood_influence_summary ? (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium">Mood influence:</span> {plan.mood_influence_summary}
          </p>
        ) : null}
        {plan.adaptation_explanation ? (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium">Why your plan looks like this:</span>{" "}
            {plan.adaptation_explanation}
          </p>
        ) : null}
      </div>

      {error ? <ErrorPanel title="Update issue" message={error} /> : null}

      <div className="space-y-4">
        {tasks.map((task) => (
          <Card key={task.id}>
            <CardHeader>
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <CardTitle className="text-xl">
                    {task.sequence_number}. {task.title}
                  </CardTitle>
                  <CardDescription className="mt-2">{task.description}</CardDescription>
                </div>
                <Badge
                  variant={
                    task.status === "completed"
                      ? "success"
                      : task.status === "skipped"
                        ? "warning"
                        : "secondary"
                  }
                >
                  {formatLabel(task.status)}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-2 text-sm sm:grid-cols-2">
                {task.resource_title ? (
                  <p>
                    <span className="font-medium">Resource:</span> {task.resource_title}
                  </p>
                ) : null}
                {task.resource_source ? (
                  <p className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">Source:</span> {task.resource_source}
                    {task.resource_source === "YouTube" ? (
                      <Badge variant="secondary">YouTube</Badge>
                    ) : null}
                  </p>
                ) : null}
                {task.resource_channel ? (
                  <p>
                    <span className="font-medium">Channel:</span> {task.resource_channel}
                  </p>
                ) : null}
                <p>
                  <span className="font-medium">Type:</span>{" "}
                  {task.content_type || formatLabel(task.activity_type)}
                </p>
                <p>
                  <span className="font-medium">Duration:</span> {task.estimated_minutes} min
                </p>
                <p>
                  <span className="font-medium">Difficulty:</span>{" "}
                  {formatLabel(task.difficulty)}
                </p>
                <p>
                  <span className="font-medium">Expected outcome:</span> {task.expected_outcome}
                </p>
              </div>
              {task.resource_thumbnail_url ? (
                <div className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={task.resource_thumbnail_url}
                    alt=""
                    className="h-40 w-full object-cover"
                  />
                </div>
              ) : null}
              <div className="space-y-2 rounded-xl bg-blue-50/70 p-4 text-sm dark:bg-slate-800/60">
                <p>
                  <span className="font-medium">Why selected:</span> {task.why_selected}
                </p>
                <p>
                  <span className="font-medium">Milestone connection:</span>{" "}
                  {task.milestone_connection}
                </p>
                <p>
                  <span className="font-medium">Mood rationale:</span> {task.mood_rationale}
                </p>
              </div>

              {task.resource_url ? (
                <Button asChild variant="outline" size="sm">
                  <a href={task.resource_url} target="_blank" rel="noreferrer">
                    Open Resource
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </Button>
              ) : null}

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor={`percent-${task.id}`}>Completion %</Label>
                  <Input
                    id={`percent-${task.id}`}
                    type="number"
                    min={0}
                    max={100}
                    value={drafts[task.id]?.percent ?? "100"}
                    onChange={(e) =>
                      setDrafts((prev) => ({
                        ...prev,
                        [task.id]: {
                          percent: e.target.value,
                          minutes: prev[task.id]?.minutes ?? String(task.estimated_minutes),
                        },
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`minutes-${task.id}`}>Actual minutes</Label>
                  <Input
                    id={`minutes-${task.id}`}
                    type="number"
                    min={0}
                    value={drafts[task.id]?.minutes ?? String(task.estimated_minutes)}
                    onChange={(e) =>
                      setDrafts((prev) => ({
                        ...prev,
                        [task.id]: {
                          percent: prev[task.id]?.percent ?? "100",
                          minutes: e.target.value,
                        },
                      }))
                    }
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  disabled={busyTaskId === task.id}
                  onClick={() => void markTask(task, "completed")}
                >
                  Mark complete
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  disabled={busyTaskId === task.id}
                  onClick={() => void markTask(task, "in_progress")}
                >
                  Mark partial
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={busyTaskId === task.id}
                  onClick={() => void markTask(task, "skipped")}
                >
                  Mark skipped
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="flex flex-wrap gap-3">
        <Button type="button" size="lg" onClick={() => router.push("/reflect")}>
          Finish session
        </Button>
        <Button asChild variant="outline" size="lg">
          <Link href="/check-in">New check-in</Link>
        </Button>
      </div>
    </div>
  );
}
