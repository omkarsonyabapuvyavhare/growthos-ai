"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  EmptyPanel,
  ErrorPanel,
  MissingUserPanel,
  PageSkeleton,
} from "@/components/state-panels";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ApiClientError, getDashboard, runDemoDayLoop } from "@/lib/api";
import {
  getDemoResult,
  getSessionDays,
  getUserId,
  saveDemoResult,
  setGoalTitle,
  setProgressPercent,
  upsertSessionDay,
} from "@/lib/user-state";
import { formatLabel } from "@/lib/utils";
import type { DashboardResponse, DemoDayLoopResponse, SessionDaySummary } from "@/types";

function activityMix(tasks: { activity_type: string }[]): string {
  const counts = new Map<string, number>();
  tasks.forEach((task) => {
    const key = String(task.activity_type);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  });
  return Array.from(counts.entries())
    .map(([key, count]) => `${count} ${formatLabel(key).toLowerCase()}`)
    .join(" · ");
}

function practiceCount(tasks: { activity_type: string }[]): number {
  return tasks.filter((task) => String(task.activity_type) === "practice").length;
}

export default function DashboardPage() {
  const [userId, setUserIdState] = useState<number | null>(null);
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [demo, setDemo] = useState<DemoDayLoopResponse | null>(null);
  const [sessionDays, setSessionDays] = useState<SessionDaySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [demoLoading, setDemoLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);

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
      const dashboard = await getDashboard(id);
      setData(dashboard);
      if (dashboard.active_goal) setGoalTitle(dashboard.active_goal.title);
      setProgressPercent(dashboard.overall_progress_percent);
      setSessionDays(getSessionDays());
      setDemo(getDemoResult());
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not load dashboard.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const comparison = useMemo(() => {
    if (demo) {
      return {
        source: "demo" as const,
        day1: {
          mood: demo.day1_checkin.mood,
          energy: demo.day1_checkin.energy_level,
          available_minutes: demo.day1_checkin.available_minutes,
          task_count: demo.day1_tasks.length,
          practice_task_count: practiceCount(demo.day1_tasks),
          completion: demo.reflection.completion_status,
          summary: demo.day1_plan.summary,
          activity_mix: activityMix(demo.day1_tasks),
        },
        day2: {
          mood: demo.day2_checkin.mood,
          energy: demo.day2_checkin.energy_level,
          available_minutes: demo.day2_checkin.available_minutes,
          task_count: demo.day2_tasks.length,
          practice_task_count: practiceCount(demo.day2_tasks),
          completion: "planned",
          summary: demo.day2_plan.summary,
          activity_mix: activityMix(demo.day2_tasks),
        },
        why: demo.adaptation_explanation,
        learned:
          demo.reflection_insight ||
          demo.reflection.insight ||
          "Session evidence was saved for the next plan.",
        patterns: demo.detected_patterns,
        goalUnchanged: demo.goal_unchanged,
        earlySignal: demo.is_early_signal,
        nextAction: demo.recommended_next_action,
      };
    }
    if (sessionDays.length >= 2) {
      const sorted = [...sessionDays].sort((a, b) => a.plan_date.localeCompare(b.plan_date));
      const day1 = sorted[sorted.length - 2];
      const day2 = sorted[sorted.length - 1];
      return {
        source: "organic" as const,
        day1: {
          mood: String(day1.mood),
          energy: String(day1.energy ?? "unknown"),
          available_minutes: day1.available_minutes,
          task_count: day1.task_count,
          practice_task_count: day1.practice_task_count ?? 0,
          completion: String(day1.completion_status ?? "unknown"),
          summary: day1.plan_summary ?? "",
          activity_mix: day1.activity_mix ?? `${day1.task_count} tasks`,
        },
        day2: {
          mood: String(day2.mood),
          energy: String(day2.energy ?? "unknown"),
          available_minutes: day2.available_minutes,
          task_count: day2.task_count,
          practice_task_count: day2.practice_task_count ?? 0,
          completion: String(day2.completion_status ?? "planned"),
          summary: day2.plan_summary ?? "",
          activity_mix: day2.activity_mix ?? `${day2.task_count} tasks`,
        },
        why:
          data?.plan_change_explanation ||
          "GrowthOS adapted using your recent session evidence.",
        learned:
          data?.growthos_knows_you[0] ||
          data?.ai_insight ||
          "Keep reflecting so GrowthOS can learn what works for you.",
        patterns: data?.detected_patterns ?? [],
        goalUnchanged: true,
        earlySignal: false,
        nextAction: data?.recommended_next_action,
      };
    }
    return null;
  }, [demo, sessionDays, data]);

  async function handleDemo() {
    if (!userId) return;
    setDemoLoading(true);
    setDemoError(null);
    try {
      const result = await runDemoDayLoop(userId, {});
      saveDemoResult(result);
      setDemo(result);
      upsertSessionDay({
        plan_date: result.day1_plan.plan_date,
        mood: result.day1_checkin.mood,
        energy: result.day1_checkin.energy_level,
        available_minutes: result.day1_checkin.available_minutes,
        task_count: result.day1_tasks.length,
        practice_task_count: practiceCount(result.day1_tasks),
        completion_status: result.reflection.completion_status,
        plan_summary: result.day1_plan.summary,
        activity_mix: activityMix(result.day1_tasks),
      });
      upsertSessionDay({
        plan_date: result.day2_plan.plan_date,
        mood: result.day2_checkin.mood,
        energy: result.day2_checkin.energy_level,
        available_minutes: result.day2_checkin.available_minutes,
        task_count: result.day2_tasks.length,
        practice_task_count: practiceCount(result.day2_tasks),
        plan_summary: result.day2_plan.summary,
        activity_mix: activityMix(result.day2_tasks),
      });
      setSessionDays(getSessionDays());
      await load();
    } catch (err) {
      setDemoError(err instanceof ApiClientError ? err.message : "Demo failed.");
    } finally {
      setDemoLoading(false);
    }
  }

  if (loading) return <PageSkeleton rows={6} />;
  if (userId === null) return <MissingUserPanel />;
  if (error) {
    return (
      <ErrorPanel
        title="Dashboard unavailable"
        message={error}
        onRetry={() => void load()}
        recoveryHref="/onboarding"
        recoveryLabel="Start over"
      />
    );
  }
  if (!data) {
    return (
      <EmptyPanel
        title="Empty dashboard"
        description="Complete onboarding to see your growth picture."
        actionHref="/onboarding"
        actionLabel="Start onboarding"
      />
    );
  }

  const goalTitle = data.active_goal?.title ?? "Not set";
  const knowsYou = data.growthos_knows_you;
  const nextAction =
    comparison?.nextAction ||
    data.recommended_next_action ||
    "Complete today’s check-in to keep adapting.";

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-blue-100 bg-white/80 p-6 shadow-sm backdrop-blur-sm dark:border-slate-800 dark:bg-slate-900/70 sm:p-8">
        <p className="text-sm font-medium uppercase tracking-[0.18em] text-blue-600 dark:text-blue-300">
          Dashboard
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Your growth at a glance</h1>
        <p className="mt-3 text-lg text-slate-800 dark:text-slate-100">
          <span className="font-medium text-slate-500 dark:text-slate-400">Original goal:</span>{" "}
          {goalTitle}
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge variant="secondary">
            Milestone: {data.current_milestone?.title ?? "Not set"}
          </Badge>
          <Badge variant="outline">
            Progress: {Math.round(data.overall_progress_percent)}%
          </Badge>
          <Badge>
            {data.completed_sessions} session{data.completed_sessions === 1 ? "" : "s"}
          </Badge>
          {comparison?.goalUnchanged ? (
            <Badge variant="success">Goal unchanged</Badge>
          ) : null}
        </div>
        <div className="mt-4">
          <Progress value={data.overall_progress_percent} aria-label="Roadmap progress" />
        </div>
      </section>

      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="text-2xl">Day 1 vs Day 2</CardTitle>
          <CardDescription>
            Mood changes today’s method and duration - not your long-term goal.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {comparison ? (
            <div className="space-y-4">
              {comparison.source === "demo" ? (
                <Badge variant="warning">Demo-generated comparison</Badge>
              ) : (
                <Badge variant="secondary">From your recent sessions</Badge>
              )}
              <div className="grid gap-4 md:grid-cols-2">
                <DayColumn
                  title="Day 1"
                  mood={comparison.day1.mood}
                  energy={comparison.day1.energy}
                  minutes={comparison.day1.available_minutes}
                  tasks={comparison.day1.task_count}
                  practice={comparison.day1.practice_task_count}
                  completion={comparison.day1.completion}
                  mix={comparison.day1.activity_mix}
                  summary={comparison.day1.summary}
                />
                <DayColumn
                  title="Day 2"
                  mood={comparison.day2.mood}
                  energy={comparison.day2.energy}
                  minutes={comparison.day2.available_minutes}
                  tasks={comparison.day2.task_count}
                  practice={comparison.day2.practice_task_count}
                  completion={comparison.day2.completion}
                  mix={comparison.day2.activity_mix}
                  summary={comparison.day2.summary}
                  highlight
                />
              </div>
            </div>
          ) : (
            <EmptyPanel
              title="Complete another session to see how GrowthOS adapts."
              description="After two days of evidence - or the hackathon demo - you will see a side-by-side comparison here."
              actionHref="/check-in"
              actionLabel="Start today’s check-in"
            />
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-blue-200 bg-gradient-to-br from-blue-50/90 to-white dark:border-blue-900 dark:from-slate-900 dark:to-slate-950">
          <CardHeader>
            <CardTitle>What GrowthOS learned</CardTitle>
            <CardDescription>Backend evidence only - never invented here.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {comparison?.learned ? <p>{comparison.learned}</p> : null}
            {(comparison?.patterns?.length ? comparison.patterns : knowsYou).length > 0 ? (
              <ul className="space-y-2">
                {(comparison?.patterns?.length ? comparison.patterns : knowsYou).map((item) => (
                  <li
                    key={item}
                    className="rounded-xl border border-blue-100/80 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-950/50"
                  >
                    {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500">Complete a reflection to surface patterns.</p>
            )}
            {comparison?.earlySignal ? (
              <Badge variant="warning">Early signal - preferences stay cautious</Badge>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Why your plan changed</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <AlertTitle>Adaptation explanation</AlertTitle>
              <AlertDescription>
                {comparison?.why ||
                  data.plan_change_explanation ||
                  "Adaptation explanations appear after your first reflection."}
              </AlertDescription>
            </Alert>
            <div>
              <p className="text-sm font-medium">Next recommended action</p>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{nextAction}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-blue-100 dark:border-slate-800">
        <CardHeader>
          <CardTitle>GrowthOS Knows You</CardTitle>
          <CardDescription>
            Progress is based on completion, usefulness, focus, and difficulty fit.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {knowsYou.length > 0 ? (
            <ul className="space-y-2 text-sm">
              {knowsYou.map((item) => (
                <li
                  key={item}
                  className="rounded-xl border border-blue-100/80 px-3 py-2 dark:border-slate-800"
                >
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-500">
              Insights appear after GrowthOS has enough session evidence.
            </p>
          )}
        </CardContent>
      </Card>

      <Card className="border-dashed border-amber-300 dark:border-amber-800">
        <CardHeader>
          <CardTitle>Hackathon demo</CardTitle>
          <CardDescription>
            Fast Day 1 to Day 2 loop for judges. Runs real workflows - not a fake result.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-slate-600 dark:text-slate-300">
            Day 1: tired · low energy · 15 minutes · short plan · partial completion · longer
            resource less useful · practice useful. Day 2: focused · higher energy · 30 minutes
            · more practice · changed plan. Goal stays the same.
          </p>
          {demoError ? <ErrorPanel title="Demo failed" message={demoError} /> : null}
          <Button
            type="button"
            size="lg"
            disabled={demoLoading}
            onClick={() => void handleDemo()}
          >
            {demoLoading ? "Running demo..." : "Run Day 1 to Day 2 Demo"}
          </Button>
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-3">
        <Button asChild>
          <Link href="/check-in">Continue with today’s check-in</Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/roadmap">View roadmap</Link>
        </Button>
      </div>
    </div>
  );
}

function DayColumn({
  title,
  mood,
  energy,
  minutes,
  tasks,
  practice,
  completion,
  mix,
  summary,
  highlight = false,
}: {
  title: string;
  mood: string;
  energy: string;
  minutes: number;
  tasks: number;
  practice: number;
  completion: string;
  mix: string;
  summary: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={
        highlight
          ? "rounded-2xl border-2 border-blue-500 bg-blue-50/70 p-4 dark:border-blue-400 dark:bg-blue-950/30"
          : "rounded-2xl border border-blue-100/80 bg-white/70 p-4 dark:border-slate-800 dark:bg-slate-950/50"
      }
    >
      <h3 className="mb-3 font-semibold">{title}</h3>
      <dl className="space-y-2 text-sm">
        <Row label="Mood" value={formatLabel(mood)} />
        <Row label="Energy" value={formatLabel(energy)} />
        <Row label="Available time" value={`${minutes} min`} />
        <Row label="Task count" value={String(tasks)} />
        <Row label="Practice tasks" value={String(practice)} />
        <Row label="Completion" value={formatLabel(completion)} />
        <Row label="Activity mix" value={mix || "n/a"} />
      </dl>
      {summary ? <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{summary}</p> : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-medium">{value}</dd>
    </div>
  );
}
