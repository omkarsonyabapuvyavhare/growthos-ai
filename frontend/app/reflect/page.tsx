"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  EmptyPanel,
  ErrorPanel,
  MissingUserPanel,
  PageSkeleton,
} from "@/components/state-panels";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { ApiClientError, getTodayPlan, submitReflection } from "@/lib/api";
import {
  getCurrentPlanId,
  getUserId,
  saveLastReflection,
  setProgressPercent,
  upsertSessionDay,
} from "@/lib/user-state";
import { formatLabel } from "@/lib/utils";
import type {
  CompletionStatus,
  DailyPlanResponse,
  DailyPostSessionWorkflowResult,
  DifficultyFeedback,
  Mood,
  ReflectionTaskUpdate,
  TaskStatus,
} from "@/types";

const MOODS: Mood[] = [
  "focused",
  "motivated",
  "curious",
  "calm",
  "tired",
  "stressed",
  "distracted",
  "low_energy",
];

export default function ReflectPage() {
  const [userId, setUserIdState] = useState<number | null>(null);
  const [plan, setPlan] = useState<DailyPlanResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DailyPostSessionWorkflowResult | null>(null);

  const [completionStatus, setCompletionStatus] = useState<CompletionStatus>("partial");
  const [learningSummary, setLearningSummary] = useState("");
  const [focusRating, setFocusRating] = useState(3);
  const [resourceEffectiveness, setResourceEffectiveness] = useState(3);
  const [difficulty, setDifficulty] = useState<DifficultyFeedback>("suitable");
  const [moodMatch, setMoodMatch] = useState("true");
  const [distractions, setDistractions] = useState("");
  const [wantsSimilar, setWantsSimilar] = useState("true");
  const [moodAfter, setMoodAfter] = useState<Mood | "">("");
  const [actualMinutes, setActualMinutes] = useState("");

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
      const today = await getTodayPlan(id);
      setPlan(today);
      const completed = today.tasks.filter((task) => task.status === "completed").length;
      const total = today.tasks.length || 1;
      if (completed === total) setCompletionStatus("completed");
      else if (completed === 0) setCompletionStatus("partial");
      else setCompletionStatus("partial");
      setActualMinutes(String(today.total_estimated_minutes));
      const learnedBits = today.tasks
        .filter((task) => task.status === "completed" || task.status === "in_progress")
        .map((task) => task.title);
      if (learnedBits.length) {
        setLearningSummary(`Worked on: ${learnedBits.join("; ")}`);
      }
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not load today’s plan.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function handleSubmit() {
    if (!userId || !plan) return;
    setSubmitting(true);
    setError(null);
    try {
      const taskUpdates: ReflectionTaskUpdate[] = plan.tasks.map((task) => {
        let status: TaskStatus = task.status;
        if (status === "pending") status = "skipped";
        return {
          task_id: task.id,
          update: {
            status,
            completion_percent:
              status === "completed" ? 100 : status === "in_progress" ? 50 : 0,
            duration_minutes: Math.max(
              1,
              Math.round(plan.total_estimated_minutes / Math.max(plan.tasks.length, 1)),
            ),
          },
        };
      });

      const payload = {
        daily_plan_id: plan.id,
        completion_status: completionStatus,
        learning_summary: learningSummary.trim(),
        focus_rating: focusRating,
        resource_effectiveness: resourceEffectiveness,
        difficulty_feedback: difficulty,
        mood_match: moodMatch === "true",
        distractions: distractions
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
        wants_similar_resources: wantsSimilar === "true",
        mood_after: moodAfter || undefined,
        task_updates: taskUpdates,
        actual_minutes_spent: actualMinutes ? Number(actualMinutes) : null,
      };

      const response = await submitReflection(userId, payload);
      saveLastReflection(response);
      setResult(response);
      setProgressPercent(response.reflection_result.roadmap_progress_after);
      upsertSessionDay({
        plan_date: plan.plan_date,
        mood: "session",
        available_minutes: plan.total_estimated_minutes,
        task_count: plan.tasks.length,
        completion_status: completionStatus,
        plan_summary: plan.summary,
      });
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Reflection failed.");
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <PageSkeleton />;
  if (userId === null) return <MissingUserPanel />;
  if (!plan && error) {
    return (
      <ErrorPanel
        title="Nothing to reflect on"
        message={error}
        onRetry={() => void load()}
        recoveryHref="/check-in"
        recoveryLabel="Create today’s plan"
      />
    );
  }
  if (!plan) {
    const planId = getCurrentPlanId();
    return (
      <EmptyPanel
        title="No session to reflect on"
        description={
          planId
            ? "Today’s plan wasn’t found. Create a fresh check-in."
            : "Finish a daily plan before reflecting."
        }
        actionHref="/check-in"
        actionLabel="Go to check-in"
      />
    );
  }

  if (result) {
    return (
      <div className="mx-auto max-w-2xl space-y-6">
        <Alert variant="success">
          <AlertTitle>What GrowthOS learned</AlertTitle>
          <AlertDescription>
            {result.reflection.insight ||
              "Your session was saved. GrowthOS will use this evidence for the next plan."}
          </AlertDescription>
        </Alert>
        <Card>
          <CardHeader>
            <CardTitle>Why your plan will change</CardTitle>
            <CardDescription>
              Your long-term goal stayed the same:{" "}
              {result.goal_unchanged ? "confirmed" : "please verify on the dashboard"}.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>{result.adaptation_explanation}</p>
            {result.adaptation.detected_patterns.length > 0 ? (
              <ul className="list-disc space-y-1 pl-5">
                {result.adaptation.detected_patterns.map((pattern) => (
                  <li key={pattern}>{pattern}</li>
                ))}
              </ul>
            ) : null}
            <Button asChild size="lg">
              <Link href="/dashboard">View dashboard</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Reflect on today’s session</h1>
        <p className="text-slate-600 dark:text-slate-300">
          Progress comes from completion and honest reflection  -  not scroll time.
        </p>
      </div>

      {error ? <ErrorPanel title="Couldn’t save reflection" message={error} /> : null}

      <Card>
        <CardHeader>
          <CardTitle>Session evidence</CardTitle>
          <CardDescription>
            Prefilling from today’s {plan.tasks.length} tasks where available.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="completion">Completion status</Label>
            <Select
              value={completionStatus}
              onValueChange={(value) => setCompletionStatus(value as CompletionStatus)}
            >
              <SelectTrigger id="completion" aria-label="Completion status">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="partial">Partial</SelectItem>
                <SelectItem value="skipped">Skipped</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="learned">What was learned</Label>
            <Textarea
              id="learned"
              value={learningSummary}
              onChange={(e) => setLearningSummary(e.target.value)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between">
              <Label>Focus rating</Label>
              <span className="text-sm font-medium">{focusRating}/5</span>
            </div>
            <Slider
              min={1}
              max={5}
              step={1}
              value={[focusRating]}
              onValueChange={(value) => setFocusRating(value[0] ?? 3)}
              aria-label="Focus rating"
            />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between">
              <Label>Resource effectiveness</Label>
              <span className="text-sm font-medium">{resourceEffectiveness}/5</span>
            </div>
            <Slider
              min={1}
              max={5}
              step={1}
              value={[resourceEffectiveness]}
              onValueChange={(value) => setResourceEffectiveness(value[0] ?? 3)}
              aria-label="Resource effectiveness"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="difficulty">Difficulty feedback</Label>
            <Select
              value={difficulty}
              onValueChange={(value) => setDifficulty(value as DifficultyFeedback)}
            >
              <SelectTrigger id="difficulty" aria-label="Difficulty feedback">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="too_easy">Too easy</SelectItem>
                <SelectItem value="suitable">Suitable</SelectItem>
                <SelectItem value="too_difficult">Too difficult</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="mood-match">Did the plan match your mood?</Label>
              <Select value={moodMatch} onValueChange={setMoodMatch}>
                <SelectTrigger id="mood-match" aria-label="Mood match">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="true">Yes</SelectItem>
                  <SelectItem value="false">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="similar">Want similar resources?</Label>
              <Select value={wantsSimilar} onValueChange={setWantsSimilar}>
                <SelectTrigger id="similar" aria-label="Wants similar resources">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="true">Yes</SelectItem>
                  <SelectItem value="false">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="distractions">Distractions (comma-separated)</Label>
            <Input
              id="distractions"
              value={distractions}
              onChange={(e) => setDistractions(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mood-after">Mood after</Label>
            <Select
              value={moodAfter}
              onValueChange={(value) => setMoodAfter(value as Mood)}
            >
              <SelectTrigger id="mood-after" aria-label="Mood after">
                <SelectValue placeholder="Optional" />
              </SelectTrigger>
              <SelectContent>
                {MOODS.map((mood) => (
                  <SelectItem key={mood} value={mood}>
                    {formatLabel(mood)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="actual-minutes">Actual minutes (optional)</Label>
            <Input
              id="actual-minutes"
              type="number"
              min={0}
              value={actualMinutes}
              onChange={(e) => setActualMinutes(e.target.value)}
            />
          </div>

          <Button
            type="button"
            size="lg"
            disabled={submitting}
            onClick={() => void handleSubmit()}
          >
            {submitting ? "Saving reflection…" : "Save reflection"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
