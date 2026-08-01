"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ErrorPanel, MissingUserPanel } from "@/components/state-panels";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
import { ApiClientError, createDailyPlan } from "@/lib/api";
import {
  getUserId,
  setCurrentPlanId,
  upsertSessionDay,
} from "@/lib/user-state";
import { cn, formatLabel } from "@/lib/utils";
import type { ActivityType, EnergyLevel, Mood } from "@/types";

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

export default function CheckInPage() {
  const router = useRouter();
  const [userId, setUserIdState] = useState<number | null>(null);
  const [mood, setMood] = useState<Mood | "">("");
  const [energy, setEnergy] = useState<EnergyLevel | "">("");
  const [focus, setFocus] = useState(3);
  const [minutes, setMinutes] = useState(20);
  const [activity, setActivity] = useState<ActivityType | "">("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setUserIdState(getUserId());
  }, []);

  if (userId === null) {
    return <MissingUserPanel />;
  }

  async function handleSubmit() {
    const id = userId;
    if (id === null) return;
    if (!mood || !energy || !activity) {
      setError("Choose mood, energy, and preferred activity to continue.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const result = await createDailyPlan(id, {
        mood,
        energy_level: energy,
        focus_level: focus,
        available_minutes: minutes,
        preferred_activity: activity,
        notes: notes.trim(),
        refresh: true,
      });
      setCurrentPlanId(result.plan.id);
      upsertSessionDay({
        plan_date: result.plan.plan_date,
        mood: result.checkin.mood,
        energy: result.checkin.energy_level,
        available_minutes: result.checkin.available_minutes,
        task_count: result.tasks.length,
        practice_task_count: result.tasks.filter(
          (task) => task.activity_type === "practice",
        ).length,
        plan_summary: result.plan.summary,
        activity_mix: result.tasks
          .map((task) => task.activity_type)
          .join(", "),
      });
      router.push("/today");
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not create today’s plan.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Daily check-in</h1>
        <p className="text-slate-600 dark:text-slate-300">
          A quick pulse check so GrowthOS can shape today’s session.
        </p>
      </div>

      <Alert>
        <AlertTitle>Mood shapes the day  -  not the destination</AlertTitle>
        <AlertDescription>
          Your mood changes today’s format, duration, difficulty, and task count  -  not your
          long-term goal.
        </AlertDescription>
      </Alert>

      {error ? <ErrorPanel title="Check-in failed" message={error} /> : null}

      <Card>
        <CardHeader>
          <CardTitle>How are you feeling?</CardTitle>
          <CardDescription>Pick the mood that best matches right now.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          {MOODS.map((option) => {
            const selected = mood === option;
            return (
              <button
                key={option}
                type="button"
                onClick={() => setMood(option)}
                className={cn(
                  "rounded-2xl border px-4 py-4 text-left transition-all",
                  selected
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-blue-100 bg-white/80 hover:border-blue-300 dark:border-slate-700 dark:bg-slate-950",
                )}
                aria-pressed={selected}
              >
                <span className="block font-medium">{formatLabel(option)}</span>
              </button>
            );
          })}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Capacity & preference</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="energy">Energy level</Label>
            <Select value={energy} onValueChange={(value) => setEnergy(value as EnergyLevel)}>
              <SelectTrigger id="energy" aria-label="Energy level">
                <SelectValue placeholder="Select energy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="focus">Focus level</Label>
              <span className="text-sm font-medium">{focus}/5</span>
            </div>
            <Slider
              id="focus"
              min={1}
              max={5}
              step={1}
              value={[focus]}
              onValueChange={(value) => setFocus(value[0] ?? 3)}
              aria-label="Focus level"
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="minutes">Available minutes</Label>
              <span className="text-sm font-medium">{minutes} min</span>
            </div>
            <Slider
              id="minutes"
              min={5}
              max={90}
              step={5}
              value={[minutes]}
              onValueChange={(value) => setMinutes(value[0] ?? 20)}
              aria-label="Available minutes"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="activity">Preferred activity</Label>
            <Select
              value={activity}
              onValueChange={(value) => setActivity(value as ActivityType)}
            >
              <SelectTrigger id="activity" aria-label="Preferred activity">
                <SelectValue placeholder="Select activity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="watch">Watch</SelectItem>
                <SelectItem value="read">Read</SelectItem>
                <SelectItem value="listen">Listen</SelectItem>
                <SelectItem value="practice">Practice</SelectItem>
                <SelectItem value="review">Review</SelectItem>
                <SelectItem value="mixed">Mixed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Optional note</Label>
            <Textarea
              id="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Anything GrowthOS should know for today?"
            />
          </div>

          <Button
            type="button"
            size="lg"
            className="w-full sm:w-auto"
            disabled={submitting}
            onClick={() => void handleSubmit()}
          >
            {submitting ? "Building today’s plan…" : "Generate today’s plan"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
