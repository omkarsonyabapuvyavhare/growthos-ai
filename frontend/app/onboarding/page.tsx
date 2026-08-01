"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { ErrorPanel } from "@/components/state-panels";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ApiClientError, onboardUser } from "@/lib/api";
import {
  setGoalTitle,
  setProgressPercent,
  setUserId,
} from "@/lib/user-state";
import { cn } from "@/lib/utils";
import type {
  CurrentLevel,
  LearningStyle,
  OnboardingRequest,
  PreferredLearningTime,
} from "@/types";

const FORMAT_OPTIONS = ["video", "article", "podcast", "interactive", "practice", "docs"];

type FormState = {
  display_name: string;
  learning_goal: string;
  aspiration: string;
  motivation: string;
  current_level: CurrentLevel | "";
  target_outcome: string;
  preferred_formats: string[];
  learning_style: LearningStyle | "";
  daily_available_minutes: string;
  preferred_session_minutes: string;
  attention_span_minutes: string;
  preferred_learning_time: PreferredLearningTime | "";
  habits: string;
  distractions: string;
};

const INITIAL: FormState = {
  display_name: "",
  learning_goal: "",
  aspiration: "",
  motivation: "",
  current_level: "",
  target_outcome: "",
  preferred_formats: [],
  learning_style: "",
  daily_available_minutes: "45",
  preferred_session_minutes: "20",
  attention_span_minutes: "15",
  preferred_learning_time: "",
  habits: "",
  distractions: "",
};

function splitList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const progress = useMemo(() => ((step + 1) / 3) * 100, [step]);

  function update<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
    setErrors((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }

  function toggleFormat(format: string) {
    setForm((prev) => {
      const exists = prev.preferred_formats.includes(format);
      return {
        ...prev,
        preferred_formats: exists
          ? prev.preferred_formats.filter((item) => item !== format)
          : [...prev.preferred_formats, format],
      };
    });
    setErrors((prev) => {
      const next = { ...prev };
      delete next.preferred_formats;
      return next;
    });
  }

  function validateStep(current: number): boolean {
    const nextErrors: Record<string, string> = {};
    if (current === 0) {
      if (!form.display_name.trim()) nextErrors.display_name = "Enter your name.";
      if (!form.learning_goal.trim()) nextErrors.learning_goal = "Enter any learning goal.";
      if (!form.aspiration.trim()) nextErrors.aspiration = "Share your aspiration.";
      if (!form.motivation.trim()) nextErrors.motivation = "Share what motivates you.";
    }
    if (current === 1) {
      if (!form.current_level) nextErrors.current_level = "Select your current level.";
      if (!form.target_outcome.trim()) nextErrors.target_outcome = "Describe your target outcome.";
      if (form.preferred_formats.length === 0) {
        nextErrors.preferred_formats = "Choose at least one preferred format.";
      }
      if (!form.learning_style) nextErrors.learning_style = "Select a learning style.";
    }
    if (current === 2) {
      const daily = Number(form.daily_available_minutes);
      const session = Number(form.preferred_session_minutes);
      const attention = Number(form.attention_span_minutes);
      if (!Number.isFinite(daily) || daily <= 0) {
        nextErrors.daily_available_minutes = "Enter daily minutes greater than 0.";
      }
      if (!Number.isFinite(session) || session <= 0) {
        nextErrors.preferred_session_minutes = "Enter preferred session minutes.";
      }
      if (!Number.isFinite(attention) || attention <= 0) {
        nextErrors.attention_span_minutes = "Enter attention span minutes.";
      }
      if (!form.preferred_learning_time) {
        nextErrors.preferred_learning_time = "Select a preferred learning time.";
      }
    }
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  }

  async function handleSubmit() {
    if (!validateStep(2)) return;
    setSubmitting(true);
    setSubmitError(null);
    const payload: OnboardingRequest = {
      display_name: form.display_name.trim(),
      learning_goal: form.learning_goal.trim(),
      aspiration: form.aspiration.trim(),
      motivation: form.motivation.trim(),
      current_level: form.current_level as CurrentLevel,
      target_outcome: form.target_outcome.trim(),
      preferred_formats: form.preferred_formats,
      learning_style: form.learning_style as LearningStyle,
      daily_available_minutes: Number(form.daily_available_minutes),
      preferred_session_minutes: Number(form.preferred_session_minutes),
      attention_span_minutes: Number(form.attention_span_minutes),
      preferred_learning_time: form.preferred_learning_time as PreferredLearningTime,
      habits: splitList(form.habits),
      distractions: splitList(form.distractions),
    };

    try {
      const result = await onboardUser(payload);
      setUserId(result.user.id);
      setGoalTitle(result.goal.title);
      setProgressPercent(result.roadmap.progress_percent);
      router.push("/roadmap");
    } catch (error) {
      const message =
        error instanceof ApiClientError
          ? error.message
          : "Onboarding failed. Please try again.";
      setSubmitError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Your growth profile</h1>
        <p className="text-slate-600 dark:text-slate-300">
          Tell GrowthOS what you want to learn. Your goal stays free text  -  any topic works.
        </p>
        <Progress value={progress} aria-label="Onboarding progress" />
        <p className="text-xs text-slate-500">Step {step + 1} of 3</p>
      </div>

      {submitError ? (
        <ErrorPanel title="Couldn’t start your journey" message={submitError} />
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>
            {step === 0 && "Who you are & what you want"}
            {step === 1 && "How you like to learn"}
            {step === 2 && "Time, habits & focus"}
          </CardTitle>
          <CardDescription>
            {step === 0 && "Your long-term goal will not be rewritten by mood or daily plans."}
            {step === 1 && "Formats and style help GrowthOS pick a small set of focused resources."}
            {step === 2 && "These constraints shape pacing  -  not your destination."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {step === 0 ? (
            <>
              <Field
                id="display_name"
                label="Display name"
                error={errors.display_name}
              >
                <Input
                  id="display_name"
                  value={form.display_name}
                  onChange={(e) => update("display_name", e.target.value)}
                  autoComplete="name"
                />
              </Field>
              <Field
                id="learning_goal"
                label="Learning goal (free text  -  any topic)"
                error={errors.learning_goal}
              >
                <Textarea
                  id="learning_goal"
                  value={form.learning_goal}
                  onChange={(e) => update("learning_goal", e.target.value)}
                  placeholder="e.g. Become confident presenting product demos"
                />
              </Field>
              <Field id="aspiration" label="Aspiration" error={errors.aspiration}>
                <Textarea
                  id="aspiration"
                  value={form.aspiration}
                  onChange={(e) => update("aspiration", e.target.value)}
                />
              </Field>
              <Field id="motivation" label="Motivation" error={errors.motivation}>
                <Textarea
                  id="motivation"
                  value={form.motivation}
                  onChange={(e) => update("motivation", e.target.value)}
                />
              </Field>
            </>
          ) : null}

          {step === 1 ? (
            <>
              <Field id="current_level" label="Current level" error={errors.current_level}>
                <Select
                  value={form.current_level}
                  onValueChange={(value) => update("current_level", value as CurrentLevel)}
                >
                  <SelectTrigger id="current_level" aria-label="Current level">
                    <SelectValue placeholder="Select level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="beginner">Beginner</SelectItem>
                    <SelectItem value="intermediate">Intermediate</SelectItem>
                    <SelectItem value="advanced">Advanced</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field
                id="target_outcome"
                label="Target outcome"
                error={errors.target_outcome}
              >
                <Textarea
                  id="target_outcome"
                  value={form.target_outcome}
                  onChange={(e) => update("target_outcome", e.target.value)}
                />
              </Field>
              <div className="space-y-2">
                <Label>Preferred formats</Label>
                <div className="flex flex-wrap gap-2">
                  {FORMAT_OPTIONS.map((format) => {
                    const selected = form.preferred_formats.includes(format);
                    return (
                      <button
                        key={format}
                        type="button"
                        onClick={() => toggleFormat(format)}
                        className={cn(
                          "rounded-xl border px-3 py-2 text-sm capitalize transition-colors",
                          selected
                            ? "border-blue-600 bg-blue-600 text-white"
                            : "border-blue-100 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200",
                        )}
                        aria-pressed={selected}
                      >
                        {format}
                      </button>
                    );
                  })}
                </div>
                {errors.preferred_formats ? (
                  <p className="text-sm text-red-600">{errors.preferred_formats}</p>
                ) : null}
              </div>
              <Field id="learning_style" label="Learning style" error={errors.learning_style}>
                <Select
                  value={form.learning_style}
                  onValueChange={(value) => update("learning_style", value as LearningStyle)}
                >
                  <SelectTrigger id="learning_style" aria-label="Learning style">
                    <SelectValue placeholder="Select style" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="visual">Visual</SelectItem>
                    <SelectItem value="auditory">Auditory</SelectItem>
                    <SelectItem value="reading">Reading</SelectItem>
                    <SelectItem value="kinesthetic">Kinesthetic</SelectItem>
                    <SelectItem value="mixed">Mixed</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
            </>
          ) : null}

          {step === 2 ? (
            <>
              <div className="grid gap-4 sm:grid-cols-3">
                <Field
                  id="daily_available_minutes"
                  label="Daily minutes"
                  error={errors.daily_available_minutes}
                >
                  <Input
                    id="daily_available_minutes"
                    type="number"
                    min={1}
                    value={form.daily_available_minutes}
                    onChange={(e) => update("daily_available_minutes", e.target.value)}
                  />
                </Field>
                <Field
                  id="preferred_session_minutes"
                  label="Session minutes"
                  error={errors.preferred_session_minutes}
                >
                  <Input
                    id="preferred_session_minutes"
                    type="number"
                    min={1}
                    value={form.preferred_session_minutes}
                    onChange={(e) => update("preferred_session_minutes", e.target.value)}
                  />
                </Field>
                <Field
                  id="attention_span_minutes"
                  label="Attention span"
                  error={errors.attention_span_minutes}
                >
                  <Input
                    id="attention_span_minutes"
                    type="number"
                    min={1}
                    value={form.attention_span_minutes}
                    onChange={(e) => update("attention_span_minutes", e.target.value)}
                  />
                </Field>
              </div>
              <Field
                id="preferred_learning_time"
                label="Preferred learning time"
                error={errors.preferred_learning_time}
              >
                <Select
                  value={form.preferred_learning_time}
                  onValueChange={(value) =>
                    update("preferred_learning_time", value as PreferredLearningTime)
                  }
                >
                  <SelectTrigger id="preferred_learning_time" aria-label="Preferred learning time">
                    <SelectValue placeholder="Select time" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="morning">Morning</SelectItem>
                    <SelectItem value="afternoon">Afternoon</SelectItem>
                    <SelectItem value="evening">Evening</SelectItem>
                    <SelectItem value="night">Night</SelectItem>
                    <SelectItem value="flexible">Flexible</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field id="habits" label="Habits (comma-separated, optional)">
                <Input
                  id="habits"
                  value={form.habits}
                  onChange={(e) => update("habits", e.target.value)}
                  placeholder="morning coffee, weekend practice"
                />
              </Field>
              <Field id="distractions" label="Distractions (comma-separated, optional)">
                <Input
                  id="distractions"
                  value={form.distractions}
                  onChange={(e) => update("distractions", e.target.value)}
                  placeholder="phone, social media"
                />
              </Field>
            </>
          ) : null}

          <div className="flex flex-wrap justify-between gap-3 pt-2">
            <Button
              type="button"
              variant="outline"
              disabled={step === 0 || submitting}
              onClick={() => setStep((value) => Math.max(0, value - 1))}
            >
              Back
            </Button>
            {step < 2 ? (
              <Button
                type="button"
                disabled={submitting}
                onClick={() => {
                  if (validateStep(step)) setStep((value) => value + 1);
                }}
              >
                Continue
              </Button>
            ) : (
              <Button type="button" disabled={submitting} onClick={handleSubmit}>
                {submitting ? "Creating your roadmap…" : "Create my roadmap"}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Field({
  id,
  label,
  error,
  children,
}: {
  id: string;
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      {children}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
    </div>
  );
}
