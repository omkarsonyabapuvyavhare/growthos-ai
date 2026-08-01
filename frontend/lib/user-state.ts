"use client";

import type {
  DailyPostSessionWorkflowResult,
  DemoDayLoopResponse,
  SessionDaySummary,
} from "@/types";

const USER_ID_KEY = "growthos_user_id";
const GOAL_TITLE_KEY = "growthos_goal_title";
const PLAN_ID_KEY = "growthos_current_plan_id";
const REFLECTION_KEY = "growthos_last_reflection";
const DEMO_KEY = "growthos_demo_result";
const SESSION_DAYS_KEY = "growthos_session_days";
const PROGRESS_KEY = "growthos_progress_percent";

function canUseStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function readJson<T>(key: string): T | null {
  if (!canUseStorage()) return null;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(key, JSON.stringify(value));
}

export function getUserId(): number | null {
  if (!canUseStorage()) return null;
  const raw = window.localStorage.getItem(USER_ID_KEY);
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export function setUserId(userId: number): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(USER_ID_KEY, String(userId));
}

export function clearUserState(): void {
  if (!canUseStorage()) return;
  [
    USER_ID_KEY,
    GOAL_TITLE_KEY,
    PLAN_ID_KEY,
    REFLECTION_KEY,
    DEMO_KEY,
    SESSION_DAYS_KEY,
    PROGRESS_KEY,
  ].forEach((key) => window.localStorage.removeItem(key));
}

export function getGoalTitle(): string | null {
  if (!canUseStorage()) return null;
  return window.localStorage.getItem(GOAL_TITLE_KEY);
}

export function setGoalTitle(title: string): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(GOAL_TITLE_KEY, title);
}

export function getCurrentPlanId(): number | null {
  if (!canUseStorage()) return null;
  const raw = window.localStorage.getItem(PLAN_ID_KEY);
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export function setCurrentPlanId(planId: number): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(PLAN_ID_KEY, String(planId));
}

export function getProgressPercent(): number {
  if (!canUseStorage()) return 0;
  const raw = window.localStorage.getItem(PROGRESS_KEY);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(100, Math.max(0, parsed));
}

export function setProgressPercent(value: number): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(
    PROGRESS_KEY,
    String(Math.min(100, Math.max(0, value))),
  );
}

export function saveLastReflection(result: DailyPostSessionWorkflowResult): void {
  writeJson(REFLECTION_KEY, result);
}

export function getLastReflection(): DailyPostSessionWorkflowResult | null {
  return readJson<DailyPostSessionWorkflowResult>(REFLECTION_KEY);
}

export function saveDemoResult(result: DemoDayLoopResponse): void {
  writeJson(DEMO_KEY, result);
}

export function getDemoResult(): DemoDayLoopResponse | null {
  return readJson<DemoDayLoopResponse>(DEMO_KEY);
}

export function upsertSessionDay(summary: SessionDaySummary): void {
  const existing = readJson<SessionDaySummary[]>(SESSION_DAYS_KEY) ?? [];
  const prior = existing.find((item) => item.plan_date === summary.plan_date);
  const merged: SessionDaySummary = {
    plan_date: summary.plan_date,
    mood:
      summary.mood && summary.mood !== "session"
        ? summary.mood
        : prior?.mood ?? summary.mood,
    energy: summary.energy ?? prior?.energy,
    available_minutes:
      summary.mood === "session" && prior
        ? prior.available_minutes
        : summary.available_minutes,
    task_count: summary.task_count || prior?.task_count || 0,
    practice_task_count:
      summary.practice_task_count ?? prior?.practice_task_count,
    completion_status: summary.completion_status ?? prior?.completion_status,
    plan_summary: summary.plan_summary || prior?.plan_summary,
    activity_mix: summary.activity_mix || prior?.activity_mix,
  };
  const next = existing.filter((item) => item.plan_date !== summary.plan_date);
  next.push(merged);
  next.sort((a, b) => a.plan_date.localeCompare(b.plan_date));
  writeJson(SESSION_DAYS_KEY, next.slice(-7));
}

export function getSessionDays(): SessionDaySummary[] {
  return readJson<SessionDaySummary[]>(SESSION_DAYS_KEY) ?? [];
}
