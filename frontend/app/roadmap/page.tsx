"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  EmptyPanel,
  ErrorPanel,
  MissingUserPanel,
  PageSkeleton,
} from "@/components/state-panels";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ApiClientError, getActiveRoadmap, getDashboard } from "@/lib/api";
import {
  getUserId,
  setGoalTitle,
  setProgressPercent,
} from "@/lib/user-state";
import { formatLabel } from "@/lib/utils";
import type { GoalResponse, RoadmapResponse } from "@/types";
import { cn } from "@/lib/utils";

export default function RoadmapPage() {
  const [userId, setUserIdState] = useState<number | null>(null);
  const [roadmap, setRoadmap] = useState<RoadmapResponse | null>(null);
  const [goal, setGoal] = useState<GoalResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      const [roadmapData, dashboard] = await Promise.all([
        getActiveRoadmap(id),
        getDashboard(id).catch(() => null),
      ]);
      setRoadmap(roadmapData);
      setProgressPercent(roadmapData.progress_percent);
      if (dashboard?.active_goal) {
        setGoal(dashboard.active_goal);
        setGoalTitle(dashboard.active_goal.title);
      }
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not load roadmap.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  if (loading) return <PageSkeleton />;
  if (userId === null) return <MissingUserPanel />;
  if (error) {
    return (
      <ErrorPanel
        title="Roadmap unavailable"
        message={error}
        onRetry={() => void load()}
        recoveryHref="/onboarding"
        recoveryLabel="Restart onboarding"
      />
    );
  }
  if (!roadmap) {
    return (
      <EmptyPanel
        title="No active roadmap"
        description="Complete onboarding to generate your personalized growth path."
        actionHref="/onboarding"
        actionLabel="Start onboarding"
      />
    );
  }

  const activeId = roadmap.current_active_milestone_id;

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="text-sm font-medium uppercase tracking-[0.18em] text-blue-600 dark:text-blue-300">
          Your roadmap
        </p>
        <h1 className="text-3xl font-semibold tracking-tight">{roadmap.title}</h1>
        {goal ? (
          <p className="text-lg text-slate-700 dark:text-slate-200">
            <span className="font-medium">Goal:</span> {goal.title}
          </p>
        ) : null}
        <p className="max-w-3xl text-slate-600 dark:text-slate-300">{roadmap.summary}</p>
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600 dark:text-slate-300">
          <Badge variant="secondary">~{roadmap.estimated_duration_weeks} weeks</Badge>
          <span>{Math.round(roadmap.progress_percent)}% overall progress</span>
        </div>
        <Progress value={roadmap.progress_percent} aria-label="Overall roadmap progress" />
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Mood changes today’s plan  -  not this long-term goal.
        </p>
      </div>

      <div className="space-y-4">
        {roadmap.phases
          .slice()
          .sort((a, b) => a.sequence_number - b.sequence_number)
          .map((phase) => (
            <Card key={phase.id}>
              <CardHeader>
                <CardTitle className="text-xl">
                  Phase {phase.sequence_number}: {phase.title}
                </CardTitle>
                <CardDescription>{phase.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm">
                  <span className="font-medium">Expected outcome:</span>{" "}
                  {phase.expected_outcome}
                </p>
                <div className="space-y-3">
                  {phase.milestones
                    .slice()
                    .sort((a, b) => a.sequence_number - b.sequence_number)
                    .map((milestone) => {
                      const active = milestone.id === activeId;
                      return (
                        <div
                          key={milestone.id}
                          className={cn(
                            "rounded-2xl border p-4 transition-colors",
                            active
                              ? "border-blue-500 bg-blue-50/80 ring-2 ring-blue-200 dark:border-blue-400 dark:bg-blue-950/30 dark:ring-blue-900"
                              : "border-blue-100/80 dark:border-slate-800",
                          )}
                        >
                          <div className="mb-2 flex flex-wrap items-center gap-2">
                            <h3 className="font-semibold">{milestone.title}</h3>
                            {active ? <Badge>Active milestone</Badge> : null}
                            <Badge variant="outline">{formatLabel(milestone.status)}</Badge>
                            <Badge variant="secondary">
                              {formatLabel(milestone.difficulty)}
                            </Badge>
                          </div>
                          <p className="mb-3 text-sm text-slate-600 dark:text-slate-300">
                            {milestone.description}
                          </p>
                          <p className="mb-2 text-sm">
                            <span className="font-medium">Completion criteria:</span>{" "}
                            {milestone.completion_criteria}
                          </p>
                          <div className="mb-2 flex flex-wrap gap-2">
                            {milestone.skills.map((skill) => (
                              <Badge key={skill} variant="secondary">
                                {skill}
                              </Badge>
                            ))}
                          </div>
                          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-300">
                            {milestone.suggested_activities.map((activity) => (
                              <li key={activity}>{activity}</li>
                            ))}
                          </ul>
                          <div className="mt-3">
                            <Progress value={milestone.progress_percent} />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </CardContent>
            </Card>
          ))}
      </div>

      <Button asChild size="lg">
        <Link href="/check-in">Continue to Today’s Check-In</Link>
      </Button>
    </div>
  );
}
