"""LangGraph daily-loop workflow tests (deterministic fakes)."""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exceptions import (  # noqa: E402
    AdaptationAgentError,
    PlannerAgentError,
    ReflectionAgentError,
    WorkflowExecutionError,
)
from models import (  # noqa: E402
    ActivityType,
    AdaptationAgentResult,
    CompletionStatus,
    DailyCheckInRequest,
    DailyCheckInResponse,
    DailyPlanResponse,
    DailyTaskResponse,
    Difficulty,
    DifficultyFeedback,
    EnergyLevel,
    Mood,
    PlanStatus,
    PlannerAgentResult,
    ReflectionAgentResult,
    ReflectionRequest,
    ReflectionResponse,
    TaskStatus,
)
from workflows.daily_loop import (  # noqa: E402
    DailyLoopWorkflow,
    build_planning_graph,
    build_post_session_graph,
)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _task(task_id: int, plan_id: int, minutes: int = 10) -> DailyTaskResponse:
    now = _now()
    return DailyTaskResponse(
        id=task_id,
        daily_plan_id=plan_id,
        resource_id=1,
        sequence_number=task_id,
        title=f"Task {task_id}",
        description="Do the thing",
        activity_type=ActivityType.practice if task_id > 1 else ActivityType.watch,
        estimated_minutes=minutes,
        difficulty=Difficulty.beginner,
        status=TaskStatus.pending,
        created_at=now,
        updated_at=now,
    )


def _plan(plan_id: int = 50, minutes: int = 20) -> DailyPlanResponse:
    now = _now()
    tasks = [_task(1, plan_id, 10), _task(2, plan_id, 10)]
    return DailyPlanResponse(
        id=plan_id,
        user_id=1,
        roadmap_id=100,
        milestone_id=200,
        checkin_id=5,
        plan_date=date(2026, 8, 1),
        summary="Focused plan",
        total_estimated_minutes=minutes,
        status=PlanStatus.pending,
        tasks=tasks,
        adaptation_explanation="Based on mood",
        created_at=now,
        updated_at=now,
    )


class FakePlanner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.last_refresh: bool | None = None
        self._plan = _plan()

    def create_daily_plan(
        self,
        user_id: int,
        *,
        checkin: DailyCheckInRequest,
        plan_date: date | None = None,
        refresh: bool = False,
    ) -> PlannerAgentResult:
        self.calls += 1
        self.last_refresh = refresh
        if self.fail:
            raise PlannerAgentError("planner down")
        reused = self.calls > 1 and not refresh
        now = _now()
        checkin_resp = DailyCheckInResponse(
            id=5,
            user_id=user_id,
            mood=checkin.mood,
            energy_level=checkin.energy_level,
            focus_level=checkin.focus_level,
            available_minutes=checkin.available_minutes,
            preferred_activity=checkin.preferred_activity,
            notes=checkin.notes,
            created_at=now,
        )
        plan = self._plan if reused else _plan(plan_id=50 + self.calls)
        if refresh:
            plan = _plan(plan_id=77)
        return PlannerAgentResult(
            checkin=checkin_resp,
            plan=plan,
            goal_title="Improve public speaking",
            milestone_title="Breath basics",
            reused_existing=reused,
            created_at=now,
        )


class FakeReflection:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self._created = False

    def reflect_on_plan(self, user_id: int, request: ReflectionRequest) -> ReflectionAgentResult:
        self.calls += 1
        if self.fail:
            raise ReflectionAgentError("reflection down")
        now = _now()
        reused = self._created
        self._created = True
        reflection = ReflectionResponse(
            id=900,
            user_id=user_id,
            daily_plan_id=request.daily_plan_id,
            completion_status=request.completion_status,
            learning_summary=request.learning_summary,
            focus_rating=request.focus_rating,
            resource_effectiveness=request.resource_effectiveness,
            difficulty_feedback=request.difficulty_feedback,
            mood_match=request.mood_match,
            distractions=list(request.distractions),
            wants_similar_resources=request.wants_similar_resources,
            mood_after=str(request.mood_after),
            insight="Partial completion with useful practice.",
            created_at=now,
        )
        return ReflectionAgentResult(
            reflection=reflection,
            plan_completion_percent=50.0,
            reused_existing=reused,
            created_at=now,
        )


class FakeAdaptation:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[int] = []
        self._seen: set[int] = set()

    def adapt_from_reflection(
        self,
        user_id: int,
        reflection_id: int,
        *,
        force: bool = False,
    ) -> AdaptationAgentResult:
        self.calls.append(reflection_id)
        if self.fail:
            raise AdaptationAgentError("adaptation down")
        reused = reflection_id in self._seen and not force
        self._seen.add(reflection_id)
        return AdaptationAgentResult(
            adaptation_explanation=(
                "Your next plan will use shorter resources because focus dropped "
                "during a longer video."
            ),
            detected_patterns=["Longer resources reduce focus"],
            is_early_signal=True,
            confidence_score=0.4,
            reflection_id=reflection_id,
            reused_existing=reused,
            goal_unchanged=True,
            created_at=_now(),
        )


class DailyLoopWorkflowTests(unittest.TestCase):
    def test_planning_stops_at_awaiting_completion(self) -> None:
        planner = FakePlanner()
        reflection = FakeReflection()
        adaptation = FakeAdaptation()
        graph = build_planning_graph(planner)
        workflow = DailyLoopWorkflow(
            planner_agent=planner,
            reflection_agent=reflection,
            adaptation_agent=adaptation,
            planning_graph=graph,
        )
        result = workflow.run_planning(
            1,
            DailyCheckInRequest(
                mood=Mood.tired,
                energy_level=EnergyLevel.low,
                focus_level=2,
                available_minutes=15,
                preferred_activity=ActivityType.watch,
            ),
        )
        self.assertEqual(planner.calls, 1)
        self.assertEqual(reflection.calls, 0)
        self.assertEqual(adaptation.calls, [])
        self.assertTrue(result.awaiting_user_completion)
        self.assertEqual(result.current_stage, "awaiting_user_completion")
        self.assertEqual(result.completed_steps, ["plan", "await_user_completion"])
        self.assertEqual(len(result.tasks), 2)

    def test_planning_failure_and_reuse(self) -> None:
        planner = FakePlanner(fail=True)
        workflow = DailyLoopWorkflow(
            planner_agent=planner,
            reflection_agent=FakeReflection(),
            adaptation_agent=FakeAdaptation(),
        )
        with self.assertRaises(WorkflowExecutionError) as ctx:
            workflow.run_planning(
                1,
                DailyCheckInRequest(
                    mood=Mood.focused,
                    energy_level=EnergyLevel.medium,
                    focus_level=4,
                    available_minutes=30,
                    preferred_activity=ActivityType.practice,
                ),
            )
        self.assertEqual(ctx.exception.stage, "planning")

        ok_planner = FakePlanner()
        workflow2 = DailyLoopWorkflow(
            planner_agent=ok_planner,
            reflection_agent=FakeReflection(),
            adaptation_agent=FakeAdaptation(),
        )
        checkin = DailyCheckInRequest(
            mood=Mood.focused,
            energy_level=EnergyLevel.medium,
            focus_level=4,
            available_minutes=30,
            preferred_activity=ActivityType.practice,
        )
        first = workflow2.run_planning(1, checkin)
        second = workflow2.run_planning(1, checkin, refresh=False)
        self.assertFalse(first.planner_result.reused_existing)
        self.assertTrue(second.planner_result.reused_existing)
        third = workflow2.run_planning(1, checkin, refresh=True)
        self.assertTrue(ok_planner.last_refresh)
        self.assertEqual(third.plan.id, 77)

    def test_post_session_order_and_adaptation_id(self) -> None:
        reflection = FakeReflection()
        adaptation = FakeAdaptation()
        graph = build_post_session_graph(reflection, adaptation)
        workflow = DailyLoopWorkflow(
            planner_agent=FakePlanner(),
            reflection_agent=reflection,
            adaptation_agent=adaptation,
            post_session_graph=graph,
        )
        result = workflow.run_post_session(
            1,
            ReflectionRequest(
                daily_plan_id=50,
                completion_status=CompletionStatus.partial,
                learning_summary="Practice helped",
                focus_rating=2,
                resource_effectiveness=4,
                difficulty_feedback=DifficultyFeedback.suitable,
                mood_match=False,
                distractions=["phone"],
                wants_similar_resources=True,
                mood_after=Mood.tired,
            ),
        )
        self.assertEqual(reflection.calls, 1)
        self.assertEqual(adaptation.calls, [900])
        self.assertEqual(result.completed_steps, ["reflection", "adaptation"])
        self.assertEqual(result.current_stage, "completed")
        self.assertTrue(result.goal_unchanged)
        self.assertIn("shorter resources", result.adaptation_explanation.lower())

    def test_reflection_failure_blocks_adaptation(self) -> None:
        reflection = FakeReflection(fail=True)
        adaptation = FakeAdaptation()
        workflow = DailyLoopWorkflow(
            planner_agent=FakePlanner(),
            reflection_agent=reflection,
            adaptation_agent=adaptation,
        )
        with self.assertRaises(WorkflowExecutionError) as ctx:
            workflow.run_post_session(
                1,
                ReflectionRequest(
                    daily_plan_id=50,
                    completion_status=CompletionStatus.partial,
                    learning_summary="x",
                    focus_rating=3,
                    resource_effectiveness=3,
                    difficulty_feedback=DifficultyFeedback.suitable,
                    mood_match=True,
                    distractions=[],
                    wants_similar_resources=False,
                    mood_after=Mood.calm,
                ),
            )
        self.assertEqual(ctx.exception.stage, "reflecting")
        self.assertEqual(adaptation.calls, [])

    def test_adaptation_failure_preserves_reflection(self) -> None:
        reflection = FakeReflection()
        adaptation = FakeAdaptation(fail=True)
        workflow = DailyLoopWorkflow(
            planner_agent=FakePlanner(),
            reflection_agent=reflection,
            adaptation_agent=adaptation,
        )
        with self.assertRaises(WorkflowExecutionError) as ctx:
            workflow.run_post_session(
                1,
                ReflectionRequest(
                    daily_plan_id=50,
                    completion_status=CompletionStatus.partial,
                    learning_summary="x",
                    focus_rating=3,
                    resource_effectiveness=3,
                    difficulty_feedback=DifficultyFeedback.suitable,
                    mood_match=True,
                    distractions=[],
                    wants_similar_resources=False,
                    mood_after=Mood.calm,
                ),
            )
        self.assertEqual(ctx.exception.stage, "adapting")
        self.assertIsNotNone(ctx.exception.partial_result)
        self.assertEqual(ctx.exception.partial_result.reflection.id, 900)  # type: ignore[union-attr]

    def test_repeated_post_session_idempotent(self) -> None:
        reflection = FakeReflection()
        adaptation = FakeAdaptation()
        workflow = DailyLoopWorkflow(
            planner_agent=FakePlanner(),
            reflection_agent=reflection,
            adaptation_agent=adaptation,
        )
        request = ReflectionRequest(
            daily_plan_id=50,
            completion_status=CompletionStatus.partial,
            learning_summary="x",
            focus_rating=3,
            resource_effectiveness=3,
            difficulty_feedback=DifficultyFeedback.suitable,
            mood_match=True,
            distractions=[],
            wants_similar_resources=False,
            mood_after=Mood.calm,
        )
        first = workflow.run_post_session(1, request)
        second = workflow.run_post_session(1, request)
        self.assertFalse(first.reflection_result.reused_existing)
        self.assertTrue(second.reflection_result.reused_existing)
        self.assertTrue(second.adaptation.reused_existing)
        self.assertNotIn("api_key", str(first).lower())


if __name__ == "__main__":
    unittest.main()
