"""LangGraph onboarding workflow tests (deterministic fakes)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exceptions import ProfileAgentError, RoadmapAgentError, WorkflowExecutionError  # noqa: E402
from models import (  # noqa: E402
    CurrentLevel,
    Difficulty,
    GoalResponse,
    GoalStatus,
    LearningStyle,
    MilestoneResponse,
    MilestoneStatus,
    OnboardingRequest,
    PhaseStatus,
    PreferredLearningTime,
    ProfileAgentResult,
    ProfileInterpretation,
    RoadmapAgentResult,
    RoadmapPhaseResponse,
    RoadmapResponse,
    RoadmapStatus,
    UserProfileResponse,
    UserResponse,
)
from workflows.onboarding import OnboardingWorkflow, build_onboarding_graph  # noqa: E402


def _interpretation() -> ProfileInterpretation:
    return ProfileInterpretation(
        identity_summary="Learner",
        aspiration_summary="Present well",
        motivation_summary="Career growth",
        current_state_summary="Beginner",
        target_state_summary="Confident speaker",
        strengths=["curious"],
        likely_challenges=["nerves"],
        learning_preferences_summary="mixed",
        recommended_pacing="steady",
        attention_strategy="short blocks",
        consistency_strategy="daily practice",
        initial_personalization_insights=["start small"],
    )


class FakeProfileAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def process_onboarding(self, request: OnboardingRequest) -> ProfileAgentResult:
        self.calls += 1
        if self.fail:
            raise ProfileAgentError("profile down")
        now = datetime.now(timezone.utc)
        user = UserResponse(id=1, display_name=request.display_name, created_at=now, updated_at=now)
        profile = UserProfileResponse(
            id=1,
            user_id=1,
            aspiration=request.aspiration,
            motivation=request.motivation,
            current_level=request.current_level,
            target_outcome=request.target_outcome,
            learning_style=request.learning_style,
            preferred_formats=list(request.preferred_formats),
            daily_available_minutes=request.daily_available_minutes,
            preferred_session_minutes=request.preferred_session_minutes,
            attention_span_minutes=request.attention_span_minutes,
            preferred_learning_time=request.preferred_learning_time,
            habits=list(request.habits),
            distractions=list(request.distractions),
            created_at=now,
            updated_at=now,
        )
        goal = GoalResponse(
            id=10,
            user_id=1,
            title=request.learning_goal,
            description="desc",
            status=GoalStatus.active,
            created_at=now,
            updated_at=now,
        )
        return ProfileAgentResult(
            user=user,
            profile=profile,
            goal=goal,
            interpretation=_interpretation(),
            created_at=now,
        )


class FakeRoadmapAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[int, int]] = []
        self._created = 0

    def generate_roadmap(
        self,
        user_id: int,
        goal_id: int,
        *,
        regenerate: bool = False,
    ) -> RoadmapAgentResult:
        self.calls.append((user_id, goal_id))
        if self.fail:
            raise RoadmapAgentError("roadmap down")
        if self._created and not regenerate:
            reused = True
        else:
            self._created += 1
            reused = False
        now = datetime.now(timezone.utc)
        roadmap = RoadmapResponse(
            id=100,
            user_id=user_id,
            goal_id=goal_id,
            title="Roadmap",
            summary="Summary",
            estimated_duration_weeks=6,
            status=RoadmapStatus.active,
            progress_percent=0,
            created_at=now,
            updated_at=now,
        )
        milestone = MilestoneResponse(
            id=200,
            phase_id=1,
            sequence_number=1,
            title="Start",
            description="Begin",
            skills=["speak"],
            suggested_activities=["practice"],
            completion_criteria="Done",
            estimated_sessions=2,
            estimated_minutes=20,
            difficulty=Difficulty.beginner,
            status=MilestoneStatus.in_progress,
            progress_percent=0,
            created_at=now,
            updated_at=now,
        )
        phase = RoadmapPhaseResponse(
            id=1,
            roadmap_id=100,
            sequence_number=1,
            title="Phase 1",
            description="Foundations",
            expected_outcome="Basics",
            status=PhaseStatus.in_progress,
            milestones=[milestone],
            created_at=now,
            updated_at=now,
        )
        return RoadmapAgentResult(
            roadmap=roadmap,
            phases=[phase],
            milestones=[milestone],
            active_milestone=milestone,
            pacing_rationale="steady",
            personalization_rationale="beginner",
            reused_existing=reused,
            created_at=now,
        )


def _request() -> OnboardingRequest:
    return OnboardingRequest(
        display_name="Ada",
        learning_goal="Improve public speaking",
        aspiration="Calm presenter",
        motivation="Lead meetings",
        current_level=CurrentLevel.beginner,
        target_outcome="5-minute talk",
        preferred_formats=["video", "practice"],
        learning_style=LearningStyle.mixed,
        daily_available_minutes=30,
        preferred_session_minutes=15,
        attention_span_minutes=10,
        preferred_learning_time=PreferredLearningTime.evening,
        habits=["review"],
        distractions=["phone"],
    )


class OnboardingWorkflowTests(unittest.TestCase):
    def test_profile_before_roadmap_and_ids(self) -> None:
        profile = FakeProfileAgent()
        roadmap = FakeRoadmapAgent()
        graph = build_onboarding_graph(profile, roadmap)
        self.assertTrue(hasattr(graph, "invoke"))

        workflow = OnboardingWorkflow(profile, roadmap, compiled_graph=graph)
        result = workflow.run(_request())

        self.assertEqual(profile.calls, 1)
        self.assertEqual(roadmap.calls, [(1, 10)])
        self.assertEqual(result.completed_steps, ["profile", "roadmap", "finish"])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.goal.title, "Improve public speaking")
        self.assertEqual(result.roadmap.id, 100)
        self.assertIsNotNone(result.active_milestone)

    def test_profile_failure_blocks_roadmap(self) -> None:
        profile = FakeProfileAgent(fail=True)
        roadmap = FakeRoadmapAgent()
        workflow = OnboardingWorkflow(profile, roadmap)
        with self.assertRaises(WorkflowExecutionError) as ctx:
            workflow.run(_request())
        self.assertEqual(ctx.exception.stage, "profile")
        self.assertEqual(roadmap.calls, [])
        self.assertNotIn("secret", str(ctx.exception).lower())
        self.assertNotIn("api_key", str(ctx.exception).lower())

    def test_roadmap_failure_preserves_profile_partial(self) -> None:
        profile = FakeProfileAgent()
        roadmap = FakeRoadmapAgent(fail=True)
        workflow = OnboardingWorkflow(profile, roadmap)
        with self.assertRaises(WorkflowExecutionError) as ctx:
            workflow.run(_request())
        self.assertEqual(ctx.exception.stage, "roadmap")
        self.assertIsNotNone(ctx.exception.partial_result)
        self.assertEqual(ctx.exception.partial_result.user.id, 1)  # type: ignore[union-attr]

    def test_repeated_invocation_reuses_roadmap(self) -> None:
        profile = FakeProfileAgent()
        roadmap = FakeRoadmapAgent()
        workflow = OnboardingWorkflow(profile, roadmap)
        first = workflow.run(_request())
        second = workflow.run(_request())
        self.assertFalse(first.roadmap_result.reused_existing)
        self.assertTrue(second.roadmap_result.reused_existing)
        self.assertEqual(roadmap._created, 1)


if __name__ == "__main__":
    unittest.main()
