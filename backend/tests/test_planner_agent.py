"""Unit tests for DailyPlannerAgent (no live Gemini/network)."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Type

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.planner_agent import (  # noqa: E402
    DailyPlannerAgent,
    build_adaptation_explanation,
    build_constrained_planner_retry_prompt,
    mood_capacity_rules,
    validate_plan_generation,
)
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from exceptions import (  # noqa: E402
    CuratorAgentError,
    GeminiInvocationError,
    GeminiResponseError,
    PlannerBudgetError,
    PlannerContextError,
    PlannerGenerationError,
    PlannerPersistenceError,
)
from models import (  # noqa: E402
    ActivityType,
    CuratedRecommendation,
    CuratorAgentResult,
    DailyCheckInRequest,
    DailyPlanGeneration,
    Difficulty,
    EnergyLevel,
    MilestoneGeneration,
    Mood,
    PlannerTaskGeneration,
    RecommendationStatus,
    RoadmapGeneration,
    RoadmapPhaseGeneration,
)
from services.database import (  # noqa: E402
    count_completed_plans,
    create_adaptation_insight,
    create_onboarding_records,
    create_user,
    get_connection,
    get_daily_plan_by_date,
    init_db,
    list_active_adaptation_insights,
    upsert_user_preference,
)
from tests.test_embedding import _settings  # noqa: E402


def _milestone(seq: int, title: str, skills: list[str]) -> MilestoneGeneration:
    return MilestoneGeneration(
        sequence_number=seq,
        title=title,
        description=f"Description for {title}",
        skills=skills,
        suggested_activities=[f"practice {title.lower()}"],
        completion_criteria=f"Complete {title}",
        estimated_sessions=2,
        estimated_minutes=20,
        difficulty=Difficulty.beginner,
    )


def _roadmap_generation() -> RoadmapGeneration:
    return RoadmapGeneration(
        title="Public Speaking Foundations",
        summary="Practical speaking roadmap.",
        estimated_duration_weeks=6,
        pacing_rationale="Fits short evening sessions.",
        personalization_rationale="Beginner friendly.",
        phases=[
            RoadmapPhaseGeneration(
                sequence_number=1,
                title="Foundations",
                description="Breath and posture.",
                expected_outcome="Short talk confidence.",
                milestones=[
                    _milestone(1, "Breath and posture basics", ["breath", "posture", "public speaking"]),
                    _milestone(2, "Two-minute outline drill", ["speech structure"]),
                ],
            ),
            RoadmapPhaseGeneration(
                sequence_number=2,
                title="Delivery",
                description="Practice delivery.",
                expected_outcome="Deliver a 5-minute talk.",
                milestones=[_milestone(1, "Story framing", ["storytelling"])],
            ),
        ],
    )


class FakeRoadmapGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        assert response_model is RoadmapGeneration
        return _roadmap_generation()


class FakeCurator:
    def __init__(self, recommendations: list[CuratedRecommendation], *, raise_error: Exception | None = None) -> None:
        self.recommendations = recommendations
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def recommend_resources(self, user_id: int, roadmap_id: int, milestone_id: int, **kwargs: Any) -> CuratorAgentResult:
        self.calls.append({"user_id": user_id, "roadmap_id": roadmap_id, "milestone_id": milestone_id, **kwargs})
        if self.raise_error is not None:
            raise self.raise_error
        return CuratorAgentResult(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            recommendations=self.recommendations,
            candidate_count=len(self.recommendations),
            created_at=datetime.now(timezone.utc),
        )


class FakePlannerGemini:
    def __init__(
        self,
        *,
        resource_ids: list[int],
        force_generation: DailyPlanGeneration | None = None,
        raise_error: Exception | None = None,
        fail_times: int = 0,
        fail_error: Exception | None = None,
    ) -> None:
        self.resource_ids = resource_ids
        self.force_generation = force_generation
        self.raise_error = raise_error
        self.fail_times = max(0, int(fail_times))
        self.fail_error = fail_error or GeminiResponseError("empty structured output")
        self.prompts: list[str] = []
        self.call_count = 0

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[DailyPlanGeneration],
        *,
        system_instruction: str | None = None,
    ) -> DailyPlanGeneration:
        self.prompts.append(prompt)
        self.call_count += 1
        if self.raise_error is not None:
            raise self.raise_error
        if self.call_count <= self.fail_times:
            raise self.fail_error
        assert response_model is DailyPlanGeneration
        if self.force_generation is not None:
            return self.force_generation

        mood = "focused"
        available = 30
        min_tasks = 1
        max_tasks = 5
        for line in prompt.splitlines():
            stripped = line.strip()
            if '"mood":' in stripped and "influence" not in stripped and "rule" not in stripped:
                mood = stripped.split(":", 1)[1].strip().strip(",").strip('"')
            if '"available_minutes":' in stripped:
                try:
                    available = int(stripped.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass
            if '"min_tasks":' in stripped:
                try:
                    min_tasks = int(stripped.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass
            if '"max_tasks":' in stripped:
                try:
                    max_tasks = int(stripped.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass

        if mood in {"tired", "low_energy", "stressed", "distracted"}:
            target_count = max(min_tasks, min(max_tasks, 2))
            slice_minutes = max(3, min(10, available) // target_count)
            tasks = [
                PlannerTaskGeneration(
                    sequence_number=1,
                    title="Short calm video",
                    description="Watch one short tip",
                    activity_type=ActivityType.watch,
                    resource_id=self.resource_ids[0],
                    estimated_minutes=slice_minutes,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One calm cue",
                    why_selected="Short trusted resource for low energy",
                    milestone_connection="Breath and posture basics",
                    mood_rationale="Low-pressure watch task",
                    content_type="video",
                ),
            ]
            if target_count >= 2:
                tasks.append(
                    PlannerTaskGeneration(
                        sequence_number=2,
                        title="One-minute practice",
                        description="Practice the cue once",
                        activity_type=ActivityType.practice,
                        resource_id=None,
                        estimated_minutes=slice_minutes,
                        difficulty=Difficulty.beginner,
                        expected_outcome="One calm attempt",
                        why_selected="Micro practice improves completion",
                        milestone_connection="Breath and posture basics",
                        mood_rationale="Structured micro-task",
                        content_type="practice",
                    )
                )
            while sum(t.estimated_minutes for t in tasks) > available and len(tasks) > min_tasks:
                tasks.pop()
            if sum(t.estimated_minutes for t in tasks) > available:
                tasks[0].estimated_minutes = available
            for index, task in enumerate(tasks, start=1):
                task.sequence_number = index
            tone = "calm" if mood != "stressed" else "reassuring"
            return DailyPlanGeneration(
                summary="A short low-pressure session.",
                guidance_tone=tone,
                mood_influence_summary="Fewer shorter tasks for today's mood.",
                task_count_rationale="1-2 tasks fit low energy/focus.",
                adaptation_explanation="Based on current mood and available time.",
                tasks=tasks,
            )

        # focused / motivated / curious
        tasks = [
            PlannerTaskGeneration(
                sequence_number=1,
                title="Study curated tip",
                description="Watch/read the curated resource",
                activity_type=ActivityType.watch,
                resource_id=self.resource_ids[0],
                estimated_minutes=10,
                difficulty=Difficulty.beginner,
                expected_outcome="Clear technique",
                why_selected="Trusted curated resource",
                milestone_connection="Breath and posture basics",
                mood_rationale="Supports focused study",
                content_type="video",
            ),
            PlannerTaskGeneration(
                sequence_number=2,
                title="Applied practice",
                description="Practice for two minutes",
                activity_type=ActivityType.practice,
                resource_id=self.resource_ids[1] if len(self.resource_ids) > 1 else None,
                estimated_minutes=10,
                difficulty=Difficulty.intermediate,
                expected_outcome="Confident short delivery",
                why_selected="Focused mood allows deeper practice",
                milestone_connection="Breath and posture basics",
                mood_rationale="Higher engagement practice",
                content_type="practice",
            ),
            PlannerTaskGeneration(
                sequence_number=3,
                title="Quick review",
                description="Note one improvement",
                activity_type=ActivityType.review,
                resource_id=None,
                estimated_minutes=5,
                difficulty=Difficulty.beginner,
                expected_outcome="One improvement note",
                why_selected="Synthesis consolidates learning",
                milestone_connection="Breath and posture basics",
                mood_rationale="Focused review",
                content_type="review",
            ),
        ]
        # Fit budget and capacity band from the prompt.
        selected: list[PlannerTaskGeneration] = []
        used = 0
        for task in tasks:
            if len(selected) >= max_tasks:
                break
            if used + task.estimated_minutes <= available:
                selected.append(task)
                used += task.estimated_minutes
        while len(selected) < min_tasks and tasks:
            candidate = tasks[len(selected) % len(tasks)].model_copy(deep=True)
            remaining = available - used
            if remaining < 3:
                break
            candidate.estimated_minutes = min(candidate.estimated_minutes, remaining)
            selected.append(candidate)
            used += candidate.estimated_minutes
        if not selected:
            tasks[0].estimated_minutes = available
            selected = [tasks[0]]
        for index, task in enumerate(selected, start=1):
            task.sequence_number = index
        return DailyPlanGeneration(
            summary="A focused practice-oriented session.",
            guidance_tone="direct",
            mood_influence_summary="Deeper practice because focus is high.",
            task_count_rationale="2-3 tasks fit available minutes.",
            adaptation_explanation="Based on current mood and available time.",
            tasks=selected,
        )


class PlannerAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_planner_")
        self.db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self.db_path)
        user, profile, goal = create_onboarding_records(
            display_name="Ada Lovelace",
            aspiration="Become a confident presenter",
            motivation="Lead team updates calmly",
            current_level="beginner",
            target_outcome="Deliver a 10-minute talk",
            learning_style="mixed",
            preferred_formats=["video", "practice"],
            daily_available_minutes=45,
            preferred_session_minutes=20,
            attention_span_minutes=15,
            preferred_learning_time="evening",
            habits=["journal"],
            distractions=["phone"],
            goal_title="Improve public speaking",
            db_path=self.db_path,
        )
        self.user = user
        self.profile = profile
        self.goal = goal
        roadmap_agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeRoadmapGemini(),
            memory_service=None,
            db_path=self.db_path,
            skip_memory_retrieval=True,
        )
        self.roadmap_result = roadmap_agent.generate_roadmap(int(user["id"]), int(goal["id"]))
        self.milestone = self.roadmap_result.active_milestone
        assert self.milestone is not None

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.resource_ids: list[int] = []
        with get_connection(self.db_path) as conn:
            for index in range(3):
                cursor = conn.execute(
                    """
                    INSERT INTO resources (
                        title, source, resource_type, url, description, difficulty,
                        estimated_duration_minutes, is_free, metadata, created_at, updated_at
                    )
                    VALUES (?, 'YouTube', 'video', ?, ?, 'beginner', ?, 1, '{}', ?, ?)
                    """,
                    (
                        f"Resource {index+1}",
                        f"https://www.youtube.com/watch?v=planner{index}",
                        "Trusted free tip",
                        8 + index,
                        now,
                        now,
                    ),
                )
                self.resource_ids.append(int(cursor.lastrowid))

        self.recommendations = [
            CuratedRecommendation(
                id=index + 1,
                user_id=int(user["id"]),
                roadmap_id=self.roadmap_result.roadmap.id,
                milestone_id=self.milestone.id,
                resource_id=resource_id,
                catalog_id=f"cat-{resource_id}",
                title=f"Resource {index+1}",
                source="YouTube",
                resource_type="video",
                url=f"https://www.youtube.com/watch?v=planner{index}",
                description="Trusted free tip",
                difficulty=Difficulty.beginner,
                estimated_duration_minutes=8 + index,
                relevance_score=0.8,
                reason="Fits milestone",
                milestone_fit="Breath basics",
                mood_suitability="Suitable",
                suggested_use="Watch then practice",
                estimated_effort="10 minutes",
                score_breakdown={"final": 0.8},
                status=RecommendationStatus.suggested,
                recommended_at=datetime.now(timezone.utc),
            )
            for index, resource_id in enumerate(self.resource_ids)
        ]
        self.curator = FakeCurator(self.recommendations)
        self.gemini = FakePlannerGemini(resource_ids=self.resource_ids)
        self.agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 1),
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _checkin(
        self,
        *,
        mood: Mood = Mood.focused,
        energy: EnergyLevel = EnergyLevel.medium,
        minutes: int = 30,
        activity: ActivityType = ActivityType.watch,
        focus: int = 4,
    ) -> DailyCheckInRequest:
        return DailyCheckInRequest(
            mood=mood,
            energy_level=energy,
            focus_level=focus,
            available_minutes=minutes,
            preferred_activity=activity,
            notes="test",
        )

    def test_valid_plan_creation(self) -> None:
        result = self.agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertEqual(result.goal_title, "Improve public speaking")
        self.assertEqual(result.plan.user_id, int(self.user["id"]))
        self.assertEqual(result.plan.roadmap_id, self.roadmap_result.roadmap.id)
        self.assertEqual(result.plan.milestone_id, self.milestone.id)
        self.assertEqual(result.checkin.user_id, int(self.user["id"]))
        self.assertGreaterEqual(len(result.plan.tasks), 1)
        self.assertLessEqual(len(result.plan.tasks), 5)
        self.assertLessEqual(result.plan.total_estimated_minutes, 30)
        self.assertEqual(
            [task.sequence_number for task in result.plan.tasks],
            list(range(1, len(result.plan.tasks) + 1)),
        )
        self.assertTrue(result.plan.tasks[0].why_selected)
        self.assertTrue(result.plan.tasks[0].milestone_connection)
        self.assertTrue(result.plan.tasks[0].expected_outcome)

    def test_tired_vs_focused_capacity(self) -> None:
        tired_rules = mood_capacity_rules(
            Mood.tired, EnergyLevel.low, available_minutes=15, focus_level=2
        )
        focused_rules = mood_capacity_rules(
            Mood.focused, EnergyLevel.high, available_minutes=30, focus_level=5
        )
        self.assertLessEqual(tired_rules.max_tasks, focused_rules.max_tasks)
        self.assertEqual(tired_rules.max_difficulty, Difficulty.beginner)

        tired = self.agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.tired, energy=EnergyLevel.low, minutes=15, focus=2),
            plan_date=date(2026, 8, 2),
        )
        focused = self.agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.focused, energy=EnergyLevel.high, minutes=30, focus=5),
            plan_date=date(2026, 8, 3),
        )
        self.assertLessEqual(len(tired.plan.tasks), len(focused.plan.tasks))
        self.assertLessEqual(tired.plan.total_estimated_minutes, 15)
        self.assertLessEqual(focused.plan.total_estimated_minutes, 30)
        self.assertIn(tired.plan.guidance_tone, {"calm", "reassuring", "structured"})

    def test_budget_and_unknown_resource_rejected(self) -> None:
        with self.assertRaises(PlannerBudgetError):
            validate_plan_generation(
                DailyPlanGeneration(
                    summary="x",
                    guidance_tone="calm",
                    mood_influence_summary="x",
                    task_count_rationale="x",
                    adaptation_explanation="x",
                    tasks=[
                        PlannerTaskGeneration(
                            sequence_number=1,
                            title="Too long",
                            description="d",
                            activity_type=ActivityType.watch,
                            resource_id=self.resource_ids[0],
                            estimated_minutes=40,
                            difficulty=Difficulty.beginner,
                            expected_outcome="o",
                            why_selected="w",
                            milestone_connection="m",
                            mood_rationale="m",
                        )
                    ],
                ),
                allowed_resource_ids=set(self.resource_ids),
                available_minutes=15,
                rules=mood_capacity_rules(
                    Mood.tired, EnergyLevel.low, available_minutes=15, focus_level=2
                ),
                attention_span_minutes=15,
            )

        bad = FakePlannerGemini(
            resource_ids=self.resource_ids,
            force_generation=DailyPlanGeneration(
                summary="x",
                guidance_tone="calm",
                mood_influence_summary="x",
                task_count_rationale="x",
                adaptation_explanation="x",
                tasks=[
                    PlannerTaskGeneration(
                        sequence_number=1,
                        title="Bad id",
                        description="d",
                        activity_type=ActivityType.watch,
                        resource_id=999999,
                        estimated_minutes=10,
                        difficulty=Difficulty.beginner,
                        expected_outcome="o",
                        why_selected="w",
                        milestone_connection="m",
                        mood_rationale="m",
                    )
                ],
            ),
        )
        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=bad,
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 4),
        )
        with self.assertRaises(PlannerGenerationError):
            agent.create_daily_plan(
                int(self.user["id"]),
                checkin=self._checkin(
                    mood=Mood.tired,
                    energy=EnergyLevel.low,
                    minutes=15,
                    focus=2,
                ),
            )
        self.assertIsNone(
            get_daily_plan_by_date(int(self.user["id"]), "2026-08-04", db_path=self.db_path)
        )

    def test_curator_and_gemini_failures(self) -> None:
        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            curator_agent=FakeCurator(self.recommendations, raise_error=CuratorAgentError("down")),
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 5),
        )
        with self.assertRaises(CuratorAgentError):
            agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertIsNone(
            get_daily_plan_by_date(int(self.user["id"]), "2026-08-05", db_path=self.db_path)
        )

        agent2 = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakePlannerGemini(
                resource_ids=self.resource_ids,
                raise_error=GeminiInvocationError("down"),
            ),
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 6),
        )
        with self.assertRaises(GeminiInvocationError):
            agent2.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertIsNone(
            get_daily_plan_by_date(int(self.user["id"]), "2026-08-06", db_path=self.db_path)
        )

    def test_persistence_rollback(self) -> None:
        def boom(**_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("insert failed")

        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            curator_agent=self.curator,
            db_path=self.db_path,
            persist_plan=boom,
            date_provider=lambda: date(2026, 8, 7),
        )
        with self.assertRaises(PlannerPersistenceError):
            agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertIsNone(
            get_daily_plan_by_date(int(self.user["id"]), "2026-08-07", db_path=self.db_path)
        )

    def test_refresh_behavior(self) -> None:
        first = self.agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        second = self.agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.tired, minutes=15),
            refresh=False,
        )
        self.assertTrue(second.reused_existing)
        self.assertEqual(first.plan.id, second.plan.id)

        third = self.agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.tired, energy=EnergyLevel.low, minutes=15, focus=2),
            refresh=True,
        )
        self.assertFalse(third.reused_existing)
        self.assertNotEqual(first.plan.id, third.plan.id)
        self.assertEqual(third.checkin.mood, Mood.tired)

    def test_context_and_ownership(self) -> None:
        with self.assertRaises(PlannerContextError):
            self.agent.create_daily_plan(99999, checkin=self._checkin())
        other = create_user("Other", db_path=self.db_path)
        with self.assertRaises(PlannerContextError):
            # Other user has no profile/roadmap/goal — clear context failure.
            DailyPlannerAgent(
                settings=_settings(gemini_api_key="test-key"),
                gemini_service=self.gemini,
                curator_agent=self.curator,
                db_path=self.db_path,
            ).create_daily_plan(int(other["id"]), checkin=self._checkin())

    def test_adaptation_explanation(self) -> None:
        create_adaptation_insight(
            user_id=int(self.user["id"]),
            insight_type="pattern",
            insight="Short sessions work better.",
            evidence=["focus_rating=2"],
            db_path=self.db_path,
        )
        upsert_user_preference(
            user_id=int(self.user["id"]),
            preference_key="preferred_content_type",
            preference_value="short_video",
            db_path=self.db_path,
        )
        text = build_adaptation_explanation(
            list_active_adaptation_insights(int(self.user["id"]), db_path=self.db_path),
            [{"preference_key": "preferred_content_type", "preference_value": "short_video"}],
            mood=Mood.tired,
            available_minutes=15,
        )
        self.assertIn("Short sessions", text)

        empty = build_adaptation_explanation([], [], mood=Mood.curious, available_minutes=20)
        self.assertIn("No prior adaptation", empty)

        result = self.agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.tired, minutes=15, focus=2),
            plan_date=date(2026, 8, 8),
        )
        self.assertTrue(result.plan.adaptation_explanation)

    def test_goal_unchanged_and_curator_scoped(self) -> None:
        result = self.agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertEqual(result.goal_title, "Improve public speaking")
        self.assertEqual(self.curator.calls[0]["user_id"], int(self.user["id"]))
        self.assertEqual(self.curator.calls[0]["milestone_id"], self.milestone.id)
        for task in result.plan.tasks:
            if task.resource_id is not None:
                self.assertIn(task.resource_id, self.resource_ids)
        # No URL invention in Gemini contract prompt
        self.assertIn("never_invent_urls", self.gemini.prompts[0])
        self.assertNotIn("example.com", self.gemini.prompts[0])

    def test_secrets_not_logged(self) -> None:
        with self.assertLogs(level=logging.INFO) as captured:
            self.agent.create_daily_plan(
                int(self.user["id"]),
                checkin=self._checkin(),
                plan_date=date(2026, 8, 9),
            )
        joined = "\n".join(captured.output)
        self.assertNotIn("test-key", joined)
        self.assertNotIn("GEMINI_API_KEY=", joined)
        self.assertEqual(count_completed_plans(int(self.user["id"]), db_path=self.db_path), 0)

    def test_retry_after_empty_structured_output(self) -> None:
        gemini = FakePlannerGemini(
            resource_ids=self.resource_ids,
            fail_times=1,
            fail_error=GeminiResponseError("empty structured output"),
        )
        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=gemini,
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 10),
        )
        result = agent.create_daily_plan(
            int(self.user["id"]),
            checkin=self._checkin(mood=Mood.tired, energy=EnergyLevel.low, minutes=15, focus=2),
        )
        self.assertEqual(gemini.call_count, 2)
        self.assertGreaterEqual(len(result.plan.tasks), 1)
        self.assertIn("CRITICAL RETRY CONSTRAINTS", gemini.prompts[1])
        self.assertIn("JSON schema", gemini.prompts[1])
        self.assertIn(str(self.resource_ids[0]), gemini.prompts[1])
        self.assertIn("Never invent URLs", gemini.prompts[1])

    def test_both_structured_attempts_fail(self) -> None:
        gemini = FakePlannerGemini(
            resource_ids=self.resource_ids,
            fail_times=2,
            fail_error=GeminiResponseError("empty structured output"),
        )
        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=gemini,
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 11),
        )
        with self.assertRaises(GeminiResponseError):
            agent.create_daily_plan(
                int(self.user["id"]),
                checkin=self._checkin(),
            )
        self.assertEqual(gemini.call_count, 2)
        self.assertIsNone(
            get_daily_plan_by_date(int(self.user["id"]), "2026-08-11", db_path=self.db_path)
        )

    def test_non_retryable_invocation_error_does_not_retry(self) -> None:
        gemini = FakePlannerGemini(
            resource_ids=self.resource_ids,
            raise_error=GeminiInvocationError("invalid api key"),
        )
        agent = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=gemini,
            curator_agent=self.curator,
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 12),
        )
        with self.assertRaises(GeminiInvocationError):
            agent.create_daily_plan(int(self.user["id"]), checkin=self._checkin())
        self.assertEqual(gemini.call_count, 1)

    def test_retry_prompt_includes_bounds_and_trusted_ids(self) -> None:
        rules = mood_capacity_rules(Mood.tired, EnergyLevel.low, available_minutes=15, focus_level=2)
        prompt = build_constrained_planner_retry_prompt(
            base_prompt="BASE",
            allowed_resource_ids={11, 22},
            available_minutes=15,
            attention_span_minutes=12,
            rules=rules,
        )
        self.assertIn("BASE", prompt)
        self.assertIn("[11, 22]", prompt)
        self.assertIn("<= 15", prompt)
        self.assertIn("<= 12", prompt)
        self.assertIn("JSON schema", prompt)
        self.assertNotIn("http://", prompt)


if __name__ == "__main__":
    unittest.main()
