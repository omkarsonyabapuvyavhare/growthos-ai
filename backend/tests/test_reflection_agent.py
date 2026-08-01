"""Unit tests for ReflectionAgent (no live Gemini/network)."""

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

from agents.planner_agent import DailyPlannerAgent  # noqa: E402
from agents.reflection_agent import (  # noqa: E402
    ReflectionAgent,
    compute_milestone_progress_after,
    compute_plan_completion_percent,
    compute_roadmap_progress,
)
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from exceptions import (  # noqa: E402
    GeminiInvocationError,
    ReflectionContextError,
    ReflectionEvidenceError,
    ReflectionMemoryError,
    ReflectionOwnershipError,
    ReflectionPersistenceError,
)
from models import (  # noqa: E402
    ActivityType,
    CompletionStatus,
    CuratedRecommendation,
    CuratorAgentResult,
    DailyCheckInRequest,
    DailyPlanGeneration,
    Difficulty,
    DifficultyFeedback,
    EnergyLevel,
    MilestoneGeneration,
    Mood,
    PlannerTaskGeneration,
    RecommendationStatus,
    ReflectionInsightGeneration,
    ReflectionRequest,
    ReflectionTaskUpdate,
    ResourceInteractionRequest,
    RoadmapGeneration,
    RoadmapPhaseGeneration,
    TaskCompletionRequest,
    TaskStatus,
)
from services.database import (  # noqa: E402
    create_onboarding_records,
    create_user,
    get_connection,
    get_daily_plan_by_id,
    get_milestone_by_id,
    get_reflection_for_plan,
    get_roadmap_by_id,
    init_db,
    list_resource_interactions_for_plan,
    update_milestone_progress,
)
from services.vector_models import VectorMemoryRecord  # noqa: E402
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
                    _milestone(1, "Breath and posture basics", ["breath", "posture"]),
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
    def __init__(self, recommendations: list[CuratedRecommendation]) -> None:
        self.recommendations = recommendations

    def recommend_resources(self, user_id: int, roadmap_id: int, milestone_id: int, **kwargs: Any) -> CuratorAgentResult:
        return CuratorAgentResult(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            recommendations=self.recommendations,
            candidate_count=len(self.recommendations),
            created_at=datetime.now(timezone.utc),
        )


class FakePlannerGemini:
    def __init__(self, resource_ids: list[int]) -> None:
        self.resource_ids = resource_ids

    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        assert response_model is DailyPlanGeneration
        return DailyPlanGeneration(
            summary="Two-task practice plan.",
            guidance_tone="direct",
            mood_influence_summary="Focused practice.",
            task_count_rationale="Two tasks fit the session.",
            adaptation_explanation="Based on current mood and available time.",
            tasks=[
                PlannerTaskGeneration(
                    sequence_number=1,
                    title="Watch tip",
                    description="Watch curated tip",
                    activity_type=ActivityType.watch,
                    resource_id=self.resource_ids[0],
                    estimated_minutes=10,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One cue",
                    why_selected="Trusted resource",
                    milestone_connection="Breath basics",
                    mood_rationale="Fits focus",
                    content_type="video",
                ),
                PlannerTaskGeneration(
                    sequence_number=2,
                    title="Practice cue",
                    description="Practice once",
                    activity_type=ActivityType.practice,
                    resource_id=None,
                    estimated_minutes=10,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One attempt",
                    why_selected="Practice consolidates",
                    milestone_connection="Breath basics",
                    mood_rationale="Active recall",
                    content_type="practice",
                ),
            ],
        )


class FakeReflectionGemini:
    def __init__(
        self,
        *,
        force: ReflectionInsightGeneration | None = None,
        raise_error: Exception | None = None,
        invent_completion: bool = False,
    ) -> None:
        self.force = force
        self.raise_error = raise_error
        self.invent_completion = invent_completion
        self.prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[ReflectionInsightGeneration],
        *,
        system_instruction: str | None = None,
    ) -> ReflectionInsightGeneration:
        self.prompts.append(prompt)
        if self.raise_error is not None:
            raise self.raise_error
        assert response_model is ReflectionInsightGeneration
        if self.force is not None:
            return self.force
        insight = (
            "You completed all tasks today despite evidence."
            if self.invent_completion
            else (
                "You completed one of two tasks. Practice helped, but focus dipped "
                "during the longer video. A shorter practice-heavy session may help."
            )
        )
        return ReflectionInsightGeneration(
            insight=insight,
            learning_progress_summary="Partial progress on breath control.",
            completion_observation="One task completed; one not completed.",
            focus_observation="Focus rating indicates mid-session drop.",
            difficulty_observation="Difficulty feedback preserved from user report.",
            resource_observation="Resource usefulness rated by the user.",
            distraction_observation="Distractions listed by the user.",
            mood_observation="Mood match reported by the user.",
            positive_signals=["practice usefulness"],
            friction_signals=["focus drop"],
            evidence_for_adaptation=["prefer shorter video + more practice"],
            recommended_next_session_adjustments=["shorter video", "more practice"],
            confidence_score=0.8,
        )


class FakeMemory:
    def __init__(self, *, raise_error: Exception | None = None) -> None:
        self.raise_error = raise_error
        self.records: list[VectorMemoryRecord] = []

    def add_text_memories(self, records: list[VectorMemoryRecord]) -> list[VectorMemoryRecord]:
        if self.raise_error is not None:
            raise self.raise_error
        self.records.extend(records)
        return list(records)


class ReflectionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_reflection_")
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
            for index in range(2):
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
                        f"https://www.youtube.com/watch?v=reflect{index}",
                        "Trusted free tip",
                        8 + index,
                        now,
                        now,
                    ),
                )
                self.resource_ids.append(int(cursor.lastrowid))

        recommendations = [
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
                url=f"https://www.youtube.com/watch?v=reflect{index}",
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
        planner = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakePlannerGemini(self.resource_ids),
            curator_agent=FakeCurator(recommendations),
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 1),
        )
        self.plan_result = planner.create_daily_plan(
            int(user["id"]),
            checkin=DailyCheckInRequest(
                mood=Mood.focused,
                energy_level=EnergyLevel.medium,
                focus_level=4,
                available_minutes=30,
                preferred_activity=ActivityType.watch,
                notes="ready",
            ),
        )
        self.plan = self.plan_result.plan
        self.tasks = self.plan.tasks
        self.memory = FakeMemory()
        self.gemini = FakeReflectionGemini()
        self.agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _request(
        self,
        *,
        complete_first: bool = True,
        complete_second: bool = False,
        invent: bool = False,
    ) -> ReflectionRequest:
        updates: list[ReflectionTaskUpdate] = []
        if complete_first:
            updates.append(
                ReflectionTaskUpdate(
                    task_id=self.tasks[0].id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.completed,
                        completion_percent=100,
                        duration_minutes=9,
                        effectiveness_rating=4,
                        notes="Useful tip",
                    ),
                )
            )
        if complete_second:
            updates.append(
                ReflectionTaskUpdate(
                    task_id=self.tasks[1].id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.completed,
                        completion_percent=100,
                        duration_minutes=8,
                    ),
                )
            )
        elif not complete_first:
            pass
        else:
            updates.append(
                ReflectionTaskUpdate(
                    task_id=self.tasks[1].id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.skipped,
                        completion_percent=0,
                    ),
                )
            )
        return ReflectionRequest(
            daily_plan_id=self.plan.id,
            completion_status=(
                CompletionStatus.completed
                if complete_first and complete_second
                else CompletionStatus.partial
                if complete_first
                else CompletionStatus.skipped
            ),
            learning_summary="Practiced one breath cue.",
            focus_rating=3,
            resource_effectiveness=4,
            difficulty_feedback=DifficultyFeedback.suitable,
            mood_match=True,
            distractions=["phone"],
            wants_similar_resources=True,
            mood_after=Mood.calm,
            task_updates=updates,
            resource_interactions=[
                ResourceInteractionRequest(
                    resource_id=self.resource_ids[0],
                    daily_plan_id=self.plan.id,
                    interaction_type="watched",
                    completion_percent=80,
                    effectiveness_rating=4,
                    duration_minutes=9,
                )
            ]
            if complete_first
            else [],
            actual_minutes_spent=18 if complete_first else None,
        )

    def test_valid_reflection_persisted(self) -> None:
        result = self.agent.reflect_on_plan(int(self.user["id"]), self._request())
        self.assertEqual(result.reflection.user_id, int(self.user["id"]))
        self.assertEqual(result.reflection.daily_plan_id, self.plan.id)
        self.assertTrue(result.reflection.insight)
        self.assertEqual(result.reflection.focus_rating, 3)
        self.assertEqual(result.reflection.difficulty_feedback, DifficultyFeedback.suitable)
        self.assertEqual(result.reflection.distractions, ["phone"])
        self.assertFalse(result.reused_existing)
        stored = get_reflection_for_plan(self.plan.id, db_path=self.db_path)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertIn("completed one of two", stored["insight"].lower())

    def test_task_and_interaction_persistence(self) -> None:
        self.agent.reflect_on_plan(int(self.user["id"]), self._request())
        plan = get_daily_plan_by_id(self.plan.id, db_path=self.db_path)
        assert plan is not None
        statuses = {int(t["id"]): t["status"] for t in plan["tasks"]}
        self.assertEqual(statuses[self.tasks[0].id], "completed")
        self.assertEqual(statuses[self.tasks[1].id], "skipped")
        interactions = list_resource_interactions_for_plan(
            self.plan.id, user_id=int(self.user["id"]), db_path=self.db_path
        )
        self.assertGreaterEqual(len(interactions), 1)
        self.assertEqual(int(interactions[0]["resource_id"]), self.resource_ids[0])

    def test_ownership_and_invalid_tasks(self) -> None:
        with self.assertRaises(ReflectionContextError):
            self.agent.reflect_on_plan(99999, self._request())
        other = create_user("Other", db_path=self.db_path)
        with self.assertRaises(ReflectionOwnershipError):
            self.agent.reflect_on_plan(int(other["id"]), self._request())
        bad = self._request()
        bad = bad.model_copy(
            update={
                "task_updates": [
                    ReflectionTaskUpdate(
                        task_id=999999,
                        update=TaskCompletionRequest(status=TaskStatus.completed),
                    )
                ]
            }
        )
        with self.assertRaises(ReflectionEvidenceError):
            self.agent.reflect_on_plan(int(self.user["id"]), bad)

    def test_gemini_evidence_and_no_invention(self) -> None:
        request = self._request(complete_first=False)
        request = request.model_copy(
            update={
                "task_updates": [],
                "resource_interactions": [],
                "actual_minutes_spent": None,
                "completion_status": CompletionStatus.skipped,
            }
        )
        agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeReflectionGemini(invent_completion=True),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        with self.assertRaises(Exception):
            agent.reflect_on_plan(int(self.user["id"]), request)
        self.assertIsNone(get_reflection_for_plan(self.plan.id, db_path=self.db_path))

        ok_agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        result = ok_agent.reflect_on_plan(int(self.user["id"]), request)
        self.assertEqual(result.plan_completion_percent, 0.0)
        prompt = self.gemini.prompts[0]
        self.assertIn('"actual_minutes_spent": null', prompt)
        self.assertIn("do not invent", prompt.lower())
        self.assertIn(str(self.plan.id), prompt)

    def test_progress_rules(self) -> None:
        self.assertEqual(compute_plan_completion_percent([]), 0.0)
        zero = compute_plan_completion_percent(
            [
                {"estimated_minutes": 10, "status": "pending"},
                {"estimated_minutes": 10, "status": "skipped"},
            ]
        )
        self.assertEqual(zero, 0.0)
        partial = compute_plan_completion_percent(
            [
                {"estimated_minutes": 10, "status": "completed"},
                {"estimated_minutes": 10, "status": "skipped"},
            ]
        )
        full = compute_plan_completion_percent(
            [
                {"estimated_minutes": 10, "status": "completed"},
                {"estimated_minutes": 10, "status": "completed"},
            ]
        )
        self.assertEqual(partial, 50.0)
        self.assertEqual(full, 100.0)
        self.assertGreater(full, partial)
        self.assertTrue(0 <= partial <= 100)
        before, status = compute_milestone_progress_after(
            10.0, plan_completion_percent=50.0, completed_task_count=1
        )
        self.assertGreaterEqual(before, 10.0)
        self.assertLess(before, 100.0)
        same, _ = compute_milestone_progress_after(
            40.0, plan_completion_percent=80.0, completed_task_count=0
        )
        self.assertEqual(same, 40.0)
        avg = compute_roadmap_progress(
            [{"progress_percent": 20.0}, {"progress_percent": 40.0}]
        )
        self.assertEqual(avg, 30.0)

        milestone_before = float(
            get_milestone_by_id(self.milestone.id, db_path=self.db_path)["progress_percent"]
        )
        roadmap_before = float(
            get_roadmap_by_id(self.roadmap_result.roadmap.id, db_path=self.db_path)[
                "progress_percent"
            ]
        )
        partial_result = self.agent.reflect_on_plan(
            int(self.user["id"]), self._request(complete_first=True, complete_second=False)
        )
        self.assertGreater(partial_result.milestone_progress_after, milestone_before)
        self.assertGreaterEqual(
            partial_result.milestone_progress_after,
            partial_result.milestone_progress_before,
        )
        self.assertGreaterEqual(
            partial_result.roadmap_progress_after,
            roadmap_before,
        )
        self.assertLess(partial_result.milestone_progress_after, 100.0)
        self.assertAlmostEqual(partial_result.plan_completion_percent, 50.0)

    def test_progress_never_decreases(self) -> None:
        update_milestone_progress(
            self.milestone.id, progress_percent=55.0, db_path=self.db_path
        )
        result = self.agent.reflect_on_plan(
            int(self.user["id"]),
            self._request(complete_first=False),
        )
        # No completed tasks => no increase, but also no decrease.
        self.assertGreaterEqual(result.milestone_progress_after, 55.0)

    def test_duplicate_reflection_returns_existing(self) -> None:
        first = self.agent.reflect_on_plan(int(self.user["id"]), self._request())
        second = self.agent.reflect_on_plan(
            int(self.user["id"]),
            self._request(complete_first=True, complete_second=True),
        )
        self.assertTrue(second.reused_existing)
        self.assertEqual(first.reflection.id, second.reflection.id)

    def test_gemini_failure_writes_nothing(self) -> None:
        agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeReflectionGemini(
                raise_error=GeminiInvocationError("down")
            ),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        with self.assertRaises(GeminiInvocationError):
            agent.reflect_on_plan(int(self.user["id"]), self._request())
        self.assertIsNone(get_reflection_for_plan(self.plan.id, db_path=self.db_path))
        plan = get_daily_plan_by_id(self.plan.id, db_path=self.db_path)
        assert plan is not None
        self.assertTrue(all(t["status"] == "pending" for t in plan["tasks"]))

    def test_transaction_rollback(self) -> None:
        def boom(**_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("insert failed")

        agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
            persist_reflection=boom,
        )
        with self.assertRaises(ReflectionPersistenceError):
            agent.reflect_on_plan(int(self.user["id"]), self._request())
        self.assertIsNone(get_reflection_for_plan(self.plan.id, db_path=self.db_path))
        plan = get_daily_plan_by_id(self.plan.id, db_path=self.db_path)
        assert plan is not None
        self.assertTrue(all(t["status"] == "pending" for t in plan["tasks"]))
        self.assertEqual(
            list_resource_interactions_for_plan(self.plan.id, db_path=self.db_path),
            [],
        )

    def test_memory_user_scope_and_partial_failure(self) -> None:
        result = self.agent.reflect_on_plan(int(self.user["id"]), self._request())
        self.assertTrue(result.memories_complete)
        self.assertEqual(len(result.memory_ids), 3)
        for record in self.memory.records:
            self.assertEqual(record.user_id, int(self.user["id"]))
            self.assertEqual(record.metadata.get("source"), "reflection_agent")

        # Fresh plan date for second reflection path via new plan is heavy;
        # test memory failure on a second user/plan setup would be large.
        # Use persist success then memory boom on a new agent with existing cleared plan:
        # create another plan day.
        planner = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakePlannerGemini(self.resource_ids),
            curator_agent=FakeCurator(
                [
                    CuratedRecommendation(
                        id=1,
                        user_id=int(self.user["id"]),
                        roadmap_id=self.roadmap_result.roadmap.id,
                        milestone_id=self.milestone.id,
                        resource_id=self.resource_ids[0],
                        catalog_id="cat-x",
                        title="Resource 1",
                        source="YouTube",
                        resource_type="video",
                        url="https://www.youtube.com/watch?v=reflect0",
                        description="Trusted free tip",
                        difficulty=Difficulty.beginner,
                        estimated_duration_minutes=8,
                        relevance_score=0.8,
                        reason="Fits",
                        milestone_fit="Breath",
                        mood_suitability="ok",
                        suggested_use="watch",
                        estimated_effort="10",
                        score_breakdown={},
                        status=RecommendationStatus.suggested,
                        recommended_at=datetime.now(timezone.utc),
                    )
                ]
            ),
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 2),
        )
        plan2 = planner.create_daily_plan(
            int(self.user["id"]),
            checkin=DailyCheckInRequest(
                mood=Mood.focused,
                energy_level=EnergyLevel.medium,
                focus_level=4,
                available_minutes=30,
                preferred_activity=ActivityType.watch,
            ),
        ).plan
        failing_memory = FakeMemory(raise_error=RuntimeError("faiss down"))
        agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeReflectionGemini(),
            memory_service=failing_memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        request = ReflectionRequest(
            daily_plan_id=plan2.id,
            completion_status=CompletionStatus.partial,
            learning_summary="Did one task",
            focus_rating=3,
            resource_effectiveness=3,
            difficulty_feedback=DifficultyFeedback.suitable,
            mood_match=True,
            distractions=[],
            wants_similar_resources=False,
            mood_after=Mood.tired,
            task_updates=[
                ReflectionTaskUpdate(
                    task_id=plan2.tasks[0].id,
                    update=TaskCompletionRequest(status=TaskStatus.completed),
                )
            ],
        )
        with self.assertRaises(ReflectionMemoryError) as ctx:
            agent.reflect_on_plan(int(self.user["id"]), request)
        self.assertIsNotNone(ctx.exception.result)
        self.assertIsNotNone(get_reflection_for_plan(plan2.id, db_path=self.db_path))
        self.assertFalse(ctx.exception.result.memories_complete)  # type: ignore[union-attr]

    def test_secrets_not_logged(self) -> None:
        with self.assertLogs(level=logging.INFO) as captured:
            self.agent.reflect_on_plan(int(self.user["id"]), self._request())
        joined = "\n".join(captured.output)
        self.assertNotIn("test-key", joined)
        self.assertNotIn("GEMINI_API_KEY=", joined)


if __name__ == "__main__":
    unittest.main()
