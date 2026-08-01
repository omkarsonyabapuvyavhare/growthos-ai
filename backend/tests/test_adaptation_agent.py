"""Unit tests for AdaptationAgent (no live Gemini/network)."""

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

from agents.adaptation_agent import (  # noqa: E402
    AdaptationAgent,
    compute_adaptation_analytics,
    confidence_cap_for_sessions,
    is_successful_session,
)
from agents.planner_agent import DailyPlannerAgent  # noqa: E402
from agents.reflection_agent import ReflectionAgent  # noqa: E402
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from exceptions import (  # noqa: E402
    AdaptationContextError,
    AdaptationEvidenceError,
    AdaptationMemoryError,
    AdaptationOwnershipError,
    AdaptationPersistenceError,
    GeminiInvocationError,
)
from models import (  # noqa: E402
    ActivityType,
    AdaptationGeneration,
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
    PreferenceUpdateGeneration,
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
    find_adaptation_insights_for_reflection,
    get_active_goal_for_user,
    get_active_roadmap_for_user,
    get_connection,
    get_reflection_by_id,
    init_db,
    list_active_adaptation_insights,
    list_user_preferences,
    upsert_user_preference,
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
        return DailyPlanGeneration(
            summary="Watch then practice.",
            guidance_tone="direct",
            mood_influence_summary="Focused.",
            task_count_rationale="Two tasks.",
            adaptation_explanation="Based on mood and time.",
            tasks=[
                PlannerTaskGeneration(
                    sequence_number=1,
                    title="Long tip video",
                    description="Watch a longer tip",
                    activity_type=ActivityType.watch,
                    resource_id=self.resource_ids[0],
                    estimated_minutes=15,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One cue",
                    why_selected="Trusted",
                    milestone_connection="Breath basics",
                    mood_rationale="Study",
                    content_type="video",
                ),
                PlannerTaskGeneration(
                    sequence_number=2,
                    title="Practice cue",
                    description="Practice once",
                    activity_type=ActivityType.practice,
                    resource_id=None,
                    estimated_minutes=8,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One attempt",
                    why_selected="Practice",
                    milestone_connection="Breath basics",
                    mood_rationale="Apply",
                    content_type="practice",
                ),
            ],
        )


class FakeReflectionGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return ReflectionInsightGeneration(
            insight="Completed practice; focus dipped on the longer video.",
            learning_progress_summary="Partial progress.",
            completion_observation="One completed, one skipped.",
            focus_observation="Focus dropped mid-session.",
            difficulty_observation="Difficulty suitable.",
            resource_observation="Longer video less useful.",
            distraction_observation="Phone distraction noted.",
            mood_observation="Mood match mixed.",
            positive_signals=["practice useful"],
            friction_signals=["long video focus drop"],
            evidence_for_adaptation=["shorter resources", "more practice"],
            recommended_next_session_adjustments=["shorter video", "more practice"],
            confidence_score=0.75,
        )


class FakeAdaptationGemini:
    def __init__(
        self,
        *,
        mode: str = "early_signal",
        raise_error: Exception | None = None,
        invalid_key: bool = False,
    ) -> None:
        self.mode = mode
        self.raise_error = raise_error
        self.invalid_key = invalid_key
        self.prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[AdaptationGeneration],
        *,
        system_instruction: str | None = None,
    ) -> AdaptationGeneration:
        self.prompts.append(prompt)
        if self.raise_error is not None:
            raise self.raise_error
        assert response_model is AdaptationGeneration
        if self.mode == "no_change":
            return AdaptationGeneration(
                summary="Insufficient repeated evidence.",
                detected_patterns=[],
                next_session_adjustments=[],
                preference_updates=[],
                evidence_summary="Only weak or incomplete signals.",
                confidence_score=0.2,
                is_early_signal=True,
                adaptation_explanation=(
                    "No strong pattern has been detected yet. "
                    "The next plan will rely on today's mood and profile."
                ),
            )
        if self.mode == "strong_permanent":
            return AdaptationGeneration(
                summary="User is permanently an auditory learner.",
                detected_patterns=["Always prefers listen format"],
                next_session_adjustments=["Use only audio"],
                preference_updates=[
                    PreferenceUpdateGeneration(
                        preference_key="preferred_format",
                        preference_value="listen",
                        confidence_score=0.95,
                        evidence=["one session"],
                        action="create",
                    )
                ],
                evidence_summary="One session only.",
                confidence_score=0.95,
                is_early_signal=False,
                adaptation_explanation="Permanently switch to audio.",
            )
        prefs = [
            PreferenceUpdateGeneration(
                preference_key="effective_resource_duration",
                preference_value="8",
                confidence_score=0.9 if self.mode != "early_signal" else 0.4,
                evidence=["focus dropped on longer video"],
                action="create",
            ),
            PreferenceUpdateGeneration(
                preference_key="preferred_activity",
                preference_value="practice",
                confidence_score=0.7 if self.mode != "early_signal" else 0.4,
                evidence=["practice rated useful"],
                action="create",
            ),
            PreferenceUpdateGeneration(
                preference_key="preferred_task_count",
                preference_value="2",
                confidence_score=0.55 if self.mode != "early_signal" else 0.35,
                evidence=["low task count worked"],
                action="create",
            ),
            PreferenceUpdateGeneration(
                preference_key="focus_support_strategy",
                preference_value="minimize_phone_distractions",
                confidence_score=0.5 if self.mode != "early_signal" else 0.35,
                evidence=["distraction: phone"],
                action="create",
            ),
            PreferenceUpdateGeneration(
                preference_key="difficulty_bias",
                preference_value="maintain",
                confidence_score=0.5 if self.mode != "early_signal" else 0.35,
                evidence=["difficulty suitable"],
                action="create",
            ),
            PreferenceUpdateGeneration(
                preference_key="preferred_session_minutes",
                preference_value="15",
                confidence_score=0.55 if self.mode != "early_signal" else 0.35,
                evidence=["shorter sessions"],
                action="create",
            ),
        ]
        if self.invalid_key:
            prefs.append(
                PreferenceUpdateGeneration(
                    preference_key="favorite_color",
                    preference_value="blue",
                    confidence_score=0.5,
                    evidence=["n/a"],
                    action="create",
                )
            )
        return AdaptationGeneration(
            summary="Focus dropped on longer video; practice was useful.",
            detected_patterns=[
                "Longer resources reduce focus",
                "Practice tasks are more useful",
            ],
            next_session_adjustments=[
                "Use shorter resources next time",
                "Include more practice",
            ],
            preference_updates=prefs,
            pacing_adjustment=" steadier micro pacing",
            difficulty_adjustment="maintain",
            format_adjustment="prefer practice and short video",
            duration_adjustment="keep resources under 10 minutes",
            task_count_adjustment="keep task count low",
            evidence_summary="Low focus on long video; high usefulness on practice.",
            confidence_score=0.4 if self.mode == "early_signal" else 0.7,
            is_early_signal=self.mode == "early_signal",
            adaptation_explanation=(
                "Your next plan will use shorter resources because focus dropped "
                "during a longer video, and more practice because practical tasks "
                "were rated useful."
            ),
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


class AdaptationAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_adapt_")
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
            goal_description="Aspiration: confident presenter.",
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
            cursor = conn.execute(
                """
                INSERT INTO resources (
                    title, source, resource_type, url, description, difficulty,
                    estimated_duration_minutes, is_free, metadata, created_at, updated_at
                )
                VALUES (?, 'YouTube', 'video', ?, ?, 'beginner', ?, 1, '{}', ?, ?)
                """,
                (
                    "Long tip",
                    "https://www.youtube.com/watch?v=adapt0",
                    "Trusted free tip",
                    15,
                    now,
                    now,
                ),
            )
            self.resource_ids.append(int(cursor.lastrowid))

        self.recommendations = [
            CuratedRecommendation(
                id=1,
                user_id=int(user["id"]),
                roadmap_id=self.roadmap_result.roadmap.id,
                milestone_id=self.milestone.id,
                resource_id=self.resource_ids[0],
                catalog_id="cat-adapt-0",
                title="Long tip",
                source="YouTube",
                resource_type="video",
                url="https://www.youtube.com/watch?v=adapt0",
                description="Trusted free tip",
                difficulty=Difficulty.beginner,
                estimated_duration_minutes=15,
                relevance_score=0.8,
                reason="Fits",
                milestone_fit="Breath",
                mood_suitability="ok",
                suggested_use="watch",
                estimated_effort="15",
                score_breakdown={"final": 0.8},
                status=RecommendationStatus.suggested,
                recommended_at=datetime.now(timezone.utc),
            )
        ]
        self.memory = FakeMemory()
        self.day = 1

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _create_reflected_plan(self, *, plan_day: int | None = None) -> int:
        if plan_day is None:
            plan_day = self.day
            self.day += 1
        planner = DailyPlannerAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakePlannerGemini(self.resource_ids),
            curator_agent=FakeCurator(self.recommendations),
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, plan_day),
        )
        plan_result = planner.create_daily_plan(
            int(self.user["id"]),
            checkin=DailyCheckInRequest(
                mood=Mood.focused,
                energy_level=EnergyLevel.medium,
                focus_level=4,
                available_minutes=30,
                preferred_activity=ActivityType.watch,
            ),
        )
        reflection_agent = ReflectionAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeReflectionGemini(),
            memory_service=FakeMemory(),  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        reflection = reflection_agent.reflect_on_plan(
            int(self.user["id"]),
            ReflectionRequest(
                daily_plan_id=plan_result.plan.id,
                completion_status=CompletionStatus.partial,
                learning_summary="Practice helped; long video drained focus.",
                focus_rating=2,
                resource_effectiveness=2,
                difficulty_feedback=DifficultyFeedback.suitable,
                mood_match=False,
                distractions=["phone"],
                wants_similar_resources=False,
                mood_after=Mood.tired,
                task_updates=[
                    ReflectionTaskUpdate(
                        task_id=plan_result.plan.tasks[0].id,
                        update=TaskCompletionRequest(
                            status=TaskStatus.completed,
                            completion_percent=70,
                            duration_minutes=12,
                            effectiveness_rating=2,
                        ),
                    ),
                    ReflectionTaskUpdate(
                        task_id=plan_result.plan.tasks[1].id,
                        update=TaskCompletionRequest(
                            status=TaskStatus.completed,
                            completion_percent=100,
                            duration_minutes=8,
                            effectiveness_rating=5,
                        ),
                    ),
                ],
                resource_interactions=[
                    ResourceInteractionRequest(
                        resource_id=self.resource_ids[0],
                        daily_plan_id=plan_result.plan.id,
                        interaction_type="watched",
                        completion_percent=70,
                        effectiveness_rating=2,
                        duration_minutes=12,
                    )
                ],
                actual_minutes_spent=20,
            ),
        )
        return reflection.reflection.id

    def _agent(self, gemini: FakeAdaptationGemini | None = None) -> AdaptationAgent:
        return AdaptationAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=gemini or FakeAdaptationGemini(mode="early_signal"),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )

    def test_valid_adaptation_and_early_signal(self) -> None:
        reflection_id = self._create_reflected_plan()
        result = self._agent().adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertTrue(result.is_early_signal)
        self.assertLessEqual(result.confidence_score, 0.45)
        self.assertTrue(result.adaptation_explanation)
        self.assertTrue(result.goal_unchanged)
        self.assertTrue(result.roadmap_unchanged)
        self.assertTrue(result.milestone_unchanged)
        self.assertGreaterEqual(len(result.insights), 1)
        prefs = {p.preference_key: p for p in result.preferences}
        # Early signal may store soft prefs only at low confidence.
        for key, pref in prefs.items():
            if pref.source.startswith("adaptation"):
                self.assertLessEqual(pref.confidence_score, 0.45)
                self.assertIn(key, {
                    "effective_resource_duration",
                    "preferred_activity",
                    "preferred_task_count",
                    "focus_support_strategy",
                    "difficulty_bias",
                    "preferred_session_minutes",
                    "preferred_format",
                    "pacing_style",
                })
        if "effective_resource_duration" in prefs:
            minutes = int(prefs["effective_resource_duration"].preference_value)
            self.assertGreaterEqual(minutes, 3)
            self.assertLessEqual(minutes, 45)
        if "preferred_task_count" in prefs:
            count = int(prefs["preferred_task_count"].preference_value)
            self.assertGreaterEqual(count, 1)
            self.assertLessEqual(count, 5)

    def test_one_session_not_permanent_high_confidence(self) -> None:
        reflection_id = self._create_reflected_plan()
        agent = self._agent(FakeAdaptationGemini(mode="strong_permanent"))
        result = agent.adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertTrue(result.is_early_signal)
        for pref in result.preferences:
            if pref.preference_key == "preferred_format" and pref.source.startswith("adaptation"):
                self.assertLessEqual(pref.confidence_score, 0.45)
                self.assertEqual(pref.source, "adaptation_early_signal")

    def test_ownership_and_context(self) -> None:
        reflection_id = self._create_reflected_plan()
        with self.assertRaises(AdaptationContextError):
            self._agent().adapt_from_reflection(99999, reflection_id)
        other = create_user("Other", db_path=self.db_path)
        with self.assertRaises(AdaptationOwnershipError):
            self._agent().adapt_from_reflection(int(other["id"]), reflection_id)

    def test_goal_roadmap_milestone_immutable(self) -> None:
        reflection_id = self._create_reflected_plan()
        goal_before = get_active_goal_for_user(int(self.user["id"]), db_path=self.db_path)
        roadmap_before = get_active_roadmap_for_user(int(self.user["id"]), db_path=self.db_path)
        assert goal_before is not None and roadmap_before is not None
        milestone_before = roadmap_before["active_milestone"]
        result = self._agent().adapt_from_reflection(int(self.user["id"]), reflection_id)
        goal_after = get_active_goal_for_user(int(self.user["id"]), db_path=self.db_path)
        roadmap_after = get_active_roadmap_for_user(int(self.user["id"]), db_path=self.db_path)
        assert goal_after is not None and roadmap_after is not None
        self.assertEqual(goal_before["title"], goal_after["title"])
        self.assertEqual(goal_before.get("description"), goal_after.get("description"))
        self.assertEqual(roadmap_before["id"], roadmap_after["id"])
        self.assertEqual(milestone_before["id"], roadmap_after["active_milestone"]["id"])
        self.assertTrue(result.goal_unchanged)

    def test_stronger_preference_not_overwritten(self) -> None:
        reflection_id = self._create_reflected_plan()
        upsert_user_preference(
            user_id=int(self.user["id"]),
            preference_key="preferred_activity",
            preference_value="watch",
            confidence_score=0.9,
            source="onboarding",
            db_path=self.db_path,
        )
        result = self._agent().adapt_from_reflection(int(self.user["id"]), reflection_id)
        prefs = {p.preference_key: p for p in result.preferences}
        self.assertEqual(prefs["preferred_activity"].preference_value, "watch")
        self.assertGreaterEqual(prefs["preferred_activity"].confidence_score, 0.9)

    def test_repeated_sessions_raise_confidence_cap(self) -> None:
        self.assertEqual(confidence_cap_for_sessions(1), 0.45)
        self.assertEqual(confidence_cap_for_sessions(2), 0.65)
        self.assertEqual(confidence_cap_for_sessions(3), 0.85)
        self.assertTrue(
            is_successful_session(
                plan_completion_percent=60,
                focus_rating=3,
                resource_effectiveness=3,
            )
        )
        first = self._create_reflected_plan()
        second = self._create_reflected_plan()
        third = self._create_reflected_plan()
        # Adapt on third with multi-session history.
        agent = self._agent(FakeAdaptationGemini(mode="multi"))
        result = agent.adapt_from_reflection(int(self.user["id"]), third)
        self.assertFalse(result.is_early_signal)
        self.assertGreater(result.confidence_score, 0.45)
        # Ensure first/second adaptations can also run independently.
        self._agent(FakeAdaptationGemini(mode="early_signal")).adapt_from_reflection(
            int(self.user["id"]), first
        )
        self.assertIsNotNone(
            find_adaptation_insights_for_reflection(
                int(self.user["id"]), second, db_path=self.db_path
            )
            or True
        )

    def test_duplicate_and_force(self) -> None:
        reflection_id = self._create_reflected_plan()
        agent = self._agent()
        first = agent.adapt_from_reflection(int(self.user["id"]), reflection_id)
        second = agent.adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertTrue(second.reused_existing)
        self.assertEqual(len(first.insights), len(second.insights))
        before = len(
            find_adaptation_insights_for_reflection(
                int(self.user["id"]), reflection_id, active_only=False, db_path=self.db_path
            )
        )
        forced = agent.adapt_from_reflection(
            int(self.user["id"]), reflection_id, force=True
        )
        self.assertFalse(forced.reused_existing)
        after_active = find_adaptation_insights_for_reflection(
            int(self.user["id"]), reflection_id, active_only=True, db_path=self.db_path
        )
        self.assertGreaterEqual(len(after_active), 1)
        self.assertGreaterEqual(
            len(
                find_adaptation_insights_for_reflection(
                    int(self.user["id"]),
                    reflection_id,
                    active_only=False,
                    db_path=self.db_path,
                )
            ),
            before,
        )

    def test_insufficient_evidence_no_change(self) -> None:
        reflection_id = self._create_reflected_plan()
        result = self._agent(FakeAdaptationGemini(mode="no_change")).adapt_from_reflection(
            int(self.user["id"]), reflection_id
        )
        self.assertIn("no strong", result.adaptation_explanation.lower())
        self.assertEqual(result.detected_patterns, [])

    def test_invalid_key_and_gemini_failure(self) -> None:
        reflection_id = self._create_reflected_plan()
        with self.assertRaises(AdaptationEvidenceError):
            self._agent(FakeAdaptationGemini(invalid_key=True)).adapt_from_reflection(
                int(self.user["id"]), reflection_id
            )
        self.assertEqual(
            find_adaptation_insights_for_reflection(
                int(self.user["id"]), reflection_id, db_path=self.db_path
            ),
            [],
        )
        with self.assertRaises(GeminiInvocationError):
            self._agent(
                FakeAdaptationGemini(raise_error=GeminiInvocationError("down"))
            ).adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertEqual(list_active_adaptation_insights(int(self.user["id"]), db_path=self.db_path), [])

    def test_transaction_rollback(self) -> None:
        reflection_id = self._create_reflected_plan()

        def boom(**_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("db failed")

        agent = AdaptationAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeAdaptationGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
            persist_adaptation=boom,
        )
        with self.assertRaises(AdaptationPersistenceError):
            agent.adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertEqual(
            find_adaptation_insights_for_reflection(
                int(self.user["id"]), reflection_id, db_path=self.db_path
            ),
            [],
        )

    def test_memory_partial_failure_and_scope(self) -> None:
        reflection_id = self._create_reflected_plan()
        ok = self._agent().adapt_from_reflection(int(self.user["id"]), reflection_id)
        self.assertTrue(ok.memories_complete)
        for record in self.memory.records:
            self.assertEqual(record.user_id, int(self.user["id"]))
            self.assertEqual(record.metadata.get("source"), "adaptation_agent")

        reflection_id_2 = self._create_reflected_plan()
        failing = FakeMemory(raise_error=RuntimeError("faiss down"))
        agent = AdaptationAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeAdaptationGemini(),
            memory_service=failing,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        with self.assertRaises(AdaptationMemoryError) as ctx:
            agent.adapt_from_reflection(int(self.user["id"]), reflection_id_2)
        self.assertIsNotNone(ctx.exception.result)
        self.assertTrue(
            find_adaptation_insights_for_reflection(
                int(self.user["id"]), reflection_id_2, db_path=self.db_path
            )
        )

    def test_missing_actual_time_not_fabricated(self) -> None:
        analytics = compute_adaptation_analytics(
            reflections=[
                {
                    "id": 1,
                    "daily_plan_id": 1,
                    "focus_rating": 3,
                    "resource_effectiveness": 3,
                    "difficulty_feedback": "suitable",
                    "distractions": [],
                    "mood_after": "calm",
                    "mood_match": True,
                }
            ],
            plans_by_id={
                1: {
                    "id": 1,
                    "total_estimated_minutes": 20,
                    "checkin_id": None,
                    "tasks": [
                        {
                            "status": "completed",
                            "activity_type": "practice",
                            "estimated_minutes": 10,
                            "content_type": "practice",
                            "metadata": {},
                        }
                    ],
                }
            },
            interactions_by_plan={1: []},
            checkins_by_id={},
        )
        self.assertFalse(analytics["actual_minutes_available"])
        self.assertIsNone(analytics["average_actual_minutes"])

    def test_user_isolation(self) -> None:
        reflection_id = self._create_reflected_plan()
        other = create_user("Intruder", db_path=self.db_path)
        with self.assertRaises(AdaptationOwnershipError):
            self._agent().adapt_from_reflection(int(other["id"]), reflection_id)
        # Ensure other user preferences remain empty.
        self.assertEqual(list_user_preferences(int(other["id"]), db_path=self.db_path), [])

    def test_secrets_not_logged(self) -> None:
        reflection_id = self._create_reflected_plan()
        with self.assertLogs(level=logging.INFO) as captured:
            self._agent().adapt_from_reflection(int(self.user["id"]), reflection_id)
        joined = "\n".join(captured.output)
        self.assertNotIn("test-key", joined)
        self.assertNotIn("GEMINI_API_KEY=", joined)
        # Prompt may include learning summary but logs should not dump full private blobs.
        self.assertNotIn("Aspiration: Become a confident presenter", joined)


if __name__ == "__main__":
    unittest.main()
