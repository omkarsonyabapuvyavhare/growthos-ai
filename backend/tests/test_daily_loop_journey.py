"""
Day 1 → Day 2 journey test through LangGraph workflows.

Uses real agents with deterministic fake Gemini/Curator/memory services.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Type

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.adaptation_agent import AdaptationAgent  # noqa: E402
from agents.planner_agent import DailyPlannerAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.reflection_agent import ReflectionAgent  # noqa: E402
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from models import (  # noqa: E402
    ActivityType,
    AdaptationGeneration,
    CompletionStatus,
    CuratedRecommendation,
    CuratorAgentResult,
    CurrentLevel,
    DailyCheckInRequest,
    DailyPlanGeneration,
    Difficulty,
    DifficultyFeedback,
    EnergyLevel,
    LearningStyle,
    MilestoneGeneration,
    Mood,
    OnboardingRequest,
    PlannerTaskGeneration,
    PreferenceUpdateGeneration,
    PreferredLearningTime,
    ProfileInterpretation,
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
    get_active_goal_for_user,
    get_connection,
    init_db,
    list_active_adaptation_insights,
)
from services.vector_models import VectorMemoryRecord  # noqa: E402
from tests.test_embedding import _settings  # noqa: E402
from workflows.daily_loop import DailyLoopWorkflow  # noqa: E402
from workflows.onboarding import OnboardingWorkflow  # noqa: E402

GOAL_TITLE = "Improve public speaking"


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


class FakeProfileGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return ProfileInterpretation(
            identity_summary="Curious beginner presenter",
            aspiration_summary="Become calm on stage",
            motivation_summary="Lead updates confidently",
            current_state_summary="Beginner with limited practice",
            target_state_summary="Deliver a short talk",
            strengths=["motivation"],
            likely_challenges=["nerves", "phone distraction"],
            learning_preferences_summary="Prefers video and practice",
            recommended_pacing="Short consistent sessions",
            attention_strategy="Keep resources short",
            consistency_strategy="Practice daily",
            initial_personalization_insights=["Start with breath work"],
        )


class FakeRoadmapGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
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


class FakeCurator:
    def __init__(self, resource_ids: list[int], user_id: int, roadmap_id: int, milestone_id: int) -> None:
        self.resource_ids = resource_ids
        self.user_id = user_id
        self.roadmap_id = roadmap_id
        self.milestone_id = milestone_id
        self.calls = 0

    def recommend_resources(self, user_id: int, roadmap_id: int, milestone_id: int, **kwargs: Any) -> CuratorAgentResult:
        self.calls += 1
        recs = [
            CuratedRecommendation(
                id=index + 1,
                user_id=user_id,
                roadmap_id=roadmap_id,
                milestone_id=milestone_id,
                resource_id=resource_id,
                catalog_id=f"cat-{resource_id}",
                title=f"Trusted resource {index + 1}",
                source="YouTube",
                resource_type="video",
                url=f"https://www.youtube.com/watch?v=journey{index}",
                description="Trusted free tip",
                difficulty=Difficulty.beginner,
                estimated_duration_minutes=8 if index == 0 else 15,
                relevance_score=0.85,
                reason="Fits active milestone",
                milestone_fit="Breath basics",
                mood_suitability="Suitable",
                suggested_use="Watch then practice",
                estimated_effort="10 minutes",
                score_breakdown={"final": 0.85},
                status=RecommendationStatus.suggested,
                recommended_at=datetime.now(timezone.utc),
            )
            for index, resource_id in enumerate(self.resource_ids)
        ]
        return CuratorAgentResult(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            recommendations=recs,
            candidate_count=len(recs),
            created_at=datetime.now(timezone.utc),
        )


class FakePlannerGemini:
    def __init__(self, resource_ids: list[int]) -> None:
        self.resource_ids = resource_ids
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        mood = "focused"
        available = 30
        for line in prompt.splitlines():
            if '"mood":' in line and "influence" not in line and "rule" not in line:
                mood = line.split(":", 1)[1].strip().strip(",").strip('"')
            if '"available_minutes":' in line:
                try:
                    available = int(line.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass

        # Prefer adaptation-driven practice when explanation mentions practice/shorter.
        wants_practice = "practice" in prompt.lower()
        shorter = "shorter" in prompt.lower() or "15 minutes" in prompt or available <= 15

        if mood in {"tired", "low_energy", "stressed"} or available <= 15:
            mins = min(8, available // 2) if available >= 6 else max(3, available // 2)
            tasks = [
                PlannerTaskGeneration(
                    sequence_number=1,
                    title="Short calm tip",
                    description="Watch a short trusted tip",
                    activity_type=ActivityType.watch,
                    resource_id=self.resource_ids[0],
                    estimated_minutes=max(3, mins),
                    difficulty=Difficulty.beginner,
                    expected_outcome="One calm cue",
                    why_selected="Short resource for low energy",
                    milestone_connection="Breath and posture basics",
                    mood_rationale="Tired capacity",
                    content_type="video",
                ),
                PlannerTaskGeneration(
                    sequence_number=2,
                    title="Tiny practice",
                    description="Practice once",
                    activity_type=ActivityType.practice,
                    resource_id=None,
                    estimated_minutes=max(3, available - max(3, mins)),
                    difficulty=Difficulty.beginner,
                    expected_outcome="One attempt",
                    why_selected="Micro practice",
                    milestone_connection="Breath and posture basics",
                    mood_rationale="Low pressure",
                    content_type="practice",
                ),
            ]
            while sum(t.estimated_minutes for t in tasks) > available and len(tasks) > 1:
                tasks.pop()
            return DailyPlanGeneration(
                summary="A short tired-friendly plan.",
                guidance_tone="calm",
                mood_influence_summary="Fewer shorter tasks for low energy.",
                task_count_rationale="1-2 easy tasks fit 15 minutes.",
                adaptation_explanation=(
                    "Based on current mood and available time."
                    if "adaptation" not in prompt.lower()
                    else "Early signal: keep sessions short."
                ),
                tasks=tasks,
            )

        # Focused day — lean into practice when adaptation evidence exists.
        # Keep each task within a typical 10-minute attention span.
        tasks = [
            PlannerTaskGeneration(
                sequence_number=1,
                title="Short study tip",
                description="Review a short curated tip",
                activity_type=ActivityType.watch,
                resource_id=self.resource_ids[0],
                estimated_minutes=6 if shorter else 8,
                difficulty=Difficulty.beginner,
                expected_outcome="Clear technique",
                why_selected="Trusted curated resource",
                milestone_connection="Breath and posture basics",
                mood_rationale="Focused study",
                content_type="video",
            ),
            PlannerTaskGeneration(
                sequence_number=2,
                title="Applied practice drill",
                description="Practice the cue aloud",
                activity_type=ActivityType.practice,
                resource_id=None,
                estimated_minutes=10 if wants_practice else 8,
                difficulty=Difficulty.intermediate,
                expected_outcome="Confident short delivery",
                why_selected="Practice was useful yesterday",
                milestone_connection="Breath and posture basics",
                mood_rationale="Deeper practice",
                content_type="practice",
            ),
            PlannerTaskGeneration(
                sequence_number=3,
                title="Quick synthesis",
                description="Note one improvement",
                activity_type=ActivityType.review,
                resource_id=None,
                estimated_minutes=5,
                difficulty=Difficulty.beginner,
                expected_outcome="One improvement note",
                why_selected="Consolidate learning",
                milestone_connection="Breath and posture basics",
                mood_rationale="Focused review",
                content_type="review",
            ),
        ]
        selected: list[PlannerTaskGeneration] = []
        used = 0
        for task in tasks:
            if used + task.estimated_minutes <= available:
                selected.append(task)
                used += task.estimated_minutes
        for index, task in enumerate(selected, start=1):
            task.sequence_number = index
        return DailyPlanGeneration(
            summary="A focused practice-oriented plan.",
            guidance_tone="direct",
            mood_influence_summary="Deeper practice with focused energy.",
            task_count_rationale="2-3 tasks fit available minutes.",
            adaptation_explanation=(
                "Your next plan will use shorter resources because focus dropped "
                "during a longer video, and more practice because practical tasks "
                "were rated useful."
                if wants_practice or shorter
                else "Based on current mood and available time."
            ),
            tasks=selected,
        )


class FakeReflectionGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return ReflectionInsightGeneration(
            insight=(
                "You completed part of the plan. Practice was useful, but focus "
                "dropped during the longer resource."
            ),
            learning_progress_summary="Partial progress on breath control.",
            completion_observation="Partial task completion.",
            focus_observation="Focus rating was low on the longer task.",
            difficulty_observation="Difficulty felt suitable.",
            resource_observation="Longer resource was less useful.",
            distraction_observation="Phone distraction noted.",
            mood_observation="Mood match was imperfect.",
            positive_signals=["practice usefulness"],
            friction_signals=["long resource focus drop"],
            evidence_for_adaptation=["shorter resources", "more practice"],
            recommended_next_session_adjustments=["shorter video", "more practice"],
            confidence_score=0.7,
        )


class FakeAdaptationGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
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
            preference_updates=[
                PreferenceUpdateGeneration(
                    preference_key="effective_resource_duration",
                    preference_value="8",
                    confidence_score=0.4,
                    evidence=["focus dropped on longer video"],
                    action="create",
                ),
                PreferenceUpdateGeneration(
                    preference_key="preferred_activity",
                    preference_value="practice",
                    confidence_score=0.4,
                    evidence=["practice rated useful"],
                    action="create",
                ),
            ],
            evidence_summary="Day 1 low focus on longer resource; practice useful.",
            confidence_score=0.4,
            is_early_signal=True,
            adaptation_explanation=(
                "Your next plan will use shorter resources because focus dropped "
                "during a longer video, and more practice because practical tasks "
                "were rated useful."
            ),
        )


class FakeMemory:
    def __init__(self) -> None:
        self.records: list[VectorMemoryRecord] = []

    def add_text_memories(self, records: list[VectorMemoryRecord]) -> list[VectorMemoryRecord]:
        self.records.extend(records)
        return list(records)

    def semantic_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class DailyLoopJourneyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_journey_")
        self.db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self.db_path)
        self.memory = FakeMemory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _seed_resources(self) -> list[int]:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ids: list[int] = []
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
                        f"Trusted resource {index + 1}",
                        f"https://www.youtube.com/watch?v=journey{index}",
                        "Trusted free tip",
                        8 if index == 0 else 15,
                        now,
                        now,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def test_day1_to_day2_adaptation_journey(self) -> None:
        settings = _settings(gemini_api_key="test-key")
        profile_agent = ProfileAgent(
            settings=settings,
            gemini_service=FakeProfileGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        roadmap_agent = RoadmapAgent(
            settings=settings,
            gemini_service=FakeRoadmapGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
            skip_memory_retrieval=True,
        )
        onboard = OnboardingWorkflow(profile_agent, roadmap_agent).run(
            OnboardingRequest(
                display_name="Journey User",
                learning_goal=GOAL_TITLE,
                aspiration="Become a calm presenter",
                motivation="Lead team updates",
                current_level=CurrentLevel.beginner,
                target_outcome="Deliver a 5-minute talk",
                preferred_formats=["video", "practice"],
                learning_style=LearningStyle.mixed,
                daily_available_minutes=30,
                preferred_session_minutes=15,
                attention_span_minutes=10,
                preferred_learning_time=PreferredLearningTime.evening,
                habits=["review"],
                distractions=["phone"],
            )
        )
        self.assertEqual(onboard.goal.title, GOAL_TITLE)
        user_id = onboard.user.id
        resource_ids = self._seed_resources()
        curator = FakeCurator(
            resource_ids,
            user_id,
            onboard.roadmap.id,
            onboard.active_milestone.id if onboard.active_milestone else 0,
        )
        planner_gemini = FakePlannerGemini(resource_ids)
        planner = DailyPlannerAgent(
            settings=settings,
            gemini_service=planner_gemini,
            curator_agent=curator,  # type: ignore[arg-type]
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 1),
        )
        reflection_agent = ReflectionAgent(
            settings=settings,
            gemini_service=FakeReflectionGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        adaptation_agent = AdaptationAgent(
            settings=settings,
            gemini_service=FakeAdaptationGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        loop = DailyLoopWorkflow(
            planner_agent=planner,
            reflection_agent=reflection_agent,
            adaptation_agent=adaptation_agent,
        )

        # --- Day 1 planning: tired / low energy / 15 minutes ---
        day1 = loop.run_planning(
            user_id,
            DailyCheckInRequest(
                mood=Mood.tired,
                energy_level=EnergyLevel.low,
                focus_level=2,
                available_minutes=15,
                preferred_activity=ActivityType.watch,
                notes="tired evening",
            ),
            plan_date=date(2026, 8, 1),
        )
        self.assertTrue(day1.awaiting_user_completion)
        self.assertLessEqual(len(day1.tasks), 5)
        self.assertLessEqual(day1.plan.total_estimated_minutes, 15)
        self.assertTrue(all(t.difficulty == Difficulty.beginner for t in day1.tasks))
        self.assertLessEqual(len(day1.tasks), 2)
        for task in day1.tasks:
            if task.resource_id is not None:
                self.assertIn(task.resource_id, resource_ids)

        # --- Day 1 post-session: partial completion + adaptation ---
        updates = [
            ReflectionTaskUpdate(
                task_id=day1.tasks[0].id,
                update=TaskCompletionRequest(
                    status=TaskStatus.completed,
                    completion_percent=70,
                    duration_minutes=min(12, day1.tasks[0].estimated_minutes + 4),
                    effectiveness_rating=2,
                ),
            )
        ]
        if len(day1.tasks) > 1:
            updates.append(
                ReflectionTaskUpdate(
                    task_id=day1.tasks[1].id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.completed,
                        completion_percent=100,
                        duration_minutes=day1.tasks[1].estimated_minutes,
                        effectiveness_rating=5,
                    ),
                )
            )
        interactions = []
        if day1.tasks[0].resource_id is not None:
            interactions.append(
                ResourceInteractionRequest(
                    resource_id=day1.tasks[0].resource_id,
                    daily_plan_id=day1.plan.id,
                    interaction_type="watched",
                    completion_percent=70,
                    effectiveness_rating=2,
                    duration_minutes=12,
                )
            )
        post1 = loop.run_post_session(
            user_id,
            ReflectionRequest(
                daily_plan_id=day1.plan.id,
                completion_status=CompletionStatus.partial,
                learning_summary="Practice helped; longer tip drained focus.",
                focus_rating=2,
                resource_effectiveness=2,
                difficulty_feedback=DifficultyFeedback.suitable,
                mood_match=False,
                distractions=["phone"],
                wants_similar_resources=False,
                mood_after=Mood.tired,
                task_updates=updates,
                resource_interactions=interactions,
                actual_minutes_spent=14,
            ),
        )
        self.assertTrue(post1.reflection.insight)
        self.assertTrue(post1.adaptation.is_early_signal)
        self.assertIn("practice", post1.adaptation_explanation.lower())
        self.assertTrue(post1.goal_unchanged)
        self.assertGreaterEqual(len(list_active_adaptation_insights(user_id, db_path=self.db_path)), 1)
        progress_after_day1 = post1.reflection_result.milestone_progress_after

        # Point planner date provider at Day 2 and rebuild planner/loop with same agents
        # but updated date for same-day uniqueness.
        planner_day2 = DailyPlannerAgent(
            settings=settings,
            gemini_service=planner_gemini,
            curator_agent=curator,  # type: ignore[arg-type]
            db_path=self.db_path,
            date_provider=lambda: date(2026, 8, 2),
        )
        loop2 = DailyLoopWorkflow(
            planner_agent=planner_day2,
            reflection_agent=reflection_agent,
            adaptation_agent=adaptation_agent,
        )

        # --- Day 2 planning: focused / 30 minutes, adaptation-aware ---
        day2 = loop2.run_planning(
            user_id,
            DailyCheckInRequest(
                mood=Mood.focused,
                energy_level=EnergyLevel.high,
                focus_level=5,
                available_minutes=30,
                preferred_activity=ActivityType.practice,
                notes="ready to practice",
            ),
            plan_date=date(2026, 8, 2),
        )
        self.assertLessEqual(len(day2.tasks), 5)
        self.assertLessEqual(day2.plan.total_estimated_minutes, 30)
        self.assertNotEqual(day1.plan.id, day2.plan.id)
        self.assertNotEqual(
            [t.title for t in day1.tasks],
            [t.title for t in day2.tasks],
        )
        self.assertGreaterEqual(len(day2.tasks), len(day1.tasks))
        practice_count = sum(1 for t in day2.tasks if t.activity_type == ActivityType.practice)
        self.assertGreaterEqual(practice_count, 1)
        explanation = (day2.plan.adaptation_explanation or "").lower()
        self.assertTrue(
            "practice" in explanation or "shorter" in explanation or "focus" in explanation
        )
        for task in day2.tasks:
            if task.resource_id is not None:
                self.assertIn(task.resource_id, resource_ids)

        goal = get_active_goal_for_user(user_id, db_path=self.db_path)
        assert goal is not None
        self.assertEqual(goal["title"], GOAL_TITLE)
        self.assertGreaterEqual(progress_after_day1, 0.0)

        # User isolation: another user has no access to this plan via ownership rules
        # (covered deeply in agent tests; journey asserts goal uniqueness).
        self.assertEqual(onboard.completed_steps, ["profile", "roadmap", "finish"])
        self.assertEqual(day1.completed_steps, ["plan", "await_user_completion"])
        self.assertEqual(post1.completed_steps, ["reflection", "adaptation"])


if __name__ == "__main__":
    unittest.main()
