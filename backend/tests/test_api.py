"""FastAPI REST API tests with injectable fake services (no live Gemini)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Type

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.deps import AppServices  # noqa: E402
from agents.adaptation_agent import AdaptationAgent  # noqa: E402
from agents.planner_agent import DailyPlannerAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.reflection_agent import ReflectionAgent  # noqa: E402
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from config import Settings  # noqa: E402
from exceptions import PlannerAgentError, WorkflowExecutionError  # noqa: E402
from main import create_app  # noqa: E402
from models import (  # noqa: E402
    ActivityType,
    AdaptationGeneration,
    CompletionStatus,
    CuratedRecommendation,
    CuratorAgentResult,
    DailyPlanGeneration,
    Difficulty,
    EnergyLevel,
    MilestoneGeneration,
    Mood,
    PlannerTaskGeneration,
    PreferenceUpdateGeneration,
    ProfileInterpretation,
    RecommendationStatus,
    ReflectionInsightGeneration,
    RoadmapGeneration,
    RoadmapPhaseGeneration,
    TaskStatus,
)
from services.database import get_connection, init_db  # noqa: E402
from services.vector_models import VectorMemoryRecord  # noqa: E402
from workflows.daily_loop import DailyLoopWorkflow  # noqa: E402
from workflows.onboarding import OnboardingWorkflow  # noqa: E402


def _settings(db_path: Path, **overrides: Any) -> Settings:
    values = dict(
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        gemini_temperature=0.2,
        gemini_max_retries=1,
        gemini_request_timeout_seconds=10,
        gemini_embedding_model="models/gemini-embedding-001",
        database_url=f"sqlite:///{db_path.as_posix()}",
        frontend_origin="http://localhost:3000",
        faiss_index_path=str(db_path.parent / "faiss" / "index.faiss"),
        faiss_metadata_path=str(db_path.parent / "faiss" / "metadata.json"),
        faiss_catalog_index_path=str(db_path.parent / "catalog" / "index.faiss"),
        faiss_catalog_metadata_path=str(db_path.parent / "catalog" / "metadata.json"),
        resource_catalog_path=str(BACKEND_ROOT / "data" / "sample_resources.json"),
    )
    values.update(overrides)
    return Settings(**values)


class FakeMemory:
    def __init__(self) -> None:
        self.records: list[VectorMemoryRecord] = []

    def add_text_memories(self, records: list[VectorMemoryRecord]) -> list[VectorMemoryRecord]:
        self.records.extend(records)
        return list(records)

    def semantic_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class FakeProfileGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return ProfileInterpretation(
            identity_summary="Learner",
            aspiration_summary="Present calmly",
            motivation_summary="Lead updates",
            current_state_summary="Beginner",
            target_state_summary="Short talk",
            strengths=["curious"],
            likely_challenges=["nerves"],
            learning_preferences_summary="mixed",
            recommended_pacing="steady",
            attention_strategy="short",
            consistency_strategy="daily",
            initial_personalization_insights=["start small"],
        )


class FakeRoadmapGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return RoadmapGeneration(
            title="Speaking Foundations",
            summary="Practical roadmap",
            estimated_duration_weeks=6,
            pacing_rationale="short sessions",
            personalization_rationale="beginner",
            phases=[
                RoadmapPhaseGeneration(
                    sequence_number=1,
                    title="Foundations",
                    description="Breath",
                    expected_outcome="Calm start",
                    milestones=[
                        MilestoneGeneration(
                            sequence_number=1,
                            title="Breath basics",
                            description="Breath work",
                            skills=["breath"],
                            suggested_activities=["practice"],
                            completion_criteria="Done",
                            estimated_sessions=2,
                            estimated_minutes=20,
                            difficulty=Difficulty.beginner,
                        ),
                        MilestoneGeneration(
                            sequence_number=2,
                            title="Outline drill",
                            description="Structure",
                            skills=["structure"],
                            suggested_activities=["outline"],
                            completion_criteria="Done",
                            estimated_sessions=2,
                            estimated_minutes=20,
                            difficulty=Difficulty.beginner,
                        ),
                    ],
                ),
                RoadmapPhaseGeneration(
                    sequence_number=2,
                    title="Delivery",
                    description="Speak",
                    expected_outcome="5-minute talk",
                    milestones=[
                        MilestoneGeneration(
                            sequence_number=1,
                            title="Story framing",
                            description="Story",
                            skills=["story"],
                            suggested_activities=["practice"],
                            completion_criteria="Done",
                            estimated_sessions=2,
                            estimated_minutes=20,
                            difficulty=Difficulty.beginner,
                        )
                    ],
                ),
            ],
        )


class FakePlannerGemini:
    def __init__(self, resource_ids: list[int]) -> None:
        self.resource_ids = resource_ids

    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        available = 30
        mood = "focused"
        for line in prompt.splitlines():
            if '"available_minutes":' in line:
                try:
                    available = int(line.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass
            if '"mood":' in line and "influence" not in line and "rule" not in line:
                mood = line.split(":", 1)[1].strip().strip(",").strip('"')
        wants_practice = "practice" in prompt.lower()
        attention = 10
        for line in prompt.splitlines():
            if '"attention_span_minutes":' in line:
                try:
                    attention = int(line.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass
        per_task_cap = max(3, min(available, attention))
        if mood in {"tired", "low_energy", "stressed"} or available <= 15:
            watch_mins = min(per_task_cap, max(3, available // 2))
            practice_mins = min(per_task_cap, max(3, available - watch_mins))
            tasks = [
                PlannerTaskGeneration(
                    sequence_number=1,
                    title="Short calm tip",
                    description="Watch a short trusted tip",
                    activity_type=ActivityType.watch,
                    resource_id=self.resource_ids[0],
                    estimated_minutes=watch_mins,
                    difficulty=Difficulty.beginner,
                    expected_outcome="Cue",
                    why_selected="Short trusted resource",
                    milestone_connection="Breath",
                    mood_rationale="Low energy",
                    content_type="video",
                ),
                PlannerTaskGeneration(
                    sequence_number=2,
                    title="Tiny practice",
                    description="Practice once",
                    activity_type=ActivityType.practice,
                    resource_id=None,
                    estimated_minutes=practice_mins,
                    difficulty=Difficulty.beginner,
                    expected_outcome="One attempt",
                    why_selected="Micro practice",
                    milestone_connection="Breath",
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
                task_count_rationale="Fits available minutes.",
                adaptation_explanation="Based on current mood and available time.",
                tasks=tasks,
            )
        practice_mins = min(per_task_cap, 10 if wants_practice else 8)
        tasks = [
            PlannerTaskGeneration(
                sequence_number=1,
                title="Short study tip",
                description="Review a curated tip",
                activity_type=ActivityType.watch,
                resource_id=self.resource_ids[0],
                estimated_minutes=min(6, per_task_cap),
                difficulty=Difficulty.beginner,
                expected_outcome="Cue",
                why_selected="Trusted",
                milestone_connection="Breath",
                mood_rationale="Focused",
                content_type="video",
            ),
            PlannerTaskGeneration(
                sequence_number=2,
                title="Applied practice drill",
                description="Practice aloud",
                activity_type=ActivityType.practice,
                resource_id=None,
                estimated_minutes=practice_mins,
                difficulty=Difficulty.intermediate,
                expected_outcome="Confident attempt",
                why_selected="Practice was useful",
                milestone_connection="Breath",
                mood_rationale="Deeper practice",
                content_type="practice",
            ),
            PlannerTaskGeneration(
                sequence_number=3,
                title="Quick synthesis",
                description="Note one improvement",
                activity_type=ActivityType.review,
                resource_id=None,
                estimated_minutes=min(5, per_task_cap),
                difficulty=Difficulty.beginner,
                expected_outcome="One note",
                why_selected="Consolidate",
                milestone_connection="Breath",
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
            task_count_rationale="Fits available minutes.",
            adaptation_explanation=(
                "Your next plan will use shorter resources because focus dropped "
                "during a longer video, and more practice because practical tasks "
                "were rated useful."
            ),
            tasks=selected,
        )


class FakeReflectionGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        return ReflectionInsightGeneration(
            insight="Partial completion with useful practice.",
            learning_progress_summary="Progress",
            completion_observation="Partial",
            focus_observation="Focus dipped",
            difficulty_observation="Suitable",
            resource_observation="Mixed",
            distraction_observation="Phone",
            mood_observation="Tired after",
            positive_signals=["practice"],
            friction_signals=["focus"],
            evidence_for_adaptation=["shorter"],
            recommended_next_session_adjustments=["practice"],
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
                "Use shorter resources next time because focus dropped during a longer "
                "video, and include more practice because the practical task was rated useful."
            ),
        )


class FakeCurator:
    def __init__(self, resource_ids: list[int]) -> None:
        self.resource_ids = resource_ids

    def recommend_resources(self, user_id: int, roadmap_id: int, milestone_id: int, **kwargs: Any) -> CuratorAgentResult:
        return CuratorAgentResult(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            recommendations=[
                CuratedRecommendation(
                    id=1,
                    user_id=user_id,
                    roadmap_id=roadmap_id,
                    milestone_id=milestone_id,
                    resource_id=self.resource_ids[0],
                    catalog_id="cat-api-1",
                    title="Tip",
                    source="YouTube",
                    resource_type="video",
                    url="https://www.youtube.com/watch?v=api1",
                    description="Trusted",
                    difficulty=Difficulty.beginner,
                    estimated_duration_minutes=8,
                    relevance_score=0.8,
                    reason="Fit",
                    milestone_fit="Breath",
                    mood_suitability="ok",
                    suggested_use="watch",
                    estimated_effort="8",
                    score_breakdown={"final": 0.8},
                    status=RecommendationStatus.suggested,
                    recommended_at=datetime.now(timezone.utc),
                )
            ],
            candidate_count=1,
            created_at=datetime.now(timezone.utc),
        )


class FailingPlanner(DailyPlannerAgent):
    def create_daily_plan(self, *args: Any, **kwargs: Any) -> Any:
        raise PlannerAgentError("planner unavailable")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_api_")
        self.db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self.db_path)
        self.settings = _settings(self.db_path)
        self.memory = FakeMemory()
        self.resource_ids = self._seed_resource()
        self.services = self._build_services()
        self.app = create_app(settings=self.settings, services=self.services)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self._tmpdir.cleanup()

    def _seed_resource(self) -> list[int]:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO resources (
                    title, source, resource_type, url, description, difficulty,
                    estimated_duration_minutes, is_free, metadata, created_at, updated_at
                )
                VALUES ('Tip', 'YouTube', 'video', 'https://www.youtube.com/watch?v=api1',
                        'Trusted', 'beginner', 8, 1, '{}', ?, ?)
                """,
                (now, now),
            )
            return [int(cursor.lastrowid)]

    def _build_services(self, *, failing_planner: bool = False) -> AppServices:
        profile = ProfileAgent(
            settings=self.settings,
            gemini_service=FakeProfileGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        roadmap = RoadmapAgent(
            settings=self.settings,
            gemini_service=FakeRoadmapGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
            skip_memory_retrieval=True,
        )
        curator = FakeCurator(self.resource_ids)
        if failing_planner:
            planner: DailyPlannerAgent = FailingPlanner(
                settings=self.settings,
                gemini_service=FakePlannerGemini(self.resource_ids),
                curator_agent=curator,  # type: ignore[arg-type]
                db_path=self.db_path,
                date_provider=lambda: date(2026, 8, 1),
            )
        else:
            planner = DailyPlannerAgent(
                settings=self.settings,
                gemini_service=FakePlannerGemini(self.resource_ids),
                curator_agent=curator,  # type: ignore[arg-type]
                db_path=self.db_path,
                date_provider=lambda: date(2026, 8, 1),
            )
        reflection = ReflectionAgent(
            settings=self.settings,
            gemini_service=FakeReflectionGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        adaptation = AdaptationAgent(
            settings=self.settings,
            gemini_service=FakeAdaptationGemini(),
            memory_service=self.memory,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        return AppServices(
            settings=self.settings,
            profile_agent=profile,
            roadmap_agent=roadmap,
            curator_agent=curator,  # type: ignore[arg-type]
            planner_agent=planner,
            reflection_agent=reflection,
            adaptation_agent=adaptation,
            onboarding_workflow=OnboardingWorkflow(profile, roadmap),
            daily_loop_workflow=DailyLoopWorkflow(
                planner_agent=planner,
                reflection_agent=reflection,
                adaptation_agent=adaptation,
            ),
            db_path=self.db_path,
        )

    def _onboard_payload(self) -> dict[str, Any]:
        return {
            "display_name": "API User",
            "learning_goal": "Improve public speaking",
            "aspiration": "Calm presenter",
            "motivation": "Lead meetings",
            "current_level": "beginner",
            "target_outcome": "5-minute talk",
            "preferred_formats": ["video", "practice"],
            "learning_style": "mixed",
            "daily_available_minutes": 30,
            "preferred_session_minutes": 15,
            "attention_span_minutes": 10,
            "preferred_learning_time": "evening",
            "habits": ["review"],
            "distractions": ["phone"],
        }

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "GrowthOS AI API")
        self.assertIn("youtube_enabled", body)
        self.assertIn("youtube_configured", body)
        self.assertIsInstance(body["youtube_enabled"], bool)
        self.assertIsInstance(body["youtube_configured"], bool)

    def test_onboarding_and_roadmap_get(self) -> None:
        created = self.client.post("/onboarding", json=self._onboard_payload())
        self.assertEqual(created.status_code, 201, created.text)
        body = created.json()
        user_id = body["user"]["id"]
        self.assertEqual(body["goal"]["title"], "Improve public speaking")
        self.assertIn("profile", body["completed_steps"])
        self.assertIn("roadmap", body["completed_steps"])

        roadmap = self.client.get(f"/users/{user_id}/roadmap")
        self.assertEqual(roadmap.status_code, 200)
        self.assertEqual(roadmap.json()["user_id"], user_id)

        # Duplicate roadmap create reuses active roadmap
        again = self.client.post(
            f"/users/{user_id}/roadmaps",
            json={"regenerate": False},
        )
        self.assertEqual(again.status_code, 201)
        self.assertTrue(again.json()["reused_existing"])

    def test_validation_errors(self) -> None:
        bad = self.client.post("/onboarding", json={"display_name": ""})
        self.assertEqual(bad.status_code, 422)

    def test_ownership_and_not_found(self) -> None:
        created = self.client.post("/onboarding", json=self._onboard_payload())
        user_id = created.json()["user"]["id"]
        plan = self.client.post(
            f"/users/{user_id}/daily-plans",
            json={
                "mood": "tired",
                "energy_level": "low",
                "focus_level": 2,
                "available_minutes": 15,
                "preferred_activity": "watch",
                "plan_date": "2026-08-01",
            },
        )
        self.assertEqual(plan.status_code, 201, plan.text)
        task_id = plan.json()["tasks"][0]["id"]
        forbidden = self.client.patch(
            f"/users/{user_id + 99}/tasks/{task_id}",
            json={"status": "completed"},
        )
        self.assertEqual(forbidden.status_code, 403)
        missing = self.client.get("/users/99999/dashboard")
        self.assertEqual(missing.status_code, 404)

    def test_daily_loop_reflection_adaptation_dashboard(self) -> None:
        user_id = self.client.post("/onboarding", json=self._onboard_payload()).json()["user"]["id"]
        checkin = self.client.post(
            f"/users/{user_id}/checkins",
            json={
                "mood": "tired",
                "energy_level": "low",
                "focus_level": 2,
                "available_minutes": 15,
                "preferred_activity": "watch",
            },
        )
        self.assertEqual(checkin.status_code, 201)

        plan = self.client.post(
            f"/users/{user_id}/daily-plans",
            json={
                "mood": "tired",
                "energy_level": "low",
                "focus_level": 2,
                "available_minutes": 15,
                "preferred_activity": "watch",
                "plan_date": "2026-08-01",
            },
        )
        self.assertEqual(plan.status_code, 201, plan.text)
        self.assertTrue(plan.json()["awaiting_user_completion"])
        plan_id = plan.json()["plan"]["id"]
        task_id = plan.json()["tasks"][0]["id"]

        # Same-day reuse
        reused = self.client.post(
            f"/users/{user_id}/daily-plans",
            json={
                "mood": "focused",
                "energy_level": "high",
                "focus_level": 5,
                "available_minutes": 30,
                "preferred_activity": "practice",
                "plan_date": "2026-08-01",
                "refresh": False,
            },
        )
        self.assertEqual(reused.status_code, 201)
        self.assertTrue(reused.json()["planner_result"]["reused_existing"])

        patched = self.client.patch(
            f"/users/{user_id}/tasks/{task_id}",
            json={"status": "completed", "completion_percent": 100, "duration_minutes": 8},
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.json()["status"], "completed")

        reflection = self.client.post(
            f"/users/{user_id}/reflections",
            json={
                "daily_plan_id": plan_id,
                "completion_status": "partial",
                "learning_summary": "Practice helped",
                "focus_rating": 2,
                "resource_effectiveness": 3,
                "difficulty_feedback": "suitable",
                "mood_match": False,
                "distractions": ["phone"],
                "wants_similar_resources": True,
                "mood_after": "tired",
                "task_updates": [
                    {
                        "task_id": task_id,
                        "update": {"status": "completed", "completion_percent": 100},
                    }
                ],
            },
        )
        self.assertEqual(reflection.status_code, 201, reflection.text)
        body = reflection.json()
        self.assertIn("reflection", body)
        self.assertIn("adaptation", body)
        self.assertIn("adaptation_explanation", body)
        reflection_id = body["reflection"]["id"]
        self.assertTrue(body["goal_unchanged"])

        # Duplicate reflection reuse (post-session workflow reuses both steps)
        again = self.client.post(
            f"/users/{user_id}/reflections",
            json={
                "daily_plan_id": plan_id,
                "completion_status": "partial",
                "learning_summary": "Practice helped",
                "focus_rating": 2,
                "resource_effectiveness": 3,
                "difficulty_feedback": "suitable",
                "mood_match": False,
                "distractions": ["phone"],
                "wants_similar_resources": True,
                "mood_after": "tired",
            },
        )
        self.assertEqual(again.status_code, 201)
        self.assertTrue(again.json()["reflection_result"]["reused_existing"])
        self.assertEqual(again.json()["reflection"]["id"], reflection_id)

        adaptation = self.client.post(
            f"/users/{user_id}/adaptations/run",
            json={"reflection_id": reflection_id},
        )
        self.assertEqual(adaptation.status_code, 201, adaptation.text)
        self.assertTrue(adaptation.json()["is_early_signal"])

        dashboard = self.client.get(f"/users/{user_id}/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(dashboard.json()["user"]["id"], user_id)
        self.assertEqual(dashboard.json()["active_goal"]["title"], "Improve public speaking")

    def test_workflow_failure_safe_error(self) -> None:
        failing = self._build_services(failing_planner=True)
        app = create_app(settings=self.settings, services=failing)
        with TestClient(app) as client:
            user_id = client.post("/onboarding", json=self._onboard_payload()).json()["user"]["id"]
            response = client.post(
                f"/users/{user_id}/daily-plans",
                json={
                    "mood": "focused",
                    "energy_level": "medium",
                    "focus_level": 4,
                    "available_minutes": 30,
                    "preferred_activity": "practice",
                    "plan_date": "2026-08-01",
                },
            )
            self.assertIn(response.status_code, {400, 502})
            detail = response.json().get("detail", "")
            self.assertNotIn("test-key", detail)
            self.assertNotIn("GEMINI_API_KEY", detail)
            self.assertNotIn(str(self.db_path), detail)

    def test_demo_day_loop(self) -> None:
        # Use a planner date provider that can handle day1/day2 via request plan_date.
        # Rebuild planner without fixed date provider for demo endpoint dates.
        profile = self.services.profile_agent
        roadmap = self.services.roadmap_agent
        curator = FakeCurator(self.resource_ids)
        planner = DailyPlannerAgent(
            settings=self.settings,
            gemini_service=FakePlannerGemini(self.resource_ids),
            curator_agent=curator,  # type: ignore[arg-type]
            db_path=self.db_path,
        )
        reflection = self.services.reflection_agent
        adaptation = self.services.adaptation_agent
        services = AppServices(
            settings=self.settings,
            profile_agent=profile,
            roadmap_agent=roadmap,
            curator_agent=curator,  # type: ignore[arg-type]
            planner_agent=planner,
            reflection_agent=reflection,
            adaptation_agent=adaptation,
            onboarding_workflow=OnboardingWorkflow(profile, roadmap),
            daily_loop_workflow=DailyLoopWorkflow(
                planner_agent=planner,
                reflection_agent=reflection,
                adaptation_agent=adaptation,
            ),
            db_path=self.db_path,
        )
        app = create_app(settings=self.settings, services=services)
        with TestClient(app) as client:
            user_id = client.post("/onboarding", json=self._onboard_payload()).json()["user"]["id"]
            demo = client.post(f"/users/{user_id}/demo/day-loop", json={})
            self.assertEqual(demo.status_code, 200, demo.text)
            body = demo.json()
            self.assertTrue(body["goal_unchanged"])
            self.assertEqual(body["goal_title"], "Improve public speaking")
            self.assertEqual(body["day1_checkin"]["mood"], "tired")
            self.assertEqual(body["day1_checkin"]["available_minutes"], 15)
            self.assertEqual(body["day2_checkin"]["mood"], "focused")
            self.assertEqual(body["day2_checkin"]["available_minutes"], 30)
            self.assertNotEqual(body["day1_plan"]["id"], body["day2_plan"]["id"])
            self.assertLessEqual(len(body["day1_tasks"]), 5)
            self.assertLessEqual(len(body["day2_tasks"]), 5)
            self.assertTrue(body["is_early_signal"])
            self.assertTrue(body["adaptation_explanation"])
            self.assertIn("adaptation", body)
            self.assertTrue(body["detected_patterns"])


if __name__ == "__main__":
    unittest.main()
