"""
MVP Day 1 → Day 2 judge-demo journey through the FastAPI layer.

Uses injectable fake Gemini/Curator services — no internet or real API key.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
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
from exceptions import GeminiResponseError  # noqa: E402
from main import create_app  # noqa: E402
from services.database import get_connection, get_daily_plan_by_date, init_db  # noqa: E402
from tests.test_api import (  # noqa: E402
    FakeAdaptationGemini,
    FakeCurator,
    FakeMemory,
    FakePlannerGemini,
    FakeProfileGemini,
    FakeReflectionGemini,
    FakeRoadmapGemini,
    _settings,
)
from workflows.daily_loop import DailyLoopWorkflow  # noqa: E402
from workflows.onboarding import OnboardingWorkflow  # noqa: E402

GOAL_TITLE = "Improve public speaking for product demos"


class FlakyDay2PlannerGemini(FakePlannerGemini):
    """Fail Day-2 planner structured calls a fixed number of times, then succeed."""

    def __init__(self, resource_ids: list[int], *, day2_fail_times: int = 1) -> None:
        super().__init__(resource_ids)
        self.day2_fail_times = max(0, int(day2_fail_times))
        self.day2_failures_raised = 0
        self.day2_calls = 0
        self.total_calls = 0

    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        self.total_calls += 1
        is_day2 = '"available_minutes": 30' in prompt or '"mood": "focused"' in prompt
        if is_day2:
            self.day2_calls += 1
            if self.day2_failures_raised < self.day2_fail_times:
                self.day2_failures_raised += 1
                raise GeminiResponseError("empty structured output")
        return super().generate_structured(prompt, response_model, **kwargs)


class MvpDemoJourneyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_mvp_")
        self.db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self.db_path)
        self.settings = _settings(self.db_path)
        self.memory = FakeMemory()
        self.resource_ids = self._seed_resources()
        self.services = self._build_services()
        self.app = create_app(settings=self.settings, services=self.services)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
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
                        f"Trusted MVP resource {index + 1}",
                        f"https://www.youtube.com/watch?v=mvp{index}",
                        "Trusted free tip",
                        8 if index == 0 else 15,
                        now,
                        now,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def _build_services(
        self,
        *,
        planner_gemini: Any | None = None,
    ) -> AppServices:
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
        planner = DailyPlannerAgent(
            settings=self.settings,
            gemini_service=planner_gemini or FakePlannerGemini(self.resource_ids),
            curator_agent=curator,  # type: ignore[arg-type]
            db_path=self.db_path,
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
            "display_name": "Judge Demo User",
            "learning_goal": GOAL_TITLE,
            "aspiration": "Present calmly",
            "motivation": "Lead product demos",
            "current_level": "beginner",
            "target_outcome": "Deliver a short demo",
            "preferred_formats": ["video", "practice"],
            "learning_style": "mixed",
            "daily_available_minutes": 45,
            "preferred_session_minutes": 20,
            "attention_span_minutes": 12,
            "preferred_learning_time": "evening",
            "habits": ["review"],
            "distractions": ["phone"],
        }

    def test_mvp_day1_day2_demo_journey(self) -> None:
        onboard = self.client.post("/onboarding", json=self._onboard_payload())
        self.assertEqual(onboard.status_code, 201, onboard.text)
        user_id = onboard.json()["user"]["id"]
        self.assertEqual(onboard.json()["goal"]["title"], GOAL_TITLE)

        # Isolation: second user cannot access first user's roadmap.
        other = self.client.post(
            "/onboarding",
            json={**self._onboard_payload(), "display_name": "Other User", "learning_goal": "Learn SQL"},
        )
        other_id = other.json()["user"]["id"]
        forbidden = self.client.get(f"/users/{other_id}/roadmap")
        self.assertEqual(forbidden.status_code, 200)
        self.assertNotEqual(forbidden.json().get("goal_id"), onboard.json()["goal"]["id"])

        demo = self.client.post(f"/users/{user_id}/demo/day-loop", json={})
        self.assertEqual(demo.status_code, 200, demo.text)
        body = demo.json()

        # 1. Free-text goal preserved exactly
        self.assertEqual(body["goal_title"], GOAL_TITLE)
        self.assertTrue(body["goal_unchanged"])

        # 2–4. Day 1 tired / 15 minutes / 1–5 tasks / trusted resources
        self.assertEqual(body["day1_checkin"]["mood"], "tired")
        self.assertEqual(body["day1_checkin"]["energy_level"], "low")
        self.assertEqual(body["day1_checkin"]["available_minutes"], 15)
        self.assertLessEqual(body["day1_plan"]["total_estimated_minutes"], 15)
        day1_tasks = body["day1_tasks"]
        self.assertGreaterEqual(len(day1_tasks), 1)
        self.assertLessEqual(len(day1_tasks), 5)
        for task in day1_tasks:
            if task.get("resource_id") is not None:
                self.assertIn(task["resource_id"], self.resource_ids)

        # 5–6. Partial completion controls progress; reflection does not invent completion
        self.assertEqual(body["reflection"]["completion_status"], "partial")
        self.assertEqual(body["reflection"]["focus_rating"], 2)
        self.assertLess(body["reflection"]["resource_effectiveness"], 5)

        # 7–8. Early signal + explanation cites evidence
        self.assertTrue(body["is_early_signal"])
        explanation = (body["adaptation_explanation"] or "").lower()
        self.assertTrue(explanation)
        self.assertTrue(
            "practice" in explanation
            or "shorter" in explanation
            or "focus" in explanation
            or "resource" in explanation
        )
        self.assertTrue(body["detected_patterns"])

        # 9–11. Day 2 focused / 30 minutes / differs / more practice when supported
        self.assertEqual(body["day2_checkin"]["mood"], "focused")
        self.assertIn(body["day2_checkin"]["energy_level"], {"medium", "high"})
        self.assertEqual(body["day2_checkin"]["available_minutes"], 30)
        self.assertLessEqual(body["day2_plan"]["total_estimated_minutes"], 30)
        self.assertNotEqual(body["day1_plan"]["id"], body["day2_plan"]["id"])
        day2_tasks = body["day2_tasks"]
        self.assertGreaterEqual(len(day2_tasks), 1)
        self.assertLessEqual(len(day2_tasks), 5)
        day1_practice = sum(1 for t in day1_tasks if t["activity_type"] == "practice")
        day2_practice = sum(1 for t in day2_tasks if t["activity_type"] == "practice")
        self.assertGreaterEqual(day2_practice, day1_practice)
        self.assertNotEqual(
            [t["title"] for t in day1_tasks],
            [t["title"] for t in day2_tasks],
        )

        # 12. Goal unchanged after demo
        dash = self.client.get(f"/users/{user_id}/dashboard")
        self.assertEqual(dash.status_code, 200, dash.text)
        dashboard = dash.json()
        self.assertEqual(dashboard["active_goal"]["title"], GOAL_TITLE)

        # 14–15. Dashboard patterns + plan-change explanation
        self.assertTrue(
            dashboard.get("detected_patterns")
            or dashboard.get("growthos_knows_you")
            or dashboard.get("adaptation_insights")
        )
        self.assertTrue(
            dashboard.get("plan_change_explanation")
            or body["adaptation_explanation"]
        )

        # 16. Duplicate reflection/adaptation for the same plan does not duplicate rows
        day1_plan_id = body["day1_plan"]["id"]
        reflection_id = body["reflection"]["id"]
        with get_connection(self.db_path) as conn:
            reflections_for_plan = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections WHERE user_id = ? AND daily_plan_id = ?",
                (user_id, day1_plan_id),
            ).fetchone()["c"]
            insights_before = conn.execute(
                "SELECT COUNT(*) AS c FROM adaptation_insights WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
        self.assertEqual(reflections_for_plan, 1)

        reuse = self.client.post(
            f"/users/{user_id}/reflections",
            json={
                "daily_plan_id": day1_plan_id,
                "completion_status": "partial",
                "learning_summary": "Practice helped; longer resource drained focus.",
                "focus_rating": 2,
                "resource_effectiveness": 2,
                "difficulty_feedback": "suitable",
                "mood_match": False,
                "distractions": ["phone"],
                "wants_similar_resources": False,
                "mood_after": "tired",
            },
        )
        self.assertEqual(reuse.status_code, 201, reuse.text)
        self.assertEqual(reuse.json()["reflection"]["id"], reflection_id)
        self.assertTrue(reuse.json()["reflection_result"]["reused_existing"])

        adapt_again = self.client.post(
            f"/users/{user_id}/adaptations/run",
            json={"reflection_id": reflection_id, "force": False},
        )
        self.assertEqual(adapt_again.status_code, 201, adapt_again.text)
        self.assertTrue(adapt_again.json()["reused_existing"])

        with get_connection(self.db_path) as conn:
            reflections_after = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections WHERE user_id = ? AND daily_plan_id = ?",
                (user_id, day1_plan_id),
            ).fetchone()["c"]
            insights_after = conn.execute(
                "SELECT COUNT(*) AS c FROM adaptation_insights WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
        self.assertEqual(reflections_after, 1)
        self.assertEqual(insights_after, insights_before)

        # 13. User isolation remains intact
        other_dash = self.client.get(f"/users/{other_id}/dashboard")
        self.assertEqual(other_dash.status_code, 200)
        self.assertEqual(other_dash.json()["active_goal"]["title"], "Learn SQL")
        self.assertNotEqual(other_dash.json()["user"]["id"], user_id)

    def test_demo_retries_empty_day2_structured_output(self) -> None:
        flaky = FlakyDay2PlannerGemini(self.resource_ids, day2_fail_times=1)
        self.services = self._build_services(planner_gemini=flaky)
        self.app = create_app(settings=self.settings, services=self.services)
        self.client = TestClient(self.app)

        user_id = self.client.post("/onboarding", json=self._onboard_payload()).json()["user"]["id"]
        demo = self.client.post(f"/users/{user_id}/demo/day-loop", json={})
        self.assertEqual(demo.status_code, 200, demo.text)
        body = demo.json()
        self.assertTrue(body["goal_unchanged"])
        self.assertEqual(body["goal_title"], GOAL_TITLE)
        self.assertEqual(flaky.day2_failures_raised, 1)
        self.assertGreaterEqual(flaky.day2_calls, 2)
        from datetime import date, timedelta

        day2 = (date.today() + timedelta(days=1)).isoformat()
        with get_connection(self.db_path) as conn:
            reflections = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            day2_plans = conn.execute(
                "SELECT COUNT(*) AS c FROM daily_plans WHERE user_id = ? AND plan_date = ?",
                (user_id, day2),
            ).fetchone()["c"]
        self.assertEqual(reflections, 1)
        self.assertEqual(day2_plans, 1)
        self.assertIsNotNone(get_daily_plan_by_date(user_id, day2, db_path=self.db_path))

    def test_demo_day2_both_failures_preserves_day1(self) -> None:
        flaky = FlakyDay2PlannerGemini(self.resource_ids, day2_fail_times=2)
        self.services = self._build_services(planner_gemini=flaky)
        self.app = create_app(settings=self.settings, services=self.services)
        self.client = TestClient(self.app)

        user_id = self.client.post("/onboarding", json=self._onboard_payload()).json()["user"]["id"]
        demo = self.client.post(f"/users/{user_id}/demo/day-loop", json={})
        self.assertEqual(demo.status_code, 502, demo.text)
        detail = demo.json().get("detail") or ""
        self.assertIn("Day 1 reflection and adaptation were preserved", detail)
        self.assertNotIn("GEMINI_API_KEY", detail)

        # Day 1 reflection/adaptation remain; no Day 2 plan row.
        with get_connection(self.db_path) as conn:
            reflections = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            adaptations = conn.execute(
                "SELECT COUNT(*) AS c FROM adaptation_insights WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            plans = conn.execute(
                "SELECT plan_date, COUNT(*) AS c FROM daily_plans WHERE user_id = ? GROUP BY plan_date",
                (user_id,),
            ).fetchall()
        self.assertEqual(reflections, 1)
        self.assertGreaterEqual(adaptations, 1)
        self.assertEqual(len(plans), 1)
        from datetime import date, timedelta

        day2 = (date.today() + timedelta(days=1)).isoformat()
        self.assertIsNone(get_daily_plan_by_date(user_id, day2, db_path=self.db_path))

        dash = self.client.get(f"/users/{user_id}/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertEqual(dash.json()["active_goal"]["title"], GOAL_TITLE)


if __name__ == "__main__":
    unittest.main()
