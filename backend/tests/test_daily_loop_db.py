"""Phase A tests: daily-loop SQLite helpers and schema extensions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models import (  # noqa: E402
    DailyCheckInResponse,
    DailyPlanResponse,
    DashboardResponse,
    TaskCompletionRequest,
)
from services.database import (  # noqa: E402
    build_dashboard_snapshot,
    compute_completion_streak,
    count_completed_plans,
    create_adaptation_insight,
    create_daily_checkin,
    create_daily_plan_bundle,
    create_onboarding_records,
    create_reflection,
    create_resource_interaction,
    create_roadmap_bundle,
    get_connection,
    get_daily_plan_by_date,
    get_task_by_id,
    init_db,
    list_active_adaptation_insights,
    list_user_preferences,
    update_daily_plan_status,
    update_milestone_progress,
    update_task_status,
    upsert_user_preference,
)


class DailyLoopDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_daily_db_")
        self.db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self.db_path)
        self.user, self.profile, self.goal = create_onboarding_records(
            display_name="Ada",
            aspiration="Speak calmly",
            motivation="Lead meetings",
            current_level="beginner",
            target_outcome="5-minute talk",
            learning_style="mixed",
            preferred_formats=["video", "practice"],
            daily_available_minutes=30,
            preferred_session_minutes=15,
            attention_span_minutes=10,
            preferred_learning_time="evening",
            habits=[],
            distractions=[],
            goal_title="Improve public speaking",
            db_path=self.db_path,
        )
        self.roadmap = create_roadmap_bundle(
            user_id=int(self.user["id"]),
            goal_id=int(self.goal["id"]),
            title="Speaking path",
            summary="Foundations first",
            estimated_duration_weeks=4,
            pacing_rationale="Short sessions",
            personalization_rationale="Beginner friendly",
            phases=[
                {
                    "sequence_number": 1,
                    "title": "Foundations",
                    "description": "Basics",
                    "expected_outcome": "Short talk",
                    "status": "in_progress",
                    "milestones": [
                        {
                            "sequence_number": 1,
                            "title": "Breath basics",
                            "description": "Calm delivery",
                            "skills": ["breath", "posture"],
                            "suggested_activities": ["practice breathing"],
                            "completion_criteria": "2-minute calm delivery",
                            "estimated_sessions": 2,
                            "estimated_minutes": 15,
                            "difficulty": "beginner",
                            "status": "in_progress",
                        }
                    ],
                },
                {
                    "sequence_number": 2,
                    "title": "Delivery",
                    "description": "Practice",
                    "expected_outcome": "5-minute talk",
                    "status": "not_started",
                    "milestones": [
                        {
                            "sequence_number": 1,
                            "title": "Story drill",
                            "description": "Structure",
                            "skills": ["storytelling"],
                            "suggested_activities": ["outline a story"],
                            "completion_criteria": "Clear outline",
                            "estimated_sessions": 2,
                            "estimated_minutes": 20,
                            "difficulty": "intermediate",
                            "status": "not_started",
                        }
                    ],
                },
            ],
            db_path=self.db_path,
        )
        self.milestone_id = int(self.roadmap["active_milestone"]["id"])

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_schema_has_daily_loop_columns(self) -> None:
        with get_connection(self.db_path) as conn:
            plan_cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(daily_plans)").fetchall()
            }
            task_cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(daily_tasks)").fetchall()
            }
            reflection_cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(reflections)").fetchall()
            }
        self.assertIn("metadata", plan_cols)
        self.assertIn("metadata", task_cols)
        self.assertIn("insight", reflection_cols)

    def test_checkin_plan_task_transaction(self) -> None:
        checkin = create_daily_checkin(
            int(self.user["id"]),
            mood="tired",
            energy_level="low",
            focus_level=2,
            available_minutes=15,
            preferred_activity="watch",
            notes="Evening session",
            db_path=self.db_path,
        )
        DailyCheckInResponse(
            id=int(checkin["id"]),
            user_id=int(checkin["user_id"]),
            mood=checkin["mood"],
            energy_level=checkin["energy_level"],
            focus_level=int(checkin["focus_level"]),
            available_minutes=int(checkin["available_minutes"]),
            preferred_activity=checkin["preferred_activity"],
            notes=checkin["notes"],
            created_at=checkin["created_at"],
        )

        plan = create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-01",
            summary="Short tired-day plan",
            total_estimated_minutes=20,
            roadmap_id=int(self.roadmap["id"]),
            milestone_id=self.milestone_id,
            checkin_id=int(checkin["id"]),
            metadata={
                "guidance_tone": "gentle",
                "mood_influence_summary": "Short because tired",
                "adaptation_explanation": "",
                "task_count_rationale": "Two tasks fit 15 minutes",
            },
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "Watch a short tip",
                    "description": "Calm delivery tip",
                    "activity_type": "watch",
                    "estimated_minutes": 10,
                    "difficulty": "beginner",
                    "metadata": {
                        "why_selected": "Matches tired mood and short attention",
                        "milestone_connection": "Breath basics",
                        "expected_outcome": "One calm breathing cue",
                        "content_type": "video",
                        "mood_rationale": "Low energy → short video",
                    },
                },
                {
                    "sequence_number": 2,
                    "title": "One-minute practice",
                    "description": "Practice breath cue",
                    "activity_type": "practice",
                    "estimated_minutes": 10,
                    "difficulty": "beginner",
                    "metadata": {
                        "why_selected": "Small practice improves completion",
                        "milestone_connection": "Breath basics",
                        "expected_outcome": "One calm delivery attempt",
                        "content_type": "practice",
                        "mood_rationale": "Short practice after watching",
                    },
                },
            ],
            db_path=self.db_path,
        )
        self.assertEqual(len(plan["tasks"]), 2)
        self.assertEqual(plan["guidance_tone"], "gentle")
        self.assertEqual(plan["tasks"][0]["why_selected"], "Matches tired mood and short attention")

        mapped = DailyPlanResponse(
            id=int(plan["id"]),
            user_id=int(plan["user_id"]),
            roadmap_id=plan.get("roadmap_id"),
            milestone_id=plan.get("milestone_id"),
            checkin_id=plan.get("checkin_id"),
            plan_date=plan["plan_date"],
            summary=plan["summary"],
            total_estimated_minutes=int(plan["total_estimated_minutes"]),
            status=plan["status"],
            tasks=[],
            guidance_tone=plan.get("guidance_tone") or "",
            mood_influence_summary=plan.get("mood_influence_summary") or "",
            adaptation_explanation=plan.get("adaptation_explanation") or "",
            task_count_rationale=plan.get("task_count_rationale") or "",
            metadata=plan.get("metadata") or {},
            created_at=plan["created_at"],
            updated_at=plan["updated_at"],
        )
        self.assertEqual(mapped.guidance_tone, "gentle")

    def test_replace_same_day_plan(self) -> None:
        create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-01",
            summary="First",
            total_estimated_minutes=10,
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "Task A",
                    "description": "A",
                    "activity_type": "watch",
                    "estimated_minutes": 10,
                    "difficulty": "beginner",
                }
            ],
            db_path=self.db_path,
        )
        second = create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-01",
            summary="Second",
            total_estimated_minutes=12,
            replace_existing=True,
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "Task B",
                    "description": "B",
                    "activity_type": "practice",
                    "estimated_minutes": 12,
                    "difficulty": "beginner",
                }
            ],
            db_path=self.db_path,
        )
        loaded = get_daily_plan_by_date(
            int(self.user["id"]),
            "2026-08-01",
            db_path=self.db_path,
        )
        assert loaded is not None
        self.assertEqual(loaded["summary"], "Second")
        self.assertEqual(loaded["tasks"][0]["title"], "Task B")
        self.assertEqual(int(second["id"]), int(loaded["id"]))

    def test_task_completion_and_interaction(self) -> None:
        plan = create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-02",
            summary="Practice day",
            total_estimated_minutes=15,
            roadmap_id=int(self.roadmap["id"]),
            milestone_id=self.milestone_id,
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "Practice",
                    "description": "Do it",
                    "activity_type": "practice",
                    "estimated_minutes": 15,
                    "difficulty": "beginner",
                }
            ],
            db_path=self.db_path,
        )
        task_id = int(plan["tasks"][0]["id"])
        updated = update_task_status(task_id, "completed", db_path=self.db_path)
        self.assertEqual(updated["status"], "completed")
        self.assertIsNotNone(updated["completed_at"])
        TaskCompletionRequest(status="completed", completion_percent=100, duration_minutes=12)

        # Resource interaction requires a real resource row; skip URL path and
        # only assert helper validation via a temporary insert when resource exists.
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO resources (
                    title, source, resource_type, url, description, difficulty,
                    estimated_duration_minutes, is_free, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, '{}', ?, ?)
                """,
                (
                    "Tip",
                    "YouTube",
                    "video",
                    "https://www.youtube.com/watch?v=phase-a-test",
                    "desc",
                    "beginner",
                    10,
                    updated["completed_at"],
                    updated["completed_at"],
                ),
            )
            resource_id = int(cursor.lastrowid)
        interaction = create_resource_interaction(
            user_id=int(self.user["id"]),
            resource_id=resource_id,
            daily_plan_id=int(plan["id"]),
            interaction_type="completed",
            completion_percent=100,
            effectiveness_rating=4,
            duration_minutes=12,
            db_path=self.db_path,
        )
        self.assertEqual(interaction["completion_percent"], 100)

    def test_reflection_adaptation_preferences_dashboard(self) -> None:
        plan = create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-03",
            summary="Day plan",
            total_estimated_minutes=15,
            metadata={"adaptation_explanation": "Shortened after low focus"},
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "Watch",
                    "description": "Watch tip",
                    "activity_type": "watch",
                    "estimated_minutes": 15,
                    "difficulty": "beginner",
                }
            ],
            db_path=self.db_path,
        )
        update_task_status(int(plan["tasks"][0]["id"]), "completed", db_path=self.db_path)
        update_daily_plan_status(int(plan["id"]), "completed", db_path=self.db_path)
        update_milestone_progress(
            self.milestone_id,
            progress_percent=25,
            status="in_progress",
            db_path=self.db_path,
        )

        reflection = create_reflection(
            user_id=int(self.user["id"]),
            daily_plan_id=int(plan["id"]),
            completion_status="partial",
            learning_summary="Video felt long",
            focus_rating=2,
            resource_effectiveness=3,
            difficulty_feedback="suitable",
            mood_match=False,
            distractions=["phone"],
            wants_similar_resources=False,
            mood_after="tired",
            insight="Short sessions work better when tired.",
            db_path=self.db_path,
        )
        self.assertIn("Short sessions", reflection["insight"])

        create_adaptation_insight(
            user_id=int(self.user["id"]),
            insight_type="pattern",
            insight="Short sessions work better.",
            confidence_score=0.8,
            evidence=["focus_rating=2", "partial completion"],
            db_path=self.db_path,
        )
        upsert_user_preference(
            user_id=int(self.user["id"]),
            preference_key="preferred_content_type",
            preference_value="short_video",
            confidence_score=0.7,
            db_path=self.db_path,
        )
        self.assertEqual(len(list_active_adaptation_insights(int(self.user["id"]), db_path=self.db_path)), 1)
        self.assertEqual(len(list_user_preferences(int(self.user["id"]), db_path=self.db_path)), 1)
        self.assertEqual(count_completed_plans(int(self.user["id"]), db_path=self.db_path), 1)
        self.assertEqual(compute_completion_streak(int(self.user["id"]), db_path=self.db_path), 1)

        snapshot = build_dashboard_snapshot(
            int(self.user["id"]),
            plan_date="2026-08-03",
            db_path=self.db_path,
        )
        assert snapshot is not None
        self.assertEqual(snapshot["completed_sessions"], 1)
        self.assertEqual(snapshot["preferred_content_type"], "short_video")
        self.assertTrue(snapshot["plan_change_explanation"])
        self.assertEqual(snapshot["active_goal"]["title"], "Improve public speaking")

        # Ensure dashboard-shaped payload can map into DashboardResponse essentials
        DashboardResponse(
            user={
                "id": int(self.user["id"]),
                "display_name": self.user["display_name"],
                "created_at": self.user["created_at"],
                "updated_at": self.user["updated_at"],
            },
            overall_progress_percent=float(snapshot["overall_progress_percent"]),
            completion_streak=int(snapshot["completion_streak"]),
            completed_sessions=int(snapshot["completed_sessions"]),
            preferred_content_type=snapshot["preferred_content_type"],
            plan_change_explanation=snapshot["plan_change_explanation"],
            detected_patterns=list(snapshot["detected_patterns"]),
            growthos_knows_you=list(snapshot["growthos_knows_you"]),
        )

    def test_invalid_task_count_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_daily_plan_bundle(
                user_id=int(self.user["id"]),
                plan_date="2026-08-04",
                summary="Too many",
                total_estimated_minutes=60,
                tasks=[
                    {
                        "sequence_number": i,
                        "title": f"T{i}",
                        "description": "x",
                        "activity_type": "watch",
                        "estimated_minutes": 10,
                        "difficulty": "beginner",
                    }
                    for i in range(1, 7)
                ],
                db_path=self.db_path,
            )

    def test_task_ownership_fields(self) -> None:
        plan = create_daily_plan_bundle(
            user_id=int(self.user["id"]),
            plan_date="2026-08-05",
            summary="Own",
            total_estimated_minutes=10,
            tasks=[
                {
                    "sequence_number": 1,
                    "title": "One",
                    "description": "One",
                    "activity_type": "watch",
                    "estimated_minutes": 10,
                    "difficulty": "beginner",
                }
            ],
            db_path=self.db_path,
        )
        task = get_task_by_id(int(plan["tasks"][0]["id"]), db_path=self.db_path)
        assert task is not None
        self.assertEqual(int(task["user_id"]), int(self.user["id"]))


if __name__ == "__main__":
    unittest.main()
