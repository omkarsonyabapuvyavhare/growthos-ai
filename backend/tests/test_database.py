"""
Step 2 database validation.

Creates a temporary SQLite database, initializes the schema, exercises
minimal helpers, confirms foreign-key enforcement, then cleans up.
Does not write into the project growthos.db file.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Allow imports when running as: python tests/test_database.py from backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models import (  # noqa: E402
    CurrentLevel,
    LearningStyle,
    OnboardingRequest,
    PreferredLearningTime,
)
from services.database import (  # noqa: E402
    EXPECTED_TABLES,
    create_goal,
    create_user,
    create_user_profile,
    foreign_keys_enabled,
    get_connection,
    get_user_by_id,
    init_db,
    list_tables,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_validation() -> None:
    temp_dir = tempfile.TemporaryDirectory(prefix="growthos_db_test_")
    db_path = Path(temp_dir.name) / "test_growthos.db"

    try:
        print(f"Using temporary database: {db_path}")

        # 1-2. Initialize schema
        created_path = init_db(db_path)
        _assert(created_path.exists(), "init_db did not create a database file")
        _assert(db_path.exists(), "database file was not created at expected path")

        with get_connection(db_path) as conn:
            # 3. Confirm expected tables
            tables = set(list_tables(conn))
            missing = sorted(set(EXPECTED_TABLES) - tables)
            _assert(not missing, f"missing tables: {missing}")
            print(f"tables_ok count={len(EXPECTED_TABLES)}")

            # 4. Confirm foreign keys enabled
            _assert(foreign_keys_enabled(conn), "foreign_keys pragma is not enabled")
            print("foreign_keys_ok")

            # 5-7. Create sample user, profile, goal
            user = create_user("Ada Lovelace", conn=conn)
            _assert(user["id"] > 0, "user id was not assigned")
            print(f"user_created id={user['id']}")

            profile = create_user_profile(
                int(user["id"]),
                aspiration="Become a confident public speaker",
                motivation="Lead team presentations with clarity",
                current_level="beginner",
                target_outcome="Deliver a 10-minute talk without notes",
                learning_style="mixed",
                preferred_formats=["video", "practice"],
                daily_available_minutes=45,
                preferred_session_minutes=20,
                attention_span_minutes=15,
                preferred_learning_time="evening",
                habits=["journal", "walk"],
                distractions=["phone", "social media"],
                conn=conn,
            )
            _assert(profile["user_id"] == user["id"], "profile user_id mismatch")
            _assert(profile["preferred_formats"] == ["video", "practice"], "formats mismatch")
            print(f"profile_created id={profile['id']}")

            goal = create_goal(
                int(user["id"]),
                title="Improve public speaking",
                description="Free-text learning goal for the MVP demo",
                conn=conn,
            )
            _assert(goal["user_id"] == user["id"], "goal user_id mismatch")
            print(f"goal_created id={goal['id']}")

        # 8. Read records back through a fresh connection
        loaded_user = get_user_by_id(int(user["id"]), db_path=db_path)
        _assert(loaded_user is not None, "user not found on reload")
        _assert(loaded_user["display_name"] == "Ada Lovelace", "display_name mismatch")
        print("reload_ok")

        # 9. Foreign-key violation must be rejected
        violation_raised = False
        try:
            with get_connection(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO goals (user_id, title, description, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        999999,
                        "Orphan goal",
                        "",
                        "active",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
        except sqlite3.IntegrityError:
            violation_raised = True

        _assert(violation_raised, "expected foreign-key violation was not raised")
        print("foreign_key_violation_rejected")

        # Bonus: Pydantic onboarding model validation
        payload = OnboardingRequest(
            display_name="Ada Lovelace",
            learning_goal="Improve public speaking",
            aspiration="Become a confident public speaker",
            motivation="Lead team presentations with clarity",
            current_level=CurrentLevel.beginner,
            target_outcome="Deliver a 10-minute talk without notes",
            preferred_formats=["video", "practice"],
            learning_style=LearningStyle.mixed,
            daily_available_minutes=45,
            preferred_session_minutes=20,
            attention_span_minutes=15,
            preferred_learning_time=PreferredLearningTime.evening,
            habits=["journal"],
            distractions=["phone"],
        )
        _assert(payload.learning_goal.startswith("Improve"), "onboarding model failed")
        print("pydantic_onboarding_ok")

        print("DATABASE_VALIDATION_OK")
    finally:
        # 10. Clean up temporary database directory
        temp_dir.cleanup()
        print("temp_database_cleaned")


if __name__ == "__main__":
    run_validation()
