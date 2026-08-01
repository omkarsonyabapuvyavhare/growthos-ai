"""
SQLite database service for GrowthOS AI.

Uses the Python standard-library sqlite3 module only (no ORM).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Sequence

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_db_path(settings: Optional[Settings] = None) -> Path:
    """Resolve the configured SQLite database path."""
    cfg = settings or get_settings()
    return cfg.resolve_sqlite_path()


def _ensure_parent_dir(db_path: Path) -> None:
    """Always create the parent directory for the database file when needed."""
    parent = db_path.parent
    if str(parent) in {"", "."}:
        return
    parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection(
    db_path: Optional[Path] = None,
    *,
    settings: Optional[Settings] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Open a short-lived SQLite connection with safe defaults.

    - Enables foreign keys on every connection
    - Uses sqlite3.Row for dictionary-like access
    - Commits on success; rolls back on failure
    - Does not keep a global persistent connection
    """
    path = db_path or resolve_db_path(settings)
    _ensure_parent_dir(path)
    # Prefer POSIX form so Vercel logs show /tmp/growthos.db, not a host-skewed path.
    connect_target = path.as_posix() if path.as_posix().startswith("/") else str(path)
    logger.info("sqlite3.connect target: %s", connect_target)

    conn = sqlite3.connect(connect_target)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def foreign_keys_enabled(conn: sqlite3.Connection) -> bool:
    """Return True when PRAGMA foreign_keys is enabled on the connection."""
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    return bool(row[0]) if row is not None else False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        aspiration TEXT NOT NULL CHECK (length(trim(aspiration)) > 0),
        motivation TEXT NOT NULL CHECK (length(trim(motivation)) > 0),
        current_level TEXT NOT NULL,
        target_outcome TEXT NOT NULL CHECK (length(trim(target_outcome)) > 0),
        learning_style TEXT NOT NULL,
        preferred_formats TEXT NOT NULL DEFAULT '[]',
        daily_available_minutes INTEGER NOT NULL
            CHECK (daily_available_minutes > 0 AND daily_available_minutes <= 24 * 60),
        preferred_session_minutes INTEGER NOT NULL
            CHECK (preferred_session_minutes > 0 AND preferred_session_minutes <= 24 * 60),
        attention_span_minutes INTEGER NOT NULL
            CHECK (attention_span_minutes > 0 AND attention_span_minutes <= 24 * 60),
        preferred_learning_time TEXT NOT NULL,
        habits TEXT NOT NULL DEFAULT '[]',
        distractions TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        description TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'paused', 'completed', 'archived')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        goal_id INTEGER NOT NULL,
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        summary TEXT NOT NULL DEFAULT '',
        estimated_duration_weeks INTEGER NOT NULL
            CHECK (estimated_duration_weeks > 0 AND estimated_duration_weeks <= 520),
        progress_percent REAL NOT NULL DEFAULT 0
            CHECK (progress_percent >= 0 AND progress_percent <= 100),
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'paused', 'completed', 'archived')),
        pacing_rationale TEXT NOT NULL DEFAULT '',
        personalization_rationale TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (goal_id) REFERENCES goals(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roadmap_phases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_id INTEGER NOT NULL,
        sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        description TEXT NOT NULL DEFAULT '',
        expected_outcome TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'not_started'
            CHECK (status IN ('not_started', 'in_progress', 'completed', 'skipped')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (roadmap_id, sequence_number),
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phase_id INTEGER NOT NULL,
        sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        description TEXT NOT NULL DEFAULT '',
        skills TEXT NOT NULL DEFAULT '[]',
        suggested_activities TEXT NOT NULL DEFAULT '[]',
        completion_criteria TEXT NOT NULL DEFAULT '',
        estimated_sessions INTEGER NOT NULL DEFAULT 1
            CHECK (estimated_sessions > 0 AND estimated_sessions <= 100),
        estimated_minutes INTEGER NOT NULL DEFAULT 30
            CHECK (estimated_minutes > 0 AND estimated_minutes <= 24 * 60),
        difficulty TEXT NOT NULL DEFAULT 'beginner'
            CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
        status TEXT NOT NULL DEFAULT 'not_started'
            CHECK (status IN ('not_started', 'in_progress', 'completed', 'skipped')),
        progress_percent REAL NOT NULL DEFAULT 0
            CHECK (progress_percent >= 0 AND progress_percent <= 100),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (phase_id, sequence_number),
        FOREIGN KEY (phase_id) REFERENCES roadmap_phases(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        source TEXT NOT NULL CHECK (length(trim(source)) > 0),
        resource_type TEXT NOT NULL CHECK (length(trim(resource_type)) > 0),
        url TEXT NOT NULL UNIQUE CHECK (length(trim(url)) > 0),
        description TEXT NOT NULL DEFAULT '',
        difficulty TEXT NOT NULL DEFAULT 'beginner'
            CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
        estimated_duration_minutes INTEGER NOT NULL
            CHECK (estimated_duration_minutes > 0 AND estimated_duration_minutes <= 24 * 60),
        is_free INTEGER NOT NULL DEFAULT 1 CHECK (is_free IN (0, 1)),
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        roadmap_id INTEGER,
        milestone_id INTEGER,
        resource_id INTEGER NOT NULL,
        relevance_score REAL NOT NULL DEFAULT 0
            CHECK (relevance_score >= 0 AND relevance_score <= 1),
        reason TEXT NOT NULL DEFAULT '',
        mood_suitability TEXT NOT NULL DEFAULT '',
        recommended_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'suggested'
            CHECK (status IN ('suggested', 'accepted', 'dismissed', 'completed', 'archived')),
        metadata TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE SET NULL,
        FOREIGN KEY (milestone_id) REFERENCES milestones(id) ON DELETE SET NULL,
        FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_checkins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        mood TEXT NOT NULL,
        energy_level TEXT NOT NULL
            CHECK (energy_level IN ('low', 'medium', 'high')),
        focus_level INTEGER NOT NULL CHECK (focus_level >= 1 AND focus_level <= 5),
        available_minutes INTEGER NOT NULL
            CHECK (available_minutes > 0 AND available_minutes <= 24 * 60),
        preferred_activity TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        roadmap_id INTEGER,
        milestone_id INTEGER,
        checkin_id INTEGER,
        plan_date TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        total_estimated_minutes INTEGER NOT NULL DEFAULT 0
            CHECK (total_estimated_minutes >= 0 AND total_estimated_minutes <= 24 * 60),
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped')),
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (user_id, plan_date),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE SET NULL,
        FOREIGN KEY (milestone_id) REFERENCES milestones(id) ON DELETE SET NULL,
        FOREIGN KEY (checkin_id) REFERENCES daily_checkins(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        daily_plan_id INTEGER NOT NULL,
        resource_id INTEGER,
        sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        description TEXT NOT NULL DEFAULT '',
        activity_type TEXT NOT NULL,
        estimated_minutes INTEGER NOT NULL
            CHECK (estimated_minutes > 0 AND estimated_minutes <= 24 * 60),
        difficulty TEXT NOT NULL DEFAULT 'beginner'
            CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped')),
        completed_at TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (daily_plan_id, sequence_number),
        FOREIGN KEY (daily_plan_id) REFERENCES daily_plans(id) ON DELETE CASCADE,
        FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reflections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        daily_plan_id INTEGER NOT NULL,
        completion_status TEXT NOT NULL
            CHECK (completion_status IN ('completed', 'partial', 'skipped')),
        learning_summary TEXT NOT NULL DEFAULT '',
        focus_rating INTEGER NOT NULL CHECK (focus_rating >= 1 AND focus_rating <= 5),
        resource_effectiveness INTEGER NOT NULL
            CHECK (resource_effectiveness >= 1 AND resource_effectiveness <= 5),
        difficulty_feedback TEXT NOT NULL
            CHECK (difficulty_feedback IN ('too_easy', 'suitable', 'too_difficult')),
        mood_match INTEGER NOT NULL CHECK (mood_match IN (0, 1)),
        distractions TEXT NOT NULL DEFAULT '[]',
        wants_similar_resources INTEGER NOT NULL CHECK (wants_similar_resources IN (0, 1)),
        mood_after TEXT NOT NULL DEFAULT '',
        insight TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (daily_plan_id) REFERENCES daily_plans(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        resource_id INTEGER NOT NULL,
        daily_plan_id INTEGER,
        interaction_type TEXT NOT NULL CHECK (length(trim(interaction_type)) > 0),
        completion_percent REAL NOT NULL DEFAULT 0
            CHECK (completion_percent >= 0 AND completion_percent <= 100),
        effectiveness_rating INTEGER
            CHECK (
                effectiveness_rating IS NULL
                OR (effectiveness_rating >= 1 AND effectiveness_rating <= 5)
            ),
        duration_minutes INTEGER
            CHECK (
                duration_minutes IS NULL
                OR (duration_minutes >= 0 AND duration_minutes <= 24 * 60)
            ),
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT,
        FOREIGN KEY (daily_plan_id) REFERENCES daily_plans(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adaptation_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        insight_type TEXT NOT NULL CHECK (length(trim(insight_type)) > 0),
        insight TEXT NOT NULL CHECK (length(trim(insight)) > 0),
        confidence_score REAL NOT NULL DEFAULT 0.5
            CHECK (confidence_score >= 0 AND confidence_score <= 1),
        evidence TEXT NOT NULL DEFAULT '[]',
        is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        preference_key TEXT NOT NULL CHECK (length(trim(preference_key)) > 0),
        preference_value TEXT NOT NULL,
        confidence_score REAL NOT NULL DEFAULT 0.5
            CHECK (confidence_score >= 0 AND confidence_score <= 1),
        source TEXT NOT NULL DEFAULT 'system',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (user_id, preference_key),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_goals_user_id ON goals(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_roadmaps_user_id ON roadmaps(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_roadmaps_goal_id ON roadmaps(goal_id)",
    "CREATE INDEX IF NOT EXISTS idx_roadmap_phases_roadmap_id ON roadmap_phases(roadmap_id)",
    "CREATE INDEX IF NOT EXISTS idx_milestones_phase_id ON milestones(phase_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_milestone_id ON recommendations(milestone_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_resource_id ON recommendations(resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_daily_checkins_user_id ON daily_checkins(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_daily_plans_user_date ON daily_plans(user_id, plan_date)",
    "CREATE INDEX IF NOT EXISTS idx_daily_tasks_plan_id ON daily_tasks(daily_plan_id)",
    "CREATE INDEX IF NOT EXISTS idx_reflections_user_id ON reflections(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_resource_interactions_user_id ON resource_interactions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_resource_interactions_resource_id ON resource_interactions(resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_adaptation_insights_user_id ON adaptation_insights(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_adaptation_insights_active ON adaptation_insights(user_id, is_active)",
    "CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_resources_source ON resources(source)",
    "CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type)",
)

EXPECTED_TABLES: tuple[str, ...] = (
    "users",
    "user_profiles",
    "goals",
    "roadmaps",
    "roadmap_phases",
    "milestones",
    "resources",
    "recommendations",
    "daily_checkins",
    "daily_plans",
    "daily_tasks",
    "reflections",
    "resource_interactions",
    "adaptation_insights",
    "user_preferences",
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Idempotently add a column to an existing SQLite table."""
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_recommendations_table(conn: sqlite3.Connection) -> None:
    """
    Ensure recommendations support archived status + metadata JSON.

    SQLite cannot alter CHECK constraints in place, so rebuild when needed.
    """
    if "recommendations" not in list_tables(conn):
        return

    _ensure_column(
        conn,
        "recommendations",
        "metadata",
        "TEXT NOT NULL DEFAULT '{}'",
    )

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='recommendations'"
    ).fetchone()
    create_sql = str(row["sql"] if row is not None else "")
    if "'archived'" in create_sql and "metadata" in create_sql:
        return

    conn.execute(
        """
        CREATE TABLE recommendations__new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            roadmap_id INTEGER,
            milestone_id INTEGER,
            resource_id INTEGER NOT NULL,
            relevance_score REAL NOT NULL DEFAULT 0
                CHECK (relevance_score >= 0 AND relevance_score <= 1),
            reason TEXT NOT NULL DEFAULT '',
            mood_suitability TEXT NOT NULL DEFAULT '',
            recommended_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'suggested'
                CHECK (status IN ('suggested', 'accepted', 'dismissed', 'completed', 'archived')),
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE SET NULL,
            FOREIGN KEY (milestone_id) REFERENCES milestones(id) ON DELETE SET NULL,
            FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
        )
        """
    )
    columns = _table_columns(conn, "recommendations")
    has_metadata = "metadata" in columns
    select_metadata = "metadata" if has_metadata else "'{}'"
    conn.execute(
        f"""
        INSERT INTO recommendations__new (
            id, user_id, roadmap_id, milestone_id, resource_id,
            relevance_score, reason, mood_suitability, recommended_at, status, metadata
        )
        SELECT
            id, user_id, roadmap_id, milestone_id, resource_id,
            relevance_score, reason, mood_suitability, recommended_at, status,
            {select_metadata}
        FROM recommendations
        """
    )
    conn.execute("DROP TABLE recommendations")
    conn.execute("ALTER TABLE recommendations__new RENAME TO recommendations")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply additive schema changes for existing databases."""
    if "roadmaps" in list_tables(conn):
        _ensure_column(conn, "roadmaps", "pacing_rationale", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(
            conn,
            "roadmaps",
            "personalization_rationale",
            "TEXT NOT NULL DEFAULT ''",
        )
    if "milestones" in list_tables(conn):
        _ensure_column(
            conn,
            "milestones",
            "suggested_activities",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            "milestones",
            "completion_criteria",
            "TEXT NOT NULL DEFAULT ''",
        )
        _ensure_column(
            conn,
            "milestones",
            "estimated_sessions",
            "INTEGER NOT NULL DEFAULT 1",
        )
        _ensure_column(
            conn,
            "milestones",
            "estimated_minutes",
            "INTEGER NOT NULL DEFAULT 30",
        )
        _ensure_column(
            conn,
            "milestones",
            "difficulty",
            "TEXT NOT NULL DEFAULT 'beginner'",
        )
    if "daily_plans" in list_tables(conn):
        _ensure_column(
            conn,
            "daily_plans",
            "metadata",
            "TEXT NOT NULL DEFAULT '{}'",
        )
    if "daily_tasks" in list_tables(conn):
        _ensure_column(
            conn,
            "daily_tasks",
            "metadata",
            "TEXT NOT NULL DEFAULT '{}'",
        )
    if "reflections" in list_tables(conn):
        _ensure_column(
            conn,
            "reflections",
            "insight",
            "TEXT NOT NULL DEFAULT ''",
        )
    _migrate_recommendations_table(conn)


def init_db(
    db_path: Optional[Path] = None,
    *,
    settings: Optional[Settings] = None,
) -> Path:
    """
    Create all GrowthOS AI tables and indexes when missing.

    Returns the resolved database path.
    """
    path = (db_path or resolve_db_path(settings)).resolve()
    with get_connection(path, settings=settings) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        _migrate_schema(conn)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
    return path


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """Return user table names present in the database."""
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def dumps_json(value: Any) -> str:
    """Serialize a Python value to JSON text for SQLite storage."""
    return json.dumps(value, ensure_ascii=False)


def loads_json(value: str | None, default: Any) -> Any:
    """Deserialize JSON text from SQLite, returning default on empty input."""
    if value is None or value == "":
        return default
    return json.loads(value)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row into a plain dictionary."""
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


# ---------------------------------------------------------------------------
# Minimal helpers (enough to validate the schema in Step 2)
# ---------------------------------------------------------------------------


def create_user(
    display_name: str,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Insert a user and return the created row."""
    name = display_name.strip()
    if not name:
        raise ValueError("display_name must not be blank")

    now = utc_now_iso()

    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        cursor = active_conn.execute(
            """
            INSERT INTO users (display_name, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (name, now, now),
        )
        user_id = int(cursor.lastrowid)
        row = active_conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("Failed to load created user")
        return result

    if conn is not None:
        return _create(conn)

    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def get_user_by_id(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Fetch a user by primary key."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return row_to_dict(row)

    if conn is not None:
        return _get(conn)

    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def create_user_profile(
    user_id: int,
    *,
    aspiration: str,
    motivation: str,
    current_level: str,
    target_outcome: str,
    learning_style: str,
    preferred_formats: Iterable[str],
    daily_available_minutes: int,
    preferred_session_minutes: int,
    attention_span_minutes: int,
    preferred_learning_time: str,
    habits: Iterable[str],
    distractions: Iterable[str],
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Insert the single active profile for a user and return the created row."""
    now = utc_now_iso()
    formats_json = dumps_json(list(preferred_formats))
    habits_json = dumps_json(list(habits))
    distractions_json = dumps_json(list(distractions))

    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        cursor = active_conn.execute(
            """
            INSERT INTO user_profiles (
                user_id,
                aspiration,
                motivation,
                current_level,
                target_outcome,
                learning_style,
                preferred_formats,
                daily_available_minutes,
                preferred_session_minutes,
                attention_span_minutes,
                preferred_learning_time,
                habits,
                distractions,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                aspiration.strip(),
                motivation.strip(),
                current_level,
                target_outcome.strip(),
                learning_style,
                formats_json,
                daily_available_minutes,
                preferred_session_minutes,
                attention_span_minutes,
                preferred_learning_time,
                habits_json,
                distractions_json,
                now,
                now,
            ),
        )
        profile_id = int(cursor.lastrowid)
        row = active_conn.execute(
            "SELECT * FROM user_profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("Failed to load created user profile")
        result["preferred_formats"] = loads_json(result["preferred_formats"], [])
        result["habits"] = loads_json(result["habits"], [])
        result["distractions"] = loads_json(result["distractions"], [])
        return result

    if conn is not None:
        return _create(conn)

    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def create_goal(
    user_id: int,
    *,
    title: str,
    description: str = "",
    status: str = "active",
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Insert a learning goal and return the created row."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title must not be blank")

    now = utc_now_iso()

    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        cursor = active_conn.execute(
            """
            INSERT INTO goals (user_id, title, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, clean_title, description.strip(), status, now, now),
        )
        goal_id = int(cursor.lastrowid)
        row = active_conn.execute(
            "SELECT * FROM goals WHERE id = ?",
            (goal_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("Failed to load created goal")
        return result

    if conn is not None:
        return _create(conn)

    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def get_user_profile_by_user_id(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Fetch the active profile for a user."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            return None
        result["preferred_formats"] = loads_json(result["preferred_formats"], [])
        result["habits"] = loads_json(result["habits"], [])
        result["distractions"] = loads_json(result["distractions"], [])
        return result

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_goal_by_id(
    goal_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Fetch a goal by primary key."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            "SELECT * FROM goals WHERE id = ?",
            (goal_id,),
        ).fetchone()
        return row_to_dict(row)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def count_users(
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Return the number of users in the database."""

    def _count(active_conn: sqlite3.Connection) -> int:
        row = active_conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        return int(row["c"]) if row is not None else 0

    if conn is not None:
        return _count(conn)
    with get_connection(db_path) as owned_conn:
        return _count(owned_conn)


def create_onboarding_records(
    *,
    display_name: str,
    aspiration: str,
    motivation: str,
    current_level: str,
    target_outcome: str,
    learning_style: str,
    preferred_formats: Iterable[str],
    daily_available_minutes: int,
    preferred_session_minutes: int,
    attention_span_minutes: int,
    preferred_learning_time: str,
    habits: Iterable[str],
    distractions: Iterable[str],
    goal_title: str,
    goal_description: str = "",
    db_path: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Create user, profile, and goal in a single SQLite transaction.

    On any failure the connection context rolls back all inserts.
    """
    with get_connection(db_path) as conn:
        user = create_user(display_name, conn=conn)
        profile = create_user_profile(
            int(user["id"]),
            aspiration=aspiration,
            motivation=motivation,
            current_level=current_level,
            target_outcome=target_outcome,
            learning_style=learning_style,
            preferred_formats=preferred_formats,
            daily_available_minutes=daily_available_minutes,
            preferred_session_minutes=preferred_session_minutes,
            attention_span_minutes=attention_span_minutes,
            preferred_learning_time=preferred_learning_time,
            habits=habits,
            distractions=distractions,
            conn=conn,
        )
        goal = create_goal(
            int(user["id"]),
            title=goal_title,
            description=goal_description,
            status="active",
            conn=conn,
        )
        return user, profile, goal


def get_active_roadmap_for_goal(
    goal_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Return the active roadmap for a goal, if any."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            """
            SELECT * FROM roadmaps
            WHERE goal_id = ? AND status = 'active'
            ORDER BY id DESC
            LIMIT 1
            """,
            (goal_id,),
        ).fetchone()
        return row_to_dict(row)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def count_roadmaps(
    *,
    db_path: Optional[Path] = None,
    goal_id: Optional[int] = None,
) -> int:
    """Count roadmaps, optionally filtered by goal."""
    with get_connection(db_path) as conn:
        if goal_id is None:
            row = conn.execute("SELECT COUNT(*) AS c FROM roadmaps").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM roadmaps WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0


def archive_active_roadmaps_for_goal(
    goal_id: int,
    *,
    conn: sqlite3.Connection,
) -> int:
    """Mark active roadmaps for a goal as archived. Returns rows updated."""
    now = utc_now_iso()
    cursor = conn.execute(
        """
        UPDATE roadmaps
        SET status = 'archived', updated_at = ?
        WHERE goal_id = ? AND status = 'active'
        """,
        (now, goal_id),
    )
    return int(cursor.rowcount)


def _milestone_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = row_to_dict(row)
    assert result is not None
    result["skills"] = loads_json(result.get("skills"), [])
    result["suggested_activities"] = loads_json(
        result.get("suggested_activities"),
        [],
    )
    return result


def get_roadmap_with_details(
    roadmap_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Load a roadmap with nested phases and milestones."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        roadmap = row_to_dict(
            active_conn.execute(
                "SELECT * FROM roadmaps WHERE id = ?",
                (roadmap_id,),
            ).fetchone()
        )
        if roadmap is None:
            return None

        phase_rows = active_conn.execute(
            """
            SELECT * FROM roadmap_phases
            WHERE roadmap_id = ?
            ORDER BY sequence_number ASC
            """,
            (roadmap_id,),
        ).fetchall()

        phases: list[dict[str, Any]] = []
        active_milestone: dict[str, Any] | None = None
        for phase_row in phase_rows:
            phase = row_to_dict(phase_row)
            assert phase is not None
            milestone_rows = active_conn.execute(
                """
                SELECT * FROM milestones
                WHERE phase_id = ?
                ORDER BY sequence_number ASC
                """,
                (int(phase["id"]),),
            ).fetchall()
            milestones = [_milestone_dict(row) for row in milestone_rows]
            phase["milestones"] = milestones
            phases.append(phase)
            if active_milestone is None:
                for milestone in milestones:
                    if milestone["status"] == "in_progress":
                        active_milestone = milestone
                        break
        if active_milestone is None and phases:
            for phase in phases:
                if phase["milestones"]:
                    active_milestone = phase["milestones"][0]
                    break

        roadmap["phases"] = phases
        roadmap["active_milestone"] = active_milestone
        return roadmap

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def create_roadmap_bundle(
    *,
    user_id: int,
    goal_id: int,
    title: str,
    summary: str,
    estimated_duration_weeks: int,
    pacing_rationale: str,
    personalization_rationale: str,
    phases: Sequence[dict[str, Any]],
    archive_existing_active: bool = False,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Persist roadmap + phases + milestones in one transaction.

    Each phase dict must include:
    sequence_number, title, description, expected_outcome, status, milestones
    Each milestone dict must include skill/activity fields and status.
    """
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        if archive_existing_active:
            archive_active_roadmaps_for_goal(goal_id, conn=conn)

        cursor = conn.execute(
            """
            INSERT INTO roadmaps (
                user_id, goal_id, title, summary, estimated_duration_weeks,
                progress_percent, status, pacing_rationale, personalization_rationale,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 0, 'active', ?, ?, ?, ?)
            """,
            (
                user_id,
                goal_id,
                title.strip(),
                summary.strip(),
                estimated_duration_weeks,
                pacing_rationale.strip(),
                personalization_rationale.strip(),
                now,
                now,
            ),
        )
        roadmap_id = int(cursor.lastrowid)

        for phase in phases:
            phase_cursor = conn.execute(
                """
                INSERT INTO roadmap_phases (
                    roadmap_id, sequence_number, title, description,
                    expected_outcome, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    roadmap_id,
                    int(phase["sequence_number"]),
                    str(phase["title"]).strip(),
                    str(phase.get("description", "")).strip(),
                    str(phase.get("expected_outcome", "")).strip(),
                    str(phase.get("status", "not_started")),
                    now,
                    now,
                ),
            )
            phase_id = int(phase_cursor.lastrowid)
            for milestone in phase.get("milestones", []):
                conn.execute(
                    """
                    INSERT INTO milestones (
                        phase_id, sequence_number, title, description, skills,
                        suggested_activities, completion_criteria,
                        estimated_sessions, estimated_minutes, difficulty,
                        status, progress_percent, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        phase_id,
                        int(milestone["sequence_number"]),
                        str(milestone["title"]).strip(),
                        str(milestone.get("description", "")).strip(),
                        dumps_json(list(milestone.get("skills", []))),
                        dumps_json(list(milestone.get("suggested_activities", []))),
                        str(milestone.get("completion_criteria", "")).strip(),
                        int(milestone.get("estimated_sessions", 1)),
                        int(milestone.get("estimated_minutes", 30)),
                        str(milestone.get("difficulty", "beginner")),
                        str(milestone.get("status", "not_started")),
                        now,
                        now,
                    ),
                )

        details = get_roadmap_with_details(roadmap_id, conn=conn)
        if details is None:
            raise RuntimeError("Failed to reload created roadmap")
        return details


# ---------------------------------------------------------------------------
# Resources / recommendations
# ---------------------------------------------------------------------------


def _resource_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["is_free"] = bool(result.get("is_free"))
    result["metadata"] = loads_json(result.get("metadata"), {})
    return result


def _recommendation_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["metadata"] = loads_json(result.get("metadata"), {})
    return result


def get_roadmap_by_id(
    roadmap_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return row_to_dict(
            active_conn.execute(
                "SELECT * FROM roadmaps WHERE id = ?",
                (roadmap_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_milestone_by_id(
    milestone_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Load a milestone with roadmap_id and user_id for ownership checks."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            """
            SELECT
                m.*,
                p.roadmap_id AS roadmap_id,
                r.user_id AS user_id,
                r.goal_id AS goal_id
            FROM milestones m
            JOIN roadmap_phases p ON p.id = m.phase_id
            JOIN roadmaps r ON r.id = p.roadmap_id
            WHERE m.id = ?
            """,
            (milestone_id,),
        ).fetchone()
        if row is None:
            return None
        return _milestone_dict(row)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_resource_by_id(
    resource_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return _resource_dict(
            active_conn.execute(
                "SELECT * FROM resources WHERE id = ?",
                (resource_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_resource_by_url(
    url: str,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return _resource_dict(
            active_conn.execute(
                "SELECT * FROM resources WHERE url = ?",
                (url.strip(),),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def count_resources(
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    def _count(active_conn: sqlite3.Connection) -> int:
        row = active_conn.execute("SELECT COUNT(*) AS c FROM resources").fetchone()
        return int(row["c"])

    if conn is not None:
        return _count(conn)
    with get_connection(db_path) as owned_conn:
        return _count(owned_conn)


def count_recommendations(
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
    status: Optional[str] = None,
) -> int:
    def _count(active_conn: sqlite3.Connection) -> int:
        if status is None:
            row = active_conn.execute(
                "SELECT COUNT(*) AS c FROM recommendations"
            ).fetchone()
        else:
            row = active_conn.execute(
                "SELECT COUNT(*) AS c FROM recommendations WHERE status = ?",
                (status,),
            ).fetchone()
        return int(row["c"])

    if conn is not None:
        return _count(conn)
    with get_connection(db_path) as owned_conn:
        return _count(owned_conn)


def upsert_catalog_resources(
    catalog_rows: Sequence[dict[str, Any]],
    *,
    db_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Idempotently import catalog resources keyed by URL.

    Returns mapping catalog_id -> resource_id.
    Existing resource IDs are preserved; mutable metadata is updated.
    """
    now = utc_now_iso()
    mapping: dict[str, int] = {}
    with get_connection(db_path) as conn:
        for row in catalog_rows:
            catalog_id = str(row["catalog_id"]).strip()
            url = str(row["url"]).strip()
            metadata = dumps_json(row.get("metadata") or {})
            existing = get_resource_by_url(url, conn=conn)
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO resources (
                        title, source, resource_type, url, description,
                        difficulty, estimated_duration_minutes, is_free,
                        metadata, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row["title"]).strip(),
                        str(row["source"]).strip(),
                        str(row["resource_type"]).strip(),
                        url,
                        str(row.get("description", "")).strip(),
                        str(row.get("difficulty", "beginner")),
                        int(row["estimated_duration_minutes"]),
                        1 if row.get("is_free", True) else 0,
                        metadata,
                        now,
                        now,
                    ),
                )
                mapping[catalog_id] = int(cursor.lastrowid)
            else:
                conn.execute(
                    """
                    UPDATE resources
                    SET title = ?, source = ?, resource_type = ?, description = ?,
                        difficulty = ?, estimated_duration_minutes = ?, is_free = ?,
                        metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        str(row["title"]).strip(),
                        str(row["source"]).strip(),
                        str(row["resource_type"]).strip(),
                        str(row.get("description", "")).strip(),
                        str(row.get("difficulty", "beginner")),
                        int(row["estimated_duration_minutes"]),
                        1 if row.get("is_free", True) else 0,
                        metadata,
                        now,
                        int(existing["id"]),
                    ),
                )
                mapping[catalog_id] = int(existing["id"])
    return mapping


def list_resources(
    *,
    free_only: bool = True,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        if free_only:
            rows = active_conn.execute(
                "SELECT * FROM resources WHERE is_free = 1 ORDER BY id ASC"
            ).fetchall()
        else:
            rows = active_conn.execute(
                "SELECT * FROM resources ORDER BY id ASC"
            ).fetchall()
        return [_resource_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def archive_active_recommendations_for_milestone(
    user_id: int,
    milestone_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    def _archive(active_conn: sqlite3.Connection) -> int:
        cursor = active_conn.execute(
            """
            UPDATE recommendations
            SET status = 'archived'
            WHERE user_id = ? AND milestone_id = ? AND status = 'suggested'
            """,
            (user_id, milestone_id),
        )
        return int(cursor.rowcount)

    if conn is not None:
        return _archive(conn)
    with get_connection(db_path) as owned_conn:
        return _archive(owned_conn)


def get_active_recommendations_for_milestone(
    user_id: int,
    milestone_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Return active (suggested) recommendations joined with resource rows."""

    def _get(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(
            """
            SELECT
                rec.*,
                res.title AS resource_title,
                res.source AS resource_source,
                res.resource_type AS resource_type,
                res.url AS resource_url,
                res.description AS resource_description,
                res.difficulty AS resource_difficulty,
                res.estimated_duration_minutes AS resource_estimated_duration_minutes,
                res.metadata AS resource_metadata
            FROM recommendations rec
            JOIN resources res ON res.id = rec.resource_id
            WHERE rec.user_id = ?
              AND rec.milestone_id = ?
              AND rec.status = 'suggested'
            ORDER BY rec.relevance_score DESC, rec.id ASC
            """,
            (user_id, milestone_id),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = _recommendation_dict(row)
            assert item is not None
            resource_meta = loads_json(row["resource_metadata"], {})
            item["resource"] = {
                "id": int(item["resource_id"]),
                "title": row["resource_title"],
                "source": row["resource_source"],
                "resource_type": row["resource_type"],
                "url": row["resource_url"],
                "description": row["resource_description"],
                "difficulty": row["resource_difficulty"],
                "estimated_duration_minutes": row[
                    "resource_estimated_duration_minutes"
                ],
                "metadata": resource_meta,
            }
            results.append(item)
        return results

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def create_recommendations_bundle(
    *,
    user_id: int,
    roadmap_id: int,
    milestone_id: int,
    recommendations: Sequence[dict[str, Any]],
    archive_existing_active: bool = False,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Persist a recommendation set in one transaction.

    Each recommendation dict requires:
    resource_id, relevance_score, reason, mood_suitability, metadata
    """
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        if archive_existing_active:
            archive_active_recommendations_for_milestone(
                user_id,
                milestone_id,
                conn=conn,
            )
        for item in recommendations:
            conn.execute(
                """
                INSERT INTO recommendations (
                    user_id, roadmap_id, milestone_id, resource_id,
                    relevance_score, reason, mood_suitability,
                    recommended_at, status, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'suggested', ?)
                """,
                (
                    user_id,
                    roadmap_id,
                    milestone_id,
                    int(item["resource_id"]),
                    float(item["relevance_score"]),
                    str(item.get("reason", "")).strip(),
                    str(item.get("mood_suitability", "")).strip(),
                    now,
                    dumps_json(item.get("metadata") or {}),
                ),
            )
        return get_active_recommendations_for_milestone(
            user_id,
            milestone_id,
            conn=conn,
        )


# ---------------------------------------------------------------------------
# Daily loop: check-ins, plans, tasks, reflections, adaptations
# ---------------------------------------------------------------------------


def _checkin_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return row_to_dict(row)


def _plan_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["metadata"] = loads_json(result.get("metadata"), {})
    return result


def _task_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["metadata"] = loads_json(result.get("metadata"), {})
    meta = result["metadata"] if isinstance(result["metadata"], dict) else {}
    result["why_selected"] = str(meta.get("why_selected") or "")
    result["milestone_connection"] = str(meta.get("milestone_connection") or "")
    result["expected_outcome"] = str(meta.get("expected_outcome") or "")
    result["content_type"] = str(meta.get("content_type") or "")
    result["mood_rationale"] = str(meta.get("mood_rationale") or "")
    return result


def _reflection_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["mood_match"] = bool(result.get("mood_match"))
    result["wants_similar_resources"] = bool(result.get("wants_similar_resources"))
    result["distractions"] = loads_json(result.get("distractions"), [])
    result["insight"] = result.get("insight") or ""
    return result


def _adaptation_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    result = row_to_dict(row)
    if result is None:
        return None
    result["is_active"] = bool(result.get("is_active"))
    result["evidence"] = loads_json(result.get("evidence"), [])
    return result


def _preference_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return row_to_dict(row)


def create_daily_checkin(
    user_id: int,
    *,
    mood: str,
    energy_level: str,
    focus_level: int,
    available_minutes: int,
    preferred_activity: str,
    notes: str = "",
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Persist a daily mood check-in."""

    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        cursor = active_conn.execute(
            """
            INSERT INTO daily_checkins (
                user_id, mood, energy_level, focus_level, available_minutes,
                preferred_activity, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                str(mood).strip(),
                str(energy_level).strip(),
                int(focus_level),
                int(available_minutes),
                str(preferred_activity).strip(),
                (notes or "").strip(),
                now,
            ),
        )
        row = active_conn.execute(
            "SELECT * FROM daily_checkins WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        result = _checkin_dict(row)
        assert result is not None
        return result

    if conn is not None:
        return _create(conn)
    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def get_daily_checkin_by_id(
    checkin_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return _checkin_dict(
            active_conn.execute(
                "SELECT * FROM daily_checkins WHERE id = ?",
                (checkin_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_latest_checkin_for_user(
    user_id: int,
    *,
    on_date: Optional[str] = None,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Return the latest check-in, optionally restricted to a calendar date (UTC)."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        if on_date:
            return _checkin_dict(
                active_conn.execute(
                    """
                    SELECT * FROM daily_checkins
                    WHERE user_id = ? AND substr(created_at, 1, 10) = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (user_id, on_date),
                ).fetchone()
            )
        return _checkin_dict(
            active_conn.execute(
                """
                SELECT * FROM daily_checkins
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_daily_plan_by_date(
    user_id: int,
    plan_date: str,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        plan = _plan_dict(
            active_conn.execute(
                """
                SELECT * FROM daily_plans
                WHERE user_id = ? AND plan_date = ?
                """,
                (user_id, plan_date),
            ).fetchone()
        )
        if plan is None:
            return None
        return get_daily_plan_with_tasks(int(plan["id"]), conn=active_conn)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_daily_plan_by_id(
    plan_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return get_daily_plan_with_tasks(plan_id, conn=active_conn)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_daily_plan_with_tasks(
    plan_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Load a plan with ordered tasks and optional resource display fields."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        plan = _plan_dict(
            active_conn.execute(
                "SELECT * FROM daily_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        )
        if plan is None:
            return None

        meta = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        plan["guidance_tone"] = str(meta.get("guidance_tone") or "")
        plan["mood_influence_summary"] = str(meta.get("mood_influence_summary") or "")
        plan["adaptation_explanation"] = str(meta.get("adaptation_explanation") or "")
        plan["task_count_rationale"] = str(meta.get("task_count_rationale") or "")

        rows = active_conn.execute(
            """
            SELECT
                t.*,
                r.title AS resource_title,
                r.source AS resource_source,
                r.url AS resource_url,
                r.resource_type AS resource_type,
                r.metadata AS resource_metadata
            FROM daily_tasks t
            LEFT JOIN resources r ON r.id = t.resource_id
            WHERE t.daily_plan_id = ?
            ORDER BY t.sequence_number ASC
            """,
            (plan_id,),
        ).fetchall()
        tasks: list[dict[str, Any]] = []
        for row in rows:
            task = _task_dict(row)
            assert task is not None
            task["resource_title"] = row["resource_title"]
            task["resource_source"] = row["resource_source"]
            task["resource_url"] = row["resource_url"]
            if not task.get("content_type") and row["resource_type"]:
                task["content_type"] = str(row["resource_type"])
            resource_meta = loads_json(row["resource_metadata"], {})
            catalog_meta = (
                resource_meta.get("catalog_metadata")
                if isinstance(resource_meta.get("catalog_metadata"), dict)
                else {}
            )
            thumbnail = str(catalog_meta.get("thumbnail_url") or "").strip()
            channel = str(catalog_meta.get("channel_title") or "").strip()
            task["resource_thumbnail_url"] = thumbnail or None
            task["resource_channel"] = channel or None
            task_meta = dict(task.get("metadata") or {})
            if thumbnail:
                task_meta.setdefault("thumbnail_url", thumbnail)
            if channel:
                task_meta.setdefault("channel_title", channel)
            discovery = str(
                catalog_meta.get("discovery_source")
                or resource_meta.get("discovery_source")
                or ""
            ).strip()
            if discovery:
                task_meta.setdefault("discovery_source", discovery)
            task["metadata"] = task_meta
            tasks.append(task)
        plan["tasks"] = tasks
        return plan

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def delete_daily_plan(
    plan_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Delete a plan and cascaded tasks (used when regenerating the same day)."""

    def _delete(active_conn: sqlite3.Connection) -> None:
        active_conn.execute("DELETE FROM daily_plans WHERE id = ?", (plan_id,))

    if conn is not None:
        _delete(conn)
        return
    with get_connection(db_path) as owned_conn:
        _delete(owned_conn)


def create_daily_plan_bundle(
    *,
    user_id: int,
    plan_date: str,
    summary: str,
    total_estimated_minutes: int,
    tasks: Sequence[dict[str, Any]],
    roadmap_id: Optional[int] = None,
    milestone_id: Optional[int] = None,
    checkin_id: Optional[int] = None,
    status: str = "pending",
    metadata: Optional[dict[str, Any]] = None,
    replace_existing: bool = False,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Persist a daily plan and its tasks in one transaction.

    Each task dict should include:
    sequence_number, title, description, activity_type, estimated_minutes,
    difficulty, optional resource_id, optional metadata, optional status.
    """
    if not 1 <= len(tasks) <= 5:
        raise ValueError("A daily plan must include between 1 and 5 tasks")

    now = utc_now_iso()
    with get_connection(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id FROM daily_plans
            WHERE user_id = ? AND plan_date = ?
            """,
            (user_id, plan_date),
        ).fetchone()
        if existing is not None:
            if not replace_existing:
                details = get_daily_plan_with_tasks(int(existing["id"]), conn=conn)
                if details is None:
                    raise RuntimeError("Existing daily plan could not be loaded")
                return details
            delete_daily_plan(int(existing["id"]), conn=conn)

        cursor = conn.execute(
            """
            INSERT INTO daily_plans (
                user_id, roadmap_id, milestone_id, checkin_id, plan_date,
                summary, total_estimated_minutes, status, metadata,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                roadmap_id,
                milestone_id,
                checkin_id,
                plan_date,
                summary.strip(),
                int(total_estimated_minutes),
                status,
                dumps_json(metadata or {}),
                now,
                now,
            ),
        )
        plan_id = int(cursor.lastrowid)
        for task in tasks:
            conn.execute(
                """
                INSERT INTO daily_tasks (
                    daily_plan_id, resource_id, sequence_number, title, description,
                    activity_type, estimated_minutes, difficulty, status,
                    completed_at, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    plan_id,
                    task.get("resource_id"),
                    int(task["sequence_number"]),
                    str(task["title"]).strip(),
                    str(task.get("description") or "").strip(),
                    str(task["activity_type"]).strip(),
                    int(task["estimated_minutes"]),
                    str(task.get("difficulty") or "beginner"),
                    str(task.get("status") or "pending"),
                    dumps_json(task.get("metadata") or {}),
                    now,
                    now,
                ),
            )
        details = get_daily_plan_with_tasks(plan_id, conn=conn)
        if details is None:
            raise RuntimeError("Failed to reload created daily plan")
        return details


def get_task_by_id(
    task_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = active_conn.execute(
            """
            SELECT t.*, p.user_id AS user_id, p.id AS plan_id
            FROM daily_tasks t
            JOIN daily_plans p ON p.id = t.daily_plan_id
            WHERE t.id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        task = _task_dict(row)
        assert task is not None
        task["user_id"] = int(row["user_id"])
        task["plan_id"] = int(row["plan_id"])
        return task

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def update_task_status(
    task_id: int,
    status: str,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Update task status; set completed_at when status is completed."""

    def _update(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        completed_at = now if status == "completed" else None
        active_conn.execute(
            """
            UPDATE daily_tasks
            SET status = ?, completed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, completed_at, now, task_id),
        )
        task = get_task_by_id(task_id, conn=active_conn)
        if task is None:
            raise RuntimeError("Task not found after update")
        return task

    if conn is not None:
        return _update(conn)
    with get_connection(db_path) as owned_conn:
        return _update(owned_conn)


def create_resource_interaction(
    *,
    user_id: int,
    resource_id: int,
    interaction_type: str,
    daily_plan_id: Optional[int] = None,
    completion_percent: float = 0,
    effectiveness_rating: Optional[int] = None,
    duration_minutes: Optional[int] = None,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        cursor = active_conn.execute(
            """
            INSERT INTO resource_interactions (
                user_id, resource_id, daily_plan_id, interaction_type,
                completion_percent, effectiveness_rating, duration_minutes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                resource_id,
                daily_plan_id,
                interaction_type.strip(),
                float(completion_percent),
                effectiveness_rating,
                duration_minutes,
                now,
            ),
        )
        row = active_conn.execute(
            "SELECT * FROM resource_interactions WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        result = row_to_dict(row)
        assert result is not None
        return result

    if conn is not None:
        return _create(conn)
    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def update_daily_plan_status(
    plan_id: int,
    status: str,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _update(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        active_conn.execute(
            """
            UPDATE daily_plans
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, plan_id),
        )
        details = get_daily_plan_with_tasks(plan_id, conn=active_conn)
        if details is None:
            raise RuntimeError("Plan not found after update")
        return details

    if conn is not None:
        return _update(conn)
    with get_connection(db_path) as owned_conn:
        return _update(owned_conn)


def update_milestone_progress(
    milestone_id: int,
    *,
    progress_percent: float,
    status: Optional[str] = None,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _update(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        progress = max(0.0, min(100.0, float(progress_percent)))
        if status is None:
            active_conn.execute(
                """
                UPDATE milestones
                SET progress_percent = ?, updated_at = ?
                WHERE id = ?
                """,
                (progress, now, milestone_id),
            )
        else:
            active_conn.execute(
                """
                UPDATE milestones
                SET progress_percent = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (progress, status, now, milestone_id),
            )
        milestone = get_milestone_by_id(milestone_id, conn=active_conn)
        if milestone is None:
            raise RuntimeError("Milestone not found after progress update")
        return milestone

    if conn is not None:
        return _update(conn)
    with get_connection(db_path) as owned_conn:
        return _update(owned_conn)


def update_roadmap_progress(
    roadmap_id: int,
    *,
    progress_percent: float,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _update(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        progress = max(0.0, min(100.0, float(progress_percent)))
        active_conn.execute(
            """
            UPDATE roadmaps
            SET progress_percent = ?, updated_at = ?
            WHERE id = ?
            """,
            (progress, now, roadmap_id),
        )
        roadmap = get_roadmap_by_id(roadmap_id, conn=active_conn)
        if roadmap is None:
            raise RuntimeError("Roadmap not found after progress update")
        return roadmap

    if conn is not None:
        return _update(conn)
    with get_connection(db_path) as owned_conn:
        return _update(owned_conn)


def create_reflection(
    *,
    user_id: int,
    daily_plan_id: int,
    completion_status: str,
    learning_summary: str,
    focus_rating: int,
    resource_effectiveness: int,
    difficulty_feedback: str,
    mood_match: bool,
    distractions: Sequence[str],
    wants_similar_resources: bool,
    mood_after: str,
    insight: str = "",
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        cursor = active_conn.execute(
            """
            INSERT INTO reflections (
                user_id, daily_plan_id, completion_status, learning_summary,
                focus_rating, resource_effectiveness, difficulty_feedback,
                mood_match, distractions, wants_similar_resources, mood_after,
                insight, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                daily_plan_id,
                completion_status,
                (learning_summary or "").strip(),
                int(focus_rating),
                int(resource_effectiveness),
                difficulty_feedback,
                1 if mood_match else 0,
                dumps_json(list(distractions)),
                1 if wants_similar_resources else 0,
                str(mood_after),
                (insight or "").strip(),
                now,
            ),
        )
        row = active_conn.execute(
            "SELECT * FROM reflections WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        result = _reflection_dict(row)
        assert result is not None
        return result

    if conn is not None:
        return _create(conn)
    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def get_reflection_by_id(
    reflection_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return _reflection_dict(
            active_conn.execute(
                "SELECT * FROM reflections WHERE id = ?",
                (reflection_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_reflection_for_plan(
    daily_plan_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    """Return the newest reflection for a daily plan, if any."""

    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return _reflection_dict(
            active_conn.execute(
                """
                SELECT * FROM reflections
                WHERE daily_plan_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (daily_plan_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def list_reflections_for_user(
    user_id: int,
    *,
    limit: int = 10,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(
            """
            SELECT * FROM reflections
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, int(limit))),
        ).fetchall()
        return [_reflection_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def list_resource_interactions_for_plan(
    daily_plan_id: int,
    *,
    user_id: Optional[int] = None,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        if user_id is None:
            rows = active_conn.execute(
                """
                SELECT * FROM resource_interactions
                WHERE daily_plan_id = ?
                ORDER BY id ASC
                """,
                (daily_plan_id,),
            ).fetchall()
        else:
            rows = active_conn.execute(
                """
                SELECT * FROM resource_interactions
                WHERE daily_plan_id = ? AND user_id = ?
                ORDER BY id ASC
                """,
                (daily_plan_id, user_id),
            ).fetchall()
        return [row_to_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def list_milestones_for_roadmap(
    roadmap_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Return all milestones for a roadmap in phase/sequence order."""

    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(
            """
            SELECT
                m.*,
                p.roadmap_id AS roadmap_id,
                r.user_id AS user_id,
                r.goal_id AS goal_id
            FROM milestones m
            JOIN roadmap_phases p ON p.id = m.phase_id
            JOIN roadmaps r ON r.id = p.roadmap_id
            WHERE p.roadmap_id = ?
            ORDER BY p.sequence_number ASC, m.sequence_number ASC
            """,
            (roadmap_id,),
        ).fetchall()
        return [_milestone_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def create_reflection_bundle(
    *,
    user_id: int,
    daily_plan_id: int,
    completion_status: str,
    learning_summary: str,
    focus_rating: int,
    resource_effectiveness: int,
    difficulty_feedback: str,
    mood_match: bool,
    distractions: Sequence[str],
    wants_similar_resources: bool,
    mood_after: str,
    insight: str,
    task_updates: Sequence[dict[str, Any]],
    resource_interactions: Sequence[dict[str, Any]],
    plan_status: str,
    milestone_id: Optional[int],
    milestone_progress_percent: Optional[float],
    milestone_status: Optional[str],
    roadmap_id: Optional[int],
    roadmap_progress_percent: Optional[float],
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Persist reflection evidence in one transaction:
    task updates, resource interactions, reflection, plan status, progress.
    """
    with get_connection(db_path) as conn:
        for item in task_updates:
            task_id = int(item["task_id"])
            status = str(item["status"])
            update_task_status(task_id, status, conn=conn)
            metadata_patch = item.get("metadata_patch") or {}
            if metadata_patch:
                task = get_task_by_id(task_id, conn=conn)
                if task is None:
                    raise RuntimeError("Task missing during reflection bundle")
                merged = dict(task.get("metadata") or {})
                merged.update(metadata_patch)
                now = utc_now_iso()
                conn.execute(
                    """
                    UPDATE daily_tasks
                    SET metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (dumps_json(merged), now, task_id),
                )

        interactions: list[dict[str, Any]] = []
        for item in resource_interactions:
            interactions.append(
                create_resource_interaction(
                    user_id=user_id,
                    resource_id=int(item["resource_id"]),
                    interaction_type=str(item["interaction_type"]),
                    daily_plan_id=daily_plan_id,
                    completion_percent=float(item.get("completion_percent") or 0),
                    effectiveness_rating=item.get("effectiveness_rating"),
                    duration_minutes=item.get("duration_minutes"),
                    conn=conn,
                )
            )

        reflection = create_reflection(
            user_id=user_id,
            daily_plan_id=daily_plan_id,
            completion_status=completion_status,
            learning_summary=learning_summary,
            focus_rating=focus_rating,
            resource_effectiveness=resource_effectiveness,
            difficulty_feedback=difficulty_feedback,
            mood_match=mood_match,
            distractions=distractions,
            wants_similar_resources=wants_similar_resources,
            mood_after=mood_after,
            insight=insight,
            conn=conn,
        )
        plan = update_daily_plan_status(daily_plan_id, plan_status, conn=conn)

        milestone = None
        if milestone_id is not None and milestone_progress_percent is not None:
            milestone = update_milestone_progress(
                milestone_id,
                progress_percent=float(milestone_progress_percent),
                status=milestone_status,
                conn=conn,
            )

        roadmap = None
        if roadmap_id is not None and roadmap_progress_percent is not None:
            roadmap = update_roadmap_progress(
                roadmap_id,
                progress_percent=float(roadmap_progress_percent),
                conn=conn,
            )

        return {
            "reflection": reflection,
            "plan": plan,
            "interactions": interactions,
            "milestone": milestone,
            "roadmap": roadmap,
        }


def create_adaptation_insight(
    *,
    user_id: int,
    insight_type: str,
    insight: str,
    confidence_score: float = 0.5,
    evidence: Optional[Sequence[str]] = None,
    is_active: bool = True,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _create(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        cursor = active_conn.execute(
            """
            INSERT INTO adaptation_insights (
                user_id, insight_type, insight, confidence_score, evidence,
                is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                insight_type.strip(),
                insight.strip(),
                float(confidence_score),
                dumps_json(list(evidence or [])),
                1 if is_active else 0,
                now,
                now,
            ),
        )
        row = active_conn.execute(
            "SELECT * FROM adaptation_insights WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        result = _adaptation_dict(row)
        assert result is not None
        return result

    if conn is not None:
        return _create(conn)
    with get_connection(db_path) as owned_conn:
        return _create(owned_conn)


def deactivate_adaptation_insights(
    user_id: int,
    *,
    insight_type: Optional[str] = None,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    def _deactivate(active_conn: sqlite3.Connection) -> int:
        now = utc_now_iso()
        if insight_type:
            cursor = active_conn.execute(
                """
                UPDATE adaptation_insights
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND insight_type = ? AND is_active = 1
                """,
                (now, user_id, insight_type),
            )
        else:
            cursor = active_conn.execute(
                """
                UPDATE adaptation_insights
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND is_active = 1
                """,
                (now, user_id),
            )
        return int(cursor.rowcount)

    if conn is not None:
        return _deactivate(conn)
    with get_connection(db_path) as owned_conn:
        return _deactivate(owned_conn)


def list_active_adaptation_insights(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(
            """
            SELECT * FROM adaptation_insights
            WHERE user_id = ? AND is_active = 1
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
        return [_adaptation_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def upsert_user_preference(
    *,
    user_id: int,
    preference_key: str,
    preference_value: str,
    confidence_score: float = 0.5,
    source: str = "adaptation",
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    def _upsert(active_conn: sqlite3.Connection) -> dict[str, Any]:
        now = utc_now_iso()
        existing = active_conn.execute(
            """
            SELECT id FROM user_preferences
            WHERE user_id = ? AND preference_key = ?
            """,
            (user_id, preference_key.strip()),
        ).fetchone()
        if existing is None:
            cursor = active_conn.execute(
                """
                INSERT INTO user_preferences (
                    user_id, preference_key, preference_value, confidence_score,
                    source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    preference_key.strip(),
                    preference_value,
                    float(confidence_score),
                    source,
                    now,
                    now,
                ),
            )
            pref_id = int(cursor.lastrowid)
        else:
            pref_id = int(existing["id"])
            active_conn.execute(
                """
                UPDATE user_preferences
                SET preference_value = ?, confidence_score = ?, source = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    preference_value,
                    float(confidence_score),
                    source,
                    now,
                    pref_id,
                ),
            )
        row = active_conn.execute(
            "SELECT * FROM user_preferences WHERE id = ?",
            (pref_id,),
        ).fetchone()
        result = _preference_dict(row)
        assert result is not None
        return result

    if conn is not None:
        return _upsert(conn)
    with get_connection(db_path) as owned_conn:
        return _upsert(owned_conn)


def list_user_preferences(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(
            """
            SELECT * FROM user_preferences
            WHERE user_id = ?
            ORDER BY preference_key ASC
            """,
            (user_id,),
        ).fetchall()
        return [_preference_dict(row) for row in rows if row is not None]

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def find_adaptation_insights_for_reflection(
    user_id: int,
    reflection_id: int,
    *,
    active_only: bool = True,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find adaptation insights linked to a reflection via evidence marker."""

    marker = f"reflection_id={int(reflection_id)}"

    def _list(active_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        if active_only:
            rows = active_conn.execute(
                """
                SELECT * FROM adaptation_insights
                WHERE user_id = ? AND is_active = 1
                ORDER BY id DESC
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = active_conn.execute(
                """
                SELECT * FROM adaptation_insights
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,),
            ).fetchall()
        matched: list[dict[str, Any]] = []
        for row in rows:
            item = _adaptation_dict(row)
            if item is None:
                continue
            evidence = item.get("evidence") or []
            if marker in evidence:
                matched.append(item)
        return matched

    if conn is not None:
        return _list(conn)
    with get_connection(db_path) as owned_conn:
        return _list(owned_conn)


def create_adaptation_bundle(
    *,
    user_id: int,
    insights: Sequence[dict[str, Any]],
    preferences: Sequence[dict[str, Any]],
    deactivate_insight_ids: Optional[Sequence[int]] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Persist adaptation insights and preference upserts in one transaction.

    Each insight dict: insight_type, insight, confidence_score, evidence, is_active
    Each preference dict: preference_key, preference_value, confidence_score, source
    """
    with get_connection(db_path) as conn:
        now = utc_now_iso()
        for insight_id in deactivate_insight_ids or []:
            conn.execute(
                """
                UPDATE adaptation_insights
                SET is_active = 0, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now, int(insight_id), user_id),
            )

        created_insights: list[dict[str, Any]] = []
        for item in insights:
            created_insights.append(
                create_adaptation_insight(
                    user_id=user_id,
                    insight_type=str(item["insight_type"]),
                    insight=str(item["insight"]),
                    confidence_score=float(item.get("confidence_score") or 0.5),
                    evidence=list(item.get("evidence") or []),
                    is_active=bool(item.get("is_active", True)),
                    conn=conn,
                )
            )

        upserted_prefs: list[dict[str, Any]] = []
        for item in preferences:
            upserted_prefs.append(
                upsert_user_preference(
                    user_id=user_id,
                    preference_key=str(item["preference_key"]),
                    preference_value=str(item["preference_value"]),
                    confidence_score=float(item.get("confidence_score") or 0.5),
                    source=str(item.get("source") or "adaptation"),
                    conn=conn,
                )
            )

        return {
            "insights": created_insights,
            "preferences": upserted_prefs,
        }


def count_completed_plans(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    def _count(active_conn: sqlite3.Connection) -> int:
        row = active_conn.execute(
            """
            SELECT COUNT(*) AS c FROM daily_plans
            WHERE user_id = ? AND status = 'completed'
            """,
            (user_id,),
        ).fetchone()
        return int(row["c"])

    if conn is not None:
        return _count(conn)
    with get_connection(db_path) as owned_conn:
        return _count(owned_conn)


def compute_completion_streak(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Count consecutive completed plan dates ending at the latest completed day."""

    def _streak(active_conn: sqlite3.Connection) -> int:
        rows = active_conn.execute(
            """
            SELECT plan_date FROM daily_plans
            WHERE user_id = ? AND status = 'completed'
            ORDER BY plan_date DESC
            """,
            (user_id,),
        ).fetchall()
        if not rows:
            return 0
        from datetime import date as date_cls, timedelta

        dates = [date_cls.fromisoformat(str(row["plan_date"])) for row in rows]
        streak = 1
        for index in range(1, len(dates)):
            if dates[index - 1] - dates[index] == timedelta(days=1):
                streak += 1
            else:
                break
        return streak

    if conn is not None:
        return _streak(conn)
    with get_connection(db_path) as owned_conn:
        return _streak(owned_conn)


def get_active_goal_for_user(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        return row_to_dict(
            active_conn.execute(
                """
                SELECT * FROM goals
                WHERE user_id = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        )

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def get_active_roadmap_for_user(
    user_id: int,
    *,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any] | None:
    def _get(active_conn: sqlite3.Connection) -> dict[str, Any] | None:
        roadmap = row_to_dict(
            active_conn.execute(
                """
                SELECT * FROM roadmaps
                WHERE user_id = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        )
        if roadmap is None:
            return None
        return get_roadmap_with_details(int(roadmap["id"]), conn=active_conn)

    if conn is not None:
        return _get(conn)
    with get_connection(db_path) as owned_conn:
        return _get(owned_conn)


def build_dashboard_snapshot(
    user_id: int,
    *,
    plan_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any] | None:
    """
    Assemble raw dashboard data from SQLite for later Pydantic mapping.

    Does not call Gemini. Agents may enrich explanation fields later.
    """
    with get_connection(db_path) as conn:
        user = get_user_by_id(user_id, conn=conn)
        if user is None:
            return None
        profile = get_user_profile_by_user_id(user_id, conn=conn)
        goal = get_active_goal_for_user(user_id, conn=conn)
        roadmap = get_active_roadmap_for_user(user_id, conn=conn)
        active_milestone = None
        if roadmap is not None:
            active_milestone = roadmap.get("active_milestone")

        today = plan_date or utc_now_iso()[:10]
        checkin = get_latest_checkin_for_user(user_id, on_date=today, conn=conn)
        plan = get_daily_plan_by_date(user_id, today, conn=conn)
        reflections = list_reflections_for_user(user_id, limit=5, conn=conn)
        adaptations = list_active_adaptation_insights(user_id, conn=conn)
        preferences = list_user_preferences(user_id, conn=conn)
        pref_map = {
            str(item["preference_key"]): str(item["preference_value"])
            for item in preferences
        }

        completed_sessions = count_completed_plans(user_id, conn=conn)
        streak = compute_completion_streak(user_id, conn=conn)

        plan_meta = (plan or {}).get("metadata") if plan else {}
        if not isinstance(plan_meta, dict):
            plan_meta = {}

        preferred_content = pref_map.get("preferred_content_type")
        if not preferred_content and profile:
            formats = list(profile.get("preferred_formats") or [])
            preferred_content = formats[0] if formats else None

        preferred_session: int | None = None
        if pref_map.get("preferred_session_minutes"):
            preferred_session = int(pref_map["preferred_session_minutes"])
        elif profile and profile.get("preferred_session_minutes") is not None:
            preferred_session = int(profile["preferred_session_minutes"])

        return {
            "user": user,
            "profile": profile,
            "active_goal": goal,
            "current_roadmap": roadmap,
            "current_milestone": active_milestone,
            "overall_progress_percent": float(
                (roadmap or {}).get("progress_percent") or 0
            ),
            "today_mood": (checkin or {}).get("mood"),
            "today_plan": plan,
            "completion_streak": streak,
            "recent_reflections": reflections,
            "preferred_content_type": preferred_content,
            "preferred_session_minutes": preferred_session,
            "average_session_minutes": None,
            "resource_effectiveness_avg": None,
            "weekly_learning_consistency": None,
            "skill_growth": list((active_milestone or {}).get("skills") or [])[:5],
            "detected_patterns": [
                str(item.get("insight") or "")
                for item in adaptations
                if item.get("insight_type") == "pattern"
            ],
            "growthos_knows_you": [
                f"{item['preference_key']}: {item['preference_value']}"
                for item in preferences
            ],
            "plan_change_explanation": plan_meta.get("adaptation_explanation")
            or (adaptations[0]["insight"] if adaptations else None),
            "ai_insight": adaptations[0]["insight"] if adaptations else None,
            "recommended_next_action": pref_map.get("recommended_next_action"),
            "adaptation_insights": adaptations,
            "completed_sessions": completed_sessions,
            "preferences": preferences,
        }

