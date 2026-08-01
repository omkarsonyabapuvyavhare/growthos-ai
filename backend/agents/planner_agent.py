"""
Daily Planner Agent for GrowthOS AI.

Deterministic orchestration:
1. Load user / profile / active roadmap / active milestone
2. Load adaptation insights and preferences
3. Persist today's check-in (unless reusing same-day plan)
4. Call Curator for trusted free resources
5. Apply mood + capacity guardrails
6. Gemini structured plan over curated IDs only
7. Validate budget/IDs/difficulty
8. Persist plan + tasks transactionally

Mood never rewrites the long-term goal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from config import Settings, get_settings
from exceptions import (
    CuratorAgentError,
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    PlannerAgentError,
    PlannerBudgetError,
    PlannerContextError,
    PlannerGenerationError,
    PlannerOwnershipError,
    PlannerPersistenceError,
)
from models import (
    ActivityType,
    CuratedRecommendation,
    CuratorAgentResult,
    DailyCheckInRequest,
    DailyCheckInResponse,
    DailyPlanGeneration,
    DailyPlanResponse,
    DailyTaskResponse,
    Difficulty,
    EnergyLevel,
    Mood,
    PlannerAgentResult,
    PlannerTaskGeneration,
    PlanStatus,
    TaskStatus,
)
from services.database import (
    create_daily_checkin,
    create_daily_plan_bundle,
    get_active_goal_for_user,
    get_active_roadmap_for_user,
    get_daily_checkin_by_id,
    get_daily_plan_by_date,
    get_user_by_id,
    get_user_profile_by_user_id,
    list_active_adaptation_insights,
    list_user_preferences,
)
from services.ai_provider import get_ai_provider

logger = logging.getLogger(__name__)

MIN_TASK_MINUTES = 3
MAX_CURATOR_LIMIT = 5
MIN_CURATOR_LIMIT = 3

PLANNER_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Daily Planner.

Create one focused daily learning plan for today's mood, energy, and available time.

Rules:
- Preserve the user's exact goal title. Never rewrite or replace it.
- Stay tied to the active milestone only.
- Use only provided curated resource IDs. Never invent URLs or resource IDs.
- Tasks without a resource are allowed only for practice, review, recall, or reflection.
- Keep total estimated minutes within the available-time budget.
- Explain why each task was selected and how mood influenced the plan.
- Do not diagnose health or invent historical completion claims.
- Optimize for learning progress, not engagement.
""".strip()

PersistPlanFn = Callable[..., dict[str, Any]]
DateProvider = Callable[[], date]


@dataclass(frozen=True)
class MoodCapacityRules:
    min_tasks: int
    max_tasks: int
    max_difficulty: Difficulty
    preferred_activities: tuple[ActivityType, ...]
    guidance_tone: str
    summary: str


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[DailyPlanGeneration],
        *,
        system_instruction: str | None = None,
    ) -> DailyPlanGeneration: ...


class SupportsCurator(Protocol):
    def recommend_resources(
        self,
        user_id: int,
        roadmap_id: int,
        milestone_id: int,
        *,
        mood: Mood | None = None,
        energy_level: EnergyLevel | None = None,
        available_minutes: int | None = None,
        preferred_format: str | None = None,
        limit: int = 5,
        refresh: bool = False,
    ) -> CuratorAgentResult: ...


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _difficulty_rank(value: Difficulty | str) -> int:
    text = value.value if isinstance(value, Difficulty) else str(value)
    return {"beginner": 0, "intermediate": 1, "advanced": 2}.get(text.lower(), 1)


def mood_capacity_rules(
    mood: Mood,
    energy_level: EnergyLevel,
    *,
    available_minutes: int,
    focus_level: int,
) -> MoodCapacityRules:
    """Deterministic mood/energy capacity guardrails for planning."""
    base = {
        Mood.tired: MoodCapacityRules(
            1,
            2,
            Difficulty.beginner,
            (ActivityType.watch, ActivityType.listen, ActivityType.review),
            "calm",
            "Fewer shorter tasks because energy is low.",
        ),
        Mood.low_energy: MoodCapacityRules(
            1,
            2,
            Difficulty.beginner,
            (ActivityType.watch, ActivityType.listen, ActivityType.review),
            "calm",
            "Low cognitive load with short review-style work.",
        ),
        Mood.stressed: MoodCapacityRules(
            1,
            2,
            Difficulty.beginner,
            (ActivityType.watch, ActivityType.review, ActivityType.practice),
            "reassuring",
            "Low-pressure plan focused on one clear outcome.",
        ),
        Mood.distracted: MoodCapacityRules(
            1,
            2,
            Difficulty.beginner,
            (ActivityType.practice, ActivityType.watch, ActivityType.review),
            "structured",
            "Micro-tasks with an explicit step order.",
        ),
        Mood.focused: MoodCapacityRules(
            2,
            4,
            Difficulty.intermediate,
            (ActivityType.practice, ActivityType.review, ActivityType.mixed),
            "direct",
            "Deeper practice with slightly higher difficulty.",
        ),
        Mood.motivated: MoodCapacityRules(
            2,
            5,
            Difficulty.intermediate,
            (ActivityType.practice, ActivityType.watch, ActivityType.mixed),
            "encouraging",
            "Ambitious but time-bounded practical pacing.",
        ),
        Mood.curious: MoodCapacityRules(
            2,
            3,
            Difficulty.intermediate,
            (ActivityType.watch, ActivityType.read, ActivityType.review),
            "exploratory",
            "Milestone-tied exploration plus a short synthesis step.",
        ),
        Mood.calm: MoodCapacityRules(
            2,
            3,
            Difficulty.beginner,
            (ActivityType.read, ActivityType.watch, ActivityType.practice),
            "steady",
            "Steady balanced session for calm focus.",
        ),
    }.get(
        mood,
        MoodCapacityRules(
            1,
            3,
            Difficulty.beginner,
            (ActivityType.watch, ActivityType.practice, ActivityType.review),
            "supportive",
            "Balanced plan based on today's check-in.",
        ),
    )

    max_tasks = base.max_tasks
    max_difficulty = base.max_difficulty
    if energy_level == EnergyLevel.low:
        max_tasks = min(max_tasks, 2)
        max_difficulty = Difficulty.beginner
    elif energy_level == EnergyLevel.high and mood in {
        Mood.focused,
        Mood.motivated,
        Mood.curious,
    }:
        max_tasks = min(5, max(max_tasks, 3))

    if focus_level <= 2:
        max_tasks = min(max_tasks, 2)
        max_difficulty = Difficulty.beginner
    elif focus_level >= 4 and energy_level != EnergyLevel.low:
        max_tasks = min(5, max(max_tasks, 2))

    # Time-based task ceiling: roughly one task per ~5 minutes of availability.
    # Short sessions (e.g. 15 minutes) may still allow 1–2 micro-tasks.
    time_ceiling = max(1, available_minutes // max(MIN_TASK_MINUTES, 5))
    max_tasks = max(1, min(max_tasks, time_ceiling, 5))
    min_tasks = min(base.min_tasks, max_tasks)
    return MoodCapacityRules(
        min_tasks=min_tasks,
        max_tasks=max_tasks,
        max_difficulty=max_difficulty,
        preferred_activities=base.preferred_activities,
        guidance_tone=base.guidance_tone,
        summary=base.summary,
    )


def activity_to_preferred_format(activity: ActivityType) -> str | None:
    mapping = {
        ActivityType.watch: "video",
        ActivityType.listen: "listen",
        ActivityType.read: "read",
        ActivityType.practice: "practice",
        ActivityType.review: "read",
        ActivityType.mixed: None,
    }
    return mapping.get(activity)


def build_adaptation_explanation(
    insights: Sequence[dict[str, Any]],
    preferences: Sequence[dict[str, Any]],
    *,
    mood: Mood,
    available_minutes: int,
) -> str:
    if not insights and not preferences:
        return (
            "Today's plan is based on your current mood, available time, and profile. "
            "No prior adaptation patterns are stored yet."
        )
    parts: list[str] = []
    for item in insights[:2]:
        text = str(item.get("insight") or "").strip()
        if text:
            parts.append(text)
    for pref in preferences[:2]:
        key = str(pref.get("preference_key") or "")
        value = str(pref.get("preference_value") or "")
        if key and value:
            parts.append(f"Learned preference: {key}={value}.")
    if not parts:
        return (
            f"Today's plan is shaped by {mood.value} mood and "
            f"{available_minutes} available minutes."
        )
    return " ".join(parts)


def build_planner_prompt(
    *,
    goal_title: str,
    milestone: dict[str, Any],
    checkin: DailyCheckInRequest,
    profile: dict[str, Any],
    preferences: Sequence[dict[str, Any]],
    insights: Sequence[dict[str, Any]],
    recommendations: Sequence[CuratedRecommendation],
    rules: MoodCapacityRules,
    adaptation_explanation: str,
) -> str:
    candidates = [
        {
            "resource_id": rec.resource_id,
            "title": rec.title,
            "source": rec.source,
            "resource_type": rec.resource_type,
            "difficulty": rec.difficulty.value,
            "estimated_duration_minutes": rec.estimated_duration_minutes,
            "reason": rec.reason,
            "milestone_fit": rec.milestone_fit,
            "suggested_use": rec.suggested_use,
        }
        for rec in recommendations
    ]
    payload = {
        "read_only_goal": {
            "title": goal_title,
            "note": "Do not rewrite or replace this goal.",
        },
        "active_milestone": {
            "title": milestone.get("title"),
            "description": milestone.get("description"),
            "skills": milestone.get("skills") or [],
            "completion_criteria": milestone.get("completion_criteria") or "",
            "difficulty": milestone.get("difficulty"),
        },
        "today_checkin": {
            "mood": checkin.mood.value,
            "energy_level": checkin.energy_level.value,
            "focus_level": checkin.focus_level,
            "available_minutes": checkin.available_minutes,
            "preferred_activity": checkin.preferred_activity.value,
            "notes": checkin.notes,
        },
        "profile_preferences": {
            "preferred_formats": profile.get("preferred_formats") or [],
            "preferred_session_minutes": profile.get("preferred_session_minutes"),
            "attention_span_minutes": profile.get("attention_span_minutes"),
            "current_level": profile.get("current_level"),
        },
        "learned_preferences": [
            {
                "key": item.get("preference_key"),
                "value": item.get("preference_value"),
            }
            for item in preferences
        ],
        "adaptation_insights": [
            {
                "type": item.get("insight_type"),
                "insight": item.get("insight"),
                "evidence": item.get("evidence") or [],
            }
            for item in insights
        ],
        "capacity_rules": {
            "min_tasks": rules.min_tasks,
            "max_tasks": rules.max_tasks,
            "max_difficulty": rules.max_difficulty.value,
            "preferred_activities": [a.value for a in rules.preferred_activities],
            "guidance_tone": rules.guidance_tone,
            "mood_rule_summary": rules.summary,
            "total_budget_minutes": checkin.available_minutes,
            "min_task_minutes": MIN_TASK_MINUTES,
        },
        "adaptation_explanation_seed": adaptation_explanation,
        "curated_candidates": candidates,
        "constraints": {
            "use_only_these_resource_ids": [c["resource_id"] for c in candidates],
            "never_invent_urls": True,
            "never_invent_resource_ids": True,
            "resourceless_tasks_only_for": [
                "practice",
                "review",
                "mixed",
            ],
        },
    }
    return (
        "Create today's focused learning plan.\n"
        "Return only structured fields. Stay inside the time budget.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def build_constrained_planner_retry_prompt(
    *,
    base_prompt: str,
    allowed_resource_ids: set[int],
    available_minutes: int,
    attention_span_minutes: int,
    rules: MoodCapacityRules,
    response_model: type[DailyPlanGeneration] = DailyPlanGeneration,
) -> str:
    """
    Strengthen a planner retry with exact schema and hard bounds.

    Used when the first structured generation is empty, invalid, or
    fails budget/ID validation. Never invents URLs or tasks.
    """
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    trusted_ids = sorted(int(item) for item in allowed_resource_ids)
    return (
        f"{base_prompt.strip()}\n\n"
        "CRITICAL RETRY CONSTRAINTS — respond with ONLY valid JSON matching "
        "the schema below. No markdown fences. No commentary or prose outside JSON.\n"
        f"- Keep between {rules.min_tasks} and {rules.max_tasks} tasks.\n"
        f"- Sum of estimated_minutes must be <= {available_minutes}.\n"
        f"- Every task estimated_minutes must be <= {attention_span_minutes}.\n"
        f"- resource_id values must come from this trusted list only: {trusted_ids}\n"
        "- resource_id may be null only for practice/review/mixed tasks.\n"
        "- Never invent URLs. Never include a url field.\n"
        "- Do not invent resource IDs outside the trusted list.\n\n"
        f"JSON schema:\n{schema}"
    )


def validate_plan_generation(
    generation: DailyPlanGeneration,
    *,
    allowed_resource_ids: set[int],
    available_minutes: int,
    rules: MoodCapacityRules,
    attention_span_minutes: int,
) -> DailyPlanGeneration:
    tasks = sorted(generation.tasks, key=lambda item: item.sequence_number)
    if not (rules.min_tasks <= len(tasks) <= rules.max_tasks):
        raise PlannerBudgetError(
            f"Task count {len(tasks)} outside allowed range "
            f"{rules.min_tasks}-{rules.max_tasks}"
        )

    numbers = [task.sequence_number for task in tasks]
    if sorted(numbers) != list(range(1, len(tasks) + 1)):
        raise PlannerGenerationError("Task sequence_number values must be contiguous from 1")

    seen_resources: set[int] = set()
    total = 0
    max_task_minutes = max(
        MIN_TASK_MINUTES,
        min(available_minutes, max(attention_span_minutes, MIN_TASK_MINUTES)),
    )
    for task in tasks:
        total += task.estimated_minutes
        if task.estimated_minutes < MIN_TASK_MINUTES:
            raise PlannerBudgetError("Task duration below minimum")
        if task.estimated_minutes > max_task_minutes:
            raise PlannerBudgetError("Task duration exceeds attention/time cap")
        if _difficulty_rank(task.difficulty) > _difficulty_rank(rules.max_difficulty):
            raise PlannerGenerationError(
                "Task difficulty exceeds mood/energy guardrails"
            )
        if task.resource_id is not None:
            if task.resource_id not in allowed_resource_ids:
                raise PlannerGenerationError("Plan references an unknown resource_id")
            if task.resource_id in seen_resources:
                raise PlannerGenerationError("Duplicate resource_id in plan tasks")
            seen_resources.add(task.resource_id)
        else:
            if task.activity_type not in {
                ActivityType.practice,
                ActivityType.review,
                ActivityType.mixed,
            }:
                raise PlannerGenerationError(
                    "Resource-less tasks are only allowed for practice/review/mixed"
                )
        if not task.milestone_connection.strip():
            raise PlannerGenerationError("Every task must connect to the active milestone")

    if total > available_minutes:
        raise PlannerBudgetError("Total estimated minutes exceed available time")
    if total <= 0:
        raise PlannerBudgetError("Plan has no actionable duration")

    return generation.model_copy(update={"tasks": tasks})


def _to_checkin_response(row: dict[str, Any]) -> DailyCheckInResponse:
    return DailyCheckInResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        mood=Mood(str(row["mood"])),
        energy_level=EnergyLevel(str(row["energy_level"])),
        focus_level=int(row["focus_level"]),
        available_minutes=int(row["available_minutes"]),
        preferred_activity=ActivityType(str(row["preferred_activity"])),
        notes=str(row.get("notes") or ""),
        created_at=_parse_dt(row["created_at"]),
    )


def _to_task_response(row: dict[str, Any]) -> DailyTaskResponse:
    return DailyTaskResponse(
        id=int(row["id"]),
        daily_plan_id=int(row["daily_plan_id"]),
        resource_id=row.get("resource_id"),
        sequence_number=int(row["sequence_number"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        activity_type=ActivityType(str(row["activity_type"])),
        estimated_minutes=int(row["estimated_minutes"]),
        difficulty=Difficulty(str(row.get("difficulty") or "beginner")),
        status=TaskStatus(str(row.get("status") or "pending")),
        completed_at=_parse_dt(row["completed_at"]) if row.get("completed_at") else None,
        why_selected=str(row.get("why_selected") or ""),
        milestone_connection=str(row.get("milestone_connection") or ""),
        expected_outcome=str(row.get("expected_outcome") or ""),
        content_type=str(row.get("content_type") or ""),
        mood_rationale=str(row.get("mood_rationale") or ""),
        resource_title=row.get("resource_title"),
        resource_source=row.get("resource_source"),
        resource_url=row.get("resource_url"),
        resource_thumbnail_url=row.get("resource_thumbnail_url"),
        resource_channel=row.get("resource_channel"),
        metadata=dict(row.get("metadata") or {}),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_plan_response(row: dict[str, Any]) -> DailyPlanResponse:
    tasks = [_to_task_response(task) for task in row.get("tasks") or []]
    return DailyPlanResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        roadmap_id=row.get("roadmap_id"),
        milestone_id=row.get("milestone_id"),
        checkin_id=row.get("checkin_id"),
        plan_date=date.fromisoformat(str(row["plan_date"])),
        summary=str(row.get("summary") or ""),
        total_estimated_minutes=int(row.get("total_estimated_minutes") or 0),
        status=PlanStatus(str(row.get("status") or "pending")),
        tasks=tasks,
        guidance_tone=str(row.get("guidance_tone") or ""),
        mood_influence_summary=str(row.get("mood_influence_summary") or ""),
        adaptation_explanation=str(row.get("adaptation_explanation") or ""),
        task_count_rationale=str(row.get("task_count_rationale") or ""),
        metadata=dict(row.get("metadata") or {}),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


class DailyPlannerAgent:
    """Create a focused, mood-aware daily plan from curated free resources."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        gemini_service: Optional[SupportsStructuredGeneration] = None,
        curator_agent: Optional[SupportsCurator] = None,
        db_path: Optional[Path] = None,
        persist_plan: Optional[PersistPlanFn] = None,
        date_provider: Optional[DateProvider] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._curator = curator_agent
        self._db_path = db_path
        self._persist_plan = persist_plan
        self._date_provider = date_provider or (lambda: datetime.now(timezone.utc).date())

    def _get_curator(self) -> SupportsCurator:
        if self._curator is None:
            from agents.curator_agent import CuratorAgent

            self._curator = CuratorAgent(
                settings=self._settings,
                db_path=self._db_path,
            )
        return self._curator

    def _load_context(
        self,
        user_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if user_id <= 0:
            raise PlannerContextError("user_id must be a positive integer")

        user = get_user_by_id(user_id, db_path=self._db_path)
        if user is None:
            raise PlannerContextError("User not found")

        profile = get_user_profile_by_user_id(user_id, db_path=self._db_path)
        if profile is None:
            raise PlannerContextError("User profile not found")

        goal = get_active_goal_for_user(user_id, db_path=self._db_path)
        if goal is None:
            raise PlannerContextError("Active goal not found")
        if int(goal["user_id"]) != user_id:
            raise PlannerOwnershipError("Goal does not belong to the requested user")

        roadmap = get_active_roadmap_for_user(user_id, db_path=self._db_path)
        if roadmap is None:
            raise PlannerContextError("Active roadmap not found")
        if int(roadmap["user_id"]) != user_id:
            raise PlannerOwnershipError("Roadmap does not belong to the requested user")

        milestone = roadmap.get("active_milestone")
        if milestone is None:
            raise PlannerContextError("Active milestone not found")
        if int(milestone.get("user_id") or roadmap["user_id"]) != user_id:
            raise PlannerOwnershipError("Milestone does not belong to the requested user")
        if int(milestone.get("roadmap_id") or roadmap["id"]) != int(roadmap["id"]):
            raise PlannerOwnershipError(
                "Milestone does not belong to the active roadmap"
            )

        return user, profile, goal, roadmap, milestone

    def create_daily_plan(
        self,
        user_id: int,
        *,
        checkin: DailyCheckInRequest,
        plan_date: date | None = None,
        refresh: bool = False,
    ) -> PlannerAgentResult:
        """
        Create today's learning plan.

        refresh=False: return the existing same-day plan when present (no new check-in).
        refresh=True: replace the same-day plan and create a new check-in.
        """
        _user, profile, goal, roadmap, milestone = self._load_context(user_id)
        goal_title = str(goal["title"])
        resolved_date = plan_date or self._date_provider()
        plan_date_text = resolved_date.isoformat()

        existing = get_daily_plan_by_date(
            user_id,
            plan_date_text,
            db_path=self._db_path,
        )
        if existing is not None and not refresh:
            checkin_row = None
            if existing.get("checkin_id"):
                checkin_row = get_daily_checkin_by_id(
                    int(existing["checkin_id"]),
                    db_path=self._db_path,
                )
            if checkin_row is None:
                raise PlannerContextError("Existing plan is missing its check-in")
            logger.info(
                "DailyPlannerAgent returning existing plan_id=%s for user_id=%s date=%s",
                existing["id"],
                user_id,
                plan_date_text,
            )
            return PlannerAgentResult(
                checkin=_to_checkin_response(checkin_row),
                plan=_to_plan_response(existing),
                goal_title=goal_title,
                milestone_title=str(milestone.get("title") or ""),
                reused_existing=True,
                created_at=_parse_dt(existing["created_at"]),
            )

        insights = list_active_adaptation_insights(user_id, db_path=self._db_path)
        preferences = list_user_preferences(user_id, db_path=self._db_path)
        rules = mood_capacity_rules(
            checkin.mood,
            checkin.energy_level,
            available_minutes=checkin.available_minutes,
            focus_level=checkin.focus_level,
        )
        adaptation_explanation = build_adaptation_explanation(
            insights,
            preferences,
            mood=checkin.mood,
            available_minutes=checkin.available_minutes,
        )

        checkin_row = create_daily_checkin(
            user_id,
            mood=checkin.mood.value,
            energy_level=checkin.energy_level.value,
            focus_level=checkin.focus_level,
            available_minutes=checkin.available_minutes,
            preferred_activity=checkin.preferred_activity.value,
            notes=checkin.notes,
            db_path=self._db_path,
        )

        curator_limit = max(MIN_CURATOR_LIMIT, min(MAX_CURATOR_LIMIT, rules.max_tasks + 1))
        preferred_format = activity_to_preferred_format(checkin.preferred_activity)
        try:
            curator_result = self._get_curator().recommend_resources(
                user_id,
                int(roadmap["id"]),
                int(milestone["id"]),
                mood=checkin.mood,
                energy_level=checkin.energy_level,
                available_minutes=checkin.available_minutes,
                preferred_format=preferred_format,
                limit=curator_limit,
                refresh=refresh,
            )
        except CuratorAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PlannerAgentError(
                f"Curator integration failed: {type(exc).__name__}"
            ) from None

        if not curator_result.recommendations:
            raise PlannerContextError("Curator returned no trusted resources")

        allowed_ids = {rec.resource_id for rec in curator_result.recommendations}
        by_resource = {rec.resource_id: rec for rec in curator_result.recommendations}
        prompt = build_planner_prompt(
            goal_title=goal_title,
            milestone=milestone,
            checkin=checkin,
            profile=profile,
            preferences=preferences,
            insights=insights,
            recommendations=curator_result.recommendations,
            rules=rules,
            adaptation_explanation=adaptation_explanation,
        )
        logger.info(
            "DailyPlannerAgent generating plan user_id=%s date=%s prompt_chars=%s",
            user_id,
            plan_date_text,
            len(prompt),
        )

        attention_span = int(profile.get("attention_span_minutes") or 15)

        def _generate_and_validate(active_prompt: str) -> DailyPlanGeneration:
            raw = self._gemini.generate_structured(
                active_prompt,
                DailyPlanGeneration,
                system_instruction=PLANNER_SYSTEM_INSTRUCTION,
            )
            return validate_plan_generation(
                raw,
                allowed_resource_ids=allowed_ids,
                available_minutes=checkin.available_minutes,
                rules=rules,
                attention_span_minutes=attention_span,
            )

        try:
            generation = _generate_and_validate(prompt)
        except (GeminiConfigurationError, GeminiInvocationError):
            # Auth / config / non-retryable provider failures — do not retry.
            raise
        except (GeminiResponseError, PlannerBudgetError, PlannerGenerationError):
            # One constrained retry for empty/invalid structured output or
            # retryable validation failures. Reuses the same trusted resource IDs.
            retry_prompt = build_constrained_planner_retry_prompt(
                base_prompt=prompt,
                allowed_resource_ids=allowed_ids,
                available_minutes=checkin.available_minutes,
                attention_span_minutes=attention_span,
                rules=rules,
            )
            logger.warning(
                "DailyPlannerAgent retrying structured plan generation user_id=%s",
                user_id,
            )
            try:
                generation = _generate_and_validate(retry_prompt)
            except (GeminiConfigurationError, GeminiInvocationError):
                raise
            except (GeminiResponseError, PlannerBudgetError, PlannerGenerationError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise PlannerGenerationError(
                    f"Plan generation failed: {type(exc).__name__}"
                ) from None
        except Exception as exc:  # noqa: BLE001
            raise PlannerGenerationError(
                f"Plan generation failed: {type(exc).__name__}"
            ) from None

        # Prefer seed explanation when Gemini invents unsupported history claims.
        if not insights and not preferences:
            final_adaptation = adaptation_explanation
        else:
            final_adaptation = generation.adaptation_explanation.strip() or adaptation_explanation

        tasks_payload: list[dict[str, Any]] = []
        total_minutes = 0
        for task in generation.tasks:
            total_minutes += task.estimated_minutes
            content_type = task.content_type
            score_breakdown: dict[str, Any] = {}
            if task.resource_id is not None:
                rec = by_resource[task.resource_id]
                content_type = content_type or rec.resource_type
                score_breakdown = dict(rec.score_breakdown or {})
            tasks_payload.append(
                {
                    "sequence_number": task.sequence_number,
                    "title": task.title,
                    "description": task.description,
                    "activity_type": task.activity_type.value,
                    "estimated_minutes": task.estimated_minutes,
                    "difficulty": task.difficulty.value,
                    "resource_id": task.resource_id,
                    "status": "pending",
                    "metadata": {
                        "why_selected": task.why_selected,
                        "milestone_connection": task.milestone_connection,
                        "expected_outcome": task.expected_outcome,
                        "content_type": content_type,
                        "mood_rationale": task.mood_rationale,
                        "score_breakdown": score_breakdown,
                    },
                }
            )

        plan_metadata = {
            "guidance_tone": generation.guidance_tone or rules.guidance_tone,
            "mood_influence_summary": generation.mood_influence_summary or rules.summary,
            "adaptation_explanation": final_adaptation,
            "task_count_rationale": generation.task_count_rationale,
            "goal_title_snapshot": goal_title,
            "mood": checkin.mood.value,
            "energy_level": checkin.energy_level.value,
            "available_minutes": checkin.available_minutes,
        }

        try:
            if self._persist_plan is not None:
                plan_row = self._persist_plan(
                    user_id=user_id,
                    plan_date=plan_date_text,
                    summary=generation.summary,
                    total_estimated_minutes=total_minutes,
                    tasks=tasks_payload,
                    roadmap_id=int(roadmap["id"]),
                    milestone_id=int(milestone["id"]),
                    checkin_id=int(checkin_row["id"]),
                    metadata=plan_metadata,
                    replace_existing=refresh,
                )
            else:
                plan_row = create_daily_plan_bundle(
                    user_id=user_id,
                    plan_date=plan_date_text,
                    summary=generation.summary,
                    total_estimated_minutes=total_minutes,
                    tasks=tasks_payload,
                    roadmap_id=int(roadmap["id"]),
                    milestone_id=int(milestone["id"]),
                    checkin_id=int(checkin_row["id"]),
                    metadata=plan_metadata,
                    replace_existing=refresh,
                    db_path=self._db_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DailyPlannerAgent persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise PlannerPersistenceError(
                f"Failed to persist daily plan: {type(exc).__name__}"
            ) from None

        # Guard: goal title in metadata must match loaded goal.
        if str(plan_row.get("metadata", {}).get("goal_title_snapshot") or "") != goal_title:
            raise PlannerGenerationError("Plan metadata altered the goal title")

        return PlannerAgentResult(
            checkin=_to_checkin_response(checkin_row),
            plan=_to_plan_response(plan_row),
            goal_title=goal_title,
            milestone_title=str(milestone.get("title") or ""),
            reused_existing=False,
            created_at=_parse_dt(plan_row["created_at"]),
        )
