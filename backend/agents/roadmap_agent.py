"""
Roadmap Agent for GrowthOS AI.

Deterministic orchestration:
1. Load user / profile / goal (ownership checks)
2. Optional same-user semantic memory retrieval
3. Gemini structured roadmap generation
4. Validate pacing/structure
5. Persist roadmap + phases + milestones transactionally
6. Optional roadmap semantic memories

Assumption: LEARNING_DAYS_PER_WEEK = 5 for weekly capacity estimates.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from config import Settings, get_settings
from exceptions import (
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    RoadmapAgentError,
    RoadmapContextError,
    RoadmapMemoryError,
    RoadmapOwnershipError,
    RoadmapPersistenceError,
)
from models import (
    Difficulty,
    GoalResponse,
    MilestoneGeneration,
    MilestoneResponse,
    MilestoneStatus,
    PhaseStatus,
    RoadmapAgentResult,
    RoadmapGeneration,
    RoadmapPhaseResponse,
    RoadmapResponse,
    RoadmapStatus,
    UserProfileResponse,
    UserResponse,
)
from services.database import (
    create_roadmap_bundle,
    get_active_roadmap_for_goal,
    get_goal_by_id,
    get_roadmap_with_details,
    get_user_by_id,
    get_user_profile_by_user_id,
)
from services.ai_provider import get_ai_provider
from services.memory import SemanticMemoryService
from services.vector_models import MemoryRecordType, VectorMemoryRecord

logger = logging.getLogger(__name__)

# MVP scheduling assumption used for capacity calculations.
LEARNING_DAYS_PER_WEEK = 5
MIN_WEEKS = 1
MAX_WEEKS = 52
MIN_PHASES = 2
MAX_PHASES = 6

ROADMAP_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Roadmap Agent.

Create a personalized learning roadmap for any free-text goal.

Rules:
- Preserve the user's exact main learning goal.
- Adapt to current level, target outcome, time, attention, and preferences.
- Use realistic, gradually increasing milestones.
- Every milestone must have concrete skills, activities, and completion criteria.
- Avoid vague phases such as "learn more".
- Do not invent personal facts, demographics, or medical/psychological claims.
- Do not include external resource URLs (the Curator Agent handles resources).
- Do not promise guaranteed mastery.
- Mood affects daily execution later, not this stable roadmap structure.
- Use 2-6 phases. Within each phase use 1-5 milestones.
- Phase sequence_number values must be contiguous starting at 1.
- Milestone sequence_number values must restart at 1 within each phase and stay <= 5.
- Return only the requested structured output.
""".strip()

PersistRoadmapFn = Callable[..., dict[str, Any]]


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[RoadmapGeneration],
        *,
        system_instruction: str | None = None,
    ) -> RoadmapGeneration: ...


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def compute_capacity(
    *,
    daily_available_minutes: int,
    preferred_session_minutes: int,
    attention_span_minutes: int,
    learning_days_per_week: int = LEARNING_DAYS_PER_WEEK,
) -> dict[str, Any]:
    """Derive weekly learning capacity from profile constraints."""
    session_minutes = min(
        preferred_session_minutes,
        attention_span_minutes,
        daily_available_minutes,
    )
    sessions_per_day = max(1, daily_available_minutes // max(session_minutes, 1))
    # Cap to one focused block per day for MVP realism when attention is short.
    if attention_span_minutes < preferred_session_minutes:
        sessions_per_day = 1
        session_minutes = min(attention_span_minutes, daily_available_minutes)
    weekly_minutes = session_minutes * sessions_per_day * learning_days_per_week
    weekly_sessions = sessions_per_day * learning_days_per_week
    return {
        "learning_days_per_week": learning_days_per_week,
        "session_minutes": session_minutes,
        "sessions_per_day": sessions_per_day,
        "weekly_minutes": weekly_minutes,
        "weekly_sessions": weekly_sessions,
    }


def validate_roadmap_generation(
    generation: RoadmapGeneration,
    *,
    weekly_minutes: int,
) -> RoadmapGeneration:
    """Apply structural and pacing guardrails to Gemini output."""
    phase_numbers = [phase.sequence_number for phase in generation.phases]
    if sorted(phase_numbers) != list(range(1, len(generation.phases) + 1)):
        raise RoadmapAgentError("Phase sequence_number values must be contiguous from 1")

    total_milestones = 0
    total_minutes = 0
    for phase in generation.phases:
        mile_numbers = [m.sequence_number for m in phase.milestones]
        if sorted(mile_numbers) != list(range(1, len(phase.milestones) + 1)):
            raise RoadmapAgentError(
                f"Milestone sequence_number values in phase {phase.sequence_number} "
                "must be contiguous from 1"
            )
        for milestone in phase.milestones:
            total_milestones += 1
            total_minutes += milestone.estimated_minutes * milestone.estimated_sessions

    if total_milestones < MIN_PHASES:
        raise RoadmapAgentError("Roadmap must include enough milestones for a practical plan")

    # Duration should roughly fit weekly capacity (soft upper bound).
    max_reasonable_weeks = max(
        MIN_WEEKS,
        min(MAX_WEEKS, (total_minutes // max(weekly_minutes, 1)) + 2),
    )
    if generation.estimated_duration_weeks > max(MAX_WEEKS, max_reasonable_weeks):
        raise RoadmapAgentError("Estimated duration exceeds validated capacity bounds")

    return generation


def build_roadmap_prompt(
    *,
    user: dict[str, Any],
    profile: dict[str, Any],
    goal: dict[str, Any],
    capacity: dict[str, Any],
    memory_snippets: Sequence[str],
) -> str:
    """Build labeled JSON context for roadmap generation."""
    payload = {
        "user_stated": {
            "display_name": user["display_name"],
            "learning_goal": goal["title"],
            "goal_description": goal.get("description", ""),
            "aspiration": profile["aspiration"],
            "motivation": profile["motivation"],
            "current_level": profile["current_level"],
            "target_outcome": profile["target_outcome"],
            "preferred_formats": profile["preferred_formats"],
            "learning_style": profile["learning_style"],
            "daily_available_minutes": profile["daily_available_minutes"],
            "preferred_session_minutes": profile["preferred_session_minutes"],
            "attention_span_minutes": profile["attention_span_minutes"],
            "preferred_learning_time": profile["preferred_learning_time"],
            "habits": profile["habits"],
            "distractions": profile["distractions"],
        },
        "derived_scheduling": {
            **capacity,
            "notes": (
                f"Assume {capacity['learning_days_per_week']} learning days/week. "
                "Fit milestones to weekly capacity; do not overwhelm."
            ),
        },
        "supplementary_memory_context": list(memory_snippets),
        "instructions": {
            "preserve_goal_exactly": goal["title"],
            "phase_count_range": [MIN_PHASES, MAX_PHASES],
            "duration_weeks_range": [MIN_WEEKS, MAX_WEEKS],
            "no_resource_urls": True,
        },
    }
    return (
        "Generate a personalized learning roadmap from this GrowthOS context.\n"
        "Treat user_stated fields as source of truth.\n"
        "Treat supplementary_memory_context as optional interpretation only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _to_user(row: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=int(row["id"]),
        display_name=str(row["display_name"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_profile(row: dict[str, Any]) -> UserProfileResponse:
    return UserProfileResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        aspiration=str(row["aspiration"]),
        motivation=str(row["motivation"]),
        current_level=row["current_level"],
        target_outcome=str(row["target_outcome"]),
        learning_style=row["learning_style"],
        preferred_formats=list(row["preferred_formats"]),
        daily_available_minutes=int(row["daily_available_minutes"]),
        preferred_session_minutes=int(row["preferred_session_minutes"]),
        attention_span_minutes=int(row["attention_span_minutes"]),
        preferred_learning_time=row["preferred_learning_time"],
        habits=list(row["habits"]),
        distractions=list(row["distractions"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_goal(row: dict[str, Any]) -> GoalResponse:
    return GoalResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        status=row["status"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_milestone(row: dict[str, Any]) -> MilestoneResponse:
    return MilestoneResponse(
        id=int(row["id"]),
        phase_id=int(row["phase_id"]),
        sequence_number=int(row["sequence_number"]),
        title=str(row["title"]),
        description=str(row["description"]),
        skills=list(row.get("skills") or []),
        suggested_activities=list(row.get("suggested_activities") or []),
        completion_criteria=str(row.get("completion_criteria") or ""),
        estimated_sessions=int(row.get("estimated_sessions") or 1),
        estimated_minutes=int(row.get("estimated_minutes") or 30),
        difficulty=row.get("difficulty") or Difficulty.beginner,
        status=row["status"],
        progress_percent=float(row.get("progress_percent") or 0),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_phase(row: dict[str, Any]) -> RoadmapPhaseResponse:
    milestones = [_to_milestone(item) for item in row.get("milestones") or []]
    return RoadmapPhaseResponse(
        id=int(row["id"]),
        roadmap_id=int(row["roadmap_id"]),
        sequence_number=int(row["sequence_number"]),
        title=str(row["title"]),
        description=str(row["description"]),
        expected_outcome=str(row["expected_outcome"]),
        status=row["status"],
        milestones=milestones,
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_roadmap(details: dict[str, Any]) -> RoadmapResponse:
    phases = [_to_phase(item) for item in details.get("phases") or []]
    active = details.get("active_milestone")
    active_id = int(active["id"]) if active else None
    return RoadmapResponse(
        id=int(details["id"]),
        user_id=int(details["user_id"]),
        goal_id=int(details["goal_id"]),
        title=str(details["title"]),
        summary=str(details["summary"]),
        estimated_duration_weeks=int(details["estimated_duration_weeks"]),
        progress_percent=float(details.get("progress_percent") or 0),
        status=details.get("status") or RoadmapStatus.active,
        pacing_rationale=str(details.get("pacing_rationale") or ""),
        personalization_rationale=str(details.get("personalization_rationale") or ""),
        phases=phases,
        current_active_milestone_id=active_id,
        created_at=_parse_dt(details["created_at"]),
        updated_at=_parse_dt(details["updated_at"]),
    )


def _assign_initial_statuses(
    generation: RoadmapGeneration,
) -> list[dict[str, Any]]:
    """Phase 1 + first milestone in_progress; everything else not_started."""
    phases_out: list[dict[str, Any]] = []
    for phase in sorted(generation.phases, key=lambda item: item.sequence_number):
        milestones_out: list[dict[str, Any]] = []
        for milestone in sorted(phase.milestones, key=lambda item: item.sequence_number):
            is_first = phase.sequence_number == 1 and milestone.sequence_number == 1
            milestones_out.append(
                {
                    "sequence_number": milestone.sequence_number,
                    "title": milestone.title,
                    "description": milestone.description,
                    "skills": list(milestone.skills),
                    "suggested_activities": list(milestone.suggested_activities),
                    "completion_criteria": milestone.completion_criteria,
                    "estimated_sessions": milestone.estimated_sessions,
                    "estimated_minutes": milestone.estimated_minutes,
                    "difficulty": _enum_value(milestone.difficulty),
                    "status": (
                        MilestoneStatus.in_progress.value
                        if is_first
                        else MilestoneStatus.not_started.value
                    ),
                }
            )
        phases_out.append(
            {
                "sequence_number": phase.sequence_number,
                "title": phase.title,
                "description": phase.description,
                "expected_outcome": phase.expected_outcome,
                "status": (
                    PhaseStatus.in_progress.value
                    if phase.sequence_number == 1
                    else PhaseStatus.not_started.value
                ),
                "milestones": milestones_out,
            }
        )
    return phases_out


class RoadmapAgent:
    """Generate and persist personalized learning roadmaps."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        gemini_service: Optional[SupportsStructuredGeneration] = None,
        memory_service: Optional[SemanticMemoryService] = None,
        db_path: Optional[Path] = None,
        persist_roadmap: Optional[PersistRoadmapFn] = None,
        skip_memory_retrieval: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._memory = memory_service
        self._db_path = db_path
        self._persist_roadmap = persist_roadmap
        self._skip_memory_retrieval = skip_memory_retrieval

    def _get_memory_service(self) -> Optional[SemanticMemoryService]:
        if self._skip_memory_retrieval and self._memory is None:
            return None
        if self._memory is None:
            self._memory = SemanticMemoryService(settings=self._settings)
        return self._memory

    def _load_context(
        self,
        user_id: int,
        goal_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if user_id <= 0 or goal_id <= 0:
            raise RoadmapContextError("user_id and goal_id must be positive integers")

        user = get_user_by_id(user_id, db_path=self._db_path)
        if user is None:
            raise RoadmapContextError("User not found")

        goal = get_goal_by_id(goal_id, db_path=self._db_path)
        if goal is None:
            raise RoadmapContextError("Goal not found")
        if int(goal["user_id"]) != user_id:
            raise RoadmapOwnershipError("Goal does not belong to the requested user")

        profile = get_user_profile_by_user_id(user_id, db_path=self._db_path)
        if profile is None:
            raise RoadmapContextError("User profile not found")

        required = [
            "aspiration",
            "motivation",
            "current_level",
            "target_outcome",
            "learning_style",
            "preferred_formats",
            "daily_available_minutes",
            "preferred_session_minutes",
            "attention_span_minutes",
            "preferred_learning_time",
        ]
        missing = [key for key in required if profile.get(key) in (None, "", [])]
        if missing:
            raise RoadmapContextError("User profile is missing required fields")

        return user, profile, goal

    def _retrieve_memory_snippets(self, user_id: int, goal_title: str) -> list[str]:
        memory_service = self._get_memory_service()
        if memory_service is None:
            return []
        try:
            hits = memory_service.semantic_search(
                query_text=goal_title,
                user_id=user_id,
                limit=5,
                record_types=[
                    MemoryRecordType.aspiration.value,
                    MemoryRecordType.goal.value,
                    MemoryRecordType.profile.value,
                    MemoryRecordType.preference.value,
                ],
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "RoadmapAgent memory retrieval failed; continuing with SQLite context"
            )
            return []

        snippets: list[str] = []
        for hit in hits:
            if hit.user_id != user_id:
                continue
            text = hit.text.strip()
            if text:
                snippets.append(text[:280])
        return snippets

    def _result_from_details(
        self,
        details: dict[str, Any],
        *,
        reused_existing: bool,
        memory_ids: list[str] | None = None,
        memories_complete: bool = True,
        memory_error: str | None = None,
    ) -> RoadmapAgentResult:
        roadmap = _to_roadmap(details)
        phases = list(roadmap.phases)
        milestones: list[MilestoneResponse] = []
        for phase in phases:
            milestones.extend(phase.milestones)
        active = None
        if details.get("active_milestone"):
            active = _to_milestone(details["active_milestone"])
        elif milestones:
            active = next(
                (item for item in milestones if item.status == MilestoneStatus.in_progress),
                milestones[0],
            )
        return RoadmapAgentResult(
            roadmap=roadmap,
            phases=phases,
            milestones=milestones,
            active_milestone=active,
            pacing_rationale=roadmap.pacing_rationale,
            personalization_rationale=roadmap.personalization_rationale,
            memory_ids=memory_ids or [],
            memories_complete=memories_complete,
            memory_error=memory_error,
            reused_existing=reused_existing,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )

    def _store_roadmap_memories(
        self,
        *,
        user_id: int,
        goal_id: int,
        details: dict[str, Any],
    ) -> list[str]:
        memory_service = self._get_memory_service()
        if memory_service is None:
            return []

        suffix = uuid.uuid4().hex[:8]
        roadmap_id = int(details["id"])
        active = details.get("active_milestone") or {}
        phase_titles = [
            f"{phase['sequence_number']}. {phase['title']}"
            for phase in details.get("phases") or []
        ]
        records = [
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-roadmap-{roadmap_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.goal.value,
                source_record_id=str(roadmap_id),
                text=(
                    f"Roadmap: {details['title']}. "
                    f"Summary: {details['summary']}. "
                    f"Duration weeks: {details['estimated_duration_weeks']}."
                ),
                metadata={
                    "source": "roadmap_agent",
                    "goal_id": goal_id,
                    "roadmap_id": roadmap_id,
                    "version": 1,
                },
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-roadmap-phases-{roadmap_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.profile.value,
                source_record_id=str(roadmap_id),
                text="Phase overview: " + " | ".join(phase_titles),
                metadata={
                    "source": "roadmap_agent",
                    "goal_id": goal_id,
                    "roadmap_id": roadmap_id,
                    "version": 1,
                },
            ),
        ]
        if active:
            records.append(
                VectorMemoryRecord(
                    memory_id=f"user-{user_id}-active-milestone-{active['id']}-{suffix}",
                    user_id=user_id,
                    record_type=MemoryRecordType.goal.value,
                    source_record_id=str(active["id"]),
                    text=(
                        f"Active milestone: {active['title']}. "
                        f"{active.get('description', '')}"
                    ),
                    metadata={
                        "source": "roadmap_agent",
                        "goal_id": goal_id,
                        "roadmap_id": roadmap_id,
                        "milestone_id": int(active["id"]),
                        "version": 1,
                    },
                )
            )
        memory_service.add_text_memories(records)
        return [record.memory_id for record in records]

    def generate_roadmap(
        self,
        user_id: int,
        goal_id: int,
        *,
        regenerate: bool = False,
    ) -> RoadmapAgentResult:
        """
        Generate a personalized roadmap for a user's goal.

        If an active roadmap already exists and regenerate=False, return it.
        If regenerate=True, archive the previous active roadmap and create a new one.
        """
        user, profile, goal = self._load_context(user_id, goal_id)

        existing = get_active_roadmap_for_goal(goal_id, db_path=self._db_path)
        if existing is not None and not regenerate:
            details = get_roadmap_with_details(
                int(existing["id"]),
                db_path=self._db_path,
            )
            if details is None:
                raise RoadmapContextError("Existing roadmap could not be loaded")
            logger.info(
                "RoadmapAgent returning existing roadmap_id=%s for goal_id=%s",
                existing["id"],
                goal_id,
            )
            return self._result_from_details(details, reused_existing=True)

        capacity = compute_capacity(
            daily_available_minutes=int(profile["daily_available_minutes"]),
            preferred_session_minutes=int(profile["preferred_session_minutes"]),
            attention_span_minutes=int(profile["attention_span_minutes"]),
        )
        memory_snippets = self._retrieve_memory_snippets(user_id, str(goal["title"]))
        prompt = build_roadmap_prompt(
            user=user,
            profile=profile,
            goal=goal,
            capacity=capacity,
            memory_snippets=memory_snippets,
        )
        logger.info(
            "RoadmapAgent generating roadmap user_id=%s goal_id=%s prompt_chars=%s",
            user_id,
            goal_id,
            len(prompt),
        )

        try:
            generation = self._gemini.generate_structured(
                prompt,
                RoadmapGeneration,
                system_instruction=ROADMAP_SYSTEM_INSTRUCTION,
            )
            generation = validate_roadmap_generation(
                generation,
                weekly_minutes=int(capacity["weekly_minutes"]),
            )
        except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
            raise
        except RoadmapAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RoadmapAgentError(
                f"Roadmap generation failed: {type(exc).__name__}"
            ) from None

        phases_payload = _assign_initial_statuses(generation)

        try:
            if self._persist_roadmap is not None:
                details = self._persist_roadmap(
                    user_id=user_id,
                    goal_id=goal_id,
                    generation=generation,
                    phases=phases_payload,
                    regenerate=regenerate,
                )
            else:
                details = create_roadmap_bundle(
                    user_id=user_id,
                    goal_id=goal_id,
                    title=generation.title,
                    summary=generation.summary,
                    estimated_duration_weeks=generation.estimated_duration_weeks,
                    pacing_rationale=generation.pacing_rationale,
                    personalization_rationale=generation.personalization_rationale,
                    phases=phases_payload,
                    archive_existing_active=regenerate,
                    db_path=self._db_path,
                )
        except RoadmapPersistenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RoadmapAgent persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise RoadmapPersistenceError(
                f"Failed to persist roadmap: {type(exc).__name__}"
            ) from None

        try:
            memory_ids = self._store_roadmap_memories(
                user_id=user_id,
                goal_id=goal_id,
                details=details,
            )
        except Exception as exc:  # noqa: BLE001
            message = (
                "Roadmap was saved, but semantic memory persistence failed: "
                f"{type(exc).__name__}"
            )
            logger.error("RoadmapAgent memory persistence failed error_type=%s", type(exc).__name__)
            partial = self._result_from_details(
                details,
                reused_existing=False,
                memory_ids=[],
                memories_complete=False,
                memory_error=message,
            )
            raise RoadmapMemoryError(message, result=partial) from None

        return self._result_from_details(
            details,
            reused_existing=False,
            memory_ids=memory_ids,
            memories_complete=True,
        )
