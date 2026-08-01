"""
Profile Agent for GrowthOS AI.

Deterministic orchestration:
1. Validate onboarding input (Pydantic)
2. Gemini structured profile interpretation
3. SQLite transaction (user + profile + goal)
4. Semantic memory writes (FAISS)

Every onboarding request creates a new user (no auth / no name dedupe).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from config import Settings, get_settings
from exceptions import (
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    ProfileAgentError,
    ProfileMemoryError,
    ProfilePersistenceError,
)
from models import (
    GoalResponse,
    OnboardingRequest,
    ProfileAgentResult,
    ProfileInterpretation,
    UserProfileResponse,
    UserResponse,
)
from services.ai_provider import get_ai_provider
from services.database import create_onboarding_records
from services.memory import SemanticMemoryService
from services.vector_models import MemoryRecordType, VectorMemoryRecord

logger = logging.getLogger(__name__)

PROFILE_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Profile Agent.

Your job is to interpret onboarding answers into a concise, practical profile
summary that later agents can use for roadmaps and daily plans.

Rules:
- Infer cautiously.
- Distinguish stated facts from reasonable interpretation.
- Do not diagnose mental or medical conditions.
- Do not fabricate achievements, habits, demographics, or personal history.
- Preserve the user's original learning goal exactly in spirit; never replace it.
- Do not invent resource URLs.
- Keep lists short and actionable.
- Return only the requested structured output.
""".strip()

PersistOnboardingFn = Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[ProfileInterpretation],
        *,
        system_instruction: str | None = None,
    ) -> ProfileInterpretation: ...


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def build_onboarding_prompt(request: OnboardingRequest) -> str:
    """Build a labeled JSON prompt from onboarding fields only."""
    payload = {
        "display_name": request.display_name,
        "learning_goal": request.learning_goal,
        "aspiration": request.aspiration,
        "motivation": request.motivation,
        "current_level": _enum_value(request.current_level),
        "target_outcome": request.target_outcome,
        "preferred_formats": list(request.preferred_formats),
        "learning_style": _enum_value(request.learning_style),
        "daily_available_minutes": request.daily_available_minutes,
        "preferred_session_minutes": request.preferred_session_minutes,
        "attention_span_minutes": request.attention_span_minutes,
        "preferred_learning_time": _enum_value(request.preferred_learning_time),
        "habits": list(request.habits),
        "distractions": list(request.distractions),
    }
    return (
        "Interpret the following GrowthOS onboarding answers.\n"
        "The learning_goal is free text and must be preserved.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _to_user_response(row: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=int(row["id"]),
        display_name=str(row["display_name"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _to_profile_response(row: dict[str, Any]) -> UserProfileResponse:
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


def _to_goal_response(row: dict[str, Any]) -> GoalResponse:
    return GoalResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        status=row["status"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


class ProfileAgent:
    """
    Transform onboarding input into structured profile artifacts + memories.

    Dependency injection:
    - gemini_service: AIProvider (or compatible fake); defaults via get_ai_provider()
    - memory_service: SemanticMemoryService or compatible fake
    - db_path: optional SQLite path override
    - persist_onboarding: optional persistence function for tests
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        gemini_service: Optional[SupportsStructuredGeneration] = None,
        memory_service: Optional[SemanticMemoryService] = None,
        db_path: Optional[Path] = None,
        persist_onboarding: Optional[PersistOnboardingFn] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._memory = memory_service
        self._db_path = db_path
        self._persist_onboarding = persist_onboarding or self._default_persist

    def _default_persist(
        self,
        request: OnboardingRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return create_onboarding_records(
            display_name=request.display_name,
            aspiration=request.aspiration,
            motivation=request.motivation,
            current_level=_enum_value(request.current_level),
            target_outcome=request.target_outcome,
            learning_style=_enum_value(request.learning_style),
            preferred_formats=request.preferred_formats,
            daily_available_minutes=request.daily_available_minutes,
            preferred_session_minutes=request.preferred_session_minutes,
            attention_span_minutes=request.attention_span_minutes,
            preferred_learning_time=_enum_value(request.preferred_learning_time),
            habits=request.habits,
            distractions=request.distractions,
            goal_title=request.learning_goal,
            goal_description=(
                f"Aspiration: {request.aspiration}. "
                f"Target outcome: {request.target_outcome}."
            ),
            db_path=self._db_path,
        )

    def _get_memory_service(self) -> SemanticMemoryService:
        if self._memory is None:
            self._memory = SemanticMemoryService(settings=self._settings)
        return self._memory

    def _build_memory_records(
        self,
        *,
        request: OnboardingRequest,
        user_id: int,
        profile_id: int,
        goal_id: int,
        interpretation: ProfileInterpretation,
    ) -> list[VectorMemoryRecord]:
        suffix = uuid.uuid4().hex[:8]
        base_meta = {
            "source": "profile_agent",
            "version": 1,
            "profile_id": profile_id,
            "goal_id": goal_id,
        }
        formats = ", ".join(request.preferred_formats)
        habits = ", ".join(request.habits) if request.habits else "none stated"
        distractions = (
            ", ".join(request.distractions) if request.distractions else "none stated"
        )
        insights = "; ".join(interpretation.initial_personalization_insights[:5])

        return [
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-aspiration-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.aspiration.value,
                source_record_id=str(profile_id),
                text=(
                    f"Aspiration: {request.aspiration}. "
                    f"Summary: {interpretation.aspiration_summary}"
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-motivation-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.profile.value,
                source_record_id=str(profile_id),
                text=(
                    f"Motivation: {request.motivation}. "
                    f"Summary: {interpretation.motivation_summary}"
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-goal-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.goal.value,
                source_record_id=str(goal_id),
                text=(
                    f"Learning goal: {request.learning_goal}. "
                    f"Target outcome: {request.target_outcome}. "
                    f"Current level: {_enum_value(request.current_level)}."
                ),
                metadata={**base_meta, "stated_or_inferred": "stated"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-preferences-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.preference.value,
                source_record_id=str(profile_id),
                text=(
                    f"Preferred formats: {formats}. "
                    f"Learning style: {_enum_value(request.learning_style)}. "
                    f"Preferred learning time: {_enum_value(request.preferred_learning_time)}. "
                    f"Summary: {interpretation.learning_preferences_summary}"
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-constraints-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.profile.value,
                source_record_id=str(profile_id),
                text=(
                    f"Daily available minutes: {request.daily_available_minutes}. "
                    f"Preferred session minutes: {request.preferred_session_minutes}. "
                    f"Attention span minutes: {request.attention_span_minutes}. "
                    f"Habits: {habits}. Distractions: {distractions}. "
                    f"Attention strategy: {interpretation.attention_strategy}. "
                    f"Consistency strategy: {interpretation.consistency_strategy}."
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-profile-summary-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.profile.value,
                source_record_id=str(profile_id),
                text=(
                    f"Identity: {interpretation.identity_summary}. "
                    f"Current state: {interpretation.current_state_summary}. "
                    f"Target state: {interpretation.target_state_summary}. "
                    f"Pacing: {interpretation.recommended_pacing}. "
                    f"Insights: {insights or 'none'}"
                ),
                metadata={**base_meta, "stated_or_inferred": "interpreted"},
            ),
        ]

    def process_onboarding(self, request: OnboardingRequest) -> ProfileAgentResult:
        """
        Process onboarding end-to-end.

        Order: Gemini interpretation → SQLite transaction → semantic memories.
        """
        if not isinstance(request, OnboardingRequest):
            request = OnboardingRequest.model_validate(request)

        prompt = build_onboarding_prompt(request)
        logger.info(
            "ProfileAgent starting onboarding prompt_chars=%s formats=%s",
            len(prompt),
            len(request.preferred_formats),
        )

        try:
            interpretation = self._gemini.generate_structured(
                prompt,
                ProfileInterpretation,
                system_instruction=PROFILE_SYSTEM_INSTRUCTION,
            )
        except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProfileAgentError(
                f"Profile interpretation failed: {type(exc).__name__}"
            ) from None

        try:
            user_row, profile_row, goal_row = self._persist_onboarding(request)
        except ProfilePersistenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ProfileAgent SQLite persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise ProfilePersistenceError(
                f"Failed to persist onboarding records: {type(exc).__name__}"
            ) from None

        user = _to_user_response(user_row)
        profile = _to_profile_response(profile_row)
        goal = _to_goal_response(goal_row)
        created_at = datetime.now(timezone.utc).replace(microsecond=0)

        memory_records = self._build_memory_records(
            request=request,
            user_id=user.id,
            profile_id=profile.id,
            goal_id=goal.id,
            interpretation=interpretation,
        )
        memory_ids = [record.memory_id for record in memory_records]

        try:
            memory_service = self._get_memory_service()
            memory_service.add_text_memories(memory_records)
        except Exception as exc:  # noqa: BLE001
            # Keep SQLite records; surface recoverable partial failure.
            message = (
                "Onboarding records were saved, but semantic memory persistence failed: "
                f"{type(exc).__name__}"
            )
            logger.error("ProfileAgent memory persistence failed error_type=%s", type(exc).__name__)
            partial = ProfileAgentResult(
                user=user,
                profile=profile,
                goal=goal,
                interpretation=interpretation,
                memory_ids=[],
                memories_complete=False,
                memory_error=message,
                created_at=created_at,
            )
            raise ProfileMemoryError(message, result=partial) from None

        logger.info(
            "ProfileAgent completed user_id=%s goal_id=%s memories=%s",
            user.id,
            goal.id,
            len(memory_ids),
        )
        return ProfileAgentResult(
            user=user,
            profile=profile,
            goal=goal,
            interpretation=interpretation,
            memory_ids=memory_ids,
            memories_complete=True,
            memory_error=None,
            created_at=created_at,
        )
