"""
Adaptation Agent for GrowthOS AI.

Deterministic orchestration:
1. Validate user and reflection ownership
2. Load reflection, plan, check-in, tasks, interactions, profile, preferences, history
3. Derive deterministic analytics in Python
4. Ask Gemini for structured interpretation of those stats
5. Validate evidence-backed guardrails
6. Persist insights + preferences transactionally
7. Store concise semantic adaptation memories
8. Return a typed result

Evidence thresholds (MVP):
- 1 session: early signal only; next-plan explanation allowed; no strong permanent preference
- 2 consistent sessions: medium-confidence preference updates (cap 0.65)
- 3+ consistent sessions: stronger preference updates (cap 0.85)

Successful session criteria:
- plan completion percent >= 50
- focus_rating >= 3
- resource_effectiveness >= 3

Never rewrites the long-term goal or roadmap/milestone ownership.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from config import Settings, get_settings
from exceptions import (
    AdaptationContextError,
    AdaptationEvidenceError,
    AdaptationGenerationError,
    AdaptationMemoryError,
    AdaptationOwnershipError,
    AdaptationPersistenceError,
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
)
from models import (
    AdaptationAgentResult,
    AdaptationGeneration,
    AdaptationInsightResponse,
    PreferenceUpdateGeneration,
    UserPreferenceResponse,
)
from services.database import (
    create_adaptation_bundle,
    find_adaptation_insights_for_reflection,
    get_active_goal_for_user,
    get_active_roadmap_for_user,
    get_daily_checkin_by_id,
    get_daily_plan_by_id,
    get_reflection_by_id,
    get_user_by_id,
    get_user_profile_by_user_id,
    list_active_adaptation_insights,
    list_reflections_for_user,
    list_resource_interactions_for_plan,
    list_user_preferences,
)
from services.ai_provider import get_ai_provider
from services.memory import SemanticMemoryService
from services.vector_models import MemoryRecordType, VectorMemoryRecord

logger = logging.getLogger(__name__)

SUPPORTED_PREFERENCE_KEYS = frozenset(
    {
        "preferred_format",
        "preferred_session_minutes",
        "preferred_task_count",
        "difficulty_bias",
        "preferred_activity",
        "effective_resource_duration",
        "pacing_style",
        "focus_support_strategy",
    }
)
SUPPORTED_FORMATS = frozenset(
    {"video", "practice", "read", "listen", "mixed", "short_video", "watch"}
)
SUPPORTED_ACTIVITIES = frozenset(
    {"watch", "listen", "read", "practice", "review", "mixed"}
)
SUPPORTED_DIFFICULTY_BIAS = frozenset(
    {"easier", "maintain", "slightly_harder", "beginner", "intermediate", "advanced"}
)
SUPPORTED_PACING = frozenset(
    {"slower", "steady", "ambitious", "micro", "standard"}
)

MIN_SESSION_MINUTES = 5
MAX_SESSION_MINUTES = 60
MIN_TASK_COUNT = 1
MAX_TASK_COUNT = 5
MIN_EFFECTIVE_DURATION = 3
MAX_EFFECTIVE_DURATION = 45

EARLY_SIGNAL_CONFIDENCE_CAP = 0.45
MEDIUM_CONFIDENCE_CAP = 0.65
STRONG_CONFIDENCE_CAP = 0.85

SUCCESS_COMPLETION_THRESHOLD = 50.0
SUCCESS_FOCUS_THRESHOLD = 3
SUCCESS_EFFECTIVENESS_THRESHOLD = 3

HISTORY_LIMIT_DEFAULT = 10
REFLECTION_MARKER_PREFIX = "reflection_id="

ADAPTATION_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Adaptation Agent.

Interpret deterministic learning statistics and propose safe next-session adjustments.

Rules:
- Do not change the user's goal or roadmap ownership.
- Do not diagnose medical, psychological, or attention conditions.
- Distinguish direct evidence from inference.
- Do not claim a permanent preference from a single weak session.
- Mark one-session conclusions as early signals.
- Do not invent missing activity, completion, or time spent.
- Use only supported preference keys supplied in the prompt.
- Return only the requested structured output.
""".strip()

PersistAdaptationFn = Callable[..., dict[str, Any]]


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[AdaptationGeneration],
        *,
        system_instruction: str | None = None,
    ) -> AdaptationGeneration: ...


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _reflection_marker(reflection_id: int) -> str:
    return f"{REFLECTION_MARKER_PREFIX}{int(reflection_id)}"


def is_successful_session(
    *,
    plan_completion_percent: float,
    focus_rating: int,
    resource_effectiveness: int,
) -> bool:
    return (
        plan_completion_percent >= SUCCESS_COMPLETION_THRESHOLD
        and int(focus_rating) >= SUCCESS_FOCUS_THRESHOLD
        and int(resource_effectiveness) >= SUCCESS_EFFECTIVENESS_THRESHOLD
    )


def confidence_cap_for_sessions(session_count: int) -> float:
    if session_count <= 1:
        return EARLY_SIGNAL_CONFIDENCE_CAP
    if session_count == 2:
        return MEDIUM_CONFIDENCE_CAP
    return STRONG_CONFIDENCE_CAP


def compute_adaptation_analytics(
    *,
    reflections: Sequence[dict[str, Any]],
    plans_by_id: dict[int, dict[str, Any]],
    interactions_by_plan: dict[int, list[dict[str, Any]]],
    checkins_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic analytics from available evidence only."""
    session_count = len(reflections)
    focus_ratings: list[int] = []
    effectiveness_ratings: list[int] = []
    planned_minutes: list[int] = []
    actual_minutes: list[int] = []
    completion_rates: list[float] = []
    successful = 0
    distraction_counter: Counter[str] = Counter()
    format_success: Counter[str] = Counter()
    activity_success: Counter[str] = Counter()
    difficulty_feedback: Counter[str] = Counter()
    task_counts_success: list[int] = []
    long_resource_low_focus = 0
    practice_useful = 0
    mood_pairs: list[dict[str, Any]] = []

    for reflection in reflections:
        plan_id = int(reflection["daily_plan_id"])
        plan = plans_by_id.get(plan_id) or {}
        tasks = list(plan.get("tasks") or [])
        interactions = list(interactions_by_plan.get(plan_id) or [])
        checkin = None
        if plan.get("checkin_id"):
            checkin = checkins_by_id.get(int(plan["checkin_id"]))

        focus = int(reflection.get("focus_rating") or 0)
        effectiveness = int(reflection.get("resource_effectiveness") or 0)
        focus_ratings.append(focus)
        effectiveness_ratings.append(effectiveness)
        difficulty_feedback[str(reflection.get("difficulty_feedback") or "")] += 1
        for item in reflection.get("distractions") or []:
            text = str(item).strip().lower()
            if text:
                distraction_counter[text] += 1

        completed = sum(1 for t in tasks if t.get("status") == "completed")
        total = len(tasks) or 1
        completion = (completed / total) * 100.0
        completion_rates.append(completion)
        if plan.get("total_estimated_minutes") is not None:
            planned_minutes.append(int(plan["total_estimated_minutes"]))

        # Actual minutes only when present on interactions or task metadata.
        session_actual = 0
        has_actual = False
        for interaction in interactions:
            if interaction.get("duration_minutes") is not None:
                session_actual += int(interaction["duration_minutes"])
                has_actual = True
        if not has_actual:
            for task in tasks:
                meta = task.get("metadata") or {}
                if meta.get("duration_minutes") is not None:
                    session_actual += int(meta["duration_minutes"])
                    has_actual = True
        if has_actual:
            actual_minutes.append(session_actual)

        success = is_successful_session(
            plan_completion_percent=completion,
            focus_rating=focus,
            resource_effectiveness=effectiveness,
        )
        if success:
            successful += 1
            task_counts_success.append(len(tasks))
            for task in tasks:
                if task.get("status") != "completed":
                    continue
                activity = str(task.get("activity_type") or "")
                if activity:
                    activity_success[activity] += 1
                content = str(task.get("content_type") or task.get("activity_type") or "")
                if content:
                    format_success[content] += 1

        # Long resource + low focus pattern.
        for task in tasks:
            minutes = int(task.get("estimated_minutes") or 0)
            if minutes >= 12 and focus <= 2:
                long_resource_low_focus += 1
                break

        # Practical task usefulness from completed practice + high effectiveness.
        practice_done = any(
            t.get("status") == "completed" and t.get("activity_type") == "practice"
            for t in tasks
        )
        if practice_done and effectiveness >= 4:
            practice_useful += 1

        mood_pairs.append(
            {
                "mood_before": (checkin or {}).get("mood"),
                "mood_after": reflection.get("mood_after"),
                "mood_match": bool(reflection.get("mood_match")),
                "completion_percent": completion,
            }
        )

    def _avg(values: Sequence[float | int]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "session_count": session_count,
        "successful_session_count": successful,
        "completion_rate": _avg(completion_rates),
        "average_planned_minutes": _avg(planned_minutes),
        "average_actual_minutes": _avg(actual_minutes),
        "actual_minutes_available": bool(actual_minutes),
        "average_focus_rating": _avg(focus_ratings),
        "average_resource_effectiveness": _avg(effectiveness_ratings),
        "most_successful_format": (
            format_success.most_common(1)[0][0] if format_success else None
        ),
        "most_successful_activity": (
            activity_success.most_common(1)[0][0] if activity_success else None
        ),
        "difficulty_feedback_counts": dict(difficulty_feedback),
        "common_distractions": [name for name, _ in distraction_counter.most_common(5)],
        "successful_task_counts": task_counts_success,
        "suggested_task_count": (
            int(round(sum(task_counts_success) / len(task_counts_success)))
            if task_counts_success
            else None
        ),
        "long_resource_low_focus_sessions": long_resource_low_focus,
        "practice_useful_sessions": practice_useful,
        "mood_session_patterns": mood_pairs,
        "is_early_signal": session_count <= 1,
        "confidence_cap": confidence_cap_for_sessions(session_count),
    }


def validate_adaptation_generation(
    generation: AdaptationGeneration,
    *,
    analytics: dict[str, Any],
    existing_preferences: Sequence[dict[str, Any]],
    evidence_tokens: Sequence[str],
) -> AdaptationGeneration:
    """Apply post-Gemini guardrails; return a sanitized generation."""
    if generation.confidence_score < 0 or generation.confidence_score > 1:
        raise AdaptationGenerationError("Invalid confidence_score")

    session_count = int(analytics.get("session_count") or 0)
    cap = float(analytics.get("confidence_cap") or EARLY_SIGNAL_CONFIDENCE_CAP)
    is_early = session_count <= 1 or bool(generation.is_early_signal)
    if session_count <= 1:
        is_early = True

    confidence = min(float(generation.confidence_score), cap)
    if is_early:
        confidence = min(confidence, EARLY_SIGNAL_CONFIDENCE_CAP)

    existing_by_key = {
        str(item.get("preference_key")): item for item in existing_preferences
    }
    allowed_updates: list[PreferenceUpdateGeneration] = []
    for update in generation.preference_updates:
        key = update.preference_key.strip()
        if key not in SUPPORTED_PREFERENCE_KEYS:
            raise AdaptationEvidenceError(f"Unsupported preference key: {key}")
        if update.action == "keep":
            continue

        value = update.preference_value.strip()
        conf = min(float(update.confidence_score), cap, confidence)
        if is_early:
            # One session may propose next-plan guidance but not a strong permanent pref.
            conf = min(conf, EARLY_SIGNAL_CONFIDENCE_CAP)
            if conf < 0.3:
                continue

        if key in {"preferred_format"} and value not in SUPPORTED_FORMATS:
            raise AdaptationEvidenceError("Unsupported preferred_format value")
        if key == "preferred_activity" and value not in SUPPORTED_ACTIVITIES:
            raise AdaptationEvidenceError("Unsupported preferred_activity value")
        if key == "difficulty_bias" and value not in SUPPORTED_DIFFICULTY_BIAS:
            raise AdaptationEvidenceError("Unsupported difficulty_bias value")
        if key == "pacing_style" and value not in SUPPORTED_PACING:
            raise AdaptationEvidenceError("Unsupported pacing_style value")
        if key in {"preferred_session_minutes", "effective_resource_duration"}:
            try:
                minutes = int(value)
            except ValueError as exc:
                raise AdaptationEvidenceError("Duration preference must be integer") from exc
            low = MIN_SESSION_MINUTES if key == "preferred_session_minutes" else MIN_EFFECTIVE_DURATION
            high = MAX_SESSION_MINUTES if key == "preferred_session_minutes" else MAX_EFFECTIVE_DURATION
            if not (low <= minutes <= high):
                raise AdaptationEvidenceError("Duration preference out of bounds")
            value = str(minutes)
        if key == "preferred_task_count":
            try:
                count = int(value)
            except ValueError as exc:
                raise AdaptationEvidenceError("Task count must be integer") from exc
            if not (MIN_TASK_COUNT <= count <= MAX_TASK_COUNT):
                raise AdaptationEvidenceError("Task count out of bounds")
            value = str(count)

        # Cited evidence should reference known tokens when provided.
        for token in update.evidence:
            text = str(token)
            if text.startswith("fabricated:"):
                raise AdaptationEvidenceError("Fabricated evidence is not allowed")

        existing = existing_by_key.get(key)
        if existing is not None:
            existing_conf = float(existing.get("confidence_score") or 0)
            existing_source = str(existing.get("source") or "")
            # Never overwrite stronger preference with weaker evidence.
            if existing_conf > conf:
                continue
            # Preserve explicit onboarding preferences unless repeated strong evidence.
            if existing_source in {"onboarding", "profile", "user"} and (
                is_early or conf < MEDIUM_CONFIDENCE_CAP
            ):
                continue

        # One session: only allow soft next-session keys at early-signal confidence.
        if is_early and key not in {
            "effective_resource_duration",
            "preferred_task_count",
            "focus_support_strategy",
            "preferred_activity",
            "preferred_format",
            "pacing_style",
            "difficulty_bias",
            "preferred_session_minutes",
        }:
            continue
        if is_early and conf > EARLY_SIGNAL_CONFIDENCE_CAP:
            conf = EARLY_SIGNAL_CONFIDENCE_CAP

        allowed_updates.append(
            PreferenceUpdateGeneration(
                preference_key=key,
                preference_value=value,
                confidence_score=conf,
                evidence=list(update.evidence),
                action=update.action,
            )
        )

    # For a true early signal with insufficient history, strip permanent-looking high prefs.
    if session_count <= 1:
        soft: list[PreferenceUpdateGeneration] = []
        for update in allowed_updates:
            # Keep only low-confidence early adjustments.
            if update.confidence_score <= EARLY_SIGNAL_CONFIDENCE_CAP:
                soft.append(update.model_copy(update={"confidence_score": min(update.confidence_score, EARLY_SIGNAL_CONFIDENCE_CAP)}))
        allowed_updates = soft

    explanation = generation.adaptation_explanation.strip()
    if not explanation:
        raise AdaptationGenerationError("adaptation_explanation is required")

    banned = re.compile(r"\b(adhd|autism|disorder|diagnos(?:e|is)|depression|anxiety disorder)\b", re.I)
    if banned.search(explanation) or banned.search(generation.summary):
        raise AdaptationGenerationError("Unsupported diagnostic language in adaptation output")

    return generation.model_copy(
        update={
            "confidence_score": confidence,
            "is_early_signal": is_early,
            "preference_updates": allowed_updates,
            "adaptation_explanation": explanation,
        }
    )


def build_no_change_explanation(analytics: dict[str, Any]) -> str:
    if int(analytics.get("session_count") or 0) <= 0:
        return (
            "No strong pattern has been detected yet. "
            "The next plan will rely on today's mood and profile preferences."
        )
    return (
        "No strong adaptation pattern has been detected yet from the available evidence. "
        "Current preferences are unchanged; the next plan will rely on today's mood and profile."
    )


def build_adaptation_prompt(
    *,
    goal_title: str,
    goal_description: str,
    analytics: dict[str, Any],
    reflection: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    preferences: Sequence[dict[str, Any]],
    existing_insights: Sequence[dict[str, Any]],
) -> str:
    payload = {
        "read_only_goal": {
            "title": goal_title,
            "description": goal_description,
            "note": "Do not rewrite or replace this goal.",
        },
        "trigger_reflection": {
            "id": reflection.get("id"),
            "completion_status": reflection.get("completion_status"),
            "focus_rating": reflection.get("focus_rating"),
            "resource_effectiveness": reflection.get("resource_effectiveness"),
            "difficulty_feedback": reflection.get("difficulty_feedback"),
            "mood_match": reflection.get("mood_match"),
            "distractions": reflection.get("distractions") or [],
            "mood_after": reflection.get("mood_after"),
            "learning_summary": reflection.get("learning_summary") or "",
            "insight": reflection.get("insight") or "",
        },
        "plan_snapshot": {
            "id": plan.get("id"),
            "summary": plan.get("summary"),
            "total_estimated_minutes": plan.get("total_estimated_minutes"),
            "task_count": len(plan.get("tasks") or []),
            "task_activities": [
                {
                    "title": t.get("title"),
                    "activity_type": t.get("activity_type"),
                    "status": t.get("status"),
                    "estimated_minutes": t.get("estimated_minutes"),
                    "content_type": t.get("content_type"),
                }
                for t in (plan.get("tasks") or [])
            ],
        },
        "profile": {
            "preferred_formats": profile.get("preferred_formats") or [],
            "preferred_session_minutes": profile.get("preferred_session_minutes"),
            "attention_span_minutes": profile.get("attention_span_minutes"),
            "learning_style": profile.get("learning_style"),
        },
        "existing_preferences": [
            {
                "key": p.get("preference_key"),
                "value": p.get("preference_value"),
                "confidence_score": p.get("confidence_score"),
                "source": p.get("source"),
            }
            for p in preferences
        ],
        "existing_active_insights": [
            {
                "type": i.get("insight_type"),
                "insight": i.get("insight"),
                "confidence_score": i.get("confidence_score"),
            }
            for i in existing_insights[:5]
        ],
        "deterministic_analytics": analytics,
        "supported_preference_keys": sorted(SUPPORTED_PREFERENCE_KEYS),
        "supported_formats": sorted(SUPPORTED_FORMATS),
        "bounds": {
            "session_minutes": [MIN_SESSION_MINUTES, MAX_SESSION_MINUTES],
            "task_count": [MIN_TASK_COUNT, MAX_TASK_COUNT],
            "effective_resource_duration": [
                MIN_EFFECTIVE_DURATION,
                MAX_EFFECTIVE_DURATION,
            ],
        },
        "constraints": {
            "never_change_goal": True,
            "never_diagnose": True,
            "one_session_is_early_signal": True,
            "do_not_invent_actual_minutes": not bool(
                analytics.get("actual_minutes_available")
            ),
        },
    }
    return (
        "Interpret the analytics and propose safe adaptation updates.\n"
        "If evidence is weak, keep preferences and explain that no strong pattern exists.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


class AdaptationAgent:
    """Learns from reflections and produces explainable next-plan adjustments."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        gemini_service: SupportsStructuredGeneration | None = None,
        memory_service: SemanticMemoryService | None = None,
        db_path: Path | str | None = None,
        persist_adaptation: PersistAdaptationFn | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._memory = memory_service
        self._db_path = Path(db_path) if db_path is not None else None
        self._persist_adaptation = persist_adaptation

    def _get_memory_service(self) -> SemanticMemoryService:
        if self._memory is None:
            self._memory = SemanticMemoryService(settings=self._settings)
        return self._memory

    def _to_insight_response(self, row: dict[str, Any]) -> AdaptationInsightResponse:
        return AdaptationInsightResponse(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            insight_type=str(row["insight_type"]),
            insight=str(row["insight"]),
            confidence_score=float(row["confidence_score"]),
            evidence=list(row.get("evidence") or []),
            is_active=bool(row.get("is_active")),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _to_pref_response(self, row: dict[str, Any]) -> UserPreferenceResponse:
        return UserPreferenceResponse(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            preference_key=str(row["preference_key"]),
            preference_value=str(row["preference_value"]),
            confidence_score=float(row["confidence_score"]),
            source=str(row.get("source") or "system"),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _load_history_context(
        self,
        user_id: int,
        *,
        limit: int,
    ) -> tuple[
        list[dict[str, Any]],
        dict[int, dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        dict[int, dict[str, Any]],
    ]:
        reflections = list_reflections_for_user(
            user_id,
            limit=max(1, limit),
            db_path=self._db_path,
        )
        plans_by_id: dict[int, dict[str, Any]] = {}
        interactions_by_plan: dict[int, list[dict[str, Any]]] = {}
        checkins_by_id: dict[int, dict[str, Any]] = {}
        for reflection in reflections:
            plan_id = int(reflection["daily_plan_id"])
            if plan_id not in plans_by_id:
                plan = get_daily_plan_by_id(plan_id, db_path=self._db_path)
                if plan is None:
                    continue
                if int(plan["user_id"]) != user_id:
                    raise AdaptationOwnershipError(
                        "Historical plan does not belong to the requested user"
                    )
                plans_by_id[plan_id] = plan
                interactions_by_plan[plan_id] = list_resource_interactions_for_plan(
                    plan_id,
                    user_id=user_id,
                    db_path=self._db_path,
                )
                if plan.get("checkin_id"):
                    checkin = get_daily_checkin_by_id(
                        int(plan["checkin_id"]),
                        db_path=self._db_path,
                    )
                    if checkin is not None:
                        checkins_by_id[int(plan["checkin_id"])] = checkin
        return reflections, plans_by_id, interactions_by_plan, checkins_by_id

    def _build_result_from_existing(
        self,
        *,
        user_id: int,
        reflection_id: int,
        insights: Sequence[dict[str, Any]],
        goal_before: dict[str, Any],
        roadmap_before: dict[str, Any] | None,
        milestone_before: dict[str, Any] | None,
    ) -> AdaptationAgentResult:
        preferences = list_user_preferences(user_id, db_path=self._db_path)
        explanation = "Existing adaptation for this reflection was reused."
        for item in insights:
            text = str(item.get("insight") or "")
            if text.startswith("next_plan:"):
                explanation = text.split(":", 1)[1].strip() or explanation
                break
            if item.get("insight_type") == "next_plan_explanation":
                explanation = text or explanation
                break
        patterns = [
            str(i.get("insight"))
            for i in insights
            if i.get("insight_type") == "pattern"
        ]
        goal_after = get_active_goal_for_user(user_id, db_path=self._db_path)
        roadmap_after = get_active_roadmap_for_user(user_id, db_path=self._db_path)
        milestone_after = (roadmap_after or {}).get("active_milestone")
        return AdaptationAgentResult(
            insights=[self._to_insight_response(i) for i in insights],
            preferences=[self._to_pref_response(p) for p in preferences],
            adaptation_explanation=explanation,
            detected_patterns=patterns,
            goal_unchanged=(
                goal_after is not None
                and str(goal_after.get("title")) == str(goal_before.get("title"))
                and str(goal_after.get("description") or "")
                == str(goal_before.get("description") or "")
            ),
            roadmap_unchanged=(
                (roadmap_before is None and roadmap_after is None)
                or (
                    roadmap_before is not None
                    and roadmap_after is not None
                    and int(roadmap_before["id"]) == int(roadmap_after["id"])
                )
            ),
            milestone_unchanged=(
                (milestone_before is None and milestone_after is None)
                or (
                    milestone_before is not None
                    and milestone_after is not None
                    and int(milestone_before["id"]) == int(milestone_after["id"])
                )
            ),
            is_early_signal=True,
            confidence_score=float(insights[0].get("confidence_score") or 0.0)
            if insights
            else 0.0,
            reflection_id=reflection_id,
            reused_existing=True,
            memory_ids=[],
            memories_complete=True,
            created_at=_parse_dt(insights[0]["created_at"])
            if insights
            else datetime.now(timezone.utc),
        )

    def _build_memory_records(
        self,
        *,
        user_id: int,
        reflection_id: int,
        insight_ids: Sequence[int],
        generation: AdaptationGeneration,
    ) -> list[VectorMemoryRecord]:
        suffix = uuid.uuid4().hex[:8]
        base_meta = {
            "source": "adaptation_agent",
            "reflection_id": reflection_id,
            "adaptation_insight_ids": list(insight_ids),
            "confidence": generation.confidence_score,
            "early_signal": generation.is_early_signal,
        }
        pattern = (
            generation.detected_patterns[0]
            if generation.detected_patterns
            else generation.summary
        )
        adjustment = (
            generation.next_session_adjustments[0]
            if generation.next_session_adjustments
            else generation.adaptation_explanation
        )
        pref_summary = ", ".join(
            f"{p.preference_key}={p.preference_value}"
            for p in generation.preference_updates
            if p.action in {"create", "update"}
        ) or "no preference updates"
        return [
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-adaptation-pattern-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.adaptation.value,
                source_record_id=str(reflection_id),
                text=f"Detected pattern: {pattern}",
                metadata={**base_meta, "kind": "pattern"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-adaptation-next-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.adaptation.value,
                source_record_id=str(reflection_id),
                text=f"Next-session adjustment: {adjustment}",
                metadata={**base_meta, "kind": "next_session"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-adaptation-prefs-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.adaptation.value,
                source_record_id=str(reflection_id),
                text=f"Preference update summary: {pref_summary}",
                metadata={**base_meta, "kind": "preferences"},
            ),
        ]

    def adapt_from_recent_history(
        self,
        user_id: int,
        *,
        limit: int = HISTORY_LIMIT_DEFAULT,
        force: bool = False,
    ) -> AdaptationAgentResult:
        """Adapt using the newest reflection, informed by recent history."""
        if user_id <= 0:
            raise AdaptationContextError("user_id must be a positive integer")
        reflections = list_reflections_for_user(
            user_id,
            limit=max(1, limit),
            db_path=self._db_path,
        )
        if not reflections:
            raise AdaptationContextError("No reflections found for user")
        return self.adapt_from_reflection(
            user_id,
            int(reflections[0]["id"]),
            force=force,
        )

    def adapt_from_reflection(
        self,
        user_id: int,
        reflection_id: int,
        *,
        force: bool = False,
    ) -> AdaptationAgentResult:
        """
        Adapt from one reflection plus recent history.

        Duplicate behavior:
        - force=False: if active insights already exist for this reflection, reuse them
        - force=True: deactivate prior insights for this reflection, then recompute
        """
        if user_id <= 0:
            raise AdaptationContextError("user_id must be a positive integer")
        if reflection_id <= 0:
            raise AdaptationContextError("reflection_id must be a positive integer")

        user = get_user_by_id(user_id, db_path=self._db_path)
        if user is None:
            raise AdaptationContextError("User not found")

        reflection = get_reflection_by_id(reflection_id, db_path=self._db_path)
        if reflection is None:
            raise AdaptationContextError("Reflection not found")
        if int(reflection["user_id"]) != user_id:
            raise AdaptationOwnershipError(
                "Reflection does not belong to the requested user"
            )

        goal_before = get_active_goal_for_user(user_id, db_path=self._db_path)
        if goal_before is None:
            raise AdaptationContextError("Active goal not found")
        roadmap_before = get_active_roadmap_for_user(user_id, db_path=self._db_path)
        milestone_before = (roadmap_before or {}).get("active_milestone")

        existing_for_reflection = find_adaptation_insights_for_reflection(
            user_id,
            reflection_id,
            active_only=True,
            db_path=self._db_path,
        )
        if existing_for_reflection and not force:
            return self._build_result_from_existing(
                user_id=user_id,
                reflection_id=reflection_id,
                insights=existing_for_reflection,
                goal_before=goal_before,
                roadmap_before=roadmap_before,
                milestone_before=milestone_before,
            )

        profile = get_user_profile_by_user_id(user_id, db_path=self._db_path)
        if profile is None:
            raise AdaptationContextError("User profile not found")

        plan = get_daily_plan_by_id(
            int(reflection["daily_plan_id"]),
            db_path=self._db_path,
        )
        if plan is None:
            raise AdaptationContextError("Daily plan not found for reflection")
        if int(plan["user_id"]) != user_id:
            raise AdaptationOwnershipError("Plan does not belong to the requested user")

        reflections, plans_by_id, interactions_by_plan, checkins_by_id = (
            self._load_history_context(user_id, limit=HISTORY_LIMIT_DEFAULT)
        )
        # Ensure trigger plan is present.
        plans_by_id[int(plan["id"])] = plan
        if int(plan["id"]) not in interactions_by_plan:
            interactions_by_plan[int(plan["id"])] = list_resource_interactions_for_plan(
                int(plan["id"]),
                user_id=user_id,
                db_path=self._db_path,
            )
        if plan.get("checkin_id") and int(plan["checkin_id"]) not in checkins_by_id:
            checkin = get_daily_checkin_by_id(
                int(plan["checkin_id"]),
                db_path=self._db_path,
            )
            if checkin is not None:
                checkins_by_id[int(plan["checkin_id"])] = checkin

        # Keep only this user's reflections (already scoped) and ensure trigger included.
        if not any(int(r["id"]) == reflection_id for r in reflections):
            reflections = [reflection, *reflections]

        analytics = compute_adaptation_analytics(
            reflections=reflections,
            plans_by_id=plans_by_id,
            interactions_by_plan=interactions_by_plan,
            checkins_by_id=checkins_by_id,
        )
        preferences = list_user_preferences(user_id, db_path=self._db_path)
        existing_insights = list_active_adaptation_insights(
            user_id,
            db_path=self._db_path,
        )

        prompt = build_adaptation_prompt(
            goal_title=str(goal_before["title"]),
            goal_description=str(goal_before.get("description") or ""),
            analytics=analytics,
            reflection=reflection,
            plan=plan,
            profile=profile,
            preferences=preferences,
            existing_insights=existing_insights,
        )
        logger.info(
            "AdaptationAgent generating user_id=%s reflection_id=%s prompt_chars=%s",
            user_id,
            reflection_id,
            len(prompt),
        )

        try:
            generation = self._gemini.generate_structured(
                prompt,
                AdaptationGeneration,
                system_instruction=ADAPTATION_SYSTEM_INSTRUCTION,
            )
            generation = validate_adaptation_generation(
                generation,
                analytics=analytics,
                existing_preferences=preferences,
                evidence_tokens=[],
            )
        except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
            raise
        except (AdaptationEvidenceError, AdaptationGenerationError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise AdaptationGenerationError(
                f"Adaptation generation failed: {type(exc).__name__}"
            ) from None

        # Insufficient / empty pattern case: still persist a truthful no-change insight.
        no_strong_pattern = (
            not generation.detected_patterns
            and not generation.preference_updates
            and "no strong" in generation.adaptation_explanation.lower()
        )
        if no_strong_pattern:
            generation = generation.model_copy(
                update={
                    "adaptation_explanation": build_no_change_explanation(analytics),
                    "is_early_signal": True,
                    "confidence_score": min(
                        generation.confidence_score,
                        EARLY_SIGNAL_CONFIDENCE_CAP,
                    ),
                }
            )

        marker = _reflection_marker(reflection_id)
        evidence_base = [
            marker,
            f"session_count={analytics['session_count']}",
            f"avg_focus={analytics.get('average_focus_rating')}",
            f"completion_rate={analytics.get('completion_rate')}",
        ]
        if analytics.get("common_distractions"):
            evidence_base.append(
                "distractions=" + ",".join(analytics["common_distractions"][:3])
            )
        if not analytics.get("actual_minutes_available"):
            evidence_base.append("actual_minutes=absent")

        insight_rows: list[dict[str, Any]] = [
            {
                "insight_type": "pattern" if generation.detected_patterns else "session",
                "insight": (
                    generation.detected_patterns[0]
                    if generation.detected_patterns
                    else generation.summary
                ),
                "confidence_score": generation.confidence_score,
                "evidence": evidence_base
                + [f"early_signal={generation.is_early_signal}"],
                "is_active": True,
            },
            {
                "insight_type": "next_plan_explanation",
                "insight": generation.adaptation_explanation,
                "confidence_score": generation.confidence_score,
                "evidence": evidence_base
                + list(generation.next_session_adjustments[:3]),
                "is_active": True,
            },
        ]
        # Skip exact active duplicates for this reflection unless force recomputes.
        if not force:
            existing_same_reflection = {
                (str(i.get("insight_type")), str(i.get("insight")))
                for i in existing_for_reflection
            }
            insight_rows = [
                row
                for row in insight_rows
                if (row["insight_type"], row["insight"]) not in existing_same_reflection
            ]

        preference_rows: list[dict[str, Any]] = []
        for update in generation.preference_updates:
            if update.action not in {"create", "update"}:
                continue
            # One-session early signal: store only low-confidence adaptations.
            source = (
                "adaptation_early_signal"
                if generation.is_early_signal
                else "adaptation"
            )
            preference_rows.append(
                {
                    "preference_key": update.preference_key,
                    "preference_value": update.preference_value,
                    "confidence_score": update.confidence_score,
                    "source": source,
                }
            )

        deactivate_ids = (
            [int(i["id"]) for i in existing_for_reflection] if force else []
        )

        try:
            if self._persist_adaptation is not None:
                bundle = self._persist_adaptation(
                    user_id=user_id,
                    insights=insight_rows,
                    preferences=preference_rows,
                    deactivate_insight_ids=deactivate_ids,
                )
            else:
                bundle = create_adaptation_bundle(
                    user_id=user_id,
                    insights=insight_rows,
                    preferences=preference_rows,
                    deactivate_insight_ids=deactivate_ids,
                    db_path=self._db_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "AdaptationAgent persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise AdaptationPersistenceError(
                f"Failed to persist adaptation: {type(exc).__name__}"
            ) from None

        goal_after = get_active_goal_for_user(user_id, db_path=self._db_path)
        roadmap_after = get_active_roadmap_for_user(user_id, db_path=self._db_path)
        milestone_after = (roadmap_after or {}).get("active_milestone")
        goal_unchanged = (
            goal_after is not None
            and str(goal_after.get("title")) == str(goal_before.get("title"))
            and str(goal_after.get("description") or "")
            == str(goal_before.get("description") or "")
        )
        roadmap_unchanged = (
            (roadmap_before is None and roadmap_after is None)
            or (
                roadmap_before is not None
                and roadmap_after is not None
                and int(roadmap_before["id"]) == int(roadmap_after["id"])
            )
        )
        milestone_unchanged = (
            (milestone_before is None and milestone_after is None)
            or (
                milestone_before is not None
                and milestone_after is not None
                and int(milestone_before["id"]) == int(milestone_after["id"])
            )
        )
        if not goal_unchanged:
            raise AdaptationEvidenceError("Adaptation unexpectedly altered the goal")

        created_insights = [self._to_insight_response(i) for i in bundle["insights"]]
        # Return all current preferences after upsert.
        current_prefs = [
            self._to_pref_response(p)
            for p in list_user_preferences(user_id, db_path=self._db_path)
        ]
        result = AdaptationAgentResult(
            insights=created_insights,
            preferences=current_prefs,
            adaptation_explanation=generation.adaptation_explanation,
            detected_patterns=list(generation.detected_patterns),
            goal_unchanged=goal_unchanged,
            roadmap_unchanged=roadmap_unchanged,
            milestone_unchanged=milestone_unchanged,
            is_early_signal=bool(generation.is_early_signal),
            confidence_score=float(generation.confidence_score),
            reflection_id=reflection_id,
            reused_existing=False,
            memory_ids=[],
            memories_complete=True,
            memory_error=None,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )

        memory_records = self._build_memory_records(
            user_id=user_id,
            reflection_id=reflection_id,
            insight_ids=[i.id for i in created_insights],
            generation=generation,
        )
        try:
            self._get_memory_service().add_text_memories(memory_records)
            result = result.model_copy(
                update={
                    "memory_ids": [r.memory_id for r in memory_records],
                    "memories_complete": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            message = (
                "Adaptation records were saved, but semantic memory persistence failed: "
                f"{type(exc).__name__}"
            )
            logger.error(
                "AdaptationAgent memory persistence failed error_type=%s",
                type(exc).__name__,
            )
            partial = result.model_copy(
                update={
                    "memory_ids": [],
                    "memories_complete": False,
                    "memory_error": message,
                }
            )
            raise AdaptationMemoryError(message, result=partial) from None

        logger.info(
            "AdaptationAgent completed user_id=%s reflection_id=%s insights=%s",
            user_id,
            reflection_id,
            len(created_insights),
        )
        return result
