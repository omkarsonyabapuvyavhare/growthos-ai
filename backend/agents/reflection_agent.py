"""
Reflection Agent for GrowthOS AI.

Deterministic orchestration:
1. Validate user and plan ownership
2. Load plan, tasks, check-in, milestone, interactions
3. Merge submitted task/resource evidence (in memory)
4. Ask Gemini for a concise structured insight
5. Persist tasks, interactions, reflection, and progress transactionally
6. Store concise semantic reflection memories in FAISS
7. Return a typed result

Evidence hierarchy:
1. persisted task status and interactions
2. task updates submitted with the reflection
3. explicit user self-report
4. Gemini interpretation (summarize only; never invent completion)
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
    ReflectionContextError,
    ReflectionEvidenceError,
    ReflectionGenerationError,
    ReflectionMemoryError,
    ReflectionOwnershipError,
    ReflectionPersistenceError,
)
from models import (
    CompletionStatus,
    ReflectionAgentResult,
    ReflectionInsightGeneration,
    ReflectionRequest,
    ReflectionResponse,
    TaskStatus,
)
from services.database import (
    create_reflection_bundle,
    get_daily_checkin_by_id,
    get_daily_plan_by_id,
    get_milestone_by_id,
    get_reflection_for_plan,
    get_roadmap_by_id,
    get_user_by_id,
    list_milestones_for_roadmap,
    list_resource_interactions_for_plan,
)
from services.ai_provider import get_ai_provider
from services.memory import SemanticMemoryService
from services.vector_models import MemoryRecordType, VectorMemoryRecord

logger = logging.getLogger(__name__)

# Max percentage points one reflection session may add to a milestone.
SESSION_MILESTONE_CONTRIBUTION_CAP = 20.0

REFLECTION_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Reflection Agent.

Summarize today's learning session using only the supplied evidence.

Rules:
- Use only supplied evidence. Distinguish direct evidence from cautious inference.
- Do not invent completed work, watched resources, or time spent.
- Do not diagnose health, mental state, or attention disorders.
- Do not change the user's goal.
- Do not generate resource URLs or new resource IDs.
- Keep the main insight short, concrete, and actionable.
- Return only the requested structured output.
""".strip()

PersistReflectionFn = Callable[..., dict[str, Any]]


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[ReflectionInsightGeneration],
        *,
        system_instruction: str | None = None,
    ) -> ReflectionInsightGeneration: ...


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


def compute_plan_completion_percent(tasks: Sequence[dict[str, Any]]) -> float:
    """
    Duration-weighted plan completion.

    - completed tasks contribute full estimated_minutes
    - in_progress tasks contribute estimated_minutes * (completion_percent/100)
      when completion_percent is present in task metadata; otherwise 0
    - skipped/pending contribute 0
    """
    if not tasks:
        return 0.0
    total_weight = 0.0
    earned = 0.0
    for task in tasks:
        weight = float(max(1, int(task.get("estimated_minutes") or 1)))
        total_weight += weight
        status = str(task.get("status") or "")
        if status == TaskStatus.completed.value:
            earned += weight
            continue
        if status == TaskStatus.in_progress.value:
            meta = task.get("metadata") or {}
            pct = meta.get("completion_percent")
            if pct is None:
                pct = task.get("completion_percent")
            if pct is not None:
                earned += weight * (max(0.0, min(100.0, float(pct))) / 100.0)
    if total_weight <= 0:
        return 0.0
    return max(0.0, min(100.0, (earned / total_weight) * 100.0))


def compute_milestone_progress_after(
    current_progress: float,
    *,
    plan_completion_percent: float,
    completed_task_count: int,
) -> tuple[float, str | None]:
    """
    Increase milestone progress from this session only.

    - never decreases
    - capped contribution per reflection
    - zero completed tasks => no increase
    - one plan never auto-completes a milestone unless progress reaches 100
    """
    current = max(0.0, min(100.0, float(current_progress)))
    if completed_task_count <= 0 or plan_completion_percent <= 0:
        status = "completed" if current >= 100.0 else None
        return current, status

    delta = (plan_completion_percent / 100.0) * SESSION_MILESTONE_CONTRIBUTION_CAP
    after = max(current, min(100.0, current + delta))
    if after >= 100.0:
        return 100.0, "completed"
    if after > 0:
        return after, "in_progress"
    return after, None


def compute_roadmap_progress(milestones: Sequence[dict[str, Any]]) -> float:
    """Roadmap progress = average of milestone progress_percent values."""
    if not milestones:
        return 0.0
    total = sum(float(item.get("progress_percent") or 0.0) for item in milestones)
    return max(0.0, min(100.0, total / len(milestones)))


def merge_task_evidence(
    plan_tasks: Sequence[dict[str, Any]],
    task_updates: Sequence[Any],
) -> list[dict[str, Any]]:
    """Apply submitted updates onto a copy of persisted tasks (updates win)."""
    by_id = {int(task["id"]): dict(task) for task in plan_tasks}
    for item in task_updates:
        task_id = int(item.task_id)
        if task_id not in by_id:
            raise ReflectionEvidenceError("Task update references an unknown task")
        update = item.update
        merged = dict(by_id[task_id])
        merged["status"] = _enum_value(update.status)
        meta = dict(merged.get("metadata") or {})
        if update.completion_percent is not None:
            meta["completion_percent"] = float(update.completion_percent)
            merged["completion_percent"] = float(update.completion_percent)
        if update.duration_minutes is not None:
            meta["duration_minutes"] = int(update.duration_minutes)
        if update.effectiveness_rating is not None:
            meta["effectiveness_rating"] = int(update.effectiveness_rating)
        if update.notes:
            meta["notes"] = update.notes
        merged["metadata"] = meta
        by_id[task_id] = merged
    return [by_id[int(task["id"])] for task in plan_tasks]


def build_reflection_prompt(
    *,
    goal_title: str,
    plan: dict[str, Any],
    tasks: Sequence[dict[str, Any]],
    checkin: dict[str, Any] | None,
    milestone: dict[str, Any] | None,
    interactions: Sequence[dict[str, Any]],
    request: ReflectionRequest,
    plan_completion_percent: float,
) -> str:
    mood_after = _enum_value(request.mood_after) if request.mood_after else ""
    payload = {
        "read_only_goal": {
            "title": goal_title,
            "note": "Do not rewrite or replace this goal.",
        },
        "plan": {
            "id": plan.get("id"),
            "summary": plan.get("summary"),
            "plan_date": plan.get("plan_date"),
            "status": plan.get("status"),
        },
        "active_milestone": {
            "title": (milestone or {}).get("title"),
            "completion_criteria": (milestone or {}).get("completion_criteria"),
            "progress_percent": (milestone or {}).get("progress_percent"),
        },
        "checkin": {
            "mood": (checkin or {}).get("mood"),
            "energy_level": (checkin or {}).get("energy_level"),
            "focus_level": (checkin or {}).get("focus_level"),
            "available_minutes": (checkin or {}).get("available_minutes"),
        }
        if checkin
        else None,
        "tasks": [
            {
                "id": task.get("id"),
                "title": task.get("title"),
                "activity_type": task.get("activity_type"),
                "estimated_minutes": task.get("estimated_minutes"),
                "status": task.get("status"),
                "resource_id": task.get("resource_id"),
                "completion_percent": (task.get("metadata") or {}).get(
                    "completion_percent"
                ),
            }
            for task in tasks
        ],
        "resource_interactions": [
            {
                "resource_id": item.get("resource_id"),
                "interaction_type": item.get("interaction_type"),
                "completion_percent": item.get("completion_percent"),
                "effectiveness_rating": item.get("effectiveness_rating"),
                "duration_minutes": item.get("duration_minutes"),
            }
            for item in interactions
        ],
        "self_report": {
            "completion_status": _enum_value(request.completion_status),
            "learning_summary": request.learning_summary,
            "focus_rating": request.focus_rating,
            "resource_effectiveness": request.resource_effectiveness,
            "difficulty_feedback": _enum_value(request.difficulty_feedback),
            "mood_match": request.mood_match,
            "distractions": list(request.distractions),
            "wants_similar_resources": request.wants_similar_resources,
            "mood_after": mood_after,
            "actual_minutes_spent": request.actual_minutes_spent,
            "actual_minutes_spent_note": (
                "absent — do not invent a time value"
                if request.actual_minutes_spent is None
                else "user-reported"
            ),
        },
        "deterministic_completion": {
            "plan_completion_percent": plan_completion_percent,
            "completed_task_count": sum(
                1 for task in tasks if task.get("status") == TaskStatus.completed.value
            ),
            "total_task_count": len(tasks),
        },
        "constraints": {
            "never_invent_completion": True,
            "never_diagnose_health": True,
            "never_generate_urls": True,
            "never_change_goal": True,
        },
    }
    return (
        "Create a concise evidence-based reflection insight.\n"
        "Do not claim work that is not listed as completed.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _to_reflection_response(row: dict[str, Any]) -> ReflectionResponse:
    return ReflectionResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        daily_plan_id=int(row["daily_plan_id"]),
        completion_status=CompletionStatus(row["completion_status"]),
        learning_summary=str(row.get("learning_summary") or ""),
        focus_rating=int(row["focus_rating"]),
        resource_effectiveness=int(row["resource_effectiveness"]),
        difficulty_feedback=row["difficulty_feedback"],
        mood_match=bool(row.get("mood_match")),
        distractions=list(row.get("distractions") or []),
        wants_similar_resources=bool(row.get("wants_similar_resources")),
        mood_after=str(row.get("mood_after") or ""),
        insight=str(row.get("insight") or "") or None,
        created_at=_parse_dt(row["created_at"]),
    )


def _plan_status_from_completion(
    completion_status: CompletionStatus,
    plan_completion_percent: float,
) -> str:
    if completion_status == CompletionStatus.completed or plan_completion_percent >= 99.9:
        return "completed"
    if completion_status == CompletionStatus.skipped:
        return "skipped"
    if plan_completion_percent > 0 or completion_status == CompletionStatus.partial:
        return "in_progress"
    return "in_progress"


class ReflectionAgent:
    """Processes post-session reflection evidence for one daily plan."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        gemini_service: SupportsStructuredGeneration | None = None,
        memory_service: SemanticMemoryService | None = None,
        db_path: Path | str | None = None,
        persist_reflection: PersistReflectionFn | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._memory = memory_service
        self._db_path = Path(db_path) if db_path is not None else None
        self._persist_reflection = persist_reflection

    def _get_memory_service(self) -> SemanticMemoryService:
        if self._memory is None:
            self._memory = SemanticMemoryService(settings=self._settings)
        return self._memory

    def _load_context(
        self,
        user_id: int,
        daily_plan_id: int,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
        str,
    ]:
        if user_id <= 0:
            raise ReflectionContextError("user_id must be a positive integer")
        if daily_plan_id <= 0:
            raise ReflectionContextError("daily_plan_id must be a positive integer")

        user = get_user_by_id(user_id, db_path=self._db_path)
        if user is None:
            raise ReflectionContextError("User not found")

        plan = get_daily_plan_by_id(daily_plan_id, db_path=self._db_path)
        if plan is None:
            raise ReflectionContextError("Daily plan not found")
        if int(plan["user_id"]) != user_id:
            raise ReflectionOwnershipError("Plan does not belong to the requested user")

        checkin = None
        if plan.get("checkin_id"):
            checkin = get_daily_checkin_by_id(
                int(plan["checkin_id"]),
                db_path=self._db_path,
            )

        roadmap = None
        milestone = None
        goal_title = str((plan.get("metadata") or {}).get("goal_title_snapshot") or "")
        if plan.get("roadmap_id"):
            roadmap = get_roadmap_by_id(int(plan["roadmap_id"]), db_path=self._db_path)
            if roadmap is None:
                raise ReflectionContextError("Roadmap not found for plan")
            if int(roadmap["user_id"]) != user_id:
                raise ReflectionOwnershipError(
                    "Roadmap does not belong to the requested user"
                )
        if plan.get("milestone_id"):
            milestone = get_milestone_by_id(
                int(plan["milestone_id"]),
                db_path=self._db_path,
            )
            if milestone is None:
                raise ReflectionContextError("Milestone not found for plan")
            if int(milestone.get("user_id") or 0) != user_id:
                raise ReflectionOwnershipError(
                    "Milestone does not belong to the requested user"
                )
            if roadmap is not None and int(milestone.get("roadmap_id") or 0) != int(
                roadmap["id"]
            ):
                raise ReflectionOwnershipError(
                    "Milestone does not belong to the plan roadmap"
                )

        interactions = list_resource_interactions_for_plan(
            daily_plan_id,
            user_id=user_id,
            db_path=self._db_path,
        )
        return user, plan, checkin, roadmap, milestone, interactions, goal_title

    def _validate_evidence(
        self,
        *,
        plan: dict[str, Any],
        request: ReflectionRequest,
    ) -> None:
        task_ids = {int(task["id"]) for task in plan.get("tasks") or []}
        plan_resource_ids = {
            int(task["resource_id"])
            for task in plan.get("tasks") or []
            if task.get("resource_id") is not None
        }
        for item in request.task_updates:
            if int(item.task_id) not in task_ids:
                raise ReflectionEvidenceError(
                    "Task update does not belong to the specified plan"
                )
        for item in request.resource_interactions:
            if int(item.resource_id) not in plan_resource_ids:
                raise ReflectionEvidenceError(
                    "Resource interaction is not associated with the plan tasks"
                )
            if item.daily_plan_id is not None and int(item.daily_plan_id) != int(
                plan["id"]
            ):
                raise ReflectionEvidenceError(
                    "Resource interaction daily_plan_id does not match the plan"
                )

    def _build_memory_records(
        self,
        *,
        user_id: int,
        reflection_id: int,
        daily_plan_id: int,
        milestone_id: int | None,
        generation: ReflectionInsightGeneration,
        request: ReflectionRequest,
        plan_completion_percent: float,
    ) -> list[VectorMemoryRecord]:
        suffix = uuid.uuid4().hex[:8]
        base_meta = {
            "source": "reflection_agent",
            "reflection_id": reflection_id,
            "daily_plan_id": daily_plan_id,
            "milestone_id": milestone_id,
            "confidence": generation.confidence_score,
        }
        distractions = ", ".join(request.distractions) if request.distractions else "none"
        return [
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-reflection-outcome-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.reflection.value,
                source_record_id=str(reflection_id),
                text=(
                    f"Session outcome: {generation.completion_observation} "
                    f"Plan completion {plan_completion_percent:.0f}%."
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-reflection-resource-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.reflection.value,
                source_record_id=str(reflection_id),
                text=(
                    f"Resource effectiveness {request.resource_effectiveness}/5. "
                    f"{generation.resource_observation}"
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
            VectorMemoryRecord(
                memory_id=f"user-{user_id}-reflection-focus-{reflection_id}-{suffix}",
                user_id=user_id,
                record_type=MemoryRecordType.reflection.value,
                source_record_id=str(reflection_id),
                text=(
                    f"Focus rating {request.focus_rating}/5. "
                    f"Distractions: {distractions}. "
                    f"{generation.focus_observation}"
                ),
                metadata={**base_meta, "stated_or_inferred": "stated+interpreted"},
            ),
        ]

    def reflect_on_plan(
        self,
        user_id: int,
        request: ReflectionRequest,
    ) -> ReflectionAgentResult:
        """
        Reflect on a completed or partially completed daily plan.

        Duplicate behavior (MVP): if a reflection already exists for the plan,
        return it without creating another or re-applying progress.
        """
        _user, plan, checkin, roadmap, milestone, existing_interactions, goal_title = (
            self._load_context(user_id, request.daily_plan_id)
        )

        existing = get_reflection_for_plan(
            request.daily_plan_id,
            db_path=self._db_path,
        )
        if existing is not None:
            if int(existing["user_id"]) != user_id:
                raise ReflectionOwnershipError(
                    "Existing reflection does not belong to the requested user"
                )
            milestone_progress = float((milestone or {}).get("progress_percent") or 0.0)
            roadmap_progress = float((roadmap or {}).get("progress_percent") or 0.0)
            tasks = list(plan.get("tasks") or [])
            return ReflectionAgentResult(
                reflection=_to_reflection_response(existing),
                plan_completion_percent=compute_plan_completion_percent(tasks),
                milestone_progress_before=milestone_progress,
                milestone_progress_after=milestone_progress,
                roadmap_progress_before=roadmap_progress,
                roadmap_progress_after=roadmap_progress,
                memory_ids=[],
                memories_complete=True,
                memory_error=None,
                reused_existing=True,
                created_at=_parse_dt(existing["created_at"]),
            )

        self._validate_evidence(plan=plan, request=request)
        merged_tasks = merge_task_evidence(plan.get("tasks") or [], request.task_updates)
        plan_completion = compute_plan_completion_percent(merged_tasks)
        completed_count = sum(
            1 for task in merged_tasks if task.get("status") == TaskStatus.completed.value
        )

        # Interactions for Gemini: persisted first, then submitted (submitted appended).
        interaction_evidence = list(existing_interactions)
        for item in request.resource_interactions:
            interaction_evidence.append(
                {
                    "resource_id": item.resource_id,
                    "interaction_type": item.interaction_type,
                    "completion_percent": item.completion_percent,
                    "effectiveness_rating": item.effectiveness_rating,
                    "duration_minutes": item.duration_minutes,
                    "daily_plan_id": request.daily_plan_id,
                }
            )

        prompt = build_reflection_prompt(
            goal_title=goal_title or "Learning goal",
            plan=plan,
            tasks=merged_tasks,
            checkin=checkin,
            milestone=milestone,
            interactions=interaction_evidence,
            request=request,
            plan_completion_percent=plan_completion,
        )
        logger.info(
            "ReflectionAgent generating insight user_id=%s plan_id=%s prompt_chars=%s",
            user_id,
            request.daily_plan_id,
            len(prompt),
        )

        try:
            generation = self._gemini.generate_structured(
                prompt,
                ReflectionInsightGeneration,
                system_instruction=REFLECTION_SYSTEM_INSTRUCTION,
            )
        except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ReflectionGenerationError(
                f"Reflection generation failed: {type(exc).__name__}"
            ) from None

        if not generation.insight.strip():
            raise ReflectionGenerationError("Reflection insight was empty")

        # Guard: Gemini must not invent full completion when evidence shows none.
        if completed_count == 0 and "completed all" in generation.insight.lower():
            raise ReflectionGenerationError(
                "Insight claimed full completion without completed-task evidence"
            )

        milestone_before = float((milestone or {}).get("progress_percent") or 0.0)
        roadmap_before = float((roadmap or {}).get("progress_percent") or 0.0)
        milestone_after, milestone_status = compute_milestone_progress_after(
            milestone_before,
            plan_completion_percent=plan_completion,
            completed_task_count=completed_count,
        )
        # Never decrease.
        milestone_after = max(milestone_before, milestone_after)

        roadmap_after = roadmap_before
        if roadmap is not None:
            milestones = list_milestones_for_roadmap(
                int(roadmap["id"]),
                db_path=self._db_path,
            )
            # Apply projected milestone update before averaging.
            projected: list[dict[str, Any]] = []
            for item in milestones:
                copy = dict(item)
                if milestone is not None and int(copy["id"]) == int(milestone["id"]):
                    copy["progress_percent"] = milestone_after
                projected.append(copy)
            roadmap_after = max(roadmap_before, compute_roadmap_progress(projected))

        task_update_rows: list[dict[str, Any]] = []
        for item in request.task_updates:
            meta_patch: dict[str, Any] = {}
            if item.update.completion_percent is not None:
                meta_patch["completion_percent"] = float(item.update.completion_percent)
            if item.update.duration_minutes is not None:
                meta_patch["duration_minutes"] = int(item.update.duration_minutes)
            if item.update.effectiveness_rating is not None:
                meta_patch["effectiveness_rating"] = int(item.update.effectiveness_rating)
            if item.update.notes:
                meta_patch["notes"] = item.update.notes
            task_update_rows.append(
                {
                    "task_id": int(item.task_id),
                    "status": _enum_value(item.update.status),
                    "metadata_patch": meta_patch,
                }
            )

        interaction_rows: list[dict[str, Any]] = []
        for item in request.resource_interactions:
            interaction_rows.append(
                {
                    "resource_id": int(item.resource_id),
                    "interaction_type": item.interaction_type,
                    "completion_percent": float(item.completion_percent),
                    "effectiveness_rating": item.effectiveness_rating,
                    "duration_minutes": item.duration_minutes,
                }
            )
        # Auto-create interactions from task updates that include resource evidence.
        plan_tasks_by_id = {int(t["id"]): t for t in plan.get("tasks") or []}
        existing_pairs = {
            (int(i["resource_id"]), str(i.get("interaction_type") or ""))
            for i in existing_interactions
        }
        submitted_pairs = {
            (int(r["resource_id"]), str(r["interaction_type"])) for r in interaction_rows
        }
        for item in request.task_updates:
            task = plan_tasks_by_id.get(int(item.task_id))
            if task is None or task.get("resource_id") is None:
                continue
            if (
                item.update.duration_minutes is None
                and item.update.effectiveness_rating is None
                and item.update.completion_percent is None
            ):
                continue
            resource_id = int(task["resource_id"])
            interaction_type = "task_completion"
            pair = (resource_id, interaction_type)
            if pair in existing_pairs or pair in submitted_pairs:
                continue
            interaction_rows.append(
                {
                    "resource_id": resource_id,
                    "interaction_type": interaction_type,
                    "completion_percent": float(item.update.completion_percent or 0),
                    "effectiveness_rating": item.update.effectiveness_rating,
                    "duration_minutes": item.update.duration_minutes,
                }
            )
            submitted_pairs.add(pair)

        mood_after = _enum_value(request.mood_after) if request.mood_after else ""
        plan_status = _plan_status_from_completion(
            request.completion_status,
            plan_completion,
        )

        try:
            if self._persist_reflection is not None:
                bundle = self._persist_reflection(
                    user_id=user_id,
                    daily_plan_id=request.daily_plan_id,
                    completion_status=_enum_value(request.completion_status),
                    learning_summary=request.learning_summary,
                    focus_rating=request.focus_rating,
                    resource_effectiveness=request.resource_effectiveness,
                    difficulty_feedback=_enum_value(request.difficulty_feedback),
                    mood_match=request.mood_match,
                    distractions=list(request.distractions),
                    wants_similar_resources=request.wants_similar_resources,
                    mood_after=mood_after,
                    insight=generation.insight.strip(),
                    task_updates=task_update_rows,
                    resource_interactions=interaction_rows,
                    plan_status=plan_status,
                    milestone_id=int(milestone["id"]) if milestone else None,
                    milestone_progress_percent=milestone_after if milestone else None,
                    milestone_status=milestone_status if milestone else None,
                    roadmap_id=int(roadmap["id"]) if roadmap else None,
                    roadmap_progress_percent=roadmap_after if roadmap else None,
                )
            else:
                bundle = create_reflection_bundle(
                    user_id=user_id,
                    daily_plan_id=request.daily_plan_id,
                    completion_status=_enum_value(request.completion_status),
                    learning_summary=request.learning_summary,
                    focus_rating=request.focus_rating,
                    resource_effectiveness=request.resource_effectiveness,
                    difficulty_feedback=_enum_value(request.difficulty_feedback),
                    mood_match=request.mood_match,
                    distractions=list(request.distractions),
                    wants_similar_resources=request.wants_similar_resources,
                    mood_after=mood_after,
                    insight=generation.insight.strip(),
                    task_updates=task_update_rows,
                    resource_interactions=interaction_rows,
                    plan_status=plan_status,
                    milestone_id=int(milestone["id"]) if milestone else None,
                    milestone_progress_percent=milestone_after if milestone else None,
                    milestone_status=milestone_status if milestone else None,
                    roadmap_id=int(roadmap["id"]) if roadmap else None,
                    roadmap_progress_percent=roadmap_after if roadmap else None,
                    db_path=self._db_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ReflectionAgent persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise ReflectionPersistenceError(
                f"Failed to persist reflection: {type(exc).__name__}"
            ) from None

        reflection_row = bundle["reflection"]
        reflection = _to_reflection_response(reflection_row)
        result = ReflectionAgentResult(
            reflection=reflection,
            plan_completion_percent=plan_completion,
            milestone_progress_before=milestone_before,
            milestone_progress_after=milestone_after if milestone else milestone_before,
            roadmap_progress_before=roadmap_before,
            roadmap_progress_after=roadmap_after if roadmap else roadmap_before,
            memory_ids=[],
            memories_complete=True,
            memory_error=None,
            reused_existing=False,
            created_at=reflection.created_at,
        )

        memory_records = self._build_memory_records(
            user_id=user_id,
            reflection_id=reflection.id,
            daily_plan_id=request.daily_plan_id,
            milestone_id=int(milestone["id"]) if milestone else None,
            generation=generation,
            request=request,
            plan_completion_percent=plan_completion,
        )
        try:
            memory_service = self._get_memory_service()
            memory_service.add_text_memories(memory_records)
            result = result.model_copy(
                update={
                    "memory_ids": [record.memory_id for record in memory_records],
                    "memories_complete": True,
                    "memory_error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            message = (
                "Reflection was saved, but semantic memory persistence failed: "
                f"{type(exc).__name__}"
            )
            logger.error(
                "ReflectionAgent memory persistence failed error_type=%s",
                type(exc).__name__,
            )
            partial = result.model_copy(
                update={
                    "memory_ids": [],
                    "memories_complete": False,
                    "memory_error": message,
                }
            )
            raise ReflectionMemoryError(message, result=partial) from None

        logger.info(
            "ReflectionAgent completed user_id=%s plan_id=%s reflection_id=%s",
            user_id,
            request.daily_plan_id,
            reflection.id,
        )
        return result
