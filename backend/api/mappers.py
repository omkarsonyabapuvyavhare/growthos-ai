"""Map SQLite/agent dict payloads to API response models."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from models import (
    ActivityType,
    AdaptationInsightResponse,
    CurrentLevel,
    DailyCheckInResponse,
    DailyPlanResponse,
    DailyTaskResponse,
    DashboardResponse,
    Difficulty,
    DifficultyFeedback,
    EnergyLevel,
    GoalResponse,
    GoalStatus,
    LearningStyle,
    MilestoneResponse,
    MilestoneStatus,
    Mood,
    PhaseStatus,
    PlanStatus,
    PreferredLearningTime,
    ReflectionResponse,
    CompletionStatus,
    RoadmapPhaseResponse,
    RoadmapResponse,
    RoadmapStatus,
    TaskStatus,
    UserProfileResponse,
    UserResponse,
)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_user(row: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=int(row["id"]),
        display_name=str(row["display_name"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_profile(row: dict[str, Any]) -> UserProfileResponse:
    return UserProfileResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        aspiration=str(row["aspiration"]),
        motivation=str(row["motivation"]),
        current_level=CurrentLevel(str(row["current_level"])),
        target_outcome=str(row["target_outcome"]),
        learning_style=LearningStyle(str(row["learning_style"])),
        preferred_formats=list(row.get("preferred_formats") or []),
        daily_available_minutes=int(row["daily_available_minutes"]),
        preferred_session_minutes=int(row["preferred_session_minutes"]),
        attention_span_minutes=int(row["attention_span_minutes"]),
        preferred_learning_time=PreferredLearningTime(str(row["preferred_learning_time"])),
        habits=list(row.get("habits") or []),
        distractions=list(row.get("distractions") or []),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_goal(row: dict[str, Any]) -> GoalResponse:
    return GoalResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        status=GoalStatus(str(row.get("status") or "active")),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_milestone(row: dict[str, Any]) -> MilestoneResponse:
    return MilestoneResponse(
        id=int(row["id"]),
        phase_id=int(row["phase_id"]),
        sequence_number=int(row["sequence_number"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        skills=list(row.get("skills") or []),
        suggested_activities=list(row.get("suggested_activities") or []),
        completion_criteria=str(row.get("completion_criteria") or ""),
        estimated_sessions=int(row.get("estimated_sessions") or 1),
        estimated_minutes=int(row.get("estimated_minutes") or 30),
        difficulty=Difficulty(str(row.get("difficulty") or "beginner")),
        status=MilestoneStatus(str(row.get("status") or "not_started")),
        progress_percent=float(row.get("progress_percent") or 0),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_phase(row: dict[str, Any]) -> RoadmapPhaseResponse:
    return RoadmapPhaseResponse(
        id=int(row["id"]),
        roadmap_id=int(row["roadmap_id"]),
        sequence_number=int(row["sequence_number"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        expected_outcome=str(row.get("expected_outcome") or ""),
        status=PhaseStatus(str(row.get("status") or "not_started")),
        milestones=[to_milestone(item) for item in row.get("milestones") or []],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_roadmap(details: dict[str, Any]) -> RoadmapResponse:
    active = details.get("active_milestone")
    return RoadmapResponse(
        id=int(details["id"]),
        user_id=int(details["user_id"]),
        goal_id=int(details["goal_id"]),
        title=str(details["title"]),
        summary=str(details.get("summary") or ""),
        estimated_duration_weeks=int(details.get("estimated_duration_weeks") or 1),
        progress_percent=float(details.get("progress_percent") or 0),
        status=RoadmapStatus(str(details.get("status") or "active")),
        pacing_rationale=str(details.get("pacing_rationale") or ""),
        personalization_rationale=str(details.get("personalization_rationale") or ""),
        phases=[to_phase(item) for item in details.get("phases") or []],
        current_active_milestone_id=int(active["id"]) if active else None,
        created_at=_parse_dt(details["created_at"]),
        updated_at=_parse_dt(details["updated_at"]),
    )


def to_checkin(row: dict[str, Any]) -> DailyCheckInResponse:
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


def to_task(row: dict[str, Any]) -> DailyTaskResponse:
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


def to_plan(row: dict[str, Any]) -> DailyPlanResponse:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
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
        tasks=[to_task(item) for item in row.get("tasks") or []],
        guidance_tone=str(row.get("guidance_tone") or meta.get("guidance_tone") or ""),
        mood_influence_summary=str(
            row.get("mood_influence_summary") or meta.get("mood_influence_summary") or ""
        ),
        adaptation_explanation=str(
            row.get("adaptation_explanation") or meta.get("adaptation_explanation") or ""
        ),
        task_count_rationale=str(
            row.get("task_count_rationale") or meta.get("task_count_rationale") or ""
        ),
        metadata=dict(meta or {}),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_reflection(row: dict[str, Any]) -> ReflectionResponse:
    return ReflectionResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        daily_plan_id=int(row["daily_plan_id"]),
        completion_status=CompletionStatus(str(row["completion_status"])),
        learning_summary=str(row.get("learning_summary") or ""),
        focus_rating=int(row["focus_rating"]),
        resource_effectiveness=int(row["resource_effectiveness"]),
        difficulty_feedback=DifficultyFeedback(str(row["difficulty_feedback"])),
        mood_match=bool(row.get("mood_match")),
        distractions=list(row.get("distractions") or []),
        wants_similar_resources=bool(row.get("wants_similar_resources")),
        mood_after=str(row.get("mood_after") or ""),
        insight=str(row.get("insight") or "") or None,
        created_at=_parse_dt(row["created_at"]),
    )


def to_adaptation_insight(row: dict[str, Any]) -> AdaptationInsightResponse:
    return AdaptationInsightResponse(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        insight_type=str(row["insight_type"]),
        insight=str(row["insight"]),
        confidence_score=float(row.get("confidence_score") or 0),
        evidence=list(row.get("evidence") or []),
        is_active=bool(row.get("is_active", True)),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def to_dashboard(snapshot: dict[str, Any]) -> DashboardResponse:
    mood_raw = snapshot.get("today_mood")
    today_mood: Optional[Mood] = Mood(str(mood_raw)) if mood_raw else None
    plan_row = snapshot.get("today_plan")
    roadmap_row = snapshot.get("current_roadmap")
    milestone_row = snapshot.get("current_milestone")
    goal_row = snapshot.get("active_goal")
    return DashboardResponse(
        user=to_user(snapshot["user"]),
        active_goal=to_goal(goal_row) if goal_row else None,
        current_roadmap=to_roadmap(roadmap_row) if roadmap_row else None,
        current_milestone=to_milestone(milestone_row) if milestone_row else None,
        overall_progress_percent=float(snapshot.get("overall_progress_percent") or 0),
        today_mood=today_mood,
        today_plan=to_plan(plan_row) if plan_row else None,
        completion_streak=int(snapshot.get("completion_streak") or 0),
        recent_reflections=[
            to_reflection(item) for item in snapshot.get("recent_reflections") or []
        ],
        preferred_content_type=snapshot.get("preferred_content_type"),
        preferred_session_minutes=snapshot.get("preferred_session_minutes"),
        average_session_minutes=snapshot.get("average_session_minutes"),
        resource_effectiveness_avg=snapshot.get("resource_effectiveness_avg"),
        weekly_learning_consistency=snapshot.get("weekly_learning_consistency"),
        skill_growth=list(snapshot.get("skill_growth") or []),
        detected_patterns=list(snapshot.get("detected_patterns") or []),
        growthos_knows_you=list(snapshot.get("growthos_knows_you") or []),
        plan_change_explanation=snapshot.get("plan_change_explanation"),
        ai_insight=snapshot.get("ai_insight"),
        recommended_next_action=snapshot.get("recommended_next_action"),
        adaptation_insights=[
            to_adaptation_insight(item)
            for item in snapshot.get("adaptation_insights") or []
        ],
        completed_sessions=int(snapshot.get("completed_sessions") or 0),
    )
