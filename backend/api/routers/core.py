"""
GrowthOS AI REST routes.

Thin wrappers over workflows/agents/database helpers — no duplicated business rules.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Path, status

from api.deps import AppServices, get_services
from api.mappers import (
    to_checkin,
    to_dashboard,
    to_plan,
    to_roadmap,
    to_task,
)
from config import get_settings
from exceptions import (
    AdaptationMemoryError,
    ProfileMemoryError,
    ReflectionMemoryError,
    RoadmapMemoryError,
    WorkflowExecutionError,
)
from models import (
    ActivityType,
    AdaptationAgentResult,
    AdaptationRunRequest,
    DailyCheckInRequest,
    DailyCheckInResponse,
    DailyPlanCreateRequest,
    DailyPlanResponse,
    DailyPlanningWorkflowResult,
    DailyPostSessionWorkflowResult,
    DailyTaskResponse,
    DashboardResponse,
    DemoDayLoopRequest,
    DemoDayLoopResponse,
    EnergyLevel,
    HealthResponse,
    Mood,
    OnboardingRequest,
    OnboardingWorkflowResult,
    ReflectionRequest,
    RoadmapAgentResult,
    RoadmapCreateRequest,
    RoadmapResponse,
    TaskCompletionRequest,
)
from services.database import (
    build_dashboard_snapshot,
    create_daily_checkin,
    create_resource_interaction,
    delete_daily_plan,
    get_active_goal_for_user,
    get_active_roadmap_for_user,
    get_daily_plan_by_date,
    get_task_by_id,
    get_user_by_id,
    update_task_status,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Return service health for local development and demos.",
    tags=["system"],
)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service="GrowthOS AI API",
        youtube_enabled=bool(settings.youtube_api_enabled),
        youtube_configured=settings.is_youtube_configured(),
    )


@router.post(
    "/onboarding",
    response_model=OnboardingWorkflowResult,
    status_code=status.HTTP_201_CREATED,
    summary="Onboard a learner",
    description="Run Profile → Roadmap onboarding workflow for a free-text learning goal.",
    tags=["onboarding"],
)
def onboard_user(
    request: OnboardingRequest,
    services: AppServices = Depends(get_services),
) -> OnboardingWorkflowResult:
    try:
        return services.onboarding_workflow.run(request)
    except (ProfileMemoryError, RoadmapMemoryError) as exc:
        # Workflow already recovers these; if they surface, prefer attached result.
        if getattr(exc, "result", None) is not None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Onboarding persisted but semantic memory failed",
            ) from None
        raise


@router.post(
    "/users/{user_id}/roadmaps",
    response_model=RoadmapAgentResult,
    status_code=status.HTTP_201_CREATED,
    summary="Generate or regenerate a roadmap",
    description="Create a personalized roadmap for the user's active (or specified) goal.",
    tags=["roadmaps"],
)
def create_roadmap(
    body: RoadmapCreateRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> RoadmapAgentResult:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    goal_id = body.goal_id
    if goal_id is None:
        goal = get_active_goal_for_user(user_id, db_path=services.db_path)
        if goal is None:
            raise HTTPException(status_code=404, detail="Active goal not found")
        goal_id = int(goal["id"])
    try:
        return services.roadmap_agent.generate_roadmap(
            user_id,
            goal_id,
            regenerate=body.regenerate,
        )
    except RoadmapMemoryError as exc:
        if exc.result is not None:
            return exc.result  # type: ignore[return-value]
        raise


@router.get(
    "/users/{user_id}/roadmap",
    response_model=RoadmapResponse,
    summary="Get active roadmap",
    description="Return the user's active roadmap with phases and milestones.",
    tags=["roadmaps"],
)
def get_roadmap(
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> RoadmapResponse:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    roadmap = get_active_roadmap_for_user(user_id, db_path=services.db_path)
    if roadmap is None:
        raise HTTPException(status_code=404, detail="Active roadmap not found")
    return to_roadmap(roadmap)


@router.post(
    "/users/{user_id}/checkins",
    response_model=DailyCheckInResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a daily check-in",
    description="Persist today's mood, energy, focus, and available time.",
    tags=["daily-loop"],
)
def create_checkin(
    body: DailyCheckInRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DailyCheckInResponse:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    row = create_daily_checkin(
        user_id,
        mood=body.mood.value,
        energy_level=body.energy_level.value,
        focus_level=body.focus_level,
        available_minutes=body.available_minutes,
        preferred_activity=body.preferred_activity.value,
        notes=body.notes,
        db_path=services.db_path,
    )
    return to_checkin(row)


@router.post(
    "/users/{user_id}/daily-plans",
    response_model=DailyPlanningWorkflowResult,
    status_code=status.HTTP_201_CREATED,
    summary="Create a daily plan",
    description=(
        "Run the planning workflow (Planner invokes Curator). "
        "Stops at awaiting_user_completion."
    ),
    tags=["daily-loop"],
)
def create_daily_plan(
    body: DailyPlanCreateRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DailyPlanningWorkflowResult:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    checkin = DailyCheckInRequest(
        mood=body.mood,
        energy_level=body.energy_level,
        focus_level=body.focus_level,
        available_minutes=body.available_minutes,
        preferred_activity=body.preferred_activity,
        notes=body.notes,
    )
    return services.daily_loop_workflow.run_planning(
        user_id,
        checkin,
        plan_date=body.plan_date,
        refresh=body.refresh,
    )


@router.get(
    "/users/{user_id}/daily-plans/today",
    response_model=DailyPlanResponse,
    summary="Get today's daily plan",
    description="Return the active plan for today (UTC date) when present.",
    tags=["daily-loop"],
)
def get_today_plan(
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DailyPlanResponse:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    today = date.today().isoformat()
    plan = get_daily_plan_by_date(user_id, today, db_path=services.db_path)
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan for today")
    return to_plan(plan)


@router.patch(
    "/users/{user_id}/tasks/{task_id}",
    response_model=DailyTaskResponse,
    summary="Update a task",
    description="Update task completion status for a task owned by the user.",
    tags=["daily-loop"],
)
def patch_task(
    body: TaskCompletionRequest,
    user_id: int = Path(..., gt=0),
    task_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DailyTaskResponse:
    task = get_task_by_id(task_id, db_path=services.db_path)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if int(task["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Task does not belong to user")
    updated = update_task_status(
        task_id,
        body.status.value,
        db_path=services.db_path,
    )
    if (
        updated.get("resource_id") is not None
        and (
            body.completion_percent is not None
            or body.duration_minutes is not None
            or body.effectiveness_rating is not None
        )
    ):
        create_resource_interaction(
            user_id=user_id,
            resource_id=int(updated["resource_id"]),
            interaction_type="task_update",
            daily_plan_id=int(updated["daily_plan_id"]),
            completion_percent=float(body.completion_percent or 0),
            effectiveness_rating=body.effectiveness_rating,
            duration_minutes=body.duration_minutes,
            db_path=services.db_path,
        )
    # Reload with display fields
    reloaded = get_task_by_id(task_id, db_path=services.db_path)
    assert reloaded is not None
    return to_task(reloaded)


@router.post(
    "/users/{user_id}/reflections",
    response_model=DailyPostSessionWorkflowResult,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a session reflection",
    description=(
        "Run the post-session workflow: reflection then adaptation. "
        "Returns insight and why the next plan will change."
    ),
    tags=["daily-loop"],
)
def create_reflection(
    body: ReflectionRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DailyPostSessionWorkflowResult:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        return services.daily_loop_workflow.run_post_session(user_id, body)
    except (ReflectionMemoryError, AdaptationMemoryError) as exc:
        # Workflow recovers attached results when possible; surface safe message otherwise.
        if getattr(exc, "result", None) is not None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session saved but semantic memory failed",
            ) from None
        raise


@router.post(
    "/users/{user_id}/adaptations/run",
    response_model=AdaptationAgentResult,
    status_code=status.HTTP_201_CREATED,
    summary="Run adaptation from a reflection",
    description="Derive next-plan adjustments and preference insights from a reflection.",
    tags=["daily-loop"],
)
def run_adaptation(
    body: AdaptationRunRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> AdaptationAgentResult:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        return services.adaptation_agent.adapt_from_reflection(
            user_id,
            body.reflection_id,
            force=body.force,
        )
    except AdaptationMemoryError as exc:
        if exc.result is not None:
            return exc.result  # type: ignore[return-value]
        raise


@router.get(
    "/users/{user_id}/dashboard",
    response_model=DashboardResponse,
    summary="Get learner dashboard",
    description="Assemble progress, today plan, reflections, and adaptation insights.",
    tags=["dashboard"],
)
def get_dashboard(
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DashboardResponse:
    snapshot = build_dashboard_snapshot(user_id, db_path=services.db_path)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="User not found")
    return to_dashboard(snapshot)


@router.post(
    "/users/{user_id}/demo/day-loop",
    response_model=DemoDayLoopResponse,
    summary="Run Day 1 → Day 2 demo loop",
    description=(
        "Demonstrate planning, reflection, adaptation, and a next-day plan "
        "for an onboarded user. Uses workflows; does not invent progress."
    ),
    tags=["demo"],
)
def run_demo_day_loop(
    body: DemoDayLoopRequest,
    user_id: int = Path(..., gt=0),
    services: AppServices = Depends(get_services),
) -> DemoDayLoopResponse:
    if get_user_by_id(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="User not found")
    goal = get_active_goal_for_user(user_id, db_path=services.db_path)
    if goal is None:
        raise HTTPException(status_code=404, detail="Active goal not found")
    if get_active_roadmap_for_user(user_id, db_path=services.db_path) is None:
        raise HTTPException(status_code=404, detail="Active roadmap not found")

    day1_date = date.today()
    day2_date = day1_date + timedelta(days=1)
    day1_checkin = body.day1_checkin or DailyCheckInRequest(
        mood=Mood.tired,
        energy_level=EnergyLevel.low,
        focus_level=2,
        available_minutes=15,
        preferred_activity=ActivityType.watch,
        notes="Demo day 1 tired session",
    )
    day2_checkin = body.day2_checkin or DailyCheckInRequest(
        mood=Mood.focused,
        energy_level=EnergyLevel.high,
        focus_level=5,
        available_minutes=30,
        preferred_activity=ActivityType.practice,
        notes="Demo day 2 focused session",
    )

    day1 = services.daily_loop_workflow.run_planning(
        user_id,
        day1_checkin,
        plan_date=day1_date,
        refresh=True,
    )

    if body.reflection is not None:
        reflection_request = body.reflection.model_copy(
            update={"daily_plan_id": day1.plan.id}
        )
    else:
        from models import (
            CompletionStatus,
            DifficultyFeedback,
            ReflectionTaskUpdate,
            TaskCompletionRequest,
            TaskStatus,
        )

        updates = []
        for task in day1.tasks:
            activity = str(task.activity_type.value if hasattr(task.activity_type, "value") else task.activity_type)
            is_practice = activity == ActivityType.practice.value
            # Judge story: longer/passive resource less useful; practical task useful.
            if is_practice:
                updates.append(
                    ReflectionTaskUpdate(
                        task_id=task.id,
                        update=TaskCompletionRequest(
                            status=TaskStatus.completed,
                            completion_percent=100,
                            duration_minutes=max(3, min(task.estimated_minutes, 8)),
                            effectiveness_rating=5,
                            notes="Practical task felt useful",
                        ),
                    )
                )
            else:
                updates.append(
                    ReflectionTaskUpdate(
                        task_id=task.id,
                        update=TaskCompletionRequest(
                            status=TaskStatus.completed,
                            completion_percent=60,
                            duration_minutes=max(3, min(task.estimated_minutes, 12)),
                            effectiveness_rating=2,
                            notes="Longer resource drained focus",
                        ),
                    )
                )
        if not updates and day1.tasks:
            first = day1.tasks[0]
            updates.append(
                ReflectionTaskUpdate(
                    task_id=first.id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.completed,
                        completion_percent=70,
                        duration_minutes=min(12, max(3, first.estimated_minutes)),
                        effectiveness_rating=2,
                    ),
                )
            )
        reflection_request = ReflectionRequest(
            daily_plan_id=day1.plan.id,
            completion_status=CompletionStatus.partial,
            learning_summary=(
                "Practice helped. The longer resource drained focus and felt less useful."
            ),
            focus_rating=2,
            resource_effectiveness=2,
            difficulty_feedback=DifficultyFeedback.suitable,
            mood_match=False,
            distractions=["phone"],
            wants_similar_resources=False,
            mood_after=Mood.tired,
            task_updates=updates,
            actual_minutes_spent=14,
        )

    post = services.daily_loop_workflow.run_post_session(user_id, reflection_request)

    # Day 2 planning uses the Planner Agent's single structured-output retry.
    # On final failure, preserve Day 1 reflection/adaptation and remove any
    # partial Day 2 plan rows (do not invent a fake plan).
    try:
        day2 = services.daily_loop_workflow.run_planning(
            user_id,
            day2_checkin,
            plan_date=day2_date,
            refresh=True,
        )
    except WorkflowExecutionError as exc:
        partial = get_daily_plan_by_date(
            user_id,
            day2_date.isoformat(),
            db_path=services.db_path,
        )
        if partial is not None:
            delete_daily_plan(int(partial["id"]), db_path=services.db_path)
        original = (exc.original_type or type(exc).__name__).strip()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Demo Day 2 planning failed after a constrained retry "
                f"({original}). Day 1 reflection and adaptation were preserved."
            ),
        ) from None
    except Exception as exc:  # noqa: BLE001
        partial = get_daily_plan_by_date(
            user_id,
            day2_date.isoformat(),
            db_path=services.db_path,
        )
        if partial is not None:
            delete_daily_plan(int(partial["id"]), db_path=services.db_path)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Demo Day 2 planning failed unexpectedly "
                f"({type(exc).__name__}). Day 1 reflection and adaptation were preserved."
            ),
        ) from None

    goal_after = get_active_goal_for_user(user_id, db_path=services.db_path)
    goal_unchanged = (
        goal_after is not None and str(goal_after.get("title")) == str(goal["title"])
    )
    patterns = list(post.adaptation.detected_patterns or [])
    if patterns:
        next_action = (
            "Use shorter resources and more practice next session - "
            "your Day 1 evidence already supports this."
        )
    elif post.adaptation_explanation:
        next_action = post.adaptation_explanation.strip()
    else:
        next_action = (
            "Check in tomorrow so GrowthOS can keep adapting your session shape."
        )
    return DemoDayLoopResponse(
        user_id=user_id,
        goal_title=str(goal["title"]),
        goal_unchanged=goal_unchanged and post.goal_unchanged,
        day1_checkin=day1.checkin,
        day1_plan=day1.plan,
        day1_tasks=list(day1.tasks),
        reflection=post.reflection,
        reflection_insight=post.reflection.insight,
        adaptation=post.adaptation,
        adaptation_explanation=post.adaptation_explanation,
        detected_patterns=list(post.adaptation.detected_patterns or []),
        is_early_signal=bool(post.adaptation.is_early_signal),
        day2_checkin=day2.checkin,
        day2_plan=day2.plan,
        day2_tasks=list(day2.tasks),
        recommended_next_action=next_action,
        completed_steps=[
            *day1.completed_steps,
            *post.completed_steps,
            *day2.completed_steps,
        ],
    )

