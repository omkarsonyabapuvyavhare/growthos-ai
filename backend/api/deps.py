"""
Application service container and FastAPI dependencies.

Agents/workflows are constructed once per app and injected for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Request

from agents.adaptation_agent import AdaptationAgent
from agents.curator_agent import CuratorAgent
from agents.planner_agent import DailyPlannerAgent
from agents.profile_agent import ProfileAgent
from agents.reflection_agent import ReflectionAgent
from agents.roadmap_agent import RoadmapAgent
from config import Settings, get_settings
from workflows.daily_loop import DailyLoopWorkflow
from workflows.onboarding import OnboardingWorkflow


@dataclass
class AppServices:
    """Injectable backend services used by API routers."""

    settings: Settings
    profile_agent: ProfileAgent
    roadmap_agent: RoadmapAgent
    curator_agent: CuratorAgent
    planner_agent: DailyPlannerAgent
    reflection_agent: ReflectionAgent
    adaptation_agent: AdaptationAgent
    onboarding_workflow: OnboardingWorkflow
    daily_loop_workflow: DailyLoopWorkflow
    db_path: Optional[Path] = None


def build_default_services(settings: Settings | None = None) -> AppServices:
    """Construct production-default agents and workflows."""
    resolved = settings or get_settings()
    db_path = resolved.resolve_sqlite_path()
    profile_agent = ProfileAgent(settings=resolved, db_path=db_path)
    roadmap_agent = RoadmapAgent(settings=resolved, db_path=db_path)
    curator_agent = CuratorAgent(settings=resolved, db_path=db_path)
    planner_agent = DailyPlannerAgent(
        settings=resolved,
        curator_agent=curator_agent,
        db_path=db_path,
    )
    reflection_agent = ReflectionAgent(settings=resolved, db_path=db_path)
    adaptation_agent = AdaptationAgent(settings=resolved, db_path=db_path)
    onboarding_workflow = OnboardingWorkflow(profile_agent, roadmap_agent)
    daily_loop_workflow = DailyLoopWorkflow(
        planner_agent=planner_agent,
        reflection_agent=reflection_agent,
        adaptation_agent=adaptation_agent,
    )
    return AppServices(
        settings=resolved,
        profile_agent=profile_agent,
        roadmap_agent=roadmap_agent,
        curator_agent=curator_agent,
        planner_agent=planner_agent,
        reflection_agent=reflection_agent,
        adaptation_agent=adaptation_agent,
        onboarding_workflow=onboarding_workflow,
        daily_loop_workflow=daily_loop_workflow,
        db_path=db_path,
    )


def get_services(request: Request) -> AppServices:
    """Resolve the app-scoped service container."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        services = build_default_services()
        request.app.state.services = services
    return services
