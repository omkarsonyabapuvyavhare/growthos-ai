"""
LangGraph-compatible typed state for GrowthOS AI workflows.

State holds serializable domain payloads and orchestration markers only.
Never store API keys, embeddings, DB connections, or service clients here.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class OnboardingState(TypedDict, total=False):
    """State for Profile → Roadmap onboarding orchestration."""

    onboarding_request: Any
    user_id: int
    goal_id: int
    profile_result: Any
    roadmap_result: Any
    active_milestone: Any
    current_stage: str
    status: str
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


class DailyPlanningState(TypedDict, total=False):
    """State for check-in → planner → await completion."""

    user_id: int
    plan_date: Optional[str]
    checkin: Any
    refresh: bool
    planner_result: Any
    daily_plan_id: int
    roadmap_id: Optional[int]
    milestone_id: Optional[int]
    awaiting_user_completion: bool
    current_stage: str
    status: str
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


class DailyPostSessionState(TypedDict, total=False):
    """State for reflection → adaptation after the user completes work."""

    user_id: int
    reflection_request: Any
    daily_plan_id: int
    reflection_result: Any
    reflection_id: int
    adaptation_result: Any
    adaptation_insight_ids: list[int]
    adaptation_explanation: str
    goal_unchanged: bool
    current_stage: str
    status: str
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
