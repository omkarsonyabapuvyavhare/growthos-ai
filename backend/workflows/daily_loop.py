"""
Daily learning-loop LangGraph workflows.

Planning stage:
  check-in → Planner (includes Curator) → await user completion → END

Post-session stage:
  Reflection → Adaptation → END

These are separate entry points because the user completes work outside the process.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from exceptions import (
    AdaptationMemoryError,
    ReflectionMemoryError,
    WorkflowExecutionError,
)
from models import (
    DailyCheckInRequest,
    DailyPlanningWorkflowResult,
    DailyPostSessionWorkflowResult,
    ReflectionRequest,
)
from workflows.state import DailyPlanningState, DailyPostSessionState


class SupportsPlannerAgent(Protocol):
    def create_daily_plan(
        self,
        user_id: int,
        *,
        checkin: DailyCheckInRequest,
        plan_date: date | None = None,
        refresh: bool = False,
    ) -> Any: ...


class SupportsReflectionAgent(Protocol):
    def reflect_on_plan(self, user_id: int, request: ReflectionRequest) -> Any: ...


class SupportsAdaptationAgent(Protocol):
    def adapt_from_reflection(
        self,
        user_id: int,
        reflection_id: int,
        *,
        force: bool = False,
    ) -> Any: ...


def build_planning_graph(planner_agent: SupportsPlannerAgent) -> Any:
    """Compile planning graph. Planner already invokes Curator internally."""

    def plan_node(state: DailyPlanningState) -> dict[str, Any]:
        if state.get("planner_result") is not None:
            return {
                "current_stage": "planning_complete",
                "completed_steps": [],
            }

        user_id = state.get("user_id")
        checkin = state.get("checkin")
        if not user_id or checkin is None:
            raise WorkflowExecutionError(
                "Planning requires user_id and checkin",
                stage="planning",
                original_type="ValueError",
            )
        if not isinstance(checkin, DailyCheckInRequest):
            checkin = DailyCheckInRequest.model_validate(checkin)

        plan_date = None
        if state.get("plan_date"):
            plan_date = date.fromisoformat(str(state["plan_date"]))

        try:
            result = planner_agent.create_daily_plan(
                int(user_id),
                checkin=checkin,
                plan_date=plan_date,
                refresh=bool(state.get("refresh", False)),
            )
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Planning stage failed: {type(exc).__name__}",
                stage="planning",
                original_type=type(exc).__name__,
            ) from None

        return {
            "planner_result": result,
            "daily_plan_id": result.plan.id,
            "roadmap_id": result.plan.roadmap_id,
            "milestone_id": result.plan.milestone_id,
            "current_stage": "planning_complete",
            "completed_steps": ["plan"],
            "status": "planning_complete",
        }

    def await_user_completion_node(state: DailyPlanningState) -> dict[str, Any]:
        return {
            "awaiting_user_completion": True,
            "current_stage": "awaiting_user_completion",
            "status": "awaiting_user_completion",
            "completed_steps": ["await_user_completion"],
        }

    graph = StateGraph(DailyPlanningState)
    graph.add_node("plan", plan_node)
    graph.add_node("await_user_completion", await_user_completion_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "await_user_completion")
    graph.add_edge("await_user_completion", END)
    return graph.compile()


def build_post_session_graph(
    reflection_agent: SupportsReflectionAgent,
    adaptation_agent: SupportsAdaptationAgent,
) -> Any:
    """Compile post-session Reflection → Adaptation graph."""

    def reflection_node(state: DailyPostSessionState) -> dict[str, Any]:
        if state.get("reflection_result") is not None and state.get("reflection_id"):
            return {
                "current_stage": "reflection_complete",
                "completed_steps": [],
            }

        user_id = state.get("user_id")
        request = state.get("reflection_request")
        if not user_id or request is None:
            raise WorkflowExecutionError(
                "Post-session requires user_id and reflection_request",
                stage="reflecting",
                original_type="ValueError",
            )
        if not isinstance(request, ReflectionRequest):
            request = ReflectionRequest.model_validate(request)

        try:
            result = reflection_agent.reflect_on_plan(int(user_id), request)
        except ReflectionMemoryError as exc:
            if exc.result is None:
                raise WorkflowExecutionError(
                    "Reflection memory failed without recoverable result",
                    stage="reflecting",
                    original_type=type(exc).__name__,
                ) from None
            result = exc.result
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Reflection stage failed: {type(exc).__name__}",
                stage="reflecting",
                original_type=type(exc).__name__,
            ) from None

        return {
            "reflection_result": result,
            "reflection_id": result.reflection.id,
            "daily_plan_id": result.reflection.daily_plan_id,
            "current_stage": "reflection_complete",
            "completed_steps": ["reflection"],
            "status": "reflecting",
        }

    def adaptation_node(state: DailyPostSessionState) -> dict[str, Any]:
        if state.get("adaptation_result") is not None:
            return {
                "current_stage": "completed",
                "status": "completed",
                "completed_steps": [],
            }

        user_id = state.get("user_id")
        reflection_id = state.get("reflection_id")
        reflection_result = state.get("reflection_result")
        if not user_id or not reflection_id:
            raise WorkflowExecutionError(
                "Adaptation stage requires reflection_id",
                stage="adapting",
                original_type="ValueError",
                partial_result=reflection_result,
            )

        try:
            result = adaptation_agent.adapt_from_reflection(
                int(user_id),
                int(reflection_id),
                force=False,
            )
        except AdaptationMemoryError as exc:
            if exc.result is None:
                raise WorkflowExecutionError(
                    "Adaptation memory failed without recoverable result",
                    stage="adapting",
                    original_type=type(exc).__name__,
                    partial_result=reflection_result,
                ) from None
            result = exc.result
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Adaptation stage failed: {type(exc).__name__}",
                stage="adapting",
                original_type=type(exc).__name__,
                partial_result=reflection_result,
            ) from None

        insight_ids = [int(item.id) for item in (result.insights or [])]
        return {
            "adaptation_result": result,
            "adaptation_insight_ids": insight_ids,
            "adaptation_explanation": result.adaptation_explanation,
            "goal_unchanged": bool(result.goal_unchanged),
            "current_stage": "completed",
            "status": "completed",
            "completed_steps": ["adaptation"],
        }

    graph = StateGraph(DailyPostSessionState)
    graph.add_node("reflection", reflection_node)
    graph.add_node("adaptation", adaptation_node)
    graph.add_edge(START, "reflection")
    graph.add_edge("reflection", "adaptation")
    graph.add_edge("adaptation", END)
    return graph.compile()


class DailyLoopWorkflow:
    """
    Daily loop orchestration with two explicit entry points.

    run_planning(...) stops at awaiting_user_completion.
    run_post_session(...) resumes after the user finishes tasks.
    """

    def __init__(
        self,
        *,
        planner_agent: SupportsPlannerAgent,
        reflection_agent: SupportsReflectionAgent,
        adaptation_agent: SupportsAdaptationAgent,
        planning_graph: Any | None = None,
        post_session_graph: Any | None = None,
    ) -> None:
        self._planner_agent = planner_agent
        self._reflection_agent = reflection_agent
        self._adaptation_agent = adaptation_agent
        self._planning_graph = planning_graph or build_planning_graph(planner_agent)
        self._post_session_graph = post_session_graph or build_post_session_graph(
            reflection_agent,
            adaptation_agent,
        )

    @property
    def planning_graph(self) -> Any:
        return self._planning_graph

    @property
    def post_session_graph(self) -> Any:
        return self._post_session_graph

    def run_planning(
        self,
        user_id: int,
        checkin: DailyCheckInRequest,
        *,
        plan_date: date | None = None,
        refresh: bool = False,
    ) -> DailyPlanningWorkflowResult:
        try:
            final_state = self._planning_graph.invoke(
                {
                    "user_id": int(user_id),
                    "checkin": checkin,
                    "plan_date": plan_date.isoformat() if plan_date else None,
                    "refresh": refresh,
                    "completed_steps": [],
                    "errors": [],
                    "awaiting_user_completion": False,
                    "current_stage": "started",
                    "status": "started",
                }
            )
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Planning workflow failed: {type(exc).__name__}",
                stage="planning",
                original_type=type(exc).__name__,
            ) from None

        planner_result = final_state.get("planner_result")
        if planner_result is None:
            raise WorkflowExecutionError(
                "Planning workflow ended without a plan",
                stage=str(final_state.get("current_stage") or "planning"),
                original_type="RuntimeError",
            )

        return DailyPlanningWorkflowResult(
            user_id=int(user_id),
            plan=planner_result.plan,
            tasks=list(planner_result.plan.tasks),
            checkin=planner_result.checkin,
            planner_result=planner_result,
            completed_steps=list(final_state.get("completed_steps") or []),
            status=str(final_state.get("status") or "awaiting_user_completion"),
            current_stage=str(
                final_state.get("current_stage") or "awaiting_user_completion"
            ),
            awaiting_user_completion=bool(
                final_state.get("awaiting_user_completion", True)
            ),
        )

    def run_post_session(
        self,
        user_id: int,
        reflection: ReflectionRequest,
    ) -> DailyPostSessionWorkflowResult:
        try:
            final_state = self._post_session_graph.invoke(
                {
                    "user_id": int(user_id),
                    "reflection_request": reflection,
                    "daily_plan_id": int(reflection.daily_plan_id),
                    "completed_steps": [],
                    "errors": [],
                    "current_stage": "started",
                    "status": "started",
                }
            )
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Post-session workflow failed: {type(exc).__name__}",
                stage="post_session",
                original_type=type(exc).__name__,
            ) from None

        reflection_result = final_state.get("reflection_result")
        adaptation_result = final_state.get("adaptation_result")
        if reflection_result is None or adaptation_result is None:
            raise WorkflowExecutionError(
                "Post-session workflow ended without reflection and adaptation",
                stage=str(final_state.get("current_stage") or "post_session"),
                original_type="RuntimeError",
                partial_result=reflection_result,
            )

        return DailyPostSessionWorkflowResult(
            reflection=reflection_result.reflection,
            adaptation=adaptation_result,
            adaptation_explanation=str(
                final_state.get("adaptation_explanation")
                or adaptation_result.adaptation_explanation
            ),
            reflection_result=reflection_result,
            completed_steps=list(final_state.get("completed_steps") or []),
            status=str(final_state.get("status") or "completed"),
            current_stage=str(final_state.get("current_stage") or "completed"),
            goal_unchanged=bool(final_state.get("goal_unchanged", True)),
        )


def build_daily_loop_workflow(
    planner_agent: SupportsPlannerAgent,
    reflection_agent: SupportsReflectionAgent,
    adaptation_agent: SupportsAdaptationAgent,
) -> DailyLoopWorkflow:
    """Factory for an injectable daily-loop workflow."""
    return DailyLoopWorkflow(
        planner_agent=planner_agent,
        reflection_agent=reflection_agent,
        adaptation_agent=adaptation_agent,
    )
