"""
Onboarding LangGraph workflow: Profile Agent → Roadmap Agent.

Orchestration only — agents own validation, Gemini, persistence, and memory.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from exceptions import (
    ProfileMemoryError,
    RoadmapMemoryError,
    WorkflowExecutionError,
)
from models import OnboardingRequest, OnboardingWorkflowResult, ProfileAgentResult
from workflows.state import OnboardingState


class SupportsProfileAgent(Protocol):
    def process_onboarding(self, request: OnboardingRequest) -> ProfileAgentResult: ...


class SupportsRoadmapAgent(Protocol):
    def generate_roadmap(
        self,
        user_id: int,
        goal_id: int,
        *,
        regenerate: bool = False,
    ) -> Any: ...


def build_onboarding_graph(
    profile_agent: SupportsProfileAgent,
    roadmap_agent: SupportsRoadmapAgent,
) -> Any:
    """Compile the onboarding StateGraph with injected agents."""

    def profile_node(state: OnboardingState) -> dict[str, Any]:
        # Idempotent skip if profile already produced in this run.
        if state.get("profile_result") is not None and state.get("user_id"):
            return {
                "current_stage": "profile_complete",
                "completed_steps": [],
            }

        request = state.get("onboarding_request")
        if request is None:
            raise WorkflowExecutionError(
                "Onboarding request is required",
                stage="profile",
                original_type="ValueError",
            )
        if not isinstance(request, OnboardingRequest):
            request = OnboardingRequest.model_validate(request)

        try:
            result = profile_agent.process_onboarding(request)
        except ProfileMemoryError as exc:
            if exc.result is None:
                raise WorkflowExecutionError(
                    "Profile memory failed without recoverable result",
                    stage="profile",
                    original_type=type(exc).__name__,
                ) from None
            result = exc.result  # type: ignore[assignment]
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Profile stage failed: {type(exc).__name__}",
                stage="profile",
                original_type=type(exc).__name__,
            ) from None

        return {
            "user_id": result.user.id,
            "goal_id": result.goal.id,
            "profile_result": result,
            "current_stage": "profile_complete",
            "completed_steps": ["profile"],
            "status": "profiling",
        }

    def roadmap_node(state: OnboardingState) -> dict[str, Any]:
        if state.get("roadmap_result") is not None:
            return {
                "current_stage": "roadmap_complete",
                "completed_steps": [],
            }

        user_id = state.get("user_id")
        goal_id = state.get("goal_id")
        if not user_id or not goal_id:
            raise WorkflowExecutionError(
                "Roadmap stage requires user_id and goal_id from profile",
                stage="roadmap",
                original_type="ValueError",
                partial_result=state.get("profile_result"),
            )

        try:
            result = roadmap_agent.generate_roadmap(
                int(user_id),
                int(goal_id),
                regenerate=False,
            )
        except RoadmapMemoryError as exc:
            if exc.result is None:
                raise WorkflowExecutionError(
                    "Roadmap memory failed without recoverable result",
                    stage="roadmap",
                    original_type=type(exc).__name__,
                    partial_result=state.get("profile_result"),
                ) from None
            result = exc.result
        except WorkflowExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkflowExecutionError(
                f"Roadmap stage failed: {type(exc).__name__}",
                stage="roadmap",
                original_type=type(exc).__name__,
                partial_result=state.get("profile_result"),
            ) from None

        return {
            "roadmap_result": result,
            "active_milestone": result.active_milestone,
            "current_stage": "roadmap_complete",
            "completed_steps": ["roadmap"],
            "status": "roadmap_ready",
        }

    def finish_node(state: OnboardingState) -> dict[str, Any]:
        return {
            "current_stage": "completed",
            "status": "completed",
            "completed_steps": ["finish"],
        }

    graph = StateGraph(OnboardingState)
    graph.add_node("profile", profile_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "roadmap")
    graph.add_edge("roadmap", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


class OnboardingWorkflow:
    """Service wrapper around the onboarding LangGraph."""

    def __init__(
        self,
        profile_agent: SupportsProfileAgent,
        roadmap_agent: SupportsRoadmapAgent,
        *,
        compiled_graph: Any | None = None,
    ) -> None:
        self._profile_agent = profile_agent
        self._roadmap_agent = roadmap_agent
        self._graph = compiled_graph or build_onboarding_graph(
            profile_agent,
            roadmap_agent,
        )

    @property
    def graph(self) -> Any:
        return self._graph

    def run(self, request: OnboardingRequest) -> OnboardingWorkflowResult:
        try:
            final_state = self._graph.invoke(
                {
                    "onboarding_request": request,
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
                f"Onboarding workflow failed: {type(exc).__name__}",
                stage="onboarding",
                original_type=type(exc).__name__,
            ) from None

        profile_result = final_state.get("profile_result")
        roadmap_result = final_state.get("roadmap_result")
        if profile_result is None or roadmap_result is None:
            raise WorkflowExecutionError(
                "Onboarding workflow ended without profile and roadmap results",
                stage=str(final_state.get("current_stage") or "unknown"),
                original_type="RuntimeError",
                partial_result=profile_result,
            )

        return OnboardingWorkflowResult(
            user=profile_result.user,
            profile=profile_result.profile,
            goal=profile_result.goal,
            roadmap=roadmap_result.roadmap,
            active_milestone=roadmap_result.active_milestone,
            profile_result=profile_result,
            roadmap_result=roadmap_result,
            completed_steps=list(final_state.get("completed_steps") or []),
            status=str(final_state.get("status") or "completed"),
            current_stage=str(final_state.get("current_stage") or "completed"),
        )


def build_onboarding_workflow(
    profile_agent: SupportsProfileAgent,
    roadmap_agent: SupportsRoadmapAgent,
) -> OnboardingWorkflow:
    """Factory for an injectable onboarding workflow."""
    return OnboardingWorkflow(profile_agent, roadmap_agent)
