"""Map GrowthOS domain exceptions to safe HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions import (
    AdaptationAgentError,
    AdaptationConflictError,
    AdaptationContextError,
    AdaptationEvidenceError,
    AdaptationGenerationError,
    AdaptationMemoryError,
    AdaptationOwnershipError,
    AdaptationPersistenceError,
    CuratorAgentError,
    CuratorContextError,
    CuratorOwnershipError,
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    GrowthOSError,
    PlannerAgentError,
    PlannerBudgetError,
    PlannerContextError,
    PlannerGenerationError,
    PlannerOwnershipError,
    PlannerPersistenceError,
    ProfileAgentError,
    ProfileMemoryError,
    ProfilePersistenceError,
    ReflectionAgentError,
    ReflectionConflictError,
    ReflectionContextError,
    ReflectionEvidenceError,
    ReflectionGenerationError,
    ReflectionMemoryError,
    ReflectionOwnershipError,
    ReflectionPersistenceError,
    RoadmapAgentError,
    RoadmapContextError,
    RoadmapMemoryError,
    RoadmapOwnershipError,
    RoadmapPersistenceError,
    WorkflowExecutionError,
)
from models import ErrorResponse


def _safe_detail(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    banned = ("api_key", "gemini_api_key", "authorization", "bearer ", "sqlite:///")
    if any(token in lowered for token in banned):
        return f"{type(exc).__name__} occurred"
    if len(text) > 300:
        return f"{type(exc).__name__} occurred"
    return text


def _status_for(exc: Exception) -> int:
    if isinstance(
        exc,
        (
            AdaptationOwnershipError,
            CuratorOwnershipError,
            PlannerOwnershipError,
            ReflectionOwnershipError,
            RoadmapOwnershipError,
        ),
    ):
        return 403
    if isinstance(
        exc,
        (
            AdaptationContextError,
            CuratorContextError,
            PlannerContextError,
            ReflectionContextError,
            RoadmapContextError,
        ),
    ):
        return 404
    if isinstance(exc, (AdaptationConflictError, ReflectionConflictError)):
        return 409
    if isinstance(
        exc,
        (
            AdaptationEvidenceError,
            PlannerBudgetError,
            ReflectionEvidenceError,
        ),
    ):
        return 422
    if isinstance(
        exc,
        (
            AdaptationGenerationError,
            GeminiConfigurationError,
            GeminiInvocationError,
            GeminiResponseError,
            PlannerGenerationError,
            ReflectionGenerationError,
        ),
    ):
        return 502
    if isinstance(
        exc,
        (
            AdaptationPersistenceError,
            PlannerPersistenceError,
            ProfilePersistenceError,
            ReflectionPersistenceError,
            RoadmapPersistenceError,
        ),
    ):
        return 500
    if isinstance(
        exc,
        (
            AdaptationMemoryError,
            ProfileMemoryError,
            ReflectionMemoryError,
            RoadmapMemoryError,
        ),
    ):
        # Recoverable when result is attached — caller may return 200 instead.
        return 503
    if isinstance(exc, WorkflowExecutionError):
        stage = (exc.stage or "").lower()
        if "own" in stage:
            return 403
        if stage in {"profile", "roadmap", "planning", "reflecting", "adapting"}:
            # Prefer mapping by original type name when available.
            original = (exc.original_type or "").lower()
            if "ownership" in original:
                return 403
            if "context" in original:
                return 404
            if "conflict" in original:
                return 409
            if "budget" in original or "evidence" in original:
                return 422
            if "gemini" in original or "generation" in original:
                return 502
            if "persistence" in original:
                return 500
            return 400
        return 400
    if isinstance(
        exc,
        (
            AdaptationAgentError,
            CuratorAgentError,
            PlannerAgentError,
            ProfileAgentError,
            ReflectionAgentError,
            RoadmapAgentError,
            GrowthOSError,
        ),
    ):
        return 400
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    """Attach domain exception handlers to the FastAPI app."""

    @app.exception_handler(WorkflowExecutionError)
    async def workflow_error_handler(
        _request: Request,
        exc: WorkflowExecutionError,
    ) -> JSONResponse:
        body = ErrorResponse(
            detail=_safe_detail(exc),
            stage=exc.stage,
            error_type=exc.original_type,
        )
        return JSONResponse(status_code=_status_for(exc), content=body.model_dump())

    @app.exception_handler(GrowthOSError)
    async def growthos_error_handler(
        _request: Request,
        exc: GrowthOSError,
    ) -> JSONResponse:
        # Memory errors with attached results are handled in route bodies when needed.
        body = ErrorResponse(
            detail=_safe_detail(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(status_code=_status_for(exc), content=body.model_dump())
