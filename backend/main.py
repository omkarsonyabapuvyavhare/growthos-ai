"""
GrowthOS AI — FastAPI application entry point.

Phase D exposes thin REST routes over completed agents and LangGraph workflows.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import AppServices, build_default_services
from api.errors import register_exception_handlers
from api.routers import router
from config import Settings, get_settings
from services.database import init_db


def _cors_origins(settings: Settings) -> list[str]:
    """Support a single origin or comma-separated FRONTEND_ORIGIN values."""
    raw = (settings.frontend_origin or "").strip()
    if not raw:
        return ["http://localhost:3000"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_app(
    *,
    settings: Optional[Settings] = None,
    services: Optional[AppServices] = None,
) -> FastAPI:
    """
    Application factory.

    Tests may inject a temporary AppServices container with fake agents.
    """
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db_path = getattr(app.state, "db_path", None) or resolved_settings.resolve_sqlite_path()
        init_db(db_path)
        yield

    application = FastAPI(
        title="GrowthOS AI API",
        description=(
            "Agentic AI Growth Curator backend. "
            "REST routes orchestrate Profile, Roadmap, Planner, Reflection, "
            "and Adaptation agents via LangGraph workflows."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.services = services or build_default_services(resolved_settings)
    application.state.db_path = application.state.services.db_path

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(resolved_settings),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)
    application.include_router(router)
    return application


app = create_app()
