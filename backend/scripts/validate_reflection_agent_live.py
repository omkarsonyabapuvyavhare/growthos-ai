"""
Optional live Reflection Agent validation.

    .\\.venv\\Scripts\\python.exe scripts\\validate_reflection_agent_live.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.curator_agent import CuratorAgent  # noqa: E402
from agents.planner_agent import DailyPlannerAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.reflection_agent import ReflectionAgent  # noqa: E402
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from models import (  # noqa: E402
    ActivityType,
    CompletionStatus,
    CurrentLevel,
    DailyCheckInRequest,
    DifficultyFeedback,
    EnergyLevel,
    LearningStyle,
    Mood,
    OnboardingRequest,
    PreferredLearningTime,
    ReflectionRequest,
    ReflectionTaskUpdate,
    TaskCompletionRequest,
    TaskStatus,
)
from services.catalog_index import CatalogSemanticIndex  # noqa: E402
from services.database import get_milestone_by_id, init_db  # noqa: E402
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.gemini import GeminiService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402


def main() -> int:
    base = get_settings()
    if not base.is_gemini_configured():
        print("LIVE_REFLECTION_AGENT_SKIPPED: GEMINI_API_KEY is not configured")
        return 0

    with tempfile.TemporaryDirectory(prefix="growthos_reflection_live_") as tmp:
        root = Path(tmp)
        db_path = root / "live.db"
        init_db(db_path)
        settings = Settings(
            gemini_api_key=base.gemini_api_key,
            gemini_model=base.gemini_model,
            gemini_temperature=base.gemini_temperature,
            gemini_max_retries=base.gemini_max_retries,
            gemini_request_timeout_seconds=base.gemini_request_timeout_seconds,
            gemini_embedding_model=base.gemini_embedding_model,
            database_url=f"sqlite:///{db_path.as_posix()}",
            frontend_origin=base.frontend_origin,
            faiss_index_path=str(root / "faiss" / "index.faiss"),
            faiss_metadata_path=str(root / "faiss" / "metadata.json"),
            faiss_catalog_index_path=str(root / "catalog" / "index.faiss"),
            faiss_catalog_metadata_path=str(root / "catalog" / "metadata.json"),
            resource_catalog_path=str(BACKEND_ROOT / "data" / "sample_resources.json"),
        )
        user_store = FAISSVectorStore(
            settings=settings,
            index_path=settings.resolve_faiss_index_path(),
            metadata_path=settings.resolve_faiss_metadata_path(),
        )
        memory = SemanticMemoryService(settings=settings, vector_store=user_store)
        gemini = GeminiService(settings=settings)
        embedding = GeminiEmbeddingService(settings=settings)
        catalog_index = CatalogSemanticIndex(
            settings=settings,
            embedding_service=embedding,
            index_path=settings.resolve_faiss_catalog_index_path(),
            metadata_path=settings.resolve_faiss_catalog_metadata_path(),
        )
        profile_agent = ProfileAgent(
            settings=settings,
            gemini_service=gemini,
            memory_service=memory,
            db_path=db_path,
        )
        roadmap_agent = RoadmapAgent(
            settings=settings,
            gemini_service=gemini,
            memory_service=memory,
            db_path=db_path,
        )
        curator = CuratorAgent(
            settings=settings,
            gemini_service=gemini,
            catalog_index=catalog_index,
            catalog_path=settings.resolve_resource_catalog_path(),
            db_path=db_path,
        )
        planner = DailyPlannerAgent(
            settings=settings,
            gemini_service=gemini,
            curator_agent=curator,
            db_path=db_path,
        )
        reflection_agent = ReflectionAgent(
            settings=settings,
            gemini_service=gemini,
            memory_service=memory,
            db_path=db_path,
        )

        request = OnboardingRequest(
            display_name="Live Reflection User",
            learning_goal="Improve public speaking",
            aspiration="Become a calm presenter",
            motivation="Share ideas confidently",
            current_level=CurrentLevel.beginner,
            target_outcome="Deliver a 5-minute update",
            preferred_formats=["video", "practice"],
            learning_style=LearningStyle.mixed,
            daily_available_minutes=30,
            preferred_session_minutes=15,
            attention_span_minutes=10,
            preferred_learning_time=PreferredLearningTime.evening,
            habits=["evening review"],
            distractions=["phone notifications"],
        )

        try:
            profile_result = profile_agent.process_onboarding(request)
            roadmap_result = roadmap_agent.generate_roadmap(
                profile_result.user.id,
                profile_result.goal.id,
            )
            active = roadmap_result.active_milestone
            if active is None:
                print("LIVE_REFLECTION_AGENT_FAILED: no active milestone")
                return 1
            milestone_before = float(
                get_milestone_by_id(active.id, db_path=db_path)["progress_percent"]
            )
            plan_result = planner.create_daily_plan(
                profile_result.user.id,
                checkin=DailyCheckInRequest(
                    mood=Mood.focused,
                    energy_level=EnergyLevel.medium,
                    focus_level=4,
                    available_minutes=25,
                    preferred_activity=ActivityType.practice,
                    notes="Live reflection validation",
                ),
            )
            tasks = plan_result.plan.tasks
            if len(tasks) < 1:
                print("LIVE_REFLECTION_AGENT_FAILED: plan has no tasks")
                return 1
            updates = [
                ReflectionTaskUpdate(
                    task_id=tasks[0].id,
                    update=TaskCompletionRequest(
                        status=TaskStatus.completed,
                        completion_percent=100,
                        duration_minutes=8,
                        effectiveness_rating=4,
                    ),
                )
            ]
            if len(tasks) > 1:
                updates.append(
                    ReflectionTaskUpdate(
                        task_id=tasks[1].id,
                        update=TaskCompletionRequest(
                            status=TaskStatus.skipped,
                            completion_percent=0,
                        ),
                    )
                )
            reflection_result = reflection_agent.reflect_on_plan(
                profile_result.user.id,
                ReflectionRequest(
                    daily_plan_id=plan_result.plan.id,
                    completion_status=CompletionStatus.partial,
                    learning_summary="Completed one focused practice task.",
                    focus_rating=3,
                    resource_effectiveness=4,
                    difficulty_feedback=DifficultyFeedback.suitable,
                    mood_match=True,
                    distractions=["phone notifications"],
                    wants_similar_resources=True,
                    mood_after=Mood.calm,
                    task_updates=updates,
                    actual_minutes_spent=12,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            name = type(exc).__name__
            if "api key" in message and ("invalid" in message or "not valid" in message):
                print(
                    "LIVE_REFLECTION_AGENT_SKIPPED: GEMINI_API_KEY is set but not valid "
                    "for live Gemini calls"
                )
                return 0
            if "429" in message or "quota" in message or "resourceexhausted" in name.lower():
                print(
                    "LIVE_REFLECTION_AGENT_SKIPPED: Gemini quota/rate limit exceeded; "
                    "retry later or raise plan limits"
                )
                print(f"DETAIL: {name}: {str(exc)[:500]}")
                return 0
            print(f"LIVE_REFLECTION_AGENT_FAILED: {name}: {str(exc)[:800]}")
            return 1

        print(f"user_id={reflection_result.reflection.user_id}")
        print(f"plan_id={reflection_result.reflection.daily_plan_id}")
        print(f"completion_status={reflection_result.reflection.completion_status.value}")
        print(f"insight={reflection_result.reflection.insight}")
        print(f"milestone_progress_before={milestone_before}")
        print(
            f"milestone_progress_after={reflection_result.milestone_progress_after}"
        )
        print(f"reflection_memories={len(reflection_result.memory_ids)}")
        print("LIVE_REFLECTION_AGENT_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
