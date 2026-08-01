"""
Optional live Curator Agent validation.

Uses temporary SQLite + FAISS paths. Invoked manually only.

    .\\.venv\\Scripts\\python.exe scripts\\validate_curator_agent_live.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.curator_agent import CuratorAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from models import (  # noqa: E402
    CurrentLevel,
    LearningStyle,
    OnboardingRequest,
    PreferredLearningTime,
)
from services.catalog_index import CatalogSemanticIndex  # noqa: E402
from services.database import init_db  # noqa: E402
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.gemini import GeminiService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402


def main() -> int:
    base = get_settings()
    if not base.is_gemini_configured():
        print("LIVE_CURATOR_AGENT_SKIPPED: GEMINI_API_KEY is not configured")
        return 0

    with tempfile.TemporaryDirectory(prefix="growthos_curator_live_") as tmp:
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
            resource_catalog_path=str(
                BACKEND_ROOT / "data" / "sample_resources.json"
            ),
        )
        user_store = FAISSVectorStore(
            settings=settings,
            index_path=settings.resolve_faiss_index_path(),
            metadata_path=settings.resolve_faiss_metadata_path(),
            autosave=True,
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

        request = OnboardingRequest(
            display_name="Live Curator User",
            learning_goal="Improve public speaking",
            aspiration="Become a calm presenter",
            motivation="Share ideas confidently in meetings",
            current_level=CurrentLevel.beginner,
            target_outcome="Deliver a 5-minute update without notes",
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
                print("LIVE_CURATOR_AGENT_FAILED: no active milestone")
                return 1
            curator_result = curator.recommend_resources(
                profile_result.user.id,
                roadmap_result.roadmap.id,
                active.id,
                limit=3,
            )
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if "API" in name or "Gemini" in name or "Embedding" in name:
                print(
                    "LIVE_CURATOR_AGENT_SKIPPED: GEMINI_API_KEY is set but not valid "
                    "for live Gemini calls"
                )
                return 0
            print(f"LIVE_CURATOR_AGENT_FAILED: {name}")
            return 1

        print(f"user_id={profile_result.user.id}")
        print(f"goal_title={profile_result.goal.title}")
        print(f"milestone_title={active.title}")
        for rec in curator_result.recommendations:
            print(
                f"- {rec.title} | source={rec.source} | "
                f"score={rec.relevance_score:.3f} | reason={rec.reason[:120]}"
            )
        print("LIVE_CURATOR_AGENT_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
