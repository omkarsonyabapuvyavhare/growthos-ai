"""
Optional live Profile Agent validation.

Uses temporary SQLite + FAISS paths so the normal project DB is not polluted.

    .\\.venv\\Scripts\\python.exe scripts\\validate_profile_agent_live.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.profile_agent import ProfileAgent  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from models import (  # noqa: E402
    CurrentLevel,
    LearningStyle,
    OnboardingRequest,
    PreferredLearningTime,
)
from services.database import init_db  # noqa: E402
from services.gemini import GeminiService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402


def main() -> int:
    base = get_settings()
    if not base.is_gemini_configured():
        print("LIVE_PROFILE_AGENT_SKIPPED: GEMINI_API_KEY is not configured")
        return 0

    with tempfile.TemporaryDirectory(prefix="growthos_profile_live_") as tmp:
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
        )

        store = FAISSVectorStore(
            settings=settings,
            index_path=settings.resolve_faiss_index_path(),
            metadata_path=settings.resolve_faiss_metadata_path(),
            autosave=True,
        )
        memory = SemanticMemoryService(settings=settings, vector_store=store)
        agent = ProfileAgent(
            settings=settings,
            gemini_service=GeminiService(settings=settings),
            memory_service=memory,
            db_path=db_path,
        )

        request = OnboardingRequest(
            display_name="Live Demo User",
            learning_goal="Improve public speaking",
            aspiration="Become a calm and clear presenter",
            motivation="Share ideas confidently in weekly meetings",
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
            result = agent.process_onboarding(request)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            # Never echo key material if a provider message includes it.
            lowered = message.lower()
            if "api key" in lowered and ("invalid" in lowered or "not valid" in lowered):
                print(
                    "LIVE_PROFILE_AGENT_SKIPPED: GEMINI_API_KEY is set but not valid "
                    "for live Gemini calls"
                )
                return 0
            print(f"LIVE_PROFILE_AGENT_FAILED: {type(exc).__name__}: {type(exc).__name__}")
            return 1

        summary = result.interpretation.identity_summary.replace("\n", " ").strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."

        print(
            "LIVE_PROFILE_AGENT_OK "
            f"user_id={result.user.id} "
            f"goal_title={result.goal.title!r} "
            f"interpretation_summary={summary!r} "
            f"memories={len(result.memory_ids)}"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
