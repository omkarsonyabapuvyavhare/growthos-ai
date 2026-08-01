"""
Manual live validation for optional YouTube Data API discovery.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\validate_youtube_live.py

Never prints the API key. Does not write to the normal database.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import get_settings  # noqa: E402
from exceptions import (  # noqa: E402
    YouTubeConfigurationError,
    YouTubeInvocationError,
    YouTubeResponseError,
)
from services.youtube import YouTubeService  # noqa: E402


def main() -> int:
    settings = get_settings()
    print("YouTube live validation")
    print(f"  enabled={settings.youtube_api_enabled}")
    print(f"  configured={settings.is_youtube_configured()}")
    print(f"  max_results={settings.youtube_max_results}")

    if not settings.youtube_api_enabled:
        print("RESULT: YouTube discovery disabled (YOUTUBE_API_ENABLED=false)")
        return 0
    if not settings.is_youtube_configured():
        print("RESULT: YOUTUBE_API_KEY is missing or blank")
        return 0

    service = YouTubeService(settings=settings)
    try:
        items = service.search_and_enrich(
            learning_goal="public speaking",
            milestone_title="Practice clear openings",
            milestone_skills=["presence", "structure"],
            current_level="beginner",
            preferred_language="en",
            preferred_format="video",
            available_minutes=20,
            attention_span_minutes=15,
            result_limit=min(5, settings.youtube_max_results),
        )
    except YouTubeConfigurationError as exc:
        print(f"RESULT: configuration error — {exc}")
        return 0
    except YouTubeInvocationError as exc:
        print(f"RESULT: invocation error — {exc}")
        return 0
    except YouTubeResponseError as exc:
        print(f"RESULT: response error — {exc}")
        return 0
    finally:
        service.close()

    def _safe(value: object) -> str:
        return str(value).encode("ascii", "replace").decode("ascii")

    print(f"RESULT: {len(items)} enriched video(s)")
    for index, item in enumerate(items, start=1):
        meta = item.metadata or {}
        print(f"{index}. {_safe(item.title)}")
        print(f"   channel: {_safe(meta.get('channel_title') or '(unknown)')}")
        print(f"   duration_minutes: {item.estimated_duration_minutes}")
        print(f"   url: {_safe(item.url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
