"""
Optional live Gemini embedding check.

Run manually:

    .\\.venv\\Scripts\\python.exe scripts\\validate_embedding_live.py

Prints only model name and vector dimension. Never prints the key or vector.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import get_settings  # noqa: E402
from exceptions import EmbeddingConfigurationError  # noqa: E402
from services.embedding import GeminiEmbeddingService  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.is_gemini_configured():
        print("LIVE_EMBEDDING_SKIPPED: GEMINI_API_KEY is not configured")
        return 0

    service = GeminiEmbeddingService(settings=settings)
    try:
        vector = service.embed_text("GrowthOS AI embedding probe")
    except EmbeddingConfigurationError as exc:
        print(f"LIVE_EMBEDDING_SKIPPED: {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"LIVE_EMBEDDING_FAILED: {type(exc).__name__}: {exc}")
        return 1

    if not vector or not all(math.isfinite(float(value)) for value in vector):
        print("LIVE_EMBEDDING_FAILED: invalid vector")
        return 1

    print(
        "LIVE_EMBEDDING_OK "
        f"model={settings.gemini_embedding_model} "
        f"dimension={len(vector)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
