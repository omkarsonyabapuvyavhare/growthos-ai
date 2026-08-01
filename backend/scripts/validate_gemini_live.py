"""
Optional live Gemini connectivity check.

Run manually only when GEMINI_API_KEY is configured:

    .\\.venv\\Scripts\\python.exe scripts\\validate_gemini_live.py

This script is intentionally not part of the default unit-test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import get_settings  # noqa: E402
from exceptions import GeminiConfigurationError  # noqa: E402
from services.gemini import GeminiService  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.is_gemini_configured():
        print("LIVE_GEMINI_SKIPPED: GEMINI_API_KEY is not configured")
        return 0

    service = GeminiService(settings=settings)
    try:
        response = service.test_connection()
    except GeminiConfigurationError as exc:
        print(f"LIVE_GEMINI_SKIPPED: {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        # Never print secrets; service already redacts key material.
        print(f"LIVE_GEMINI_FAILED: {type(exc).__name__}: {exc}")
        return 1

    preview = response.replace("\n", " ").strip()[:80]
    print(f"LIVE_GEMINI_OK response_chars={len(response)} preview={preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
