"""Unit tests for the AI provider abstraction (no live provider calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import Settings  # noqa: E402
from exceptions import AIProviderConfigurationError  # noqa: E402
from services.ai_provider import (  # noqa: E402
    GeminiProvider,
    get_ai_provider,
)


def _settings(**overrides: object) -> Settings:
    values = {
        "ai_provider": "gemini",
        "gemini_api_key": "test-key",
        "gemini_model": "gemini-flash-latest",
        "gemini_embedding_model": "models/gemini-embedding-001",
    }
    values.update(overrides)
    return Settings(**values)


class AIProviderFactoryTests(unittest.TestCase):
    def test_default_provider_is_gemini(self) -> None:
        provider = get_ai_provider(settings=_settings())
        self.assertIsInstance(provider, GeminiProvider)

    def test_gemini_provider_name_case_insensitive(self) -> None:
        provider = get_ai_provider(settings=_settings(ai_provider="Gemini"))
        self.assertIsInstance(provider, GeminiProvider)

    def test_unsupported_provider_fails_clearly(self) -> None:
        with self.assertRaises(AIProviderConfigurationError) as ctx:
            get_ai_provider(settings=_settings(ai_provider="grok"))
        message = str(ctx.exception)
        self.assertIn("Unsupported AI_PROVIDER", message)
        self.assertIn("gemini", message)

    def test_empty_provider_fails_clearly(self) -> None:
        with self.assertRaises(AIProviderConfigurationError) as ctx:
            get_ai_provider(settings=_settings(ai_provider="   "))
        self.assertIn("AI_PROVIDER is empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
