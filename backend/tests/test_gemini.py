"""
Unit tests for the Gemini service foundation.

These tests use fakes/mocks and must not call the real Gemini API.
Run with: python -m unittest tests.test_gemini
"""

from __future__ import annotations

import logging
import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import Settings  # noqa: E402
from exceptions import (  # noqa: E402
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
)
from services.gemini import GeminiService, is_quota_error  # noqa: E402


class StructuredProbe(BaseModel):
    """Test-only structured response model."""

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    confidence_score: float = Field(..., ge=0, le=1)


class FakeMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeChatModel:
    """Minimal stand-in for ChatGoogleGenerativeAI."""

    def __init__(
        self,
        *,
        text_response: Any = "hello from fake gemini",
        structured_response: Any = None,
        raise_on_invoke: Exception | None = None,
        api_key: str = "",
    ) -> None:
        self.text_response = text_response
        self.structured_response = structured_response
        self.raise_on_invoke = raise_on_invoke
        self.api_key = api_key
        self.invoke_calls = 0

    def invoke(self, _messages: Any) -> Any:
        self.invoke_calls += 1
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke
        return FakeMessage(self.text_response)

    def with_structured_output(self, _model: Type[BaseModel]) -> "FakeStructuredModel":
        return FakeStructuredModel(self)


class FakeStructuredModel:
    def __init__(self, parent: FakeChatModel) -> None:
        self.parent = parent

    def invoke(self, _messages: Any) -> Any:
        self.parent.invoke_calls += 1
        if self.parent.raise_on_invoke is not None:
            raise self.parent.raise_on_invoke
        return self.parent.structured_response


class QuotaError(Exception):
    """Test double shaped like a provider quota failure."""

    def __init__(self, message: str = "429 RESOURCE_EXHAUSTED quota exceeded") -> None:
        super().__init__(message)
        self.status_code = 429


class AuthError(Exception):
    """Test double for non-quota authentication failures."""

    def __init__(self, message: str = "401 UNAUTHENTICATED invalid api key") -> None:
        super().__init__(message)
        self.status_code = 401


def _settings(**overrides: Any) -> Settings:
    data = {
        "gemini_api_key": "",
        "gemini_api_key_1": "",
        "gemini_api_key_2": "",
        "gemini_api_key_3": "",
        "gemini_api_key_4": "",
        "gemini_model": "gemini-2.5-flash",
        "gemini_temperature": 0.2,
        "gemini_max_retries": 2,
        "gemini_request_timeout_seconds": 30,
        "database_url": "sqlite:///./growthos.db",
        "frontend_origin": "http://localhost:3000",
        "faiss_index_path": "./data/faiss_index",
    }
    data.update(overrides)
    return Settings(**data)


class GeminiServiceTests(unittest.TestCase):
    def test_missing_api_key_raises_only_when_called(self) -> None:
        service = GeminiService(settings=_settings(gemini_api_key=""))
        # Construction must not raise
        self.assertFalse(service.is_configured())
        with self.assertRaises(GeminiConfigurationError):
            service.generate_text("hello")

    def test_fastapi_app_starts_without_gemini_key(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "GrowthOS AI API")
        self.assertIn("youtube_enabled", body)
        self.assertIn("youtube_configured", body)

    def test_empty_prompt_rejected(self) -> None:
        service = GeminiService(
            settings=_settings(gemini_api_key="test-key"),
            chat_model=FakeChatModel(),
        )
        with self.assertRaises(ValueError):
            service.generate_text("   ")

    def test_plain_text_normalized(self) -> None:
        service = GeminiService(
            settings=_settings(gemini_api_key="test-key"),
            chat_model=FakeChatModel(text_response="  trimmed response  "),
        )
        text = service.generate_text("Summarize growth planning briefly.")
        self.assertEqual(text, "trimmed response")

    def test_empty_provider_response_raises_response_error(self) -> None:
        service = GeminiService(
            settings=_settings(gemini_api_key="test-key"),
            chat_model=FakeChatModel(text_response="   "),
        )
        with self.assertRaises(GeminiResponseError):
            service.generate_text("Say hello")

    def test_provider_failure_becomes_invocation_error(self) -> None:
        service = GeminiService(
            settings=_settings(gemini_api_key="secret-key-value"),
            chat_model=FakeChatModel(
                raise_on_invoke=RuntimeError("boom secret-key-value leaked")
            ),
        )
        with self.assertRaises(GeminiInvocationError) as ctx:
            service.generate_text("Say hello")
        self.assertNotIn("secret-key-value", str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))

    def test_valid_structured_output(self) -> None:
        payload = StructuredProbe(
            title="Public speaking",
            summary="A short growth summary",
            confidence_score=0.8,
        )
        service = GeminiService(
            settings=_settings(gemini_api_key="test-key"),
            chat_model=FakeChatModel(structured_response=payload),
        )
        result = service.generate_structured(
            "Return a tiny structured probe.",
            StructuredProbe,
        )
        self.assertIsInstance(result, StructuredProbe)
        self.assertEqual(result.title, "Public speaking")
        self.assertEqual(result.confidence_score, 0.8)

    def test_invalid_structured_output_rejected(self) -> None:
        service = GeminiService(
            settings=_settings(gemini_api_key="test-key"),
            chat_model=FakeChatModel(
                structured_response={
                    "title": "",
                    "summary": "x",
                    "confidence_score": 2.5,
                }
            ),
        )
        with self.assertRaises(GeminiResponseError):
            service.generate_structured("Return invalid structured data.", StructuredProbe)

    def test_api_key_never_appears_in_logs(self) -> None:
        secret = "super-secret-gemini-key"
        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = ListHandler()
        logger = logging.getLogger("services.gemini")
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            service = GeminiService(
                settings=_settings(gemini_api_key=secret),
                chat_model=FakeChatModel(
                    raise_on_invoke=RuntimeError(f"provider failed using {secret}")
                ),
            )
            with self.assertRaises(GeminiInvocationError) as ctx:
                service.generate_text("Hello")
            self.assertNotIn(secret, str(ctx.exception))
            for line in records:
                self.assertNotIn(secret, line)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def test_validate_configuration_detects_missing_key(self) -> None:
        service = GeminiService(settings=_settings(gemini_api_key=""))
        with self.assertRaises(GeminiConfigurationError):
            service.validate_configuration()


class GeminiKeyRotationTests(unittest.TestCase):
    def test_one_configured_key_backward_compatible(self) -> None:
        settings = _settings(gemini_api_key="solo-key")
        self.assertEqual(settings.gemini_api_keys(), ["solo-key"])
        service = GeminiService(
            settings=settings,
            chat_model_factory=lambda key: FakeChatModel(
                text_response=f"ok:{key}",
                api_key=key,
            ),
        )
        self.assertEqual(service.configured_key_count(), 1)
        self.assertEqual(service.generate_text("hi"), "ok:solo-key")
        self.assertEqual(service.active_key_number(), 1)

    def test_two_configured_keys(self) -> None:
        settings = _settings(
            gemini_api_key_1="key-one",
            gemini_api_key_2="key-two",
        )
        self.assertEqual(settings.gemini_api_keys(), ["key-one", "key-two"])
        service = GeminiService(settings=settings)
        self.assertEqual(service.configured_key_count(), 2)

    def test_four_configured_keys(self) -> None:
        settings = _settings(
            gemini_api_key="ignored-when-numbered-present",
            gemini_api_key_1="k1",
            gemini_api_key_2="k2",
            gemini_api_key_3="k3",
            gemini_api_key_4="k4",
        )
        self.assertEqual(settings.gemini_api_keys(), ["k1", "k2", "k3", "k4"])
        service = GeminiService(settings=settings)
        self.assertEqual(service.configured_key_count(), 4)

    def test_successful_primary_key(self) -> None:
        built: list[str] = []

        def factory(key: str) -> FakeChatModel:
            built.append(key)
            return FakeChatModel(text_response="primary-ok", api_key=key)

        service = GeminiService(
            settings=_settings(
                gemini_api_key_1="alpha-key",
                gemini_api_key_2="beta-key",
            ),
            chat_model_factory=factory,
        )
        self.assertEqual(service.generate_text("hello"), "primary-ok")
        self.assertEqual(built, ["alpha-key"])
        self.assertEqual(service.active_key_number(), 1)

    def test_automatic_failover_on_429(self) -> None:
        built: list[str] = []

        def factory(key: str) -> FakeChatModel:
            built.append(key)
            if key == "alpha-key":
                return FakeChatModel(raise_on_invoke=QuotaError(), api_key=key)
            return FakeChatModel(text_response="failover-ok", api_key=key)

        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = ListHandler()
        logger = logging.getLogger("services.gemini")
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            service = GeminiService(
                settings=_settings(
                    gemini_api_key_1="alpha-key",
                    gemini_api_key_2="beta-key",
                ),
                chat_model_factory=factory,
            )
            text = service.generate_text("hello")
            self.assertEqual(text, "failover-ok")
            self.assertEqual(built, ["alpha-key", "beta-key"])
            self.assertEqual(service.active_key_number(), 2)
            self.assertTrue(
                any("Switching to Gemini key #2 due to quota." in line for line in records)
            )
            for line in records:
                self.assertNotIn("alpha-key", line)
                self.assertNotIn("beta-key", line)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def test_all_keys_exhausted(self) -> None:
        def factory(key: str) -> FakeChatModel:
            return FakeChatModel(
                raise_on_invoke=QuotaError(f"quota for {key}"),
                api_key=key,
            )

        service = GeminiService(
            settings=_settings(
                gemini_api_key_1="alpha-key",
                gemini_api_key_2="beta-key",
                gemini_api_key_3="gamma-key",
                gemini_api_key_4="delta-key",
            ),
            chat_model_factory=factory,
        )
        with self.assertRaises(GeminiInvocationError) as ctx:
            service.generate_text("hello")
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "All configured Gemini API keys have exhausted their quota.",
        )
        self.assertNotIn("alpha-key", message)
        self.assertNotIn("beta-key", message)
        self.assertNotIn("gamma-key", message)
        self.assertNotIn("delta-key", message)

    def test_non_quota_errors_do_not_rotate(self) -> None:
        built: list[str] = []

        def factory(key: str) -> FakeChatModel:
            built.append(key)
            if key == "alpha-key":
                return FakeChatModel(
                    raise_on_invoke=AuthError("invalid api key alpha-key"),
                    api_key=key,
                )
            return FakeChatModel(text_response="should-not-reach", api_key=key)

        service = GeminiService(
            settings=_settings(
                gemini_api_key_1="alpha-key",
                gemini_api_key_2="beta-key",
            ),
            chat_model_factory=factory,
        )
        with self.assertRaises(GeminiInvocationError) as ctx:
            service.generate_text("hello")
        self.assertEqual(built, ["alpha-key"])
        self.assertNotIn("alpha-key", str(ctx.exception))
        self.assertNotIn("beta-key", str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))

    def test_quota_classifier(self) -> None:
        self.assertTrue(is_quota_error(QuotaError()))
        self.assertTrue(is_quota_error(RuntimeError("ResourceExhausted: quota")))
        self.assertFalse(is_quota_error(AuthError()))
        self.assertFalse(is_quota_error(ValueError("invalid prompt shape")))

    def test_rotation_is_thread_safe(self) -> None:
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def factory(key: str) -> FakeChatModel:
            if key == "alpha-key":
                return FakeChatModel(raise_on_invoke=QuotaError(), api_key=key)
            return FakeChatModel(text_response="ok", api_key=key)

        service = GeminiService(
            settings=_settings(
                gemini_api_key_1="alpha-key",
                gemini_api_key_2="beta-key",
            ),
            chat_model_factory=factory,
        )

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                text = service.generate_text("hello")
                if text != "ok":
                    raise AssertionError(f"unexpected text={text!r}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(service.active_key_number(), 2)


if __name__ == "__main__":
    unittest.main()
