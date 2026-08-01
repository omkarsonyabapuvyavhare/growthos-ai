"""Unit tests for GeminiEmbeddingService (no live Gemini calls)."""

from __future__ import annotations

import logging
import math
import sys
import unittest
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import Settings  # noqa: E402
from exceptions import (  # noqa: E402
    EmbeddingConfigurationError,
    EmbeddingInvocationError,
    EmbeddingResponseError,
)
from services.embedding import GeminiEmbeddingService  # noqa: E402


class FakeEmbeddingClient:
    def __init__(
        self,
        *,
        query_vector: list[float] | None = None,
        document_vectors: list[list[float]] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.query_vector = [0.0, 1.0, 0.0] if query_vector is None else query_vector
        self.document_vectors = document_vectors
        self.raise_on_call = raise_on_call
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return list(self.query_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.document_vectors is not None:
            return [list(item) for item in self.document_vectors]
        return [[float(index), 1.0, 0.0] for index, _ in enumerate(texts)]


def _settings(**overrides: Any) -> Settings:
    data = {
        "gemini_api_key": "",
        "gemini_api_key_1": "",
        "gemini_api_key_2": "",
        "gemini_api_key_3": "",
        "gemini_api_key_4": "",
        "gemini_model": "gemini-2.5-flash",
        "gemini_embedding_model": "models/gemini-embedding-001",
        "gemini_temperature": 0.2,
        "gemini_max_retries": 2,
        "gemini_request_timeout_seconds": 30,
        "database_url": "sqlite:///./growthos.db",
        "frontend_origin": "http://localhost:3000",
        "faiss_index_path": "./data/faiss_index/index.faiss",
        "faiss_metadata_path": "./data/faiss_index/metadata.json",
    }
    data.update(overrides)
    return Settings(**data)


class EmbeddingServiceTests(unittest.TestCase):
    def test_missing_key_fails_only_on_use(self) -> None:
        service = GeminiEmbeddingService(settings=_settings(gemini_api_key=""))
        self.assertFalse(service.is_configured())
        with self.assertRaises(EmbeddingConfigurationError):
            service.embed_text("hello")

    def test_blank_input_rejected(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(),
        )
        with self.assertRaises(ValueError):
            service.embed_text("  ")
        with self.assertRaises(ValueError):
            service.embed_documents([])

    def test_valid_vector_returned(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(query_vector=[1.0, 2.0, 3.0]),
        )
        vector = service.embed_text("growth plan")
        self.assertEqual(vector, [1.0, 2.0, 3.0])
        self.assertEqual(service.embedding_dimension(), 3)

    def test_batch_consistent_dimensions(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(
                document_vectors=[[1.0, 0.0], [0.0, 1.0]]
            ),
        )
        vectors = service.embed_documents(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(item) == 2 for item in vectors))

    def test_empty_vector_rejected(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(query_vector=[]),
        )
        with self.assertRaises(EmbeddingResponseError):
            service.embed_text("hello")

    def test_nan_and_inf_rejected(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(query_vector=[1.0, float("nan")]),
        )
        with self.assertRaises(EmbeddingResponseError):
            service.embed_text("hello")

        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(query_vector=[1.0, float("inf")]),
        )
        with self.assertRaises(EmbeddingResponseError):
            service.embed_text("hello")

    def test_provider_error_wrapped(self) -> None:
        secret = "secret-embed-key"
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key=secret),
            embedding_client=FakeEmbeddingClient(
                raise_on_call=RuntimeError(f"failed with {secret}")
            ),
        )
        with self.assertRaises(EmbeddingInvocationError) as ctx:
            service.embed_text("hello")
        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))

    def test_api_key_not_in_logs(self) -> None:
        secret = "another-secret-key"
        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = ListHandler()
        logger = logging.getLogger("services.embedding")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            service = GeminiEmbeddingService(
                settings=_settings(gemini_api_key=secret),
                embedding_client=FakeEmbeddingClient(
                    raise_on_call=RuntimeError(f"boom {secret}")
                ),
            )
            with self.assertRaises(EmbeddingInvocationError):
                service.embed_text("hello")
            for line in records:
                self.assertNotIn(secret, line)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

    def test_inconsistent_batch_dimensions_rejected(self) -> None:
        service = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="k"),
            embedding_client=FakeEmbeddingClient(
                document_vectors=[[1.0, 0.0], [1.0, 0.0, 0.0]]
            ),
        )
        with self.assertRaises(EmbeddingResponseError):
            service.embed_documents(["a", "b"])

    def test_finite_check_helper(self) -> None:
        self.assertTrue(math.isfinite(1.0))


if __name__ == "__main__":
    unittest.main()
