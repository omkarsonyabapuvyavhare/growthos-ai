"""
Gemini embedding service for GrowthOS AI.

Uses LangChain GoogleGenerativeAIEmbeddings (langchain-google-genai).
Does not fabricate local/random vectors when Gemini is unavailable.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Protocol, Sequence

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Settings, get_settings
from exceptions import (
    EmbeddingConfigurationError,
    EmbeddingInvocationError,
    EmbeddingResponseError,
)

logger = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    """Protocol for Gemini embedding clients and test fakes."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _redact_secrets(text: str, api_key: str) -> str:
    cleaned = text
    key = api_key.strip()
    if key:
        cleaned = cleaned.replace(key, "[REDACTED]")
    return cleaned.replace("GEMINI_API_KEY=", "GEMINI_API_KEY=[REDACTED]")


def _validate_vector(vector: Sequence[float], *, label: str) -> list[float]:
    if vector is None or len(vector) == 0:
        raise EmbeddingResponseError(f"{label} embedding vector is empty")
    values: list[float] = []
    for index, raw in enumerate(vector):
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise EmbeddingResponseError(
                f"{label} embedding contains a non-numeric value at index {index}"
            ) from None
        if not math.isfinite(number):
            raise EmbeddingResponseError(
                f"{label} embedding contains a non-finite value at index {index}"
            )
        values.append(number)
    return values


class GeminiEmbeddingService:
    """
    Lazy Gemini embedding client.

    The provider client is created on first use, not at import time.
    Tests may inject a fake client via embedding_client.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._injected_client = embedding_client
        self._client: Any | None = None
        self._dimension: int | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    def is_configured(self) -> bool:
        return self._settings.is_gemini_configured()

    def validate_configuration(self) -> None:
        if not self._settings.is_gemini_configured():
            raise EmbeddingConfigurationError(
                "GEMINI_API_KEY is not configured. "
                "Set it in the backend environment before requesting embeddings."
            )
        if not self._settings.gemini_embedding_model.strip():
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_MODEL is empty. "
                "Set a supported Gemini embedding model name."
            )

    def embedding_dimension(self) -> int | None:
        """Return the last observed embedding dimension, if any."""
        return self._dimension

    def _api_key(self) -> str:
        """Use primary configured Gemini key (supports GEMINI_API_KEY_1..4)."""
        return self._settings.primary_gemini_api_key()

    def _redact(self, text: str) -> str:
        cleaned = text
        for key in self._settings.gemini_api_keys():
            cleaned = _redact_secrets(cleaned, key)
        cleaned = _redact_secrets(cleaned, self._settings.gemini_api_key)
        return cleaned

    def _build_client(self) -> GoogleGenerativeAIEmbeddings:
        self.validate_configuration()
        return GoogleGenerativeAIEmbeddings(
            model=self._settings.gemini_embedding_model.strip(),
            google_api_key=self._api_key(),
        )

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._client is None:
            logger.info(
                "Initializing Gemini embedding client model=%s",
                self._settings.gemini_embedding_model,
            )
            self._client = self._build_client()
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Embed a single non-blank text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        cleaned = text.strip()
        try:
            client = self._get_client()
            logger.info("Embedding query started text_chars=%s", len(cleaned))
            raw = client.embed_query(cleaned)
        except EmbeddingConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001
            message = self._redact(str(exc))
            logger.error("Embedding query failed error_type=%s", type(exc).__name__)
            raise EmbeddingInvocationError(
                f"Gemini embedding query failed: {message}"
            ) from None

        vector = _validate_vector(raw, label="query")
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise EmbeddingResponseError(
                f"query embedding dimension {len(vector)} "
                f"does not match expected {self._dimension}"
            )
        logger.info("Embedding query succeeded dimension=%s", len(vector))
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents using the provider batch method.

        Rejects empty lists and blank entries.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list")

        cleaned: list[str] = []
        for index, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")
            cleaned.append(item.strip())

        try:
            client = self._get_client()
            logger.info(
                "Embedding documents started count=%s",
                len(cleaned),
            )
            raw_vectors = client.embed_documents(cleaned)
        except EmbeddingConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001
            message = self._redact(str(exc))
            logger.error("Embedding documents failed error_type=%s", type(exc).__name__)
            raise EmbeddingInvocationError(
                f"Gemini document embedding failed: {message}"
            ) from None

        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(cleaned):
            raise EmbeddingResponseError(
                "Gemini document embedding count does not match input count"
            )

        vectors: list[list[float]] = []
        expected_dim = self._dimension
        for index, raw in enumerate(raw_vectors):
            vector = _validate_vector(raw, label=f"documents[{index}]")
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise EmbeddingResponseError(
                    f"documents[{index}] dimension {len(vector)} "
                    f"does not match expected {expected_dim}"
                )
            vectors.append(vector)

        self._dimension = expected_dim
        logger.info(
            "Embedding documents succeeded count=%s dimension=%s",
            len(vectors),
            expected_dim,
        )
        return vectors


def get_embedding_service(settings: Optional[Settings] = None) -> GeminiEmbeddingService:
    """Factory for a GeminiEmbeddingService bound to settings."""
    return GeminiEmbeddingService(settings=settings)
