"""
AI provider abstraction for GrowthOS AI.

Agents and workflows depend on AIProvider, not on Gemini-specific classes.
Gemini is the default and currently only supported provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from config import Settings, get_settings
from exceptions import AIProviderConfigurationError
from services.embedding import GeminiEmbeddingService
from services.gemini import GeminiService

T = TypeVar("T", bound=BaseModel)

SUPPORTED_AI_PROVIDERS = frozenset({"gemini"})


class AIProvider(ABC):
    """Provider-agnostic interface for text generation and embeddings."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        """Generate plain text from a prompt."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_instruction: str | None = None,
    ) -> T:
        """Generate structured output validated by a Pydantic model."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single non-blank text string."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate that this provider is ready for use."""

    def is_configured(self) -> bool:
        """Return True when the provider appears configured."""
        try:
            self.validate_configuration()
        except Exception:  # noqa: BLE001 - configuration probe only
            return False
        return True


class GeminiProvider(AIProvider):
    """
    Gemini-backed AIProvider.

    Delegates chat and embedding calls to the existing Gemini services so
    behaviour stays identical to the pre-abstraction implementation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        chat_model: Any | None = None,
        embedding_client: Any | None = None,
        gemini_service: GeminiService | None = None,
        embedding_service: GeminiEmbeddingService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._chat = gemini_service or GeminiService(
            settings=self._settings,
            chat_model=chat_model,
        )
        self._embeddings = embedding_service or GeminiEmbeddingService(
            settings=self._settings,
            embedding_client=embedding_client,
        )

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def gemini_service(self) -> GeminiService:
        """Expose the underlying chat service for Gemini-specific tooling."""
        return self._chat

    @property
    def embedding_service(self) -> GeminiEmbeddingService:
        """Expose the underlying embedding service for Gemini-specific tooling."""
        return self._embeddings

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        return self._chat.generate_text(
            prompt,
            system_instruction=system_instruction,
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_instruction: str | None = None,
    ) -> T:
        return self._chat.generate_structured(
            prompt,
            response_model,
            system_instruction=system_instruction,
        )

    def embed_text(self, text: str) -> list[float]:
        return self._embeddings.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def validate_configuration(self) -> None:
        self._chat.validate_configuration()
        self._embeddings.validate_configuration()

    def is_configured(self) -> bool:
        return self._chat.is_configured() and self._embeddings.is_configured()

    def test_connection(self) -> str:
        """Optional low-cost connectivity check (chat path)."""
        return self._chat.test_connection()


def resolve_ai_provider_name(settings: Settings) -> str:
    """Normalize and return the configured provider name."""
    return settings.ai_provider.strip().lower()


def get_ai_provider(settings: Optional[Settings] = None) -> AIProvider:
    """
    Build the configured AI provider.

    Defaults to Gemini. Raises AIProviderConfigurationError for unsupported names.
    """
    resolved = settings or get_settings()
    provider_name = resolve_ai_provider_name(resolved)
    if not provider_name:
        raise AIProviderConfigurationError(
            "AI_PROVIDER is empty. Set AI_PROVIDER=gemini."
        )
    if provider_name not in SUPPORTED_AI_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_AI_PROVIDERS))
        raise AIProviderConfigurationError(
            f"Unsupported AI_PROVIDER={provider_name!r}. "
            f"Supported providers: {supported}."
        )
    if provider_name == "gemini":
        return GeminiProvider(settings=resolved)
    # Defensive: SUPPORTED_AI_PROVIDERS and branches must stay in sync.
    raise AIProviderConfigurationError(
        f"Unsupported AI_PROVIDER={provider_name!r}."
    )
