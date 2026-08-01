"""
Google Gemini service foundation for GrowthOS AI.

Uses LangChain's ChatGoogleGenerativeAI (langchain-google-genai).
Supports optional multi-key rotation for free-tier quota resilience.
Agents and workflows remain unaware of which key is active.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Optional, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from config import Settings, get_settings
from exceptions import (
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Reusable instructions for later agents (profile, roadmap, curator, planner, etc.)
NO_FABRICATED_URLS_INSTRUCTION = (
    "Never fabricate, invent, or guess resource URLs. "
    "Only reference URLs explicitly provided in the input. "
    "If a URL is unknown, omit it or mark it as unavailable."
)

STRUCTURED_OUTPUT_ONLY_INSTRUCTION = (
    "Return only the requested structured output. "
    "Do not include markdown fences, commentary, or extra keys."
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are GrowthOS AI, an agentic growth curator that optimizes for "
    "human learning and goal achievement rather than engagement. "
    f"{NO_FABRICATED_URLS_INSTRUCTION}"
)

_QUOTA_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate-limit",
    "rate limit",
    "ratelimit",
    "quota exceeded",
    "quotaexhaust",
    "exceeded your current quota",
    "too many requests",
)


def _redact_secrets(text: str, api_key: str) -> str:
    """Remove API key material from error or log text."""
    cleaned = text
    key = api_key.strip()
    if key:
        cleaned = cleaned.replace(key, "[REDACTED]")
    # Common env-style leaks
    cleaned = cleaned.replace("GEMINI_API_KEY=", "GEMINI_API_KEY=[REDACTED]")
    for index in range(1, 5):
        cleaned = cleaned.replace(
            f"GEMINI_API_KEY_{index}=",
            f"GEMINI_API_KEY_{index}=[REDACTED]",
        )
    return cleaned


def _normalize_text(content: Any) -> str:
    """Normalize LangChain message content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if str(part).strip()).strip()
    return str(content).strip()


def is_quota_error(exc: BaseException) -> bool:
    """Return True when an exception indicates quota / rate-limit exhaustion."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    try:
        if int(status) == 429:
            return True
    except (TypeError, ValueError):
        pass

    name = type(exc).__name__.lower()
    if "resourceexhausted" in name or "toomanyrequests" in name:
        return True

    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


class GeminiService:
    """
    Lazy, reusable Gemini client for plain-text and structured generation.

    Supports ordered multi-key rotation on quota/rate-limit failures only.
    The ChatGoogleGenerativeAI model is created on first use per key.
    Tests may inject a fake chat model via chat_model, or a per-key factory
    via chat_model_factory.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._injected_chat_model = chat_model
        self._chat_model_factory = chat_model_factory
        self._chat_models: dict[int, Any] = {}
        self._active_key_index = 0
        self._key_lock = threading.RLock()

    @property
    def settings(self) -> Settings:
        """Return the settings object used by this service instance."""
        return self._settings

    def _api_keys(self) -> list[str]:
        return self._settings.gemini_api_keys()

    def _redact(self, text: str) -> str:
        cleaned = text
        for key in self._api_keys():
            cleaned = _redact_secrets(cleaned, key)
        cleaned = _redact_secrets(cleaned, self._settings.gemini_api_key)
        for index in range(1, 5):
            value = getattr(self._settings, f"gemini_api_key_{index}", "")
            cleaned = _redact_secrets(cleaned, str(value or ""))
        return cleaned

    def validate_configuration(self) -> None:
        """
        Validate that Gemini can be invoked.

        Raises GeminiConfigurationError when no API key is configured.
        Safe to call repeatedly; does not initialize the model.
        """
        if not self._settings.is_gemini_configured():
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is not configured. "
                "Set GEMINI_API_KEY or GEMINI_API_KEY_1..4 in the backend "
                "environment before calling Gemini."
            )
        if not self._settings.gemini_model.strip():
            raise GeminiConfigurationError(
                "GEMINI_MODEL is empty. Set a supported Gemini model name."
            )

    def is_configured(self) -> bool:
        """Return True when at least one Gemini API key is present."""
        return self._settings.is_gemini_configured()

    def configured_key_count(self) -> int:
        """Return how many Gemini keys are available for rotation."""
        return len(self._api_keys())

    def active_key_number(self) -> int:
        """1-based index of the current key (for diagnostics/tests only)."""
        keys = self._api_keys()
        if not keys:
            return 0
        with self._key_lock:
            return (self._active_key_index % len(keys)) + 1

    def _build_chat_model(self, api_key: str) -> Any:
        """Create a chat model bound to a specific API key."""
        self.validate_configuration()
        if self._chat_model_factory is not None:
            return self._chat_model_factory(api_key)
        return ChatGoogleGenerativeAI(
            model=self._settings.gemini_model.strip(),
            google_api_key=api_key,
            temperature=self._settings.gemini_temperature,
            max_retries=self._settings.gemini_max_retries,
            timeout=self._settings.gemini_request_timeout_seconds,
        )

    def _get_chat_model_for_index(self, index: int) -> Any:
        """Return the injected model or a lazily initialized model for key index."""
        if self._injected_chat_model is not None:
            return self._injected_chat_model
        keys = self._api_keys()
        if index < 0 or index >= len(keys):
            raise GeminiConfigurationError("Gemini key index out of range")
        with self._key_lock:
            model = self._chat_models.get(index)
            if model is None:
                logger.info(
                    "Initializing Gemini chat model model=%s temperature=%s key_slot=%s",
                    self._settings.gemini_model,
                    self._settings.gemini_temperature,
                    index + 1,
                )
                model = self._build_chat_model(keys[index])
                self._chat_models[index] = model
            return model

    def _get_chat_model(self) -> Any:
        """Return the injected model or the active-key chat model."""
        if self._injected_chat_model is not None:
            return self._injected_chat_model
        keys = self._api_keys()
        if not keys:
            self.validate_configuration()
        with self._key_lock:
            index = self._active_key_index % max(len(keys), 1)
        return self._get_chat_model_for_index(index)

    def _invoke_with_key_rotation(self, operation: Callable[[Any], Any]) -> Any:
        """
        Invoke a chat operation, rotating keys only on quota/rate-limit errors.
        """
        if self._injected_chat_model is not None:
            return operation(self._injected_chat_model)

        self.validate_configuration()
        keys = self._api_keys()
        key_count = len(keys)
        with self._key_lock:
            start_index = self._active_key_index % key_count

        last_quota_error: BaseException | None = None
        for offset in range(key_count):
            index = (start_index + offset) % key_count
            model = self._get_chat_model_for_index(index)
            try:
                result = operation(model)
            except GeminiConfigurationError:
                raise
            except GeminiInvocationError:
                raise
            except GeminiResponseError:
                raise
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001 - classify then wrap/rotate
                if is_quota_error(exc):
                    last_quota_error = exc
                    next_offset = offset + 1
                    if next_offset < key_count:
                        next_index = (start_index + next_offset) % key_count
                        logger.warning(
                            "Switching to Gemini key #%s due to quota.",
                            next_index + 1,
                        )
                        with self._key_lock:
                            self._active_key_index = next_index
                        continue
                    logger.error(
                        "All configured Gemini API keys exhausted key_count=%s",
                        key_count,
                    )
                    raise GeminiInvocationError(
                        "All configured Gemini API keys have exhausted their quota."
                    ) from None
                # Non-quota failures must not rotate.
                message = self._redact(str(exc))
                logger.error(
                    "Gemini invocation failed error_type=%s",
                    type(exc).__name__,
                )
                raise GeminiInvocationError(
                    f"Gemini invocation failed: {message}"
                ) from None

            with self._key_lock:
                self._active_key_index = index
            return result

        # Defensive: loop should always return or raise.
        if last_quota_error is not None:
            raise GeminiInvocationError(
                "All configured Gemini API keys have exhausted their quota."
            ) from None
        raise GeminiInvocationError("Gemini invocation failed with no available keys")

    def _build_messages(
        self,
        *,
        prompt: str,
        system_instruction: str | None,
        structured: bool,
    ) -> list[Any]:
        system_parts = [system_instruction.strip() if system_instruction else DEFAULT_SYSTEM_INSTRUCTION]
        system_parts.append(NO_FABRICATED_URLS_INSTRUCTION)
        if structured:
            system_parts.append(STRUCTURED_OUTPUT_ONLY_INSTRUCTION)
        system_text = "\n\n".join(dict.fromkeys(system_parts))  # de-dupe, keep order
        return [
            SystemMessage(content=system_text),
            HumanMessage(content=prompt.strip()),
        ]

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        """
        Generate plain text from Gemini.

        Raises:
            ValueError: empty prompt
            GeminiConfigurationError: missing API key
            GeminiInvocationError: provider failure
            GeminiResponseError: empty model response
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        messages = self._build_messages(
            prompt=prompt,
            system_instruction=system_instruction,
            structured=False,
        )

        logger.info("Gemini text generation started prompt_chars=%s", len(prompt.strip()))
        try:
            result = self._invoke_with_key_rotation(lambda model: model.invoke(messages))
        except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
            raise
        except Exception as exc:  # noqa: BLE001 - wrap unexpected provider errors
            message = self._redact(str(exc))
            logger.error("Gemini text generation failed error_type=%s", type(exc).__name__)
            raise GeminiInvocationError(
                f"Gemini text generation failed: {message}"
            ) from None

        text = _normalize_text(getattr(result, "content", result))
        if not text:
            raise GeminiResponseError("Gemini returned an empty text response")
        logger.info("Gemini text generation succeeded response_chars=%s", len(text))
        return text

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_instruction: str | None = None,
    ) -> T:
        """
        Generate structured output validated by a Pydantic model.

        Uses LangChain with_structured_output when available.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not issubclass(response_model, BaseModel):
            raise TypeError("response_model must be a Pydantic BaseModel subclass")

        messages = self._build_messages(
            prompt=prompt,
            system_instruction=system_instruction,
            structured=True,
        )

        def _structured_invoke(model: Any) -> Any:
            if not hasattr(model, "with_structured_output"):
                raise GeminiInvocationError(
                    "Configured chat model does not support structured output"
                )
            structured_model = model.with_structured_output(response_model)
            return structured_model.invoke(messages)

        logger.info(
            "Gemini structured generation started model=%s prompt_chars=%s",
            response_model.__name__,
            len(prompt.strip()),
        )
        try:
            result = self._invoke_with_key_rotation(_structured_invoke)
        except GeminiConfigurationError:
            raise
        except GeminiInvocationError:
            raise
        except ValidationError as exc:
            message = self._redact(str(exc))
            raise GeminiResponseError(
                f"Gemini structured output failed Pydantic validation: {message}"
            ) from None
        except Exception as exc:  # noqa: BLE001 - wrap unexpected provider errors
            message = self._redact(str(exc))
            logger.error(
                "Gemini structured generation failed error_type=%s",
                type(exc).__name__,
            )
            raise GeminiInvocationError(
                f"Gemini structured generation failed: {message}"
            ) from None

        if result is None:
            logger.warning(
                "Gemini structured output empty for %s; retrying JSON text fallback",
                response_model.__name__,
            )
            return self._generate_structured_via_json(
                prompt=prompt,
                response_model=response_model,
                system_instruction=system_instruction,
            )

        if isinstance(result, response_model):
            logger.info(
                "Gemini structured generation succeeded model=%s",
                response_model.__name__,
            )
            return result

        # Some LangChain versions may return a dict
        if isinstance(result, dict):
            try:
                parsed = response_model.model_validate(result)
            except ValidationError as exc:
                message = self._redact(str(exc))
                raise GeminiResponseError(
                    f"Gemini structured output failed Pydantic validation: {message}"
                ) from None
            return parsed

        raise GeminiResponseError(
            f"Gemini structured output had unexpected type: {type(result).__name__}"
        )

    def _generate_structured_via_json(
        self,
        *,
        prompt: str,
        response_model: Type[T],
        system_instruction: str | None,
    ) -> T:
        """Fallback when with_structured_output returns empty for some Gemini models."""
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        json_prompt = (
            f"{prompt.strip()}\n\n"
            "Return ONLY valid JSON matching this schema. "
            "No markdown fences, no commentary.\n\n"
            f"JSON schema:\n{schema}"
        )
        text = self.generate_text(
            json_prompt,
            system_instruction=system_instruction,
        )
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
            return response_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            message = self._redact(str(exc))
            raise GeminiResponseError(
                f"Gemini JSON fallback failed validation: {message}"
            ) from None

    def test_connection(self) -> str:
        """
        Optional low-cost connectivity check.

        Returns a short text response. Intended for manual verification only.
        """
        return self.generate_text(
            "Reply with exactly: ok",
            system_instruction=(
                "You are a connectivity probe for GrowthOS AI. "
                "Reply with the single word ok."
            ),
        )


def get_gemini_service(settings: Optional[Settings] = None) -> GeminiService:
    """Factory for a GeminiService bound to current or provided settings."""
    return GeminiService(settings=settings)
