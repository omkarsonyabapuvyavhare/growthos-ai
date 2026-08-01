"""
Typed models for GrowthOS AI semantic vector memory.

FAISS stores vectors; these models describe memory records and search hits.
Structured domain entities remain in SQLite.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryRecordType(str, Enum):
    """Initial memory categories. Additional string values may be used later."""

    aspiration = "aspiration"
    profile = "profile"
    goal = "goal"
    reflection = "reflection"
    preference = "preference"
    adaptation = "adaptation"
    resource = "resource"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class VectorMemoryRecord(BaseModel):
    """One semantic memory entry scoped to a single user."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(..., min_length=1, max_length=128)
    user_id: int = Field(..., gt=0)
    record_type: str = Field(..., min_length=1, max_length=64)
    source_record_id: Optional[str] = Field(default=None, max_length=128)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("memory_id", "record_type", "text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("source_record_id")
    @classmethod
    def clean_source_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("metadata")
    @classmethod
    def json_serializable_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        return value

    @field_validator("created_at")
    @classmethod
    def ensure_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class VectorSearchResult(BaseModel):
    """
    One similarity hit.

    similarity_score is cosine similarity in [-1, 1] for normalized vectors
    searched with FAISS inner product (higher is more similar).
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    user_id: int
    record_type: str
    source_record_id: Optional[str] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    created_at: datetime
