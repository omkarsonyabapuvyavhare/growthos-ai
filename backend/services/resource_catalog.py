"""
Approved free-resource catalog loader for GrowthOS AI.

Loads and validates backend/data/sample_resources.json.
Does not fetch the network or invent URLs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

from pydantic import ValidationError

from config import BACKEND_ROOT, Settings, get_settings
from exceptions import ResourceCatalogError
from models import (
    Difficulty,
    EnergyLevel,
    Mood,
    ResourceCatalogItem,
)

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = BACKEND_ROOT / "data" / "sample_resources.json"

APPROVED_SOURCES = frozenset(
    {
        "YouTube",
        "Spotify",
        "RSS",
        "Dev.to",
        "Medium",
        "freeCodeCamp",
        "MDN",
        "GeeksforGeeks",
        "Real Python",
        "Hashnode",
        "Google Books",
        "Open Library",
        "Project Gutenberg",
        "arXiv",
        "LeetCode",
        "HackerRank",
        "Kaggle",
        "GitHub",
        "Reddit",
        "GitHub Discussions",
    }
)


def _validate_url_syntax(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ResourceCatalogError(f"Invalid resource URL syntax: {url}")
    host = parsed.netloc.lower()
    if host in {"example.com", "www.example.com", "localhost"} or host.endswith(
        ".example"
    ):
        raise ResourceCatalogError("Placeholder domains are not allowed in the catalog")
    return url


def default_catalog_path(settings: Optional[Settings] = None) -> Path:
    if settings is not None:
        return settings.resolve_resource_catalog_path()
    return DEFAULT_CATALOG_PATH


def load_resource_catalog(
    path: Optional[Path] = None,
    *,
    settings: Optional[Settings] = None,
) -> list[ResourceCatalogItem]:
    """
    Load and validate every catalog record.

    Malformed records fail the whole load (no silent skips).
    """
    catalog_path = (path or default_catalog_path(settings)).resolve()
    if not catalog_path.is_file():
        raise ResourceCatalogError(f"Resource catalog file not found: {catalog_path.name}")

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResourceCatalogError(
            f"Resource catalog JSON is malformed: {exc.msg}"
        ) from None
    except OSError as exc:
        raise ResourceCatalogError(
            f"Failed to read resource catalog: {type(exc).__name__}"
        ) from None

    if not isinstance(raw, list) or not raw:
        raise ResourceCatalogError("Resource catalog must be a non-empty JSON array")

    items: list[ResourceCatalogItem] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()

    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ResourceCatalogError(
                f"Catalog entry at index {index} must be an object"
            )
        try:
            item = ResourceCatalogItem.model_validate(row)
        except ValidationError as exc:
            raise ResourceCatalogError(
                f"Catalog entry at index {index} failed validation: "
                f"{exc.errors()[0].get('msg', 'invalid')}"
            ) from None

        if item.source not in APPROVED_SOURCES:
            raise ResourceCatalogError(
                f"Catalog entry {item.catalog_id} uses unapproved source '{item.source}'"
            )

        url_text = str(item.url)
        _validate_url_syntax(url_text)

        if item.catalog_id in seen_ids:
            raise ResourceCatalogError(f"Duplicate catalog_id: {item.catalog_id}")
        if url_text in seen_urls:
            raise ResourceCatalogError(f"Duplicate catalog URL for {item.catalog_id}")

        seen_ids.add(item.catalog_id)
        seen_urls.add(url_text)
        items.append(item)

    logger.info("Loaded resource catalog count=%s", len(items))
    return items


def filter_catalog(
    items: Sequence[ResourceCatalogItem],
    *,
    topics: Optional[Sequence[str]] = None,
    skills: Optional[Sequence[str]] = None,
    difficulties: Optional[Sequence[Difficulty | str]] = None,
    formats: Optional[Sequence[str]] = None,
    moods: Optional[Sequence[Mood | str]] = None,
    energy_levels: Optional[Sequence[EnergyLevel | str]] = None,
    max_duration_minutes: Optional[int] = None,
    free_only: bool = True,
) -> list[ResourceCatalogItem]:
    """Deterministic filtering helpers for catalog records."""

    topic_set = {t.strip().lower() for t in (topics or []) if t and t.strip()}
    skill_set = {s.strip().lower() for s in (skills or []) if s and s.strip()}
    difficulty_set = {
        (d.value if isinstance(d, Difficulty) else str(d)).lower()
        for d in (difficulties or [])
    }
    format_set = {f.strip().lower() for f in (formats or []) if f and f.strip()}
    mood_set = {
        (m.value if isinstance(m, Mood) else str(m)).lower() for m in (moods or [])
    }
    energy_set = {
        (e.value if isinstance(e, EnergyLevel) else str(e)).lower()
        for e in (energy_levels or [])
    }

    results: list[ResourceCatalogItem] = []
    for item in items:
        if free_only and not item.is_free:
            continue
        if topic_set:
            item_topics = {t.lower() for t in item.topics}
            if item_topics.isdisjoint(topic_set):
                # also allow skill-only matches when topics provided with skills
                if not skill_set:
                    continue
        if skill_set:
            item_skills = {s.lower() for s in item.skills}
            item_topics = {t.lower() for t in item.topics}
            if item_skills.isdisjoint(skill_set) and item_topics.isdisjoint(skill_set):
                if topic_set and not {t.lower() for t in item.topics}.isdisjoint(topic_set):
                    pass
                else:
                    continue
        if difficulty_set and item.difficulty.value not in difficulty_set:
            continue
        if format_set:
            item_formats = {f.lower() for f in item.supported_formats}
            if item_formats.isdisjoint(format_set):
                continue
        if mood_set and item.suitable_moods:
            item_moods = {m.value for m in item.suitable_moods}
            if item_moods.isdisjoint(mood_set):
                continue
        if energy_set and item.suitable_energy_levels:
            item_energy = {e.value for e in item.suitable_energy_levels}
            if item_energy.isdisjoint(energy_set):
                continue
        if (
            max_duration_minutes is not None
            and item.estimated_duration_minutes > max_duration_minutes
        ):
            continue
        results.append(item)
    return results


def catalog_item_to_metadata(item: ResourceCatalogItem) -> dict[str, Any]:
    """Serialize catalog-only fields into resources.metadata JSON."""
    return {
        "catalog_id": item.catalog_id,
        "topics": list(item.topics),
        "skills": list(item.skills),
        "supported_formats": list(item.supported_formats),
        "language": item.language,
        "requires_account": item.requires_account,
        "suitable_moods": [m.value for m in item.suitable_moods],
        "suitable_energy_levels": [e.value for e in item.suitable_energy_levels],
        "catalog_metadata": dict(item.metadata),
    }


def embedding_text_for_item(item: ResourceCatalogItem) -> str:
    """Public, non-private text used for catalog semantic indexing."""
    return (
        f"{item.title}. {item.description}. "
        f"Topics: {', '.join(item.topics)}. "
        f"Skills: {', '.join(item.skills)}. "
        f"Formats: {', '.join(item.supported_formats)}."
    )
