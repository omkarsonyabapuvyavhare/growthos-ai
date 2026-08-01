"""
Curator Agent for GrowthOS AI.

Deterministic orchestration:
1. Validate user / roadmap / milestone ownership
2. Sync approved free-resource catalog into SQLite
3. Optionally discover live YouTube candidates (fallback to catalog)
4. Filter candidates (with staged relaxation)
5. Semantic rank via dedicated catalog FAISS index
6. Gemini ranks known candidate IDs only (never URLs)
7. Composite score + transactional recommendation persistence

Semantic ranking uses a dedicated catalog FAISS index, not user memory.
YouTube discovery is optional and never invents URLs for Gemini.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from config import Settings, get_settings
from exceptions import (
    CuratorAgentError,
    CuratorContextError,
    CuratorOwnershipError,
    CuratorPersistenceError,
    CuratorRankingError,
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    ResourceCatalogError,
    YouTubeConfigurationError,
    YouTubeInvocationError,
    YouTubeResponseError,
)
from models import (
    CuratedRecommendation,
    CuratorAgentResult,
    CuratorRankedItem,
    CuratorRankingGeneration,
    Difficulty,
    EnergyLevel,
    Mood,
    RecommendationStatus,
    ResourceCandidate,
    ResourceCatalogItem,
)
from services.catalog_index import CatalogSemanticIndex
from services.database import (
    create_recommendations_bundle,
    get_active_recommendations_for_milestone,
    get_milestone_by_id,
    get_roadmap_by_id,
    get_user_by_id,
    get_user_profile_by_user_id,
    upsert_catalog_resources,
)
from services.ai_provider import get_ai_provider
from services.resource_catalog import (
    catalog_item_to_metadata,
    load_resource_catalog,
)
from services.youtube import (
    YOUTUBE_FALLBACK_NOTE,
    YouTubeService,
    merge_catalog_with_youtube,
)

logger = logging.getLogger(__name__)

# Composite ranking weights (sum = 1.0)
WEIGHT_SEMANTIC = 0.25
WEIGHT_SKILL = 0.20
WEIGHT_DIFFICULTY = 0.15
WEIGHT_FORMAT = 0.10
WEIGHT_DURATION = 0.10
WEIGHT_MOOD = 0.05
WEIGHT_ENERGY = 0.05
WEIGHT_GEMINI = 0.10

DEFAULT_CANDIDATE_POOL = 12

CURATOR_SYSTEM_INSTRUCTION = """
You are GrowthOS AI's Curator Agent.

Rank only the provided candidate learning resources for the user's current milestone.
Optimize for relevance and growth, not endless engagement.

Rules:
- Return only candidate_id values from the provided list.
- Never invent or return URLs.
- Prefer resources that fit skills, difficulty, time, and preferred formats.
- Mood and energy affect suitability notes, not forced exclusion.
- Keep reasons short and practical.
- Do not promise guaranteed mastery.
""".strip()

PersistRecommendationsFn = Callable[..., list[dict[str, Any]]]


class SupportsStructuredGeneration(Protocol):
    def generate_structured(
        self,
        prompt: str,
        response_model: type[CuratorRankingGeneration],
        *,
        system_instruction: str | None = None,
    ) -> CuratorRankingGeneration: ...


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_tokens(values: Sequence[Any]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _difficulty_rank(value: Difficulty | str) -> int:
    text = value.value if isinstance(value, Difficulty) else str(value)
    order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    return order.get(text.lower(), 1)


def difficulty_band(level: str) -> set[str]:
    rank = _difficulty_rank(level)
    allowed = {"beginner", "intermediate", "advanced"}
    selected = {name for name, value in {
        "beginner": 0,
        "intermediate": 1,
        "advanced": 2,
    }.items() if abs(value - rank) <= 1}
    return selected or allowed


def compute_skill_overlap(milestone_skills: Sequence[str], item: ResourceCatalogItem) -> float:
    target = _normalize_tokens(milestone_skills)
    if not target:
        return 0.0
    haystack = _normalize_tokens(list(item.skills) + list(item.topics) + [item.title])
    hits = 0
    for skill in target:
        if any(skill in token or token in skill for token in haystack):
            hits += 1
    return min(1.0, hits / max(len(target), 1))


def compute_difficulty_fit(user_level: str, item_difficulty: Difficulty) -> float:
    delta = abs(_difficulty_rank(user_level) - _difficulty_rank(item_difficulty))
    return {0: 1.0, 1: 0.7, 2: 0.3}.get(delta, 0.0)


def compute_format_fit(
    preferred_formats: Sequence[str],
    item_formats: Sequence[str],
) -> float:
    preferred = _normalize_tokens(preferred_formats)
    supported = _normalize_tokens(item_formats)
    if not preferred:
        return 0.5
    if preferred & supported:
        return 1.0
    # soft aliases
    aliases = {
        "video": {"watch", "video"},
        "watch": {"watch", "video"},
        "read": {"read", "article", "documentation", "book"},
        "article": {"read", "article"},
        "listen": {"listen", "podcast"},
        "podcast": {"listen", "podcast"},
        "practice": {"practice"},
    }
    expanded: set[str] = set()
    for fmt in preferred:
        expanded |= aliases.get(fmt, {fmt})
    return 1.0 if expanded & supported else 0.2


def compute_duration_fit(
    estimated_minutes: int,
    *,
    available_minutes: Optional[int],
    preferred_session_minutes: int,
    attention_span_minutes: int,
) -> float:
    budget = preferred_session_minutes
    if available_minutes is not None:
        budget = min(budget, available_minutes)
    budget = min(budget, attention_span_minutes) if attention_span_minutes > 0 else budget
    budget = max(budget, 5)
    if estimated_minutes <= budget:
        return 1.0
    if estimated_minutes <= budget * 1.5:
        return 0.7
    if estimated_minutes <= budget * 2.5:
        return 0.4
    return 0.15


def compute_mood_fit(mood: Optional[Mood], item: ResourceCatalogItem) -> float:
    if mood is None or not item.suitable_moods:
        return 0.5
    return 1.0 if mood in item.suitable_moods else 0.35


def compute_energy_fit(
    energy: Optional[EnergyLevel],
    item: ResourceCatalogItem,
) -> float:
    if energy is None or not item.suitable_energy_levels:
        return 0.5
    return 1.0 if energy in item.suitable_energy_levels else 0.35


def composite_score(
    candidate: ResourceCandidate,
    gemini_score: float,
) -> float:
    total = (
        WEIGHT_SEMANTIC * candidate.semantic_score
        + WEIGHT_SKILL * candidate.skill_overlap_score
        + WEIGHT_DIFFICULTY * candidate.difficulty_fit_score
        + WEIGHT_FORMAT * candidate.format_fit_score
        + WEIGHT_DURATION * candidate.duration_fit_score
        + WEIGHT_MOOD * candidate.mood_fit_score
        + WEIGHT_ENERGY * candidate.energy_fit_score
        + WEIGHT_GEMINI * gemini_score
    )
    return max(0.0, min(1.0, round(total, 4)))


def build_curator_prompt(
    *,
    milestone: dict[str, Any],
    profile: dict[str, Any],
    candidates: Sequence[ResourceCandidate],
    mood: Optional[Mood],
    energy_level: Optional[EnergyLevel],
    available_minutes: Optional[int],
    preferred_format: Optional[str],
    limit: int,
) -> str:
    preferred_formats = list(profile.get("preferred_formats") or [])
    if preferred_format:
        preferred_formats = [preferred_format, *preferred_formats]

    payload = {
        "user_stated": {
            "current_level": profile.get("current_level"),
            "preferred_formats": preferred_formats,
            "preferred_session_minutes": profile.get("preferred_session_minutes"),
            "attention_span_minutes": profile.get("attention_span_minutes"),
            "daily_available_minutes": profile.get("daily_available_minutes"),
        },
        "optional_session_context": {
            "mood": mood.value if mood else None,
            "energy_level": energy_level.value if energy_level else None,
            "available_minutes": available_minutes,
            "note": "Mood affects daily suitability notes, not hard exclusion.",
        },
        "milestone": {
            "title": milestone.get("title"),
            "description": milestone.get("description"),
            "skills": milestone.get("skills") or [],
            "completion_criteria": milestone.get("completion_criteria") or "",
            "difficulty": milestone.get("difficulty"),
            "estimated_minutes": milestone.get("estimated_minutes"),
        },
        "ranking_request": {
            "limit": limit,
            "return_only_candidate_ids": True,
            "never_return_urls": True,
        },
        "candidates": [
            {
                "candidate_id": c.catalog_id,
                "title": c.title,
                "source": c.source,
                "resource_type": c.resource_type,
                "difficulty": c.difficulty.value,
                "estimated_duration_minutes": c.estimated_duration_minutes,
                "supported_formats": c.supported_formats,
                "skills": c.skills,
                "topics": c.topics,
                "description": c.description,
                "deterministic_score": c.deterministic_score,
            }
            for c in candidates
        ],
    }
    return (
        "Rank the best free learning resources for this milestone.\n"
        "Use only candidate_id values from candidates.\n"
        "Do not include URLs.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def validate_gemini_ranking(
    generation: CuratorRankingGeneration,
    *,
    allowed_ids: set[str],
    limit: int,
) -> list[CuratorRankedItem]:
    if not generation.selections:
        raise CuratorRankingError("Gemini ranking returned no selections")
    if len(generation.selections) > limit:
        raise CuratorRankingError("Gemini ranking exceeded requested limit")

    seen: set[str] = set()
    validated: list[CuratorRankedItem] = []
    for item in generation.selections:
        candidate_id = item.candidate_id.strip()
        if candidate_id not in allowed_ids:
            raise CuratorRankingError("Gemini selected an unknown candidate_id")
        if candidate_id in seen:
            raise CuratorRankingError("Gemini selected duplicate candidate_id values")
        if not (0.0 <= item.relevance_score <= 1.0):
            raise CuratorRankingError("Gemini relevance_score out of range")
        seen.add(candidate_id)
        validated.append(item)
    return validated


class CuratorAgent:
    """Select and explain free resources for a roadmap milestone."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        gemini_service: Optional[SupportsStructuredGeneration] = None,
        catalog_index: Optional[CatalogSemanticIndex] = None,
        catalog_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        persist_recommendations: Optional[PersistRecommendationsFn] = None,
        allow_deterministic_fallback: bool = True,
        youtube_service: Optional[YouTubeService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gemini = gemini_service or get_ai_provider(settings=self._settings)
        self._catalog_index = catalog_index
        self._catalog_path = catalog_path
        self._db_path = db_path
        self._persist_recommendations = persist_recommendations
        self._allow_deterministic_fallback = allow_deterministic_fallback
        self._youtube = youtube_service
        self._youtube_resolved = youtube_service is not None

    def _get_catalog_index(self) -> CatalogSemanticIndex:
        if self._catalog_index is None:
            self._catalog_index = CatalogSemanticIndex(settings=self._settings)
        return self._catalog_index

    def _get_youtube_service(self) -> Optional[YouTubeService]:
        if self._youtube_resolved:
            return self._youtube
        self._youtube_resolved = True
        if self._settings.youtube_api_enabled and self._settings.is_youtube_configured():
            self._youtube = YouTubeService(settings=self._settings)
        else:
            self._youtube = None
        return self._youtube

    def _discover_youtube_candidates(
        self,
        *,
        profile: dict[str, Any],
        milestone: dict[str, Any],
        mood: Optional[Mood],
        energy_level: Optional[EnergyLevel],
        available_minutes: Optional[int],
        preferred_format: Optional[str],
    ) -> tuple[list[ResourceCatalogItem], str]:
        """
        Optionally fetch live YouTube candidates.

        Returns (items, safe_note). On any YouTube failure, returns ([], fallback note).
        """
        service = self._get_youtube_service()
        if service is None or not service.is_enabled() or not service.is_configured():
            return [], ""

        learning_goal = str(
            milestone.get("title")
            or profile.get("aspiration")
            or profile.get("target_outcome")
            or "learning"
        )
        # Prefer goal title from nested context when available
        goal_title = str(profile.get("goal_title") or "").strip()
        if goal_title:
            learning_goal = goal_title

        try:
            items = service.search_and_enrich(
                learning_goal=learning_goal,
                milestone_title=str(milestone.get("title") or ""),
                milestone_skills=list(milestone.get("skills") or []),
                current_level=str(profile.get("current_level") or "beginner"),
                preferred_language=str(profile.get("language") or "en"),
                preferred_format=preferred_format
                or (list(profile.get("preferred_formats") or ["video"])[:1] or ["video"])[0],
                available_minutes=available_minutes,
                attention_span_minutes=int(profile.get("attention_span_minutes") or 15),
                mood=mood,
                energy_level=energy_level,
                result_limit=self._settings.youtube_max_results,
            )
            return items, ""
        except (YouTubeConfigurationError, YouTubeInvocationError, YouTubeResponseError) as exc:
            logger.warning(
                "YouTube discovery unavailable; using validated catalog error_type=%s",
                type(exc).__name__,
            )
            return [], YOUTUBE_FALLBACK_NOTE
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "YouTube discovery failed unexpectedly; using validated catalog error_type=%s",
                type(exc).__name__,
            )
            return [], YOUTUBE_FALLBACK_NOTE

    def _sync_youtube_subset(
        self,
        items: Sequence[ResourceCatalogItem],
        *,
        mapping: dict[str, int],
    ) -> dict[str, int]:
        """Persist only filtered YouTube-live candidates into resources (URL-keyed)."""
        youtube_rows: list[dict[str, Any]] = []
        for item in items:
            discovery = str((item.metadata or {}).get("discovery_source") or "")
            if discovery != "youtube_live":
                continue
            if item.catalog_id in mapping:
                continue
            youtube_rows.append(
                {
                    "catalog_id": item.catalog_id,
                    "title": item.title,
                    "source": item.source,
                    "resource_type": item.resource_type,
                    "url": str(item.url),
                    "description": item.description,
                    "difficulty": item.difficulty.value,
                    "estimated_duration_minutes": item.estimated_duration_minutes,
                    "is_free": item.is_free,
                    "metadata": catalog_item_to_metadata(item),
                }
            )
        if not youtube_rows:
            return mapping
        youtube_mapping = upsert_catalog_resources(youtube_rows, db_path=self._db_path)
        mapping.update(youtube_mapping)
        return mapping

    def _load_context(
        self,
        user_id: int,
        roadmap_id: int,
        milestone_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if user_id <= 0 or roadmap_id <= 0 or milestone_id <= 0:
            raise CuratorContextError(
                "user_id, roadmap_id, and milestone_id must be positive integers"
            )

        user = get_user_by_id(user_id, db_path=self._db_path)
        if user is None:
            raise CuratorContextError("User not found")

        roadmap = get_roadmap_by_id(roadmap_id, db_path=self._db_path)
        if roadmap is None:
            raise CuratorContextError("Roadmap not found")
        if int(roadmap["user_id"]) != user_id:
            raise CuratorOwnershipError("Roadmap does not belong to the requested user")

        milestone = get_milestone_by_id(milestone_id, db_path=self._db_path)
        if milestone is None:
            raise CuratorContextError("Milestone not found")
        if int(milestone["user_id"]) != user_id:
            raise CuratorOwnershipError("Milestone does not belong to the requested user")
        if int(milestone["roadmap_id"]) != roadmap_id:
            raise CuratorOwnershipError("Milestone does not belong to the requested roadmap")

        profile = get_user_profile_by_user_id(user_id, db_path=self._db_path)
        if profile is None:
            raise CuratorContextError("User profile not found")

        return user, profile, roadmap, milestone

    def _sync_catalog(self) -> tuple[list[ResourceCatalogItem], dict[str, int]]:
        items = load_resource_catalog(
            self._catalog_path,
            settings=self._settings,
        )
        rows: list[dict[str, Any]] = []
        for item in items:
            rows.append(
                {
                    "catalog_id": item.catalog_id,
                    "title": item.title,
                    "source": item.source,
                    "resource_type": item.resource_type,
                    "url": str(item.url),
                    "description": item.description,
                    "difficulty": item.difficulty.value,
                    "estimated_duration_minutes": item.estimated_duration_minutes,
                    "is_free": item.is_free,
                    "metadata": catalog_item_to_metadata(item),
                }
            )
        mapping = upsert_catalog_resources(rows, db_path=self._db_path)
        return items, mapping

    def _filter_candidates(
        self,
        items: Sequence[ResourceCatalogItem],
        *,
        profile: dict[str, Any],
        milestone: dict[str, Any],
        mood: Optional[Mood],
        energy_level: Optional[EnergyLevel],
        available_minutes: Optional[int],
        preferred_format: Optional[str],
    ) -> list[ResourceCatalogItem]:
        """
        Staged filtering. Never relax free status.

        Relaxation order:
        1. strict (skills/topics + difficulty + format + duration + mood + energy)
        2. drop mood/energy
        3. drop format preference
        4. loosen duration to 2.5x budget
        5. keep free + difficulty band only
        """
        skills = list(milestone.get("skills") or [])
        preferred_formats = list(profile.get("preferred_formats") or [])
        if preferred_format:
            preferred_formats = [preferred_format, *preferred_formats]
        level = str(profile.get("current_level") or "beginner")
        band = difficulty_band(level)
        session = int(profile.get("preferred_session_minutes") or 20)
        attention = int(profile.get("attention_span_minutes") or session)
        budget = session
        if available_minutes is not None:
            budget = min(budget, int(available_minutes))
        budget = min(budget, attention)

        free_items = [item for item in items if item.is_free]
        if not free_items:
            raise CuratorAgentError("No free catalog resources available")

        def apply(
            *,
            require_skill: bool,
            require_format: bool,
            require_mood_energy: bool,
            duration_multiplier: float,
        ) -> list[ResourceCatalogItem]:
            max_duration = int(budget * duration_multiplier) if budget > 0 else None
            selected: list[ResourceCatalogItem] = []
            for item in free_items:
                if item.difficulty.value not in band:
                    continue
                if require_skill and compute_skill_overlap(skills, item) <= 0:
                    # soft topic match on milestone title words
                    title_tokens = _normalize_tokens(str(milestone.get("title", "")).split())
                    item_tokens = _normalize_tokens(
                        list(item.topics) + list(item.skills) + item.title.split()
                    )
                    if not (title_tokens & item_tokens):
                        continue
                if require_format and preferred_formats:
                    if compute_format_fit(preferred_formats, item.supported_formats) < 0.5:
                        continue
                if require_mood_energy:
                    if mood is not None and item.suitable_moods and mood not in item.suitable_moods:
                        continue
                    if (
                        energy_level is not None
                        and item.suitable_energy_levels
                        and energy_level not in item.suitable_energy_levels
                    ):
                        continue
                if max_duration is not None and item.estimated_duration_minutes > max_duration:
                    continue
                selected.append(item)
            return selected

        stages = [
            dict(require_skill=True, require_format=True, require_mood_energy=True, duration_multiplier=1.5),
            dict(require_skill=True, require_format=True, require_mood_energy=False, duration_multiplier=1.5),
            dict(require_skill=True, require_format=False, require_mood_energy=False, duration_multiplier=1.5),
            dict(require_skill=True, require_format=False, require_mood_energy=False, duration_multiplier=2.5),
            dict(require_skill=False, require_format=False, require_mood_energy=False, duration_multiplier=3.0),
        ]
        for stage in stages:
            matched = apply(**stage)
            if matched:
                return matched
        return list(free_items)

    def _build_candidates(
        self,
        filtered: Sequence[ResourceCatalogItem],
        *,
        mapping: dict[str, int],
        profile: dict[str, Any],
        milestone: dict[str, Any],
        mood: Optional[Mood],
        energy_level: Optional[EnergyLevel],
        available_minutes: Optional[int],
        preferred_format: Optional[str],
        semantic_scores: dict[str, float],
    ) -> list[ResourceCandidate]:
        preferred_formats = list(profile.get("preferred_formats") or [])
        if preferred_format:
            preferred_formats = [preferred_format, *preferred_formats]
        candidates: list[ResourceCandidate] = []
        for item in filtered:
            resource_id = mapping.get(item.catalog_id)
            if resource_id is None:
                continue
            skill = compute_skill_overlap(list(milestone.get("skills") or []), item)
            difficulty = compute_difficulty_fit(
                str(profile.get("current_level") or "beginner"),
                item.difficulty,
            )
            fmt = compute_format_fit(preferred_formats, item.supported_formats)
            duration = compute_duration_fit(
                item.estimated_duration_minutes,
                available_minutes=available_minutes,
                preferred_session_minutes=int(profile.get("preferred_session_minutes") or 20),
                attention_span_minutes=int(profile.get("attention_span_minutes") or 15),
            )
            mood_fit = compute_mood_fit(mood, item)
            energy_fit = compute_energy_fit(energy_level, item)
            semantic = float(semantic_scores.get(item.catalog_id, 0.0))
            deterministic = (
                WEIGHT_SEMANTIC * semantic
                + WEIGHT_SKILL * skill
                + WEIGHT_DIFFICULTY * difficulty
                + WEIGHT_FORMAT * fmt
                + WEIGHT_DURATION * duration
                + WEIGHT_MOOD * mood_fit
                + WEIGHT_ENERGY * energy_fit
            ) / (1.0 - WEIGHT_GEMINI)
            candidates.append(
                ResourceCandidate(
                    catalog_id=item.catalog_id,
                    resource_id=resource_id,
                    title=item.title,
                    source=item.source,
                    resource_type=item.resource_type,
                    url=item.url,
                    description=item.description,
                    topics=list(item.topics),
                    skills=list(item.skills),
                    difficulty=item.difficulty,
                    estimated_duration_minutes=item.estimated_duration_minutes,
                    supported_formats=list(item.supported_formats),
                    suitable_moods=list(item.suitable_moods),
                    suitable_energy_levels=list(item.suitable_energy_levels),
                    semantic_score=semantic,
                    skill_overlap_score=skill,
                    difficulty_fit_score=difficulty,
                    format_fit_score=fmt,
                    duration_fit_score=duration,
                    mood_fit_score=mood_fit,
                    energy_fit_score=energy_fit,
                    deterministic_score=max(0.0, min(1.0, round(deterministic, 4))),
                )
            )
        candidates.sort(key=lambda c: c.deterministic_score, reverse=True)
        return candidates

    def _result_from_rows(
        self,
        *,
        user_id: int,
        roadmap_id: int,
        milestone_id: int,
        rows: Sequence[dict[str, Any]],
        reused_existing: bool,
        used_deterministic_fallback: bool = False,
        ranking_notes: str = "",
        candidate_count: int = 0,
    ) -> CuratorAgentResult:
        recommendations: list[CuratedRecommendation] = []
        for row in rows:
            resource = row.get("resource") or {}
            meta = row.get("metadata") or {}
            catalog_id = str(
                meta.get("catalog_id")
                or (resource.get("metadata") or {}).get("catalog_id")
                or ""
            )
            recommendations.append(
                CuratedRecommendation(
                    id=int(row["id"]),
                    user_id=user_id,
                    roadmap_id=int(row.get("roadmap_id") or roadmap_id),
                    milestone_id=int(row.get("milestone_id") or milestone_id),
                    resource_id=int(row["resource_id"]),
                    catalog_id=catalog_id,
                    title=str(resource.get("title") or ""),
                    source=str(resource.get("source") or ""),
                    resource_type=str(resource.get("resource_type") or ""),
                    url=resource.get("url"),
                    description=str(resource.get("description") or ""),
                    difficulty=Difficulty(str(resource.get("difficulty") or "beginner")),
                    estimated_duration_minutes=int(
                        resource.get("estimated_duration_minutes") or 1
                    ),
                    relevance_score=float(row.get("relevance_score") or 0),
                    reason=str(row.get("reason") or ""),
                    milestone_fit=str(meta.get("milestone_fit") or ""),
                    mood_suitability=str(row.get("mood_suitability") or ""),
                    suggested_use=str(meta.get("suggested_use") or ""),
                    estimated_effort=str(meta.get("estimated_effort") or ""),
                    score_breakdown=dict(meta.get("score_breakdown") or {}),
                    status=RecommendationStatus(str(row.get("status") or "suggested")),
                    recommended_at=_parse_dt(row.get("recommended_at")),
                )
            )
        created_at = (
            recommendations[0].recommended_at
            if recommendations
            else datetime.now(timezone.utc)
        )
        return CuratorAgentResult(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            recommendations=recommendations,
            candidate_count=candidate_count,
            used_deterministic_fallback=used_deterministic_fallback,
            reused_existing=reused_existing,
            ranking_notes=ranking_notes,
            created_at=created_at,
        )

    def recommend_resources(
        self,
        user_id: int,
        roadmap_id: int,
        milestone_id: int,
        *,
        mood: Mood | None = None,
        energy_level: EnergyLevel | None = None,
        available_minutes: int | None = None,
        preferred_format: str | None = None,
        limit: int = 5,
        refresh: bool = False,
    ) -> CuratorAgentResult:
        if limit <= 0 or limit > 20:
            raise CuratorContextError("limit must be between 1 and 20")
        if available_minutes is not None and available_minutes <= 0:
            raise CuratorContextError("available_minutes must be positive when provided")

        _user, profile, _roadmap, milestone = self._load_context(
            user_id,
            roadmap_id,
            milestone_id,
        )

        existing = get_active_recommendations_for_milestone(
            user_id,
            milestone_id,
            db_path=self._db_path,
        )
        if existing and not refresh:
            logger.info(
                "CuratorAgent returning existing recommendations user_id=%s milestone_id=%s count=%s",
                user_id,
                milestone_id,
                len(existing),
            )
            return self._result_from_rows(
                user_id=user_id,
                roadmap_id=roadmap_id,
                milestone_id=milestone_id,
                rows=existing,
                reused_existing=True,
                candidate_count=len(existing),
                ranking_notes="Returned existing active recommendations",
            )

        try:
            catalog_items, mapping = self._sync_catalog()
        except ResourceCatalogError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ResourceCatalogError(
                f"Failed to synchronize catalog: {type(exc).__name__}"
            ) from None

        # Prefer learning goal title from SQLite goal when present on profile context.
        # Milestone-scoped discovery still uses milestone title/skills.
        youtube_items, youtube_note = self._discover_youtube_candidates(
            profile=profile,
            milestone=milestone,
            mood=mood,
            energy_level=energy_level,
            available_minutes=available_minutes,
            preferred_format=preferred_format,
        )
        combined_items = merge_catalog_with_youtube(catalog_items, youtube_items)

        filtered = self._filter_candidates(
            combined_items,
            profile=profile,
            milestone=milestone,
            mood=mood,
            energy_level=energy_level,
            available_minutes=available_minutes,
            preferred_format=preferred_format,
        )

        # Persist only YouTube-live candidates that survived filtering (not every search hit).
        mapping = self._sync_youtube_subset(filtered, mapping=mapping)

        query = (
            f"{milestone.get('title', '')}. {milestone.get('description', '')}. "
            f"Skills: {', '.join(milestone.get('skills') or [])}. "
            f"Criteria: {milestone.get('completion_criteria', '')}"
        )
        used_fallback = False
        semantic_scores: dict[str, float] = {}
        try:
            index = self._get_catalog_index()
            index.ensure_indexed(filtered)
            ranked = index.rank_catalog_ids(
                query,
                [item.catalog_id for item in filtered],
                limit=max(DEFAULT_CANDIDATE_POOL, limit),
            )
            semantic_scores = {catalog_id: score for catalog_id, score in ranked}
        except Exception as exc:  # noqa: BLE001
            if not self._allow_deterministic_fallback:
                raise CuratorAgentError(
                    f"Semantic ranking failed: {type(exc).__name__}"
                ) from None
            used_fallback = True
            logger.warning(
                "CuratorAgent semantic ranking failed; using deterministic catalog ranking error_type=%s",
                type(exc).__name__,
            )
            semantic_scores = {item.catalog_id: 0.0 for item in filtered}

        candidates = self._build_candidates(
            filtered,
            mapping=mapping,
            profile=profile,
            milestone=milestone,
            mood=mood,
            energy_level=energy_level,
            available_minutes=available_minutes,
            preferred_format=preferred_format,
            semantic_scores=semantic_scores,
        )
        if not candidates:
            raise CuratorAgentError("No suitable free resource candidates found")

        pool = candidates[: max(DEFAULT_CANDIDATE_POOL, limit)]
        allowed_ids = {c.catalog_id for c in pool}
        by_id = {c.catalog_id: c for c in pool}
        item_by_id = {item.catalog_id: item for item in filtered}

        gemini_by_id: dict[str, CuratorRankedItem] = {}
        ranking_notes = ""
        if used_fallback:
            ranking_notes = "Deterministic catalog-only ranking (semantic fallback)"
            top = pool[:limit]
            for index_pos, candidate in enumerate(top):
                gemini_by_id[candidate.catalog_id] = CuratorRankedItem(
                    candidate_id=candidate.catalog_id,
                    relevance_score=candidate.deterministic_score,
                    reason="Selected from approved free catalog using deterministic fit scores.",
                    milestone_fit="Matches available milestone skills/topics and constraints.",
                    mood_suitability=(
                        f"Mood={mood.value}" if mood else "No mood provided"
                    ),
                    suggested_use="Use during the next focused learning session for this milestone.",
                    estimated_effort=f"About {candidate.estimated_duration_minutes} minutes",
                )
        else:
            prompt = build_curator_prompt(
                milestone=milestone,
                profile=profile,
                candidates=pool,
                mood=mood,
                energy_level=energy_level,
                available_minutes=available_minutes,
                preferred_format=preferred_format,
                limit=limit,
            )
            logger.info(
                "CuratorAgent ranking user_id=%s milestone_id=%s candidates=%s prompt_chars=%s",
                user_id,
                milestone_id,
                len(pool),
                len(prompt),
            )
            try:
                generation = self._gemini.generate_structured(
                    prompt,
                    CuratorRankingGeneration,
                    system_instruction=CURATOR_SYSTEM_INSTRUCTION,
                )
                validated = validate_gemini_ranking(
                    generation,
                    allowed_ids=allowed_ids,
                    limit=limit,
                )
            except (GeminiConfigurationError, GeminiInvocationError, GeminiResponseError):
                raise
            except CuratorRankingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise CuratorRankingError(
                    f"Gemini ranking failed: {type(exc).__name__}"
                ) from None
            for item in validated:
                gemini_by_id[item.candidate_id] = item
            ranking_notes = "Composite ranking with Gemini explanations"

        if youtube_note:
            ranking_notes = f"{ranking_notes}. {youtube_note}".strip(". ")

        selected_ids = list(gemini_by_id.keys())[:limit]
        persist_rows: list[dict[str, Any]] = []
        for candidate_id in selected_ids:
            candidate = by_id[candidate_id]
            gemini_item = gemini_by_id[candidate_id]
            final_score = composite_score(candidate, gemini_item.relevance_score)
            catalog_item = item_by_id.get(candidate_id)
            discovery_source = "static_catalog"
            if catalog_item is not None:
                discovery_source = str(
                    (catalog_item.metadata or {}).get("discovery_source")
                    or "static_catalog"
                )
            elif candidate.source == "YouTube" and candidate.catalog_id.startswith("yt-"):
                # Static seed YouTube entries use descriptive catalog ids; live uses yt-{videoId}
                video_part = candidate.catalog_id.removeprefix("yt-")
                if len(video_part) == 11:
                    discovery_source = "youtube_live"
            persist_rows.append(
                {
                    "resource_id": candidate.resource_id,
                    "relevance_score": final_score,
                    "reason": gemini_item.reason,
                    "mood_suitability": gemini_item.mood_suitability,
                    "metadata": {
                        "catalog_id": candidate.catalog_id,
                        "discovery_source": discovery_source,
                        "milestone_fit": gemini_item.milestone_fit,
                        "suggested_use": gemini_item.suggested_use,
                        "estimated_effort": gemini_item.estimated_effort,
                        "score_breakdown": {
                            "semantic": candidate.semantic_score,
                            "skill_overlap": candidate.skill_overlap_score,
                            "difficulty_fit": candidate.difficulty_fit_score,
                            "format_fit": candidate.format_fit_score,
                            "duration_fit": candidate.duration_fit_score,
                            "mood_fit": candidate.mood_fit_score,
                            "energy_fit": candidate.energy_fit_score,
                            "gemini": gemini_item.relevance_score,
                            "final": final_score,
                            "weights": {
                                "semantic": WEIGHT_SEMANTIC,
                                "skill": WEIGHT_SKILL,
                                "difficulty": WEIGHT_DIFFICULTY,
                                "format": WEIGHT_FORMAT,
                                "duration": WEIGHT_DURATION,
                                "mood": WEIGHT_MOOD,
                                "energy": WEIGHT_ENERGY,
                                "gemini": WEIGHT_GEMINI,
                            },
                        },
                        "trusted_url": str(candidate.url),
                    },
                }
            )

        try:
            if self._persist_recommendations is not None:
                rows = self._persist_recommendations(
                    user_id=user_id,
                    roadmap_id=roadmap_id,
                    milestone_id=milestone_id,
                    recommendations=persist_rows,
                    archive_existing_active=refresh,
                )
            else:
                rows = create_recommendations_bundle(
                    user_id=user_id,
                    roadmap_id=roadmap_id,
                    milestone_id=milestone_id,
                    recommendations=persist_rows,
                    archive_existing_active=refresh,
                    db_path=self._db_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CuratorAgent persistence failed error_type=%s",
                type(exc).__name__,
            )
            raise CuratorPersistenceError(
                f"Failed to persist recommendations: {type(exc).__name__}"
            ) from None

        return self._result_from_rows(
            user_id=user_id,
            roadmap_id=roadmap_id,
            milestone_id=milestone_id,
            rows=rows,
            reused_existing=False,
            used_deterministic_fallback=used_fallback,
            ranking_notes=ranking_notes,
            candidate_count=len(candidates),
        )
