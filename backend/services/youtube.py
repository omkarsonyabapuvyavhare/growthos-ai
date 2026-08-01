"""
Optional YouTube Data API discovery for GrowthOS AI Curator.

Discovers real public video candidates only. Does not scrape, download,
or invent URLs. Gemini remains the mandatory AI provider separately.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional, Protocol, Sequence

import httpx

from config import Settings, get_settings
from exceptions import (
    YouTubeConfigurationError,
    YouTubeInvocationError,
    YouTubeResponseError,
)
from models import Difficulty, EnergyLevel, Mood, ResourceCatalogItem

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

PROMO_PATTERNS = (
    "sponsored",
    "buy now",
    "#ad",
    "affiliate",
    "limited offer",
    "click the link in bio",
)

YOUTUBE_FALLBACK_NOTE = (
    "Live YouTube discovery was temporarily unavailable; "
    "recommendations were selected from the validated free-resource catalog."
)


class SupportsHttpGet(Protocol):
    def get(self, url: str, *, params: dict[str, Any] | None = None) -> Any: ...


def _redact_secrets(text: str, api_key: str) -> str:
    cleaned = text
    key = api_key.strip()
    if key:
        cleaned = cleaned.replace(key, "[REDACTED]")
    return cleaned.replace("YOUTUBE_API_KEY=", "YOUTUBE_API_KEY=[REDACTED]")


def validate_youtube_video_id(video_id: str) -> str:
    cleaned = (video_id or "").strip()
    if not VIDEO_ID_RE.fullmatch(cleaned):
        raise YouTubeResponseError(f"Invalid YouTube video ID: {cleaned!r}")
    return cleaned


def canonical_youtube_watch_url(video_id: str) -> str:
    validated = validate_youtube_video_id(video_id)
    return f"https://www.youtube.com/watch?v={validated}"


def parse_iso8601_duration_minutes(duration: str) -> int:
    """
    Convert YouTube contentDetails.duration (ISO-8601) to whole minutes.

    Raises YouTubeResponseError when duration is missing or not positive/finite.
    """
    if not isinstance(duration, str) or not duration.strip():
        raise YouTubeResponseError("YouTube duration is missing")
    match = ISO8601_DURATION_RE.fullmatch(duration.strip())
    if match is None:
        raise YouTubeResponseError(f"Unrecognized YouTube duration: {duration!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0:
        raise YouTubeResponseError("YouTube duration must be positive")
    value = math.ceil(total_seconds / 60.0)
    if not math.isfinite(value) or value <= 0:
        raise YouTubeResponseError("YouTube duration minutes must be finite and positive")
    return int(value)


def youtube_catalog_id(video_id: str) -> str:
    return f"yt-{validate_youtube_video_id(video_id)}"


def build_learning_search_query(
    *,
    learning_goal: str,
    milestone_title: str,
    milestone_skills: Sequence[str] | None = None,
    current_level: str | None = None,
    preferred_format: str | None = None,
) -> str:
    """Build a focused public search query (no private reflection data)."""
    parts: list[str] = []
    goal = (learning_goal or "").strip()
    milestone = (milestone_title or "").strip()
    if goal:
        parts.append(goal)
    if milestone and milestone.lower() not in goal.lower():
        parts.append(milestone)
    skills = [str(s).strip() for s in (milestone_skills or []) if str(s).strip()]
    parts.extend(skills[:2])
    level = (current_level or "").strip().lower()
    if level == "beginner":
        parts.append("beginner tutorial")
    elif level == "advanced":
        parts.append("advanced lesson")
    else:
        parts.append("tutorial")
    fmt = (preferred_format or "").strip().lower()
    if fmt in {"practice", "exercise"}:
        parts.append("practice")
    elif fmt in {"guide", "read", "article"}:
        parts.append("guide")
    else:
        parts.append("lesson")
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(part)
    query = " ".join(ordered).strip()
    if not query:
        raise ValueError("YouTube search query requires a learning goal or milestone")
    return query[:200]


def _infer_difficulty(level: str | None) -> Difficulty:
    text = (level or "beginner").strip().lower()
    if text == "advanced":
        return Difficulty.advanced
    if text == "intermediate":
        return Difficulty.intermediate
    return Difficulty.beginner


def _moods_for_duration(minutes: int, mood: Mood | None) -> list[Mood]:
    if minutes <= 12:
        base = [
            Mood.tired,
            Mood.stressed,
            Mood.distracted,
            Mood.curious,
            Mood.focused,
            Mood.motivated,
        ]
    elif minutes <= 25:
        base = [Mood.curious, Mood.focused, Mood.motivated, Mood.tired]
    else:
        base = [Mood.focused, Mood.motivated, Mood.curious]
    if mood is not None and mood not in base and minutes <= 20:
        base.append(mood)
    return base


def _energy_for_duration(minutes: int) -> list[EnergyLevel]:
    if minutes <= 12:
        return [EnergyLevel.low, EnergyLevel.medium, EnergyLevel.high]
    if minutes <= 25:
        return [EnergyLevel.medium, EnergyLevel.high, EnergyLevel.low]
    return [EnergyLevel.high, EnergyLevel.medium]


def _looks_promotional(title: str, description: str) -> bool:
    haystack = f"{title}\n{description}".lower()
    return any(token in haystack for token in PROMO_PATTERNS)


def _thumbnail_url(snippet: dict[str, Any]) -> str:
    thumbs = snippet.get("thumbnails") or {}
    for key in ("medium", "high", "default"):
        entry = thumbs.get(key) or {}
        url = str(entry.get("url") or "").strip()
        if url.startswith("https://"):
            return url
    return ""


class YouTubeService:
    """
    Lazy YouTube Data API client for learning-video discovery.

    http_client may be injected for tests (must expose .get(url, params=...)).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._injected_client = http_client
        self._owned_client: httpx.Client | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    def is_configured(self) -> bool:
        return self._settings.is_youtube_configured()

    def is_enabled(self) -> bool:
        return bool(self._settings.youtube_api_enabled)

    def validate_configuration(self) -> None:
        if not self._settings.youtube_api_enabled:
            raise YouTubeConfigurationError(
                "YouTube discovery is disabled. Set YOUTUBE_API_ENABLED=true to use it."
            )
        if not self._settings.is_youtube_configured():
            raise YouTubeConfigurationError(
                "YOUTUBE_API_KEY is not configured. "
                "Set it in the backend environment before calling YouTube."
            )

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._owned_client is None:
            self._owned_client = httpx.Client(
                timeout=self._settings.youtube_request_timeout_seconds,
            )
        return self._owned_client

    def close(self) -> None:
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self.validate_configuration()
        request_params = dict(params)
        request_params["key"] = self._settings.youtube_api_key.strip()
        try:
            client = self._get_client()
            response = client.get(url, params=request_params)
        except httpx.TimeoutException as exc:
            message = _redact_secrets(str(exc), self._settings.youtube_api_key)
            logger.warning("YouTube request timed out error_type=%s", type(exc).__name__)
            raise YouTubeInvocationError(
                f"YouTube request timed out: {message}"
            ) from None
        except Exception as exc:  # noqa: BLE001
            message = _redact_secrets(str(exc), self._settings.youtube_api_key)
            logger.warning(
                "YouTube request failed error_type=%s",
                type(exc).__name__,
            )
            raise YouTubeInvocationError(
                f"YouTube request failed: {message}"
            ) from None

        status = int(getattr(response, "status_code", 0) or 0)
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            message = _redact_secrets(str(exc), self._settings.youtube_api_key)
            raise YouTubeResponseError(
                f"YouTube returned a non-JSON response: {message}"
            ) from None

        if status == 403 or status == 429:
            error_msg = ""
            if isinstance(payload, dict):
                error_msg = str((payload.get("error") or {}).get("message") or "")
            safe = _redact_secrets(error_msg or f"HTTP {status}", self._settings.youtube_api_key)
            logger.warning("YouTube quota or auth error status=%s", status)
            raise YouTubeInvocationError(f"YouTube API quota or auth error: {safe}")

        if status >= 400:
            error_msg = ""
            if isinstance(payload, dict):
                error_msg = str((payload.get("error") or {}).get("message") or "")
            safe = _redact_secrets(error_msg or f"HTTP {status}", self._settings.youtube_api_key)
            raise YouTubeInvocationError(f"YouTube API HTTP error: {safe}")

        if not isinstance(payload, dict):
            raise YouTubeResponseError("YouTube response payload must be an object")
        return payload

    def search_learning_videos(
        self,
        *,
        learning_goal: str,
        milestone_title: str,
        milestone_skills: Sequence[str] | None = None,
        current_level: str | None = None,
        preferred_language: str | None = None,
        preferred_format: str | None = None,
        result_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run YouTube search and return raw search items (snippet + videoId)."""
        query = build_learning_search_query(
            learning_goal=learning_goal,
            milestone_title=milestone_title,
            milestone_skills=milestone_skills,
            current_level=current_level,
            preferred_format=preferred_format,
        )
        limit = int(result_limit or self._settings.youtube_max_results)
        limit = max(1, min(limit, 25))
        language = (preferred_language or "en").strip()[:16] or "en"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": limit,
            "safeSearch": "moderate",
            "videoEmbeddable": "true",
            "relevanceLanguage": language,
        }
        logger.info(
            "YouTube search started query_chars=%s max_results=%s",
            len(query),
            limit,
        )
        payload = self._request_json(YOUTUBE_SEARCH_URL, params)
        items = payload.get("items")
        if items is None:
            raise YouTubeResponseError("YouTube search response missing items")
        if not isinstance(items, list):
            raise YouTubeResponseError("YouTube search items must be a list")

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            snippet = item.get("snippet") or {}
            if not isinstance(snippet, dict):
                continue
            live = str(snippet.get("liveBroadcastContent") or "none").lower()
            if live in {"live", "upcoming"}:
                continue
            video_id = str((item.get("id") or {}).get("videoId") or "").strip()
            try:
                video_id = validate_youtube_video_id(video_id)
            except YouTubeResponseError:
                continue
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            results.append(
                {
                    "video_id": video_id,
                    "title": str(snippet.get("title") or "").strip(),
                    "description": str(snippet.get("description") or "").strip(),
                    "channel_title": str(snippet.get("channelTitle") or "").strip(),
                    "published_at": str(snippet.get("publishedAt") or "").strip(),
                    "thumbnail_url": _thumbnail_url(snippet),
                    "search_query": query,
                }
            )
        return results

    def get_video_details(self, video_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Fetch contentDetails/status/snippet for validated video IDs."""
        cleaned: list[str] = []
        for raw in video_ids:
            try:
                cleaned.append(validate_youtube_video_id(str(raw)))
            except YouTubeResponseError:
                continue
        if not cleaned:
            return {}

        details: dict[str, dict[str, Any]] = {}
        # YouTube videos.list accepts up to 50 IDs per call
        for start in range(0, len(cleaned), 50):
            batch = cleaned[start : start + 50]
            payload = self._request_json(
                YOUTUBE_VIDEOS_URL,
                {
                    "part": "snippet,contentDetails,status",
                    "id": ",".join(batch),
                },
            )
            items = payload.get("items")
            if items is None:
                raise YouTubeResponseError("YouTube videos response missing items")
            if not isinstance(items, list):
                raise YouTubeResponseError("YouTube videos items must be a list")
            for item in items:
                if not isinstance(item, dict):
                    continue
                video_id = str(item.get("id") or "").strip()
                try:
                    video_id = validate_youtube_video_id(video_id)
                except YouTubeResponseError:
                    continue
                snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
                content = (
                    item.get("contentDetails")
                    if isinstance(item.get("contentDetails"), dict)
                    else {}
                )
                status = item.get("status") if isinstance(item.get("status"), dict) else {}
                privacy = str(status.get("privacyStatus") or "").lower()
                if privacy and privacy != "public":
                    continue
                if status.get("embeddable") is False:
                    continue
                live = str(snippet.get("liveBroadcastContent") or "none").lower()
                if live in {"live", "upcoming"}:
                    continue
                try:
                    duration_minutes = parse_iso8601_duration_minutes(
                        str(content.get("duration") or "")
                    )
                except YouTubeResponseError:
                    continue
                details[video_id] = {
                    "video_id": video_id,
                    "title": str(snippet.get("title") or "").strip(),
                    "description": str(snippet.get("description") or "").strip(),
                    "channel_title": str(snippet.get("channelTitle") or "").strip(),
                    "published_at": str(snippet.get("publishedAt") or "").strip(),
                    "thumbnail_url": _thumbnail_url(snippet),
                    "duration_minutes": duration_minutes,
                    "url": canonical_youtube_watch_url(video_id),
                }
        return details

    def to_catalog_item(
        self,
        *,
        video: dict[str, Any],
        milestone_skills: Sequence[str] | None,
        current_level: str | None,
        mood: Mood | None,
        search_query: str,
    ) -> ResourceCatalogItem | None:
        """Normalize an enriched video dict into ResourceCatalogItem."""
        try:
            video_id = validate_youtube_video_id(str(video.get("video_id") or ""))
            duration = int(video.get("duration_minutes") or 0)
            if duration <= 0:
                return None
            title = str(video.get("title") or "").strip()
            description = str(video.get("description") or "").strip() or title
            if not title:
                return None
            if _looks_promotional(title, description):
                return None
            skills = [str(s).strip() for s in (milestone_skills or []) if str(s).strip()]
            if not skills:
                skills = ["general learning"]
            topics = list(dict.fromkeys([*(skills[:3]), "youtube", "video tutorial"]))
            return ResourceCatalogItem(
                catalog_id=youtube_catalog_id(video_id),
                title=title[:300],
                source="YouTube",
                resource_type="video",
                url=canonical_youtube_watch_url(video_id),
                description=description[:4000],
                topics=topics[:8] or ["youtube"],
                skills=skills[:8],
                difficulty=_infer_difficulty(current_level),
                estimated_duration_minutes=min(duration, 24 * 60),
                supported_formats=["video", "watch"],
                language="en",
                is_free=True,
                requires_account=False,
                suitable_moods=_moods_for_duration(duration, mood),
                suitable_energy_levels=_energy_for_duration(duration),
                metadata={
                    "discovery_source": "youtube_live",
                    "youtube_video_id": video_id,
                    "channel_title": str(video.get("channel_title") or "").strip(),
                    "thumbnail_url": str(video.get("thumbnail_url") or "").strip(),
                    "published_at": str(video.get("published_at") or "").strip(),
                    "search_query": search_query[:200],
                },
            )
        except Exception:  # noqa: BLE001
            return None

    def filter_for_session(
        self,
        items: Sequence[ResourceCatalogItem],
        *,
        available_minutes: int | None,
        attention_span_minutes: int | None,
        mood: Mood | None,
        energy_level: EnergyLevel | None,
    ) -> list[ResourceCatalogItem]:
        """
        Duration/mood-aware filter with staged relaxation.

        Never removes every useful candidate when any remain after relaxation.
        """
        if not items:
            return []
        session_budget = available_minutes or attention_span_minutes or 20
        attention = attention_span_minutes or session_budget
        budget = max(5, min(int(session_budget), int(attention)))

        def stage(max_multiplier: float, prefer_short: bool) -> list[ResourceCatalogItem]:
            max_minutes = int(budget * max_multiplier)
            selected: list[ResourceCatalogItem] = []
            for item in items:
                minutes = item.estimated_duration_minutes
                if minutes <= 0 or minutes > max_minutes:
                    continue
                if prefer_short and mood in {Mood.tired, Mood.stressed, Mood.distracted}:
                    if minutes > max(budget, 15):
                        continue
                if mood == Mood.curious and minutes > max(budget * 2, 30):
                    # allow one exploratory later; strict stage keeps shorter
                    continue
                if energy_level == EnergyLevel.low and minutes > max(budget, 18):
                    continue
                selected.append(item)
            return selected

        prefer_short = mood in {Mood.tired, Mood.stressed, Mood.distracted}
        for multiplier in (1.0, 1.5, 2.5, 4.0):
            matched = stage(multiplier, prefer_short=prefer_short and multiplier <= 1.5)
            if matched:
                # Prefer shorter first for tired moods
                if prefer_short:
                    matched.sort(key=lambda i: i.estimated_duration_minutes)
                return matched
        return list(items)

    def search_and_enrich(
        self,
        *,
        learning_goal: str,
        milestone_title: str,
        milestone_skills: Sequence[str] | None = None,
        current_level: str | None = None,
        preferred_language: str | None = None,
        preferred_format: str | None = None,
        available_minutes: int | None = None,
        attention_span_minutes: int | None = None,
        mood: Mood | None = None,
        energy_level: EnergyLevel | None = None,
        result_limit: int | None = None,
    ) -> list[ResourceCatalogItem]:
        """
        Search → details → normalize → session filter.

        Returns ResourceCatalogItem candidates with backend-built watch URLs.
        """
        search_hits = self.search_learning_videos(
            learning_goal=learning_goal,
            milestone_title=milestone_title,
            milestone_skills=milestone_skills,
            current_level=current_level,
            preferred_language=preferred_language,
            preferred_format=preferred_format,
            result_limit=result_limit,
        )
        if not search_hits:
            return []

        details = self.get_video_details([hit["video_id"] for hit in search_hits])
        query = str(search_hits[0].get("search_query") or "")
        enriched: list[ResourceCatalogItem] = []
        seen_urls: set[str] = set()
        for hit in search_hits:
            detail = details.get(hit["video_id"])
            if detail is None:
                continue
            merged = {
                **hit,
                **detail,
                "thumbnail_url": detail.get("thumbnail_url") or hit.get("thumbnail_url"),
            }
            item = self.to_catalog_item(
                video=merged,
                milestone_skills=milestone_skills,
                current_level=current_level,
                mood=mood,
                search_query=query,
            )
            if item is None:
                continue
            url = str(item.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            enriched.append(item)

        filtered = self.filter_for_session(
            enriched,
            available_minutes=available_minutes,
            attention_span_minutes=attention_span_minutes,
            mood=mood,
            energy_level=energy_level,
        )
        logger.info(
            "YouTube discovery enriched=%s filtered=%s",
            len(enriched),
            len(filtered),
        )
        return filtered


def merge_catalog_with_youtube(
    catalog_items: Sequence[ResourceCatalogItem],
    youtube_items: Sequence[ResourceCatalogItem],
) -> list[ResourceCatalogItem]:
    """Prefer static catalog entries when URL or catalog_id collide."""
    by_url: dict[str, ResourceCatalogItem] = {}
    by_id: dict[str, ResourceCatalogItem] = {}
    ordered: list[ResourceCatalogItem] = []

    for item in catalog_items:
        url = str(item.url)
        by_url[url] = item
        by_id[item.catalog_id] = item
        ordered.append(item)

    for item in youtube_items:
        url = str(item.url)
        if url in by_url or item.catalog_id in by_id:
            continue
        by_url[url] = item
        by_id[item.catalog_id] = item
        ordered.append(item)
    return ordered


def get_youtube_service(settings: Optional[Settings] = None) -> YouTubeService:
    """Factory for a YouTubeService bound to settings."""
    return YouTubeService(settings=settings)
