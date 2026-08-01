"""Unit tests for optional YouTube Data API discovery (mocked HTTP only)."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import Settings  # noqa: E402
from exceptions import (  # noqa: E402
    YouTubeConfigurationError,
    YouTubeInvocationError,
    YouTubeResponseError,
)
from models import Mood, ResourceCatalogItem  # noqa: E402
from services.youtube import (  # noqa: E402
    YouTubeService,
    build_learning_search_query,
    canonical_youtube_watch_url,
    merge_catalog_with_youtube,
    parse_iso8601_duration_minutes,
    validate_youtube_video_id,
)


def _settings(**overrides: object) -> Settings:
    values: dict[str, Any] = {
        "ai_provider": "gemini",
        "gemini_api_key": "test-key",
        "youtube_api_enabled": True,
        "youtube_api_key": "yt-secret-key-123",
        "youtube_max_results": 10,
        "youtube_request_timeout_seconds": 10,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeHttpClient:
    def __init__(
        self,
        *,
        search_payload: Any | None = None,
        videos_payload: Any | None = None,
        raise_on_get: Exception | None = None,
        status_code: int = 200,
    ) -> None:
        self.search_payload = search_payload
        self.videos_payload = videos_payload
        self.raise_on_get = raise_on_get
        self.status_code = status_code
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> FakeResponse:
        params = dict(params or {})
        self.calls.append((url, params))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if "search" in url:
            return FakeResponse(self.search_payload, status_code=self.status_code)
        return FakeResponse(self.videos_payload, status_code=self.status_code)


def _search_items(*video_ids: str) -> dict[str, Any]:
    items = []
    for vid in video_ids:
        items.append(
            {
                "id": {"videoId": vid},
                "snippet": {
                    "title": f"Lesson {vid}",
                    "description": f"Tutorial about {vid}",
                    "channelTitle": "Learn Channel",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "liveBroadcastContent": "none",
                    "thumbnails": {
                        "medium": {"url": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"}
                    },
                },
            }
        )
    return {"items": items}


def _video_details(
    *entries: tuple[str, str],
) -> dict[str, Any]:
    items = []
    for video_id, duration in entries:
        items.append(
            {
                "id": video_id,
                "snippet": {
                    "title": f"Lesson {video_id}",
                    "description": "A focused tutorial",
                    "channelTitle": "Learn Channel",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "liveBroadcastContent": "none",
                    "thumbnails": {
                        "medium": {
                            "url": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                        }
                    },
                },
                "contentDetails": {"duration": duration},
                "status": {"privacyStatus": "public", "embeddable": True},
            }
        )
    return {"items": items}


class YouTubeHelperTests(unittest.TestCase):
    def test_parse_duration_minutes(self) -> None:
        self.assertEqual(parse_iso8601_duration_minutes("PT5M"), 5)
        self.assertEqual(parse_iso8601_duration_minutes("PT1H2M"), 62)
        self.assertEqual(parse_iso8601_duration_minutes("PT90S"), 2)

    def test_invalid_video_id_rejected(self) -> None:
        with self.assertRaises(YouTubeResponseError):
            validate_youtube_video_id("bad id")
        with self.assertRaises(YouTubeResponseError):
            canonical_youtube_watch_url("short")

    def test_query_builder_focused(self) -> None:
        query = build_learning_search_query(
            learning_goal="public speaking",
            milestone_title="Clear openings",
            milestone_skills=["presence", "structure", "extra"],
            current_level="beginner",
            preferred_format="video",
        )
        self.assertIn("public speaking", query)
        self.assertIn("Clear openings", query)
        self.assertIn("presence", query)
        self.assertIn("structure", query)
        self.assertNotIn("extra", query)
        self.assertIn("beginner tutorial", query)


class YouTubeServiceTests(unittest.TestCase):
    def test_missing_key_unconfigured(self) -> None:
        service = YouTubeService(settings=_settings(youtube_api_key=""))
        self.assertFalse(service.is_configured())
        with self.assertRaises(YouTubeConfigurationError):
            service.validate_configuration()

    def test_backend_starts_without_youtube_key(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("youtube_enabled", body)
        self.assertIn("youtube_configured", body)

    def test_valid_search_normalized(self) -> None:
        client = FakeHttpClient(
            search_payload=_search_items("dQw4w9WgXcQ"),
            videos_payload=_video_details(("dQw4w9WgXcQ", "PT8M")),
        )
        service = YouTubeService(settings=_settings(), http_client=client)
        items = service.search_and_enrich(
            learning_goal="public speaking",
            milestone_title="Openings",
            milestone_skills=["presence"],
            current_level="beginner",
            available_minutes=20,
            attention_span_minutes=15,
        )
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.catalog_id, "yt-dQw4w9WgXcQ")
        self.assertEqual(item.source, "YouTube")
        self.assertEqual(item.resource_type, "video")
        self.assertEqual(str(item.url), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(item.estimated_duration_minutes, 8)
        self.assertTrue(item.is_free)
        self.assertEqual(item.metadata.get("discovery_source"), "youtube_live")
        self.assertEqual(item.metadata.get("channel_title"), "Learn Channel")

    def test_video_details_provide_duration(self) -> None:
        client = FakeHttpClient(
            search_payload=_search_items("abcdefghijk"),
            videos_payload=_video_details(("abcdefghijk", "PT12M30S")),
        )
        service = YouTubeService(settings=_settings(), http_client=client)
        details = service.get_video_details(["abcdefghijk"])
        self.assertEqual(details["abcdefghijk"]["duration_minutes"], 13)

    def test_duplicate_videos_removed(self) -> None:
        search = _search_items("abcdefghijk", "abcdefghijk", "lmnopqrstuv")
        videos = _video_details(
            ("abcdefghijk", "PT5M"),
            ("lmnopqrstuv", "PT6M"),
        )
        client = FakeHttpClient(search_payload=search, videos_payload=videos)
        service = YouTubeService(settings=_settings(), http_client=client)
        items = service.search_and_enrich(
            learning_goal="python",
            milestone_title="basics",
            available_minutes=30,
            attention_span_minutes=20,
        )
        ids = [item.catalog_id for item in items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_long_videos_filtered_for_short_sessions(self) -> None:
        client = FakeHttpClient(
            search_payload=_search_items("shortVid012", "longVid0123"),
            videos_payload=_video_details(
                ("shortVid012", "PT8M"),
                ("longVid0123", "PT2H"),
            ),
        )
        service = YouTubeService(settings=_settings(), http_client=client)
        items = service.search_and_enrich(
            learning_goal="python",
            milestone_title="loops",
            available_minutes=10,
            attention_span_minutes=10,
            mood=Mood.tired,
        )
        self.assertTrue(items)
        self.assertTrue(all(i.estimated_duration_minutes <= 25 for i in items))
        self.assertTrue(all(i.catalog_id == "yt-shortVid012" for i in items))

    def test_mood_prefers_shorter_when_tired(self) -> None:
        items = [
            ResourceCatalogItem(
                catalog_id="yt-abcdefghijk",
                title="Short",
                source="YouTube",
                resource_type="video",
                url="https://www.youtube.com/watch?v=abcdefghijk",
                description="short",
                topics=["python"],
                skills=["python"],
                difficulty="beginner",
                estimated_duration_minutes=8,
                supported_formats=["video"],
                metadata={"discovery_source": "youtube_live"},
            ),
            ResourceCatalogItem(
                catalog_id="yt-lmnopqrstuv",
                title="Long",
                source="YouTube",
                resource_type="video",
                url="https://www.youtube.com/watch?v=lmnopqrstuv",
                description="long",
                topics=["python"],
                skills=["python"],
                difficulty="beginner",
                estimated_duration_minutes=40,
                supported_formats=["video"],
                metadata={"discovery_source": "youtube_live"},
            ),
        ]
        service = YouTubeService(settings=_settings())
        filtered = service.filter_for_session(
            items,
            available_minutes=15,
            attention_span_minutes=15,
            mood=Mood.tired,
            energy_level=None,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].catalog_id, "yt-abcdefghijk")

    def test_api_key_never_in_exceptions_or_logs(self) -> None:
        secret = "yt-secret-key-123"
        client = FakeHttpClient(
            search_payload={"error": {"message": f"bad key {secret}"}},
            status_code=403,
        )
        service = YouTubeService(settings=_settings(youtube_api_key=secret), http_client=client)
        logger = logging.getLogger("services.youtube")
        with self.assertLogs(logger, level="WARNING") as captured:
            with self.assertRaises(YouTubeInvocationError) as ctx:
                service.search_learning_videos(
                    learning_goal="python",
                    milestone_title="basics",
                )
        self.assertNotIn(secret, str(ctx.exception))
        self.assertTrue(all(secret not in line for line in captured.output))

    def test_timeout_becomes_invocation_error(self) -> None:
        import httpx

        client = FakeHttpClient(raise_on_get=httpx.TimeoutException("timed out"))
        service = YouTubeService(settings=_settings(), http_client=client)
        with self.assertRaises(YouTubeInvocationError):
            service.search_learning_videos(
                learning_goal="python",
                milestone_title="basics",
            )

    def test_quota_error_handled(self) -> None:
        client = FakeHttpClient(
            search_payload={"error": {"message": "quotaExceeded"}},
            status_code=429,
        )
        service = YouTubeService(settings=_settings(), http_client=client)
        with self.assertRaises(YouTubeInvocationError) as ctx:
            service.search_learning_videos(
                learning_goal="python",
                milestone_title="basics",
            )
        self.assertIn("quota", str(ctx.exception).lower())

    def test_malformed_response_fails_safely(self) -> None:
        client = FakeHttpClient(search_payload={"unexpected": True})
        service = YouTubeService(settings=_settings(), http_client=client)
        with self.assertRaises(YouTubeResponseError):
            service.search_learning_videos(
                learning_goal="python",
                milestone_title="basics",
            )

    def test_merge_prefers_catalog(self) -> None:
        catalog = [
            ResourceCatalogItem(
                catalog_id="static-1",
                title="Catalog video",
                source="YouTube",
                resource_type="video",
                url="https://www.youtube.com/watch?v=abcdefghijk",
                description="catalog",
                topics=["python"],
                skills=["python"],
                difficulty="beginner",
                estimated_duration_minutes=10,
                supported_formats=["video"],
            )
        ]
        youtube = [
            ResourceCatalogItem(
                catalog_id="yt-abcdefghijk",
                title="Live video",
                source="YouTube",
                resource_type="video",
                url="https://www.youtube.com/watch?v=abcdefghijk",
                description="live",
                topics=["python"],
                skills=["python"],
                difficulty="beginner",
                estimated_duration_minutes=10,
                supported_formats=["video"],
                metadata={"discovery_source": "youtube_live"},
            )
        ]
        merged = merge_catalog_with_youtube(catalog, youtube)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].catalog_id, "static-1")


if __name__ == "__main__":
    unittest.main()
