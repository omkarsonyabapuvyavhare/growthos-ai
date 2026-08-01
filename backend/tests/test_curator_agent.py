"""Unit tests for CuratorAgent (no live Gemini or network calls)."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Type

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.curator_agent import (  # noqa: E402
    CuratorAgent,
    composite_score,
    validate_gemini_ranking,
)
from agents.roadmap_agent import RoadmapAgent  # noqa: E402
from exceptions import (  # noqa: E402
    CuratorContextError,
    CuratorOwnershipError,
    CuratorPersistenceError,
    CuratorRankingError,
    GeminiInvocationError,
    YouTubeInvocationError,
)
from models import (  # noqa: E402
    CuratorRankedItem,
    CuratorRankingGeneration,
    Difficulty,
    EnergyLevel,
    MilestoneGeneration,
    Mood,
    ResourceCatalogItem,
    RoadmapGeneration,
    RoadmapPhaseGeneration,
)
from services.catalog_index import CatalogSemanticIndex  # noqa: E402
from services.database import (  # noqa: E402
    count_recommendations,
    count_resources,
    create_onboarding_records,
    create_user,
    get_active_recommendations_for_milestone,
    get_resource_by_url,
    init_db,
)
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402
from services.youtube import YouTubeService  # noqa: E402
from tests.test_embedding import _settings  # noqa: E402


class KeywordEmbeddingClient:
    """Deterministic embeddings that encode keyword presence."""

    KEYS = [
        "python",
        "speaking",
        "javascript",
        "finance",
        "photo",
        "writing",
        "interview",
        "design",
        "breath",
        "posture",
        "public",
    ]

    def __init__(self, *, raise_on_call: Exception | None = None) -> None:
        self.raise_on_call = raise_on_call
        self.query_calls = 0

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        # Bias term keeps vectors non-zero for FAISS L2 normalization.
        return [1.0 if key in lowered else 0.0 for key in self.KEYS] + [0.25]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return [self._vec(text) for text in texts]


def _milestone(seq: int, title: str, skills: list[str]) -> MilestoneGeneration:
    return MilestoneGeneration(
        sequence_number=seq,
        title=title,
        description=f"Description for {title}",
        skills=skills,
        suggested_activities=[f"practice {title.lower()}"],
        completion_criteria=f"Complete {title}",
        estimated_sessions=2,
        estimated_minutes=20,
        difficulty=Difficulty.beginner,
    )


def _roadmap_generation() -> RoadmapGeneration:
    return RoadmapGeneration(
        title="Public Speaking Foundations",
        summary="Practical speaking roadmap.",
        estimated_duration_weeks=6,
        pacing_rationale="Fits short evening sessions.",
        personalization_rationale="Beginner friendly.",
        phases=[
            RoadmapPhaseGeneration(
                sequence_number=1,
                title="Foundations",
                description="Breath and posture.",
                expected_outcome="Short talk confidence.",
                milestones=[
                    _milestone(1, "Breath and posture basics", ["breath", "posture", "public speaking"]),
                    _milestone(2, "Two-minute outline drill", ["speech structure", "public speaking"]),
                ],
            ),
            RoadmapPhaseGeneration(
                sequence_number=2,
                title="Delivery",
                description="Practice delivery.",
                expected_outcome="Deliver a 5-minute talk.",
                milestones=[
                    _milestone(1, "Story framing", ["storytelling", "public speaking"]),
                ],
            ),
        ],
    )


class FakeRoadmapGemini:
    def generate_structured(self, prompt: str, response_model: Type[Any], **kwargs: Any) -> Any:
        assert response_model is RoadmapGeneration
        return _roadmap_generation()


class FakeCuratorGemini:
    def __init__(
        self,
        *,
        selections: list[CuratorRankedItem] | None = None,
        raise_error: Exception | None = None,
        choose_from_prompt: bool = True,
    ) -> None:
        self.selections = selections
        self.raise_error = raise_error
        self.choose_from_prompt = choose_from_prompt
        self.prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[CuratorRankingGeneration],
        *,
        system_instruction: str | None = None,
    ) -> CuratorRankingGeneration:
        self.prompts.append(prompt)
        if self.raise_error is not None:
            raise self.raise_error
        assert response_model is CuratorRankingGeneration
        if self.selections is not None:
            return CuratorRankingGeneration(selections=self.selections)
        # Pick candidate_ids from the candidates list; honor requested limit.
        limit = 3
        for line in prompt.splitlines():
            if '"limit":' in line:
                try:
                    limit = int(line.split(":", 1)[1].strip().rstrip(","))
                except ValueError:
                    pass
                break
        ids: list[str] = []
        in_candidates = False
        for line in prompt.splitlines():
            if '"candidates":' in line:
                in_candidates = True
                continue
            if in_candidates and '"candidate_id":' in line:
                value = line.split(":", 1)[1].strip().strip(",").strip('"')
                if value and value not in ids:
                    ids.append(value)
            if len(ids) >= limit:
                break
        selections = [
            CuratorRankedItem(
                candidate_id=candidate_id,
                relevance_score=0.8 - (index * 0.1),
                reason=f"Useful for milestone ({candidate_id})",
                milestone_fit="Supports current skills",
                mood_suitability="Suitable",
                suggested_use="Study in next session",
                estimated_effort="15-30 minutes",
            )
            for index, candidate_id in enumerate(ids[:limit])
        ]
        return CuratorRankingGeneration(selections=selections)


class CuratorAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_curator_")
        root = Path(self._tmpdir.name)
        self.db_path = root / "test.db"
        init_db(self.db_path)
        user, profile, goal = create_onboarding_records(
            display_name="Ada Lovelace",
            aspiration="Become a confident presenter",
            motivation="Lead team updates calmly",
            current_level="beginner",
            target_outcome="Deliver a 10-minute talk without notes",
            learning_style="mixed",
            preferred_formats=["video", "practice"],
            daily_available_minutes=45,
            preferred_session_minutes=20,
            attention_span_minutes=15,
            preferred_learning_time="evening",
            habits=["journal"],
            distractions=["phone"],
            goal_title="Improve public speaking",
            goal_description="Free-text goal",
            db_path=self.db_path,
        )
        self.user = user
        self.profile = profile
        self.goal = goal

        self.catalog_store = FAISSVectorStore(
            index_path=root / "catalog" / "index.faiss",
            metadata_path=root / "catalog" / "metadata.json",
            autosave=False,
        )
        self.embed_client = KeywordEmbeddingClient()
        self.catalog_index = CatalogSemanticIndex(
            settings=_settings(gemini_api_key="test-key"),
            embedding_service=GeminiEmbeddingService(
                settings=_settings(gemini_api_key="test-key"),
                embedding_client=self.embed_client,
            ),
            vector_store=self.catalog_store,
        )
        roadmap_agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeRoadmapGemini(),
            memory_service=None,
            db_path=self.db_path,
            skip_memory_retrieval=True,
        )
        self.roadmap_result = roadmap_agent.generate_roadmap(
            int(user["id"]),
            int(goal["id"]),
        )
        self.milestone = self.roadmap_result.active_milestone
        assert self.milestone is not None
        self.gemini = FakeCuratorGemini()
        self.agent = CuratorAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_valid_recommendations(self) -> None:
        result = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=3,
        )
        self.assertEqual(result.user_id, int(self.user["id"]))
        self.assertEqual(result.roadmap_id, self.roadmap_result.roadmap.id)
        self.assertEqual(result.milestone_id, self.milestone.id)
        self.assertGreaterEqual(len(result.recommendations), 1)
        self.assertLessEqual(len(result.recommendations), 3)
        for rec in result.recommendations:
            self.assertTrue(str(rec.url).startswith("https://"))
            self.assertNotIn("example.com", str(rec.url))
            self.assertEqual(rec.user_id, int(self.user["id"]))

    def test_gemini_never_controls_url(self) -> None:
        result = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
        )
        prompt = self.gemini.prompts[0]
        self.assertIn("never_return_urls", prompt)
        for rec in result.recommendations:
            self.assertTrue(rec.catalog_id)
            self.assertIn("final", rec.score_breakdown)
            self.assertIn("gemini", rec.score_breakdown)

    def test_unknown_gemini_id_rejected(self) -> None:
        agent = CuratorAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeCuratorGemini(
                selections=[
                    CuratorRankedItem(
                        candidate_id="not-in-catalog",
                        relevance_score=0.9,
                        reason="x",
                        milestone_fit="x",
                        mood_suitability="x",
                        suggested_use="x",
                        estimated_effort="x",
                    )
                ]
            ),
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
        )
        with self.assertRaises(CuratorRankingError):
            agent.recommend_resources(
                int(self.user["id"]),
                self.roadmap_result.roadmap.id,
                self.milestone.id,
            )
        self.assertEqual(count_recommendations(db_path=self.db_path), 0)

    def test_duplicate_gemini_selections_rejected(self) -> None:
        with self.assertRaises(CuratorRankingError):
            validate_gemini_ranking(
                CuratorRankingGeneration(
                    selections=[
                        CuratorRankedItem(
                            candidate_id="a",
                            relevance_score=0.5,
                            reason="r",
                            milestone_fit="m",
                            mood_suitability="m",
                            suggested_use="s",
                            estimated_effort="e",
                        ),
                        CuratorRankedItem(
                            candidate_id="a",
                            relevance_score=0.4,
                            reason="r",
                            milestone_fit="m",
                            mood_suitability="m",
                            suggested_use="s",
                            estimated_effort="e",
                        ),
                    ]
                ),
                allowed_ids={"a"},
                limit=5,
            )

    def test_invalid_score_rejected(self) -> None:
        with self.assertRaises(Exception):
            CuratorRankedItem(
                candidate_id="a",
                relevance_score=1.5,
                reason="r",
                milestone_fit="m",
                mood_suitability="m",
                suggested_use="s",
                estimated_effort="e",
            )

    def test_gemini_failure_writes_no_rows(self) -> None:
        agent = CuratorAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeCuratorGemini(
                raise_error=GeminiInvocationError("down")
            ),
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
        )
        with self.assertRaises(GeminiInvocationError):
            agent.recommend_resources(
                int(self.user["id"]),
                self.roadmap_result.roadmap.id,
                self.milestone.id,
            )
        self.assertEqual(count_recommendations(db_path=self.db_path), 0)

    def test_persistence_failure_rolls_back(self) -> None:
        def boom(**_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("insert failed")

        agent = CuratorAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
            persist_recommendations=boom,
        )
        with self.assertRaises(CuratorPersistenceError):
            agent.recommend_resources(
                int(self.user["id"]),
                self.roadmap_result.roadmap.id,
                self.milestone.id,
            )
        self.assertEqual(count_recommendations(db_path=self.db_path), 0)

    def test_refresh_false_avoids_duplicate_active_set(self) -> None:
        first = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
        )
        second = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
            refresh=False,
        )
        self.assertTrue(second.reused_existing)
        self.assertEqual(
            [rec.id for rec in first.recommendations],
            [rec.id for rec in second.recommendations],
        )
        active = get_active_recommendations_for_milestone(
            int(self.user["id"]),
            self.milestone.id,
            db_path=self.db_path,
        )
        self.assertEqual(len(active), len(first.recommendations))

    def test_refresh_true_archives_previous(self) -> None:
        first = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
        )
        second = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
            refresh=True,
        )
        self.assertFalse(second.reused_existing)
        self.assertNotEqual(
            [rec.id for rec in first.recommendations],
            [rec.id for rec in second.recommendations],
        )
        self.assertGreaterEqual(
            count_recommendations(db_path=self.db_path, status="archived"),
            1,
        )

    def test_ownership_enforced(self) -> None:
        other = create_user("Other", db_path=self.db_path)
        with self.assertRaises(CuratorOwnershipError):
            self.agent.recommend_resources(
                int(other["id"]),
                self.roadmap_result.roadmap.id,
                self.milestone.id,
            )

    def test_missing_context(self) -> None:
        with self.assertRaises(CuratorContextError):
            self.agent.recommend_resources(99999, 1, 1)

    def test_mood_energy_do_not_overfilter(self) -> None:
        result = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            mood=Mood.tired,
            energy_level=EnergyLevel.low,
            available_minutes=10,
            limit=2,
        )
        self.assertGreaterEqual(len(result.recommendations), 1)

    def test_semantic_failure_uses_fallback(self) -> None:
        broken = CatalogSemanticIndex(
            settings=_settings(gemini_api_key="test-key"),
            embedding_service=GeminiEmbeddingService(
                settings=_settings(gemini_api_key="test-key"),
                embedding_client=KeywordEmbeddingClient(
                    raise_on_call=RuntimeError("embed down")
                ),
            ),
            vector_store=self.catalog_store,
        )
        agent = CuratorAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            catalog_index=broken,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
            allow_deterministic_fallback=True,
        )
        result = agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
        )
        self.assertTrue(result.used_deterministic_fallback)
        self.assertGreaterEqual(len(result.recommendations), 1)

    def test_composite_score_combines_signals(self) -> None:
        from models import ResourceCandidate
        from pydantic import HttpUrl

        candidate = ResourceCandidate(
            catalog_id="x",
            resource_id=1,
            title="t",
            source="MDN",
            resource_type="article",
            url=HttpUrl("https://developer.mozilla.org/en-US/"),
            description="d",
            difficulty=Difficulty.beginner,
            estimated_duration_minutes=10,
            semantic_score=1.0,
            skill_overlap_score=1.0,
            difficulty_fit_score=1.0,
            format_fit_score=1.0,
            duration_fit_score=1.0,
            mood_fit_score=1.0,
            energy_fit_score=1.0,
            deterministic_score=1.0,
        )
        score = composite_score(candidate, 1.0)
        self.assertEqual(score, 1.0)

    def test_secrets_not_logged(self) -> None:
        with self.assertLogs(level=logging.INFO) as captured:
            self.agent.recommend_resources(
                int(self.user["id"]),
                self.roadmap_result.roadmap.id,
                self.milestone.id,
                limit=1,
            )
        joined = "\n".join(captured.output)
        self.assertNotIn("test-key", joined)
        self.assertNotIn("GEMINI_API_KEY=", joined)

    def test_combines_catalog_and_youtube_candidates(self) -> None:
        yt_item = ResourceCatalogItem(
            catalog_id="yt-AbCdEfGhIjK",
            title="Live speaking practice",
            source="YouTube",
            resource_type="video",
            url="https://www.youtube.com/watch?v=AbCdEfGhIjK",
            description="Practice openings with a coach",
            topics=["public speaking", "breath", "posture"],
            skills=["breath", "posture", "public speaking"],
            difficulty=Difficulty.beginner,
            estimated_duration_minutes=10,
            supported_formats=["video", "watch"],
            suitable_moods=[Mood.focused, Mood.tired, Mood.curious, Mood.motivated],
            suitable_energy_levels=[EnergyLevel.low, EnergyLevel.medium, EnergyLevel.high],
            metadata={
                "discovery_source": "youtube_live",
                "youtube_video_id": "AbCdEfGhIjK",
                "channel_title": "Speak Better",
                "thumbnail_url": "https://i.ytimg.com/vi/AbCdEfGhIjK/mqdefault.jpg",
            },
        )

        class FakeYouTube(YouTubeService):
            def __init__(self) -> None:
                super().__init__(settings=_settings(youtube_api_enabled=True, youtube_api_key="yt-key"))

            def is_enabled(self) -> bool:
                return True

            def is_configured(self) -> bool:
                return True

            def search_and_enrich(self, **kwargs: Any) -> list[ResourceCatalogItem]:
                return [yt_item]

        gemini = FakeCuratorGemini()
        agent = CuratorAgent(
            settings=_settings(
                gemini_api_key="test-key",
                youtube_api_enabled=True,
                youtube_api_key="yt-key",
            ),
            gemini_service=gemini,
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
            youtube_service=FakeYouTube(),
        )
        result = agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=5,
            refresh=True,
        )
        self.assertGreaterEqual(len(result.recommendations), 1)
        self.assertLessEqual(len(result.recommendations), 5)
        prompt = gemini.prompts[0]
        self.assertIn("yt-AbCdEfGhIjK", prompt)
        self.assertIn("never_return_urls", prompt)
        # Prompt may list trusted URLs for context, but Gemini must only return candidate IDs
        for rec in result.recommendations:
            self.assertTrue(str(rec.url).startswith("https://"))
            self.assertIn("final", rec.score_breakdown)

    def test_gemini_cannot_generate_youtube_url(self) -> None:
        result = self.agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=2,
        )
        for rec in result.recommendations:
            self.assertEqual(rec.score_breakdown.get("final") is not None, True)
            # Trusted URL always comes from candidate/DB, not Gemini free text
            self.assertTrue(str(rec.url))
            self.assertNotIn("example.com", str(rec.url))

    def test_youtube_failure_falls_back_to_catalog(self) -> None:
        class FailingYouTube(YouTubeService):
            def __init__(self) -> None:
                super().__init__(
                    settings=_settings(youtube_api_enabled=True, youtube_api_key="yt-key")
                )

            def is_enabled(self) -> bool:
                return True

            def is_configured(self) -> bool:
                return True

            def search_and_enrich(self, **kwargs: Any) -> list[ResourceCatalogItem]:
                raise YouTubeInvocationError("quotaExceeded")

        agent = CuratorAgent(
            settings=_settings(
                gemini_api_key="test-key",
                youtube_api_enabled=True,
                youtube_api_key="yt-key",
            ),
            gemini_service=self.gemini,
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
            youtube_service=FailingYouTube(),
        )
        result = agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=3,
            refresh=True,
        )
        self.assertGreaterEqual(len(result.recommendations), 1)
        self.assertLessEqual(len(result.recommendations), 5)
        self.assertIn("YouTube discovery was temporarily unavailable", result.ranking_notes)

    def test_youtube_selected_resources_persist_idempotently(self) -> None:
        yt_item = ResourceCatalogItem(
            catalog_id="yt-ZyXwVuTsRqP",
            title="Idempotent speaking clip",
            source="YouTube",
            resource_type="video",
            url="https://www.youtube.com/watch?v=ZyXwVuTsRqP",
            description="Short practice clip",
            topics=["public speaking", "breath"],
            skills=["breath", "posture", "public speaking"],
            difficulty=Difficulty.beginner,
            estimated_duration_minutes=8,
            supported_formats=["video", "watch"],
            suitable_moods=[Mood.focused, Mood.motivated, Mood.curious, Mood.tired],
            suitable_energy_levels=[EnergyLevel.low, EnergyLevel.medium, EnergyLevel.high],
            metadata={
                "discovery_source": "youtube_live",
                "youtube_video_id": "ZyXwVuTsRqP",
                "channel_title": "Practice Lab",
                "thumbnail_url": "https://i.ytimg.com/vi/ZyXwVuTsRqP/mqdefault.jpg",
            },
        )

        class FakeYouTube(YouTubeService):
            def __init__(self) -> None:
                super().__init__(
                    settings=_settings(youtube_api_enabled=True, youtube_api_key="yt-key")
                )

            def is_enabled(self) -> bool:
                return True

            def is_configured(self) -> bool:
                return True

            def search_and_enrich(self, **kwargs: Any) -> list[ResourceCatalogItem]:
                return [yt_item]

        agent = CuratorAgent(
            settings=_settings(
                gemini_api_key="test-key",
                youtube_api_enabled=True,
                youtube_api_key="yt-key",
            ),
            gemini_service=FakeCuratorGemini(),
            catalog_index=self.catalog_index,
            catalog_path=BACKEND_ROOT / "data" / "sample_resources.json",
            db_path=self.db_path,
            youtube_service=FakeYouTube(),
        )
        first = agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=5,
            refresh=True,
        )
        before = count_resources(db_path=self.db_path)
        row1 = get_resource_by_url(
            "https://www.youtube.com/watch?v=ZyXwVuTsRqP",
            db_path=self.db_path,
        )
        self.assertIsNotNone(row1)
        second = agent.recommend_resources(
            int(self.user["id"]),
            self.roadmap_result.roadmap.id,
            self.milestone.id,
            limit=5,
            refresh=True,
        )
        after = count_resources(db_path=self.db_path)
        row2 = get_resource_by_url(
            "https://www.youtube.com/watch?v=ZyXwVuTsRqP",
            db_path=self.db_path,
        )
        self.assertEqual(before, after)
        assert row1 is not None and row2 is not None
        self.assertEqual(int(row1["id"]), int(row2["id"]))
        self.assertGreaterEqual(len(first.recommendations), 1)
        self.assertGreaterEqual(len(second.recommendations), 1)
        self.assertLessEqual(len(first.recommendations), 5)
        self.assertLessEqual(len(second.recommendations), 5)


if __name__ == "__main__":
    unittest.main()
