"""Unit tests for RoadmapAgent (no live Gemini calls)."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Type

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.roadmap_agent import (  # noqa: E402
    RoadmapAgent,
    build_roadmap_prompt,
    compute_capacity,
    validate_roadmap_generation,
)
from exceptions import (  # noqa: E402
    GeminiInvocationError,
    RoadmapAgentError,
    RoadmapContextError,
    RoadmapMemoryError,
    RoadmapOwnershipError,
    RoadmapPersistenceError,
)
from models import (  # noqa: E402
    Difficulty,
    MilestoneGeneration,
    MilestoneStatus,
    PhaseStatus,
    RoadmapGeneration,
    RoadmapPhaseGeneration,
)
from services.database import (  # noqa: E402
    count_roadmaps,
    create_goal,
    create_onboarding_records,
    create_user,
    get_goal_by_id,
    init_db,
)
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402
from tests.test_embedding import FakeEmbeddingClient, _settings  # noqa: E402


def _milestone(seq: int, title: str, difficulty: Difficulty = Difficulty.beginner) -> MilestoneGeneration:
    return MilestoneGeneration(
        sequence_number=seq,
        title=title,
        description=f"Description for {title}",
        skills=[f"skill-{seq}"],
        suggested_activities=[f"practice {title.lower()}"],
        completion_criteria=f"Complete {title}",
        estimated_sessions=2,
        estimated_minutes=20,
        difficulty=difficulty,
    )


def _sample_generation() -> RoadmapGeneration:
    return RoadmapGeneration(
        title="Public Speaking Foundations",
        summary="A practical path from beginner nerves to a short confident talk.",
        estimated_duration_weeks=6,
        pacing_rationale="Uses short evening sessions that fit a 15-minute attention span.",
        personalization_rationale="Matches beginner level and practice-oriented formats.",
        phases=[
            RoadmapPhaseGeneration(
                sequence_number=1,
                title="Foundations",
                description="Build comfort with structure and short delivery.",
                expected_outcome="Can outline and deliver a 2-minute talk.",
                milestones=[
                    _milestone(1, "Breath and posture basics"),
                    _milestone(2, "Two-minute outline drill"),
                ],
            ),
            RoadmapPhaseGeneration(
                sequence_number=2,
                title="Delivery practice",
                description="Increase clarity and confidence under mild pressure.",
                expected_outcome="Can deliver a 5-minute talk with notes.",
                milestones=[
                    _milestone(1, "Story framing practice", Difficulty.intermediate),
                    _milestone(2, "Five-minute timed talk", Difficulty.intermediate),
                ],
            ),
        ],
    )


class FakeGemini:
    def __init__(
        self,
        generation: RoadmapGeneration | None = None,
        *,
        raise_error: Exception | None = None,
    ) -> None:
        self.generation = generation or _sample_generation()
        self.raise_error = raise_error
        self.prompts: list[str] = []
        self.system_instructions: list[str | None] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[RoadmapGeneration],
        *,
        system_instruction: str | None = None,
    ) -> RoadmapGeneration:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        if self.raise_error is not None:
            raise self.raise_error
        assert response_model is RoadmapGeneration
        return self.generation


class RoadmapAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_roadmap_")
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
        self.store = FAISSVectorStore(
            index_path=root / "faiss" / "index.faiss",
            metadata_path=root / "faiss" / "metadata.json",
            autosave=True,
        )
        self.memory = SemanticMemoryService(
            embedding_service=GeminiEmbeddingService(
                settings=_settings(gemini_api_key="test-key"),
                embedding_client=FakeEmbeddingClient(query_vector=[1.0, 0.0, 0.0]),
            ),
            vector_store=self.store,
        )
        self.gemini = FakeGemini()
        self.agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_valid_roadmap_creation(self) -> None:
        result = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual(result.roadmap.user_id, int(self.user["id"]))
        self.assertEqual(result.roadmap.goal_id, int(self.goal["id"]))
        self.assertEqual(result.roadmap.progress_percent, 0)
        self.assertFalse(result.reused_existing)
        self.assertEqual(len(result.phases), 2)
        self.assertGreaterEqual(len(result.milestones), 4)

    def test_phase_and_milestone_order(self) -> None:
        result = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual([p.sequence_number for p in result.phases], [1, 2])
        self.assertEqual(
            [m.sequence_number for m in result.phases[0].milestones],
            [1, 2],
        )

    def test_initial_active_states(self) -> None:
        result = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual(result.phases[0].status, PhaseStatus.in_progress)
        self.assertEqual(result.phases[1].status, PhaseStatus.not_started)
        self.assertEqual(result.phases[0].milestones[0].status, MilestoneStatus.in_progress)
        self.assertEqual(result.phases[0].milestones[1].status, MilestoneStatus.not_started)
        self.assertIsNotNone(result.active_milestone)
        assert result.active_milestone is not None
        self.assertEqual(result.active_milestone.title, "Breath and posture basics")

    def test_free_text_goal_unchanged(self) -> None:
        self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        goal = get_goal_by_id(int(self.goal["id"]), db_path=self.db_path)
        assert goal is not None
        self.assertEqual(goal["title"], "Improve public speaking")

    def test_prompt_includes_level_and_time_context(self) -> None:
        self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        prompt = self.gemini.prompts[0]
        self.assertIn("beginner", prompt)
        self.assertIn("attention_span_minutes", prompt)
        self.assertIn("daily_available_minutes", prompt)
        self.assertIn("Improve public speaking", prompt)
        payload = json.loads(prompt.split("\n\n", 1)[1])
        self.assertEqual(payload["user_stated"]["current_level"], "beginner")
        self.assertIn("weekly_minutes", payload["derived_scheduling"])

    def test_no_duplicate_active_roadmap(self) -> None:
        first = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        second = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertTrue(second.reused_existing)
        self.assertEqual(first.roadmap.id, second.roadmap.id)
        self.assertEqual(count_roadmaps(db_path=self.db_path, goal_id=int(self.goal["id"])), 1)

    def test_regeneration_archives_previous(self) -> None:
        first = self.agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        second = self.agent.generate_roadmap(
            int(self.user["id"]),
            int(self.goal["id"]),
            regenerate=True,
        )
        self.assertFalse(second.reused_existing)
        self.assertNotEqual(first.roadmap.id, second.roadmap.id)
        self.assertEqual(count_roadmaps(db_path=self.db_path, goal_id=int(self.goal["id"])), 2)

    def test_gemini_failure_writes_no_rows(self) -> None:
        agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeGemini(raise_error=GeminiInvocationError("down")),
            memory_service=self.memory,
            db_path=self.db_path,
        )
        with self.assertRaises(GeminiInvocationError):
            agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual(count_roadmaps(db_path=self.db_path), 0)

    def test_persistence_failure_rolls_back(self) -> None:
        def boom(**_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("insert failed")

        agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,
            db_path=self.db_path,
            persist_roadmap=boom,
        )
        with self.assertRaises(RoadmapPersistenceError):
            agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual(count_roadmaps(db_path=self.db_path), 0)

    def test_missing_user_profile_goal_and_ownership(self) -> None:
        with self.assertRaises(RoadmapContextError):
            self.agent.generate_roadmap(99999, int(self.goal["id"]))
        with self.assertRaises(RoadmapContextError):
            self.agent.generate_roadmap(int(self.user["id"]), 99999)

        other = create_user("Other User", db_path=self.db_path)
        with self.assertRaises(RoadmapOwnershipError):
            self.agent.generate_roadmap(int(other["id"]), int(self.goal["id"]))

        # User without profile but with an owned goal
        bare = create_user("No Profile", db_path=self.db_path)
        bare_goal = create_goal(
            int(bare["id"]),
            title="Learn cooking",
            db_path=self.db_path,
        )
        with self.assertRaises(RoadmapContextError):
            self.agent.generate_roadmap(int(bare["id"]), int(bare_goal["id"]))

    def test_memory_retrieval_scoped_and_fallback(self) -> None:
        # Seed another user's memory; search must not use it.
        other_user, _, other_goal = create_onboarding_records(
            display_name="Other",
            aspiration="Other aspiration",
            motivation="Other motivation",
            current_level="beginner",
            target_outcome="Other outcome",
            learning_style="visual",
            preferred_formats=["read"],
            daily_available_minutes=30,
            preferred_session_minutes=15,
            attention_span_minutes=10,
            preferred_learning_time="morning",
            habits=[],
            distractions=[],
            goal_title="Learn cooking",
            db_path=self.db_path,
        )
        other_agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeGemini(),
            memory_service=self.memory,
            db_path=self.db_path,
        )
        other_agent.generate_roadmap(int(other_user["id"]), int(other_goal["id"]))

        class BoomMemory(SemanticMemoryService):
            def semantic_search(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                raise RuntimeError("memory down")

        agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=BoomMemory(
                embedding_service=self.memory.embedding_service,
                vector_store=self.store,
            ),
            db_path=self.db_path,
        )
        result = agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertEqual(result.roadmap.goal_id, int(self.goal["id"]))

    def test_roadmap_memory_failure_preserves_sqlite(self) -> None:
        class BrokenMemory(SemanticMemoryService):
            def add_text_memories(self, records):  # type: ignore[no-untyped-def]
                raise RuntimeError("faiss down")

            def semantic_search(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                return []

        agent = RoadmapAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=BrokenMemory(
                embedding_service=self.memory.embedding_service,
                vector_store=self.store,
            ),
            db_path=self.db_path,
        )
        with self.assertRaises(RoadmapMemoryError) as ctx:
            agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
        self.assertIsNotNone(ctx.exception.result)
        self.assertEqual(count_roadmaps(db_path=self.db_path), 1)

    def test_invalid_generation_structure_rejected(self) -> None:
        bad = _sample_generation()
        bad.phases[0].sequence_number = 3
        with self.assertRaises(RoadmapAgentError):
            validate_roadmap_generation(bad, weekly_minutes=100)

    def test_capacity_helper(self) -> None:
        capacity = compute_capacity(
            daily_available_minutes=45,
            preferred_session_minutes=20,
            attention_span_minutes=15,
        )
        self.assertEqual(capacity["learning_days_per_week"], 5)
        self.assertEqual(capacity["session_minutes"], 15)
        self.assertGreater(capacity["weekly_minutes"], 0)

    def test_secrets_not_logged(self) -> None:
        secret = "roadmap-secret-key"
        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = ListHandler()
        logger = logging.getLogger("agents.roadmap_agent")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            agent = RoadmapAgent(
                settings=_settings(gemini_api_key=secret),
                gemini_service=FakeGemini(raise_error=GeminiInvocationError("provider down")),
                memory_service=self.memory,
                db_path=self.db_path,
            )
            with self.assertRaises(GeminiInvocationError) as ctx:
                agent.generate_roadmap(int(self.user["id"]), int(self.goal["id"]))
            self.assertNotIn(secret, str(ctx.exception))
            for line in records:
                self.assertNotIn(secret, line)
            prompt = build_roadmap_prompt(
                user=self.user,
                profile=self.profile,
                goal=self.goal,
                capacity=compute_capacity(
                    daily_available_minutes=45,
                    preferred_session_minutes=20,
                    attention_span_minutes=15,
                ),
                memory_snippets=[],
            )
            self.assertNotIn(secret, prompt)
            self.assertNotIn("GEMINI_API_KEY", prompt)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)


if __name__ == "__main__":
    unittest.main()
