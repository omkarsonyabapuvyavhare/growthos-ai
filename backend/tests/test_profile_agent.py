"""Unit tests for ProfileAgent (no live Gemini calls)."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Type

from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.profile_agent import ProfileAgent, build_onboarding_prompt  # noqa: E402
from exceptions import (  # noqa: E402
    GeminiInvocationError,
    ProfileMemoryError,
    ProfilePersistenceError,
)
from models import (  # noqa: E402
    CurrentLevel,
    LearningStyle,
    OnboardingRequest,
    PreferredLearningTime,
    ProfileInterpretation,
)
from services.database import (  # noqa: E402
    count_users,
    get_user_by_id,
    get_user_profile_by_user_id,
    init_db,
)
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402
from tests.test_embedding import FakeEmbeddingClient, _settings  # noqa: E402


def _sample_request(**overrides: Any) -> OnboardingRequest:
    data = {
        "display_name": "Ada Lovelace",
        "learning_goal": "Improve public speaking",
        "aspiration": "Become a confident presenter",
        "motivation": "Lead team updates without anxiety",
        "current_level": CurrentLevel.beginner,
        "target_outcome": "Deliver a 10-minute talk without notes",
        "preferred_formats": ["video", "practice"],
        "learning_style": LearningStyle.mixed,
        "daily_available_minutes": 45,
        "preferred_session_minutes": 20,
        "attention_span_minutes": 15,
        "preferred_learning_time": PreferredLearningTime.evening,
        "habits": ["journal"],
        "distractions": ["phone"],
    }
    data.update(overrides)
    return OnboardingRequest(**data)


def _interpretation() -> ProfileInterpretation:
    return ProfileInterpretation(
        identity_summary="A beginner learner focused on public speaking growth.",
        aspiration_summary="Wants to become a confident presenter.",
        motivation_summary="Motivated by calmer team leadership moments.",
        current_state_summary="Beginner with limited speaking practice.",
        target_state_summary="Can deliver a short talk without notes.",
        strengths=["clear motivation", "willing to practice"],
        likely_challenges=["phone distractions", "short attention span"],
        learning_preferences_summary="Prefers video plus practice in the evening.",
        recommended_pacing="Short daily sessions of about 20 minutes.",
        attention_strategy="Use focused micro-tasks under 15 minutes.",
        consistency_strategy="Protect a regular evening practice slot.",
        initial_personalization_insights=[
            "Start with short speaking drills",
            "Pair videos with immediate practice",
        ],
    )


class FakeGemini:
    def __init__(
        self,
        interpretation: ProfileInterpretation | None = None,
        *,
        raise_error: Exception | None = None,
    ) -> None:
        self.interpretation = interpretation or _interpretation()
        self.raise_error = raise_error
        self.prompts: list[str] = []
        self.system_instructions: list[str | None] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[ProfileInterpretation],
        *,
        system_instruction: str | None = None,
    ) -> ProfileInterpretation:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        if self.raise_error is not None:
            raise self.raise_error
        assert response_model is ProfileInterpretation
        return self.interpretation


class ProfileAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_profile_")
        root = Path(self._tmpdir.name)
        self.db_path = root / "test.db"
        init_db(self.db_path)
        self.store = FAISSVectorStore(
            index_path=root / "faiss" / "index.faiss",
            metadata_path=root / "faiss" / "metadata.json",
            autosave=True,
        )
        self.embeddings = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="test-key"),
            embedding_client=FakeEmbeddingClient(query_vector=[1.0, 0.0, 0.0]),
        )
        self.memory = SemanticMemoryService(
            embedding_service=self.embeddings,
            vector_store=self.store,
        )
        self.gemini = FakeGemini()
        self.agent = ProfileAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_valid_onboarding_creates_user_profile_goal(self) -> None:
        result = self.agent.process_onboarding(_sample_request())
        self.assertEqual(result.user.display_name, "Ada Lovelace")
        self.assertEqual(result.profile.user_id, result.user.id)
        self.assertEqual(result.goal.user_id, result.user.id)
        self.assertEqual(count_users(db_path=self.db_path), 1)
        self.assertIsNotNone(get_user_by_id(result.user.id, db_path=self.db_path))
        self.assertIsNotNone(
            get_user_profile_by_user_id(result.user.id, db_path=self.db_path)
        )

    def test_free_text_goal_preserved_exactly(self) -> None:
        goal = "Learn sourdough bread baking from scratch"
        result = self.agent.process_onboarding(
            _sample_request(learning_goal=goal)
        )
        self.assertEqual(result.goal.title, goal)

    def test_preferred_formats_persisted_as_json(self) -> None:
        result = self.agent.process_onboarding(_sample_request())
        row = get_user_profile_by_user_id(result.user.id, db_path=self.db_path)
        assert row is not None
        self.assertEqual(row["preferred_formats"], ["video", "practice"])
        # Ensure DB stores JSON text under the hood via reload path
        self.assertIsInstance(row["preferred_formats"], list)

    def test_gemini_receives_onboarding_context(self) -> None:
        request = _sample_request()
        self.agent.process_onboarding(request)
        self.assertEqual(len(self.gemini.prompts), 1)
        prompt = self.gemini.prompts[0]
        self.assertIn("Improve public speaking", prompt)
        self.assertIn("Become a confident presenter", prompt)
        payload = json.loads(prompt.split("\n\n", 1)[1])
        self.assertEqual(payload["learning_goal"], request.learning_goal)
        self.assertIsNotNone(self.gemini.system_instructions[0])
        self.assertIn("Infer cautiously", self.gemini.system_instructions[0] or "")

    def test_structured_interpretation_returned(self) -> None:
        result = self.agent.process_onboarding(_sample_request())
        self.assertEqual(
            result.interpretation.identity_summary,
            _interpretation().identity_summary,
        )
        self.assertTrue(result.interpretation.strengths)

    def test_semantic_memories_created_for_user(self) -> None:
        result = self.agent.process_onboarding(_sample_request())
        self.assertTrue(result.memories_complete)
        self.assertGreaterEqual(len(result.memory_ids), 6)
        self.assertEqual(self.store.count(user_id=result.user.id), len(result.memory_ids))
        for memory_id in result.memory_ids:
            memory = self.store.get_memory(memory_id)
            self.assertIsNotNone(memory)
            assert memory is not None
            self.assertEqual(memory.user_id, result.user.id)

    def test_user_isolation_across_onboardings(self) -> None:
        first = self.agent.process_onboarding(
            _sample_request(display_name="User One", learning_goal="Learn Python")
        )
        second = self.agent.process_onboarding(
            _sample_request(display_name="User Two", learning_goal="Learn cooking")
        )
        hits = self.store.search([1.0, 0.0, 0.0], user_id=first.user.id, limit=20)
        self.assertTrue(hits)
        self.assertTrue(all(item.user_id == first.user.id for item in hits))
        self.assertTrue(all(item.user_id != second.user.id for item in hits))

    def test_gemini_failure_creates_no_sqlite_records(self) -> None:
        failing = ProfileAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=FakeGemini(
                raise_error=GeminiInvocationError("provider down")
            ),
            memory_service=self.memory,
            db_path=self.db_path,
        )
        with self.assertRaises(GeminiInvocationError):
            failing.process_onboarding(_sample_request())
        self.assertEqual(count_users(db_path=self.db_path), 0)

    def test_sqlite_failure_rolls_back(self) -> None:
        def boom(_request: OnboardingRequest) -> tuple[dict, dict, dict]:
            raise RuntimeError("db write failed")

        agent = ProfileAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=self.memory,
            db_path=self.db_path,
            persist_onboarding=boom,
        )
        with self.assertRaises(ProfilePersistenceError):
            agent.process_onboarding(_sample_request())
        self.assertEqual(count_users(db_path=self.db_path), 0)

    def test_memory_failure_after_sqlite_is_recoverable(self) -> None:
        class BrokenMemory(SemanticMemoryService):
            def add_text_memories(self, records):  # type: ignore[no-untyped-def]
                raise RuntimeError("faiss unavailable")

        agent = ProfileAgent(
            settings=_settings(gemini_api_key="test-key"),
            gemini_service=self.gemini,
            memory_service=BrokenMemory(
                embedding_service=self.embeddings,
                vector_store=self.store,
            ),
            db_path=self.db_path,
        )
        with self.assertRaises(ProfileMemoryError) as ctx:
            agent.process_onboarding(_sample_request())
        self.assertIsNotNone(ctx.exception.result)
        result = ctx.exception.result
        assert result is not None
        self.assertEqual(count_users(db_path=self.db_path), 1)
        self.assertFalse(result.memories_complete)
        self.assertIsNotNone(get_user_by_id(result.user.id, db_path=self.db_path))

    def test_blank_required_fields_fail_validation(self) -> None:
        with self.assertRaises(ValidationError):
            OnboardingRequest(
                display_name=" ",
                learning_goal="Learn Python",
                aspiration="Dev",
                motivation="Jobs",
                current_level=CurrentLevel.beginner,
                target_outcome="Build apps",
                preferred_formats=["video"],
                learning_style=LearningStyle.visual,
                daily_available_minutes=30,
                preferred_session_minutes=15,
                attention_span_minutes=10,
                preferred_learning_time=PreferredLearningTime.morning,
            )

    def test_minute_constraints_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            _sample_request(daily_available_minutes=0)
        with self.assertRaises(ValidationError):
            _sample_request(preferred_session_minutes=-5)

    def test_duplicate_display_names_create_separate_users(self) -> None:
        first = self.agent.process_onboarding(
            _sample_request(display_name="Same Name")
        )
        second = self.agent.process_onboarding(
            _sample_request(display_name="Same Name", learning_goal="Learn photography")
        )
        self.assertNotEqual(first.user.id, second.user.id)
        self.assertEqual(count_users(db_path=self.db_path), 2)

    def test_secrets_not_in_errors_or_logs(self) -> None:
        secret = "super-secret-profile-key"
        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = ListHandler()
        logger = logging.getLogger("agents.profile_agent")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            agent = ProfileAgent(
                settings=_settings(gemini_api_key=secret),
                gemini_service=FakeGemini(
                    raise_error=GeminiInvocationError("provider down")
                ),
                memory_service=self.memory,
                db_path=self.db_path,
            )
            with self.assertRaises(GeminiInvocationError) as ctx:
                agent.process_onboarding(_sample_request())
            self.assertNotIn(secret, str(ctx.exception))
            for line in records:
                self.assertNotIn(secret, line)
            prompt = build_onboarding_prompt(_sample_request())
            self.assertNotIn(secret, prompt)
            self.assertNotIn("GEMINI_API_KEY", prompt)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

    def test_prompt_builder_includes_only_onboarding_fields(self) -> None:
        prompt = build_onboarding_prompt(_sample_request())
        self.assertNotIn("GEMINI_API_KEY", prompt)
        self.assertNotIn("faiss", prompt.lower())
        self.assertIn("learning_goal", prompt)


if __name__ == "__main__":
    unittest.main()
