"""Unit tests for SemanticMemoryService composition."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exceptions import EmbeddingInvocationError, VectorStoreError  # noqa: E402
from services.embedding import GeminiEmbeddingService  # noqa: E402
from services.memory import SemanticMemoryService  # noqa: E402
from services.vector_models import VectorMemoryRecord  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402
from tests.test_embedding import FakeEmbeddingClient, _settings  # noqa: E402


class ExplodingStore(FAISSVectorStore):
    def add_memory(self, record, embedding):  # type: ignore[no-untyped-def]
        raise RuntimeError("faiss boom")


class MemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_memory_")
        root = Path(self._tmpdir.name)
        self.store = FAISSVectorStore(
            index_path=root / "index.faiss",
            metadata_path=root / "metadata.json",
            autosave=True,
        )
        self.client = FakeEmbeddingClient(query_vector=[1.0, 0.0, 0.0])
        self.embeddings = GeminiEmbeddingService(
            settings=_settings(gemini_api_key="test-key"),
            embedding_client=self.client,
        )
        self.memory = SemanticMemoryService(
            embedding_service=self.embeddings,
            vector_store=self.store,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_text_embedded_before_insertion(self) -> None:
        record = VectorMemoryRecord(
            memory_id="m1",
            user_id=1,
            record_type="aspiration",
            text="Become a calmer speaker",
        )
        self.memory.add_text_memory(record)
        self.assertEqual(self.client.query_calls, 1)
        self.assertIsNotNone(self.store.get_memory("m1"))

    def test_search_embeds_query_and_scopes_user(self) -> None:
        self.memory.add_text_memory(
            VectorMemoryRecord(
                memory_id="u1",
                user_id=1,
                record_type="goal",
                text="user one goal",
            )
        )
        other_store_client = FakeEmbeddingClient(query_vector=[1.0, 0.0, 0.0])
        other = SemanticMemoryService(
            embedding_service=GeminiEmbeddingService(
                settings=_settings(gemini_api_key="test-key"),
                embedding_client=other_store_client,
            ),
            vector_store=self.store,
        )
        other.add_text_memory(
            VectorMemoryRecord(
                memory_id="u2",
                user_id=2,
                record_type="goal",
                text="user two goal",
            )
        )
        before = self.client.query_calls
        results = self.memory.semantic_search("goal", user_id=1, limit=5)
        self.assertEqual(self.client.query_calls, before + 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_id, 1)

    def test_embedding_failure_does_not_mutate_faiss(self) -> None:
        failing = SemanticMemoryService(
            embedding_service=GeminiEmbeddingService(
                settings=_settings(gemini_api_key="test-key"),
                embedding_client=FakeEmbeddingClient(
                    raise_on_call=RuntimeError("embed failed")
                ),
            ),
            vector_store=self.store,
        )
        with self.assertRaises(EmbeddingInvocationError):
            failing.add_text_memory(
                VectorMemoryRecord(
                    memory_id="nope",
                    user_id=1,
                    record_type="goal",
                    text="should not be stored",
                )
            )
        self.assertEqual(self.store.count(), 0)

    def test_faiss_failure_reported_clearly(self) -> None:
        root = Path(self._tmpdir.name) / "explode"
        bad_memory = SemanticMemoryService(
            embedding_service=self.embeddings,
            vector_store=ExplodingStore(
                index_path=root / "index.faiss",
                metadata_path=root / "metadata.json",
                autosave=False,
            ),
        )
        with self.assertRaises(VectorStoreError):
            bad_memory.add_text_memory(
                VectorMemoryRecord(
                    memory_id="x1",
                    user_id=1,
                    record_type="goal",
                    text="trigger faiss failure",
                )
            )


if __name__ == "__main__":
    unittest.main()
