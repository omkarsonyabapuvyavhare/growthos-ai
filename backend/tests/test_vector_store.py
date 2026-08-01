"""Unit tests for FAISSVectorStore (deterministic fake vectors)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exceptions import VectorStoreError, VectorStoreValidationError  # noqa: E402
from services.vector_models import VectorMemoryRecord  # noqa: E402
from services.vector_store import FAISSVectorStore  # noqa: E402


def _record(
    memory_id: str,
    user_id: int,
    text: str,
    record_type: str = "goal",
) -> VectorMemoryRecord:
    return VectorMemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        record_type=record_type,
        text=text,
        metadata={"source": "test"},
    )


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_faiss_")
        root = Path(self._tmpdir.name)
        self.index_path = root / "nested" / "index.faiss"
        self.metadata_path = root / "nested" / "metadata.json"
        self.store = FAISSVectorStore(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            autosave=True,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_add_and_get(self) -> None:
        record = _record("m1", 1, "learn public speaking")
        self.store.add_memory(record, [1.0, 0.0, 0.0])
        loaded = self.store.get_memory("m1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.text, "learn public speaking")
        self.assertEqual(self.store.count(user_id=1), 1)

    def test_similarity_ranking(self) -> None:
        self.store.add_memories(
            [
                _record("near", 1, "near vector"),
                _record("far", 1, "far vector"),
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
        results = self.store.search([0.99, 0.01, 0.0], user_id=1, limit=2)
        self.assertEqual(results[0].memory_id, "near")
        self.assertGreater(results[0].similarity_score, results[1].similarity_score)

    def test_user_isolation(self) -> None:
        self.store.add_memories(
            [
                _record("a1", 1, "user one aspiration", "aspiration"),
                _record("b1", 2, "user two aspiration", "aspiration"),
            ],
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )
        results = self.store.search([1.0, 0.0, 0.0], user_id=1, limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_id, 1)
        self.assertEqual(results[0].memory_id, "a1")

    def test_record_type_filter(self) -> None:
        self.store.add_memories(
            [
                _record("g1", 1, "goal text", "goal"),
                _record("r1", 1, "reflection text", "reflection"),
            ],
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
            ],
        )
        results = self.store.search(
            [1.0, 0.0, 0.0],
            user_id=1,
            limit=5,
            record_types=["reflection"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, "r1")

    def test_limit_larger_than_available(self) -> None:
        self.store.add_memory(_record("only", 1, "only one"), [1.0, 0.0, 0.0])
        results = self.store.search([1.0, 0.0, 0.0], user_id=1, limit=50)
        self.assertEqual(len(results), 1)

    def test_duplicate_memory_id_rejected(self) -> None:
        self.store.add_memory(_record("dup", 1, "one"), [1.0, 0.0, 0.0])
        with self.assertRaises(VectorStoreValidationError):
            self.store.add_memory(_record("dup", 1, "two"), [0.0, 1.0, 0.0])

    def test_wrong_dimension_rejected(self) -> None:
        self.store.add_memory(_record("d3", 1, "dim3"), [1.0, 0.0, 0.0])
        with self.assertRaises(VectorStoreValidationError):
            self.store.add_memory(_record("d2", 1, "dim2"), [1.0, 0.0])

    def test_nan_query_rejected(self) -> None:
        self.store.add_memory(_record("ok", 1, "ok"), [1.0, 0.0, 0.0])
        with self.assertRaises(VectorStoreValidationError):
            self.store.search([float("nan"), 0.0, 0.0], user_id=1, limit=1)

    def test_save_and_load_preserves_records(self) -> None:
        self.store.add_memories(
            [
                _record("s1", 1, "alpha"),
                _record("s2", 1, "beta"),
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
        self.assertTrue(self.index_path.exists())
        self.assertTrue(self.metadata_path.exists())

        reloaded = FAISSVectorStore(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            autosave=False,
        )
        reloaded.load()
        self.assertEqual(reloaded.count(user_id=1), 2)
        results = reloaded.search([1.0, 0.0, 0.0], user_id=1, limit=1)
        self.assertEqual(results[0].memory_id, "s1")

    def test_delete_behavior(self) -> None:
        self.store.add_memories(
            [_record("keep", 1, "keep"), _record("drop", 1, "drop")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        self.assertTrue(self.store.delete_memory("drop"))
        self.assertIsNone(self.store.get_memory("drop"))
        self.assertEqual(self.store.count(user_id=1), 1)
        results = self.store.search([0.0, 1.0, 0.0], user_id=1, limit=5)
        self.assertTrue(all(item.memory_id != "drop" for item in results))

    def test_malformed_metadata_fails_safely(self) -> None:
        self.store.add_memory(_record("safe", 1, "safe"), [1.0, 0.0, 0.0])
        self.metadata_path.write_text("{not-json", encoding="utf-8")
        with self.assertRaises(VectorStoreError):
            self.store.load()
        # Previous in-memory state preserved
        self.assertIsNotNone(self.store.get_memory("safe"))

    def test_index_metadata_count_mismatch_detected_on_save_path(self) -> None:
        # Build a store, then corrupt metadata record count expectation by
        # writing metadata with an extra record lacking a matching FAISS rebuild path.
        self.store.add_memory(_record("one", 1, "one"), [1.0, 0.0, 0.0])
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        payload["records"].append(
            {
                "faiss_id": 99,
                "memory_id": "ghost",
                "user_id": 1,
                "record_type": "goal",
                "source_record_id": None,
                "text": "ghost",
                "metadata": {},
                "created_at": payload["records"][0]["created_at"],
                # missing embedding -> malformed
            }
        )
        self.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        broken = FAISSVectorStore(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            autosave=False,
        )
        with self.assertRaises(VectorStoreError):
            broken.load()

    def test_persistence_creates_directories(self) -> None:
        self.assertFalse(self.index_path.parent.exists())
        self.store.add_memory(_record("dir", 1, "creates dirs"), [1.0, 0.0, 0.0])
        self.assertTrue(self.index_path.parent.exists())

    def test_failed_reload_does_not_partially_overwrite(self) -> None:
        self.store.add_memory(_record("keep-me", 1, "keep me"), [1.0, 0.0, 0.0])
        self.metadata_path.write_text(
            json.dumps({"version": 1, "records": "bad-type"}),
            encoding="utf-8",
        )
        with self.assertRaises(VectorStoreError):
            self.store.load()
        self.assertEqual(self.store.get_memory("keep-me").text, "keep me")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
