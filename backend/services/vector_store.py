"""
FAISS-backed semantic vector store for GrowthOS AI.

Similarity strategy:
- L2-normalize embeddings
- Search with FAISS IndexFlatIP (inner product == cosine similarity)

Persistence:
- FAISS binary index file
- JSON metadata that includes normalized embeddings for local rebuild
  without requiring Gemini calls

Concurrency:
- In-process threading lock protects mutations and save/load.
- Callers must generate embeddings BEFORE acquiring the store lock.
- MVP assumes a single backend process for safe file persistence.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import faiss
import numpy as np

from config import Settings, get_settings
from exceptions import VectorStoreError, VectorStoreValidationError
from services.vector_models import VectorMemoryRecord, VectorSearchResult

logger = logging.getLogger(__name__)

METADATA_VERSION = 1
SIMILARITY_STRATEGY = "cosine_ip"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_embedding(vector: Sequence[float]) -> np.ndarray:
    """Return an L2-normalized float32 vector."""
    arr = np.asarray(list(vector), dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise VectorStoreValidationError("embedding must be a non-empty 1-D vector")
    if not np.all(np.isfinite(arr)):
        raise VectorStoreValidationError("embedding contains non-finite values")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise VectorStoreValidationError("embedding norm is zero")
    return arr / norm


class FAISSVectorStore:
    """
    User-scoped FAISS memory store with durable metadata.

    FAISS IDs are explicit integers managed by IndexIDMap2.
    memory_id values are stable application identifiers stored in metadata.
    Duplicate memory_id values are rejected.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        autosave: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._index_path = index_path or self._settings.resolve_faiss_index_path()
        self._metadata_path = metadata_path or self._settings.resolve_faiss_metadata_path()
        self._autosave = autosave
        self._lock = threading.RLock()

        self._dimension: int | None = None
        self._index: Any | None = None
        self._next_faiss_id = 1
        # memory_id -> record dict including embedding + faiss_id
        self._records: dict[str, dict[str, Any]] = {}

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _create_index(self, dimension: int) -> Any:
        base = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap2(base)

    def _ensure_index(self, dimension: int) -> Any:
        if self._index is None:
            self._dimension = dimension
            self._index = self._create_index(dimension)
            return self._index
        if self._dimension != dimension:
            raise VectorStoreValidationError(
                f"embedding dimension {dimension} does not match store dimension {self._dimension}"
            )
        return self._index

    def _record_to_search_result(
        self,
        record: dict[str, Any],
        similarity_score: float,
    ) -> VectorSearchResult:
        return VectorSearchResult(
            memory_id=record["memory_id"],
            user_id=int(record["user_id"]),
            record_type=record["record_type"],
            source_record_id=record.get("source_record_id"),
            text=record["text"],
            metadata=dict(record.get("metadata") or {}),
            similarity_score=float(similarity_score),
            created_at=_from_iso(record["created_at"]),
        )

    def add_memory(
        self,
        record: VectorMemoryRecord,
        embedding: Sequence[float],
    ) -> VectorMemoryRecord:
        """Insert one memory. Raises if memory_id already exists."""
        self.add_memories([record], [list(embedding)])
        return record

    def add_memories(
        self,
        records: Sequence[VectorMemoryRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> list[VectorMemoryRecord]:
        """Insert many memories. Embeddings must already be computed."""
        if len(records) != len(embeddings):
            raise VectorStoreValidationError("records and embeddings length mismatch")
        if not records:
            raise VectorStoreValidationError("records must not be empty")

        prepared: list[tuple[VectorMemoryRecord, np.ndarray]] = []
        for record, embedding in zip(records, embeddings):
            if not isinstance(record, VectorMemoryRecord):
                raise VectorStoreValidationError("each record must be a VectorMemoryRecord")
            prepared.append((record, normalize_embedding(embedding)))

        with self._lock:
            for record, _ in prepared:
                if record.memory_id in self._records:
                    raise VectorStoreValidationError(
                        f"duplicate memory_id is not allowed: {record.memory_id}"
                    )

            dimension = int(prepared[0][1].shape[0])
            for _, vector in prepared:
                if int(vector.shape[0]) != dimension:
                    raise VectorStoreValidationError(
                        "all embeddings in a batch must share the same dimension"
                    )

            index = self._ensure_index(dimension)
            vectors = np.vstack([vector for _, vector in prepared]).astype(np.float32)
            faiss_ids = np.asarray(
                [self._next_faiss_id + offset for offset in range(len(prepared))],
                dtype=np.int64,
            )
            index.add_with_ids(vectors, faiss_ids)

            for offset, (record, vector) in enumerate(prepared):
                faiss_id = int(faiss_ids[offset])
                self._records[record.memory_id] = {
                    "faiss_id": faiss_id,
                    "memory_id": record.memory_id,
                    "user_id": record.user_id,
                    "record_type": record.record_type,
                    "source_record_id": record.source_record_id,
                    "text": record.text,
                    "metadata": dict(record.metadata),
                    "created_at": _to_iso(record.created_at),
                    "embedding": vector.astype(float).tolist(),
                    "deleted": False,
                }
            self._next_faiss_id = int(faiss_ids[-1]) + 1

            if self._autosave:
                self._save_unlocked()

        logger.info("Added %s memories to FAISS store", len(prepared))
        return list(records)

    def search(
        self,
        query_embedding: Sequence[float],
        user_id: int,
        limit: int = 5,
        record_types: Optional[Iterable[str]] = None,
    ) -> list[VectorSearchResult]:
        """
        Semantic search restricted to one user.

        Oversamples FAISS results, then filters by user_id / record_type so
        another user's memories are never returned.
        """
        if user_id <= 0:
            raise VectorStoreValidationError("user_id must be a positive integer")
        if limit <= 0:
            raise VectorStoreValidationError("limit must be a positive integer")

        query = normalize_embedding(query_embedding)
        type_filter = {item.strip() for item in (record_types or []) if str(item).strip()}

        with self._lock:
            if self._index is None or self._dimension is None or not self._records:
                return []
            if int(query.shape[0]) != self._dimension:
                raise VectorStoreValidationError(
                    f"query dimension {query.shape[0]} does not match store dimension {self._dimension}"
                )

            active_for_user = [
                record
                for record in self._records.values()
                if not record.get("deleted")
                and int(record["user_id"]) == user_id
                and (not type_filter or record["record_type"] in type_filter)
            ]
            if not active_for_user:
                return []

            # Oversample so post-filtering still fills the requested limit.
            oversample = min(max(limit * 10, limit), len(self._records))
            scores, ids = self._index.search(query.reshape(1, -1), oversample)

            results: list[VectorSearchResult] = []
            faiss_to_record = {
                int(record["faiss_id"]): record
                for record in self._records.values()
                if not record.get("deleted")
            }

            for score, faiss_id in zip(scores[0].tolist(), ids[0].tolist()):
                if int(faiss_id) < 0:
                    continue
                record = faiss_to_record.get(int(faiss_id))
                if record is None:
                    continue
                if int(record["user_id"]) != user_id:
                    continue
                if type_filter and record["record_type"] not in type_filter:
                    continue
                if not math.isfinite(float(score)):
                    continue
                results.append(self._record_to_search_result(record, float(score)))
                if len(results) >= limit:
                    break

            return results

    def get_memory(self, memory_id: str) -> Optional[VectorMemoryRecord]:
        with self._lock:
            record = self._records.get(memory_id)
            if record is None or record.get("deleted"):
                return None
            return VectorMemoryRecord(
                memory_id=record["memory_id"],
                user_id=int(record["user_id"]),
                record_type=record["record_type"],
                source_record_id=record.get("source_record_id"),
                text=record["text"],
                metadata=dict(record.get("metadata") or {}),
                created_at=_from_iso(record["created_at"]),
            )

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by stable memory_id.

        Strategy: drop metadata and rebuild the FAISS index from remaining
        persisted embeddings so vector positions/IDs stay consistent.
        Returns False when the memory does not exist.
        """
        with self._lock:
            record = self._records.get(memory_id)
            if record is None or record.get("deleted"):
                return False

            del self._records[memory_id]
            index, records, next_id, dimension = self._rebuild_from_records(
                list(self._records.values()),
                dimension=self._dimension,
                next_faiss_id=self._next_faiss_id,
            )
            self._index = index
            self._records = records
            self._next_faiss_id = next_id
            self._dimension = dimension
            if self._autosave:
                self._save_unlocked()
            return True

    def count(self, user_id: Optional[int] = None) -> int:
        with self._lock:
            if user_id is None:
                return sum(1 for record in self._records.values() if not record.get("deleted"))
            return sum(
                1
                for record in self._records.values()
                if not record.get("deleted") and int(record["user_id"]) == user_id
            )

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        _ensure_parent(path)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        self._atomic_write_bytes(path, text.encode("utf-8"))

    def _metadata_payload(self) -> dict[str, Any]:
        return {
            "version": METADATA_VERSION,
            "dimension": self._dimension,
            "similarity": SIMILARITY_STRATEGY,
            "next_faiss_id": self._next_faiss_id,
            "records": [
                record
                for record in self._records.values()
                if not record.get("deleted")
            ],
        }

    def _save_unlocked(self) -> None:
        if self._index is None or self._dimension is None:
            # Persist empty metadata so paths/directories exist.
            payload = self._metadata_payload()
            self._atomic_write_text(
                self._metadata_path,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            return

        _ensure_parent(self._index_path)
        # FAISS write_index is not atomic; write to temp then replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=self._index_path.name + ".",
            dir=str(self._index_path.parent),
        )
        os.close(fd)
        try:
            faiss.write_index(self._index, tmp_name)
            os.replace(tmp_name, self._index_path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

        payload = self._metadata_payload()
        active_count = len(payload["records"])
        if int(self._index.ntotal) != active_count:
            raise VectorStoreError(
                f"index/metadata count mismatch on save: "
                f"index={self._index.ntotal} metadata={active_count}"
            )

        self._atomic_write_text(
            self._metadata_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        logger.info(
            "Saved FAISS store index=%s metadata=%s count=%s",
            self._index_path.name,
            self._metadata_path.name,
            active_count,
        )

    def load(self) -> None:
        """
        Load index + metadata from disk.

        On failure, the previous in-memory state is preserved.
        """
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if not self._metadata_path.exists():
            logger.info("No FAISS metadata found at %s; starting empty", self._metadata_path)
            return

        try:
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"failed to read FAISS metadata: {exc}") from None

        try:
            dimension = raw.get("dimension")
            records = raw.get("records")
            next_faiss_id = int(raw.get("next_faiss_id", 1))
            if records is None or not isinstance(records, list):
                raise VectorStoreValidationError("metadata records must be a list")
            if dimension is not None:
                dimension = int(dimension)
            parsed_records: dict[str, dict[str, Any]] = {}
            for item in records:
                if not isinstance(item, dict):
                    raise VectorStoreValidationError("metadata record must be an object")
                memory_id = str(item.get("memory_id", "")).strip()
                if not memory_id:
                    raise VectorStoreValidationError("metadata record missing memory_id")
                embedding = item.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise VectorStoreValidationError(
                        f"metadata record {memory_id} missing embedding for rebuild"
                    )
                vector = normalize_embedding(embedding)
                if dimension is None:
                    dimension = int(vector.shape[0])
                elif int(vector.shape[0]) != dimension:
                    raise VectorStoreValidationError(
                        f"metadata record {memory_id} has wrong embedding dimension"
                    )
                parsed_records[memory_id] = {
                    "faiss_id": int(item["faiss_id"]),
                    "memory_id": memory_id,
                    "user_id": int(item["user_id"]),
                    "record_type": str(item["record_type"]),
                    "source_record_id": item.get("source_record_id"),
                    "text": str(item["text"]),
                    "metadata": dict(item.get("metadata") or {}),
                    "created_at": str(item["created_at"]),
                    "embedding": vector.astype(float).tolist(),
                    "deleted": bool(item.get("deleted", False)),
                }
        except (KeyError, TypeError, ValueError, VectorStoreValidationError) as exc:
            raise VectorStoreError(f"malformed FAISS metadata: {exc}") from None

        # Prefer rebuilding from persisted embeddings for consistency after deletions.
        try:
            rebuilt_index, rebuilt_records, rebuilt_next_id, rebuilt_dim = self._rebuild_from_records(
                list(parsed_records.values()),
                dimension=dimension,
                next_faiss_id=next_faiss_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to rebuild FAISS index from metadata: {exc}") from None

        # Optional consistency check against on-disk FAISS file when present.
        if self._index_path.exists():
            try:
                disk_index = faiss.read_index(str(self._index_path))
                if int(disk_index.ntotal) != int(rebuilt_index.ntotal):
                    logger.warning(
                        "On-disk FAISS ntotal=%s differs from rebuilt ntotal=%s; using rebuild",
                        disk_index.ntotal,
                        rebuilt_index.ntotal,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read on-disk FAISS index; using rebuild (%s)", type(exc).__name__)

        # Commit only after successful rebuild (no partial overwrite on failure).
        self._index = rebuilt_index
        self._records = rebuilt_records
        self._next_faiss_id = rebuilt_next_id
        self._dimension = rebuilt_dim
        logger.info("Loaded FAISS store count=%s dimension=%s", len(self._records), self._dimension)

    def _rebuild_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        dimension: Optional[int],
        next_faiss_id: int,
    ) -> tuple[Any, dict[str, dict[str, Any]], int, Optional[int]]:
        active = [record for record in records if not record.get("deleted")]
        if not active:
            if dimension is None:
                return None, {}, max(1, next_faiss_id), None
            return self._create_index(dimension), {}, max(1, next_faiss_id), dimension

        if dimension is None:
            dimension = len(active[0]["embedding"])

        index = self._create_index(dimension)
        rebuilt: dict[str, dict[str, Any]] = {}
        vectors = []
        ids = []
        for record in active:
            vector = normalize_embedding(record["embedding"])
            if int(vector.shape[0]) != dimension:
                raise VectorStoreValidationError(
                    f"record {record['memory_id']} dimension mismatch during rebuild"
                )
            faiss_id = int(record["faiss_id"])
            vectors.append(vector)
            ids.append(faiss_id)
            rebuilt[record["memory_id"]] = {
                **record,
                "embedding": vector.astype(float).tolist(),
                "deleted": False,
            }

        index.add_with_ids(
            np.vstack(vectors).astype(np.float32),
            np.asarray(ids, dtype=np.int64),
        )
        if int(index.ntotal) != len(rebuilt):
            raise VectorStoreError(
                f"index/metadata count mismatch after rebuild: "
                f"index={index.ntotal} metadata={len(rebuilt)}"
            )
        max_id = max(ids) if ids else 0
        return index, rebuilt, max(next_faiss_id, max_id + 1), dimension

    def rebuild(self) -> None:
        """Rebuild the in-memory FAISS index from persisted embeddings and save."""
        with self._lock:
            index, records, next_id, dimension = self._rebuild_from_records(
                list(self._records.values()),
                dimension=self._dimension,
                next_faiss_id=self._next_faiss_id,
            )
            self._index = index
            self._records = records
            self._next_faiss_id = next_id
            self._dimension = dimension
            self._save_unlocked()

    def clear_for_testing(self) -> None:
        """Reset in-memory state and delete persistence files (tests only)."""
        with self._lock:
            self._index = None
            self._records = {}
            self._dimension = None
            self._next_faiss_id = 1
            for path in (self._index_path, self._metadata_path):
                if path.exists():
                    path.unlink()


def get_vector_store(settings: Optional[Settings] = None) -> FAISSVectorStore:
    """Factory that loads an existing store when persistence files are present."""
    store = FAISSVectorStore(settings=settings)
    store.load()
    return store
