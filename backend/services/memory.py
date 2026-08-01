"""
Thin semantic-memory composition for GrowthOS AI.

Combines Gemini embeddings with the FAISS vector store.
This is not an agent and performs no workflow orchestration.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from config import Settings, get_settings
from exceptions import EmbeddingInvocationError, VectorStoreError
from services.embedding import GeminiEmbeddingService
from services.vector_models import VectorMemoryRecord, VectorSearchResult
from services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class SemanticMemoryService:
    """
    Deterministic helper: embed text, then mutate/search FAISS.

    Embeddings are generated outside the vector-store lock (handled inside
    FAISSVectorStore only around local mutations).
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or GeminiEmbeddingService(
            settings=self._settings
        )
        self._vector_store = vector_store or FAISSVectorStore(settings=self._settings)

    @property
    def embedding_service(self) -> GeminiEmbeddingService:
        return self._embedding_service

    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._vector_store

    def add_text_memory(self, record: VectorMemoryRecord) -> VectorMemoryRecord:
        """Embed record.text, then insert into FAISS."""
        embedding = self._embedding_service.embed_text(record.text)
        try:
            return self._vector_store.add_memory(record, embedding)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to store memory: {exc}") from None

    def add_text_memories(
        self,
        records: Sequence[VectorMemoryRecord],
    ) -> list[VectorMemoryRecord]:
        """Batch-embed texts, then insert into FAISS."""
        texts = [record.text for record in records]
        embeddings = self._embedding_service.embed_documents(list(texts))
        try:
            return self._vector_store.add_memories(records, embeddings)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to store memories: {exc}") from None

    def semantic_search(
        self,
        query_text: str,
        user_id: int,
        limit: int = 5,
        record_types: Optional[Iterable[str]] = None,
    ) -> list[VectorSearchResult]:
        """Embed the query text, then search within one user's memories."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        try:
            query_embedding = self._embedding_service.embed_text(query_text)
        except EmbeddingInvocationError:
            # Do not mutate FAISS when embedding fails.
            raise
        return self._vector_store.search(
            query_embedding,
            user_id=user_id,
            limit=limit,
            record_types=record_types,
        )


def get_memory_service(settings: Optional[Settings] = None) -> SemanticMemoryService:
    """Factory that loads an existing FAISS store when present."""
    cfg = settings or get_settings()
    store = FAISSVectorStore(settings=cfg)
    store.load()
    return SemanticMemoryService(settings=cfg, vector_store=store)
