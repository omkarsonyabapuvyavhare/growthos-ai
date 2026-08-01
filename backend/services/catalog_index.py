"""
Dedicated FAISS index for the free-resource catalog.

This store is separate from user semantic memory. It uses a reserved
namespace user_id that exists only inside this catalog index path and
never shares files with the user-memory FAISS store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from config import Settings, get_settings
from models import ResourceCatalogItem
from services.embedding import GeminiEmbeddingService
from services.resource_catalog import embedding_text_for_item
from services.vector_models import MemoryRecordType, VectorMemoryRecord
from services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Reserved only for the dedicated catalog FAISS store (not user memory).
CATALOG_NAMESPACE_USER_ID = 1


class CatalogSemanticIndex:
    """Embed and search approved catalog resources in an isolated FAISS index."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding = embedding_service or GeminiEmbeddingService(
            settings=self._settings
        )
        if vector_store is not None:
            self._store = vector_store
        else:
            self._store = FAISSVectorStore(
                settings=self._settings,
                index_path=index_path
                or self._settings.resolve_faiss_catalog_index_path(),
                metadata_path=metadata_path
                or self._settings.resolve_faiss_catalog_metadata_path(),
                autosave=True,
            )

    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._store

    def ensure_indexed(self, items: Sequence[ResourceCatalogItem]) -> int:
        """Index any missing catalog items. Returns newly added count."""
        pending_items: list[ResourceCatalogItem] = []
        for item in items:
            memory_id = f"catalog-{item.catalog_id}"
            if self._store.get_memory(memory_id) is None:
                pending_items.append(item)
        if not pending_items:
            return 0

        texts = [embedding_text_for_item(item) for item in pending_items]
        vectors = self._embedding.embed_documents(texts)
        records = [
            VectorMemoryRecord(
                memory_id=f"catalog-{item.catalog_id}",
                user_id=CATALOG_NAMESPACE_USER_ID,
                record_type=MemoryRecordType.resource.value,
                source_record_id=item.catalog_id,
                text=text,
                metadata={
                    "catalog_id": item.catalog_id,
                    "source": item.source,
                    "difficulty": item.difficulty.value,
                },
            )
            for item, text in zip(pending_items, texts)
        ]
        previous_autosave = getattr(self._store, "_autosave", True)
        try:
            self._store._autosave = False
            self._store.add_memories(records, vectors)
        finally:
            self._store._autosave = previous_autosave
            if previous_autosave:
                self._store.save()
        logger.info("CatalogSemanticIndex added=%s", len(records))
        return len(records)

    def rank_catalog_ids(
        self,
        query_text: str,
        candidate_catalog_ids: Sequence[str],
        *,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Rank candidate catalog IDs by semantic similarity.

        Returns (catalog_id, similarity_01) sorted descending.
        """
        if not candidate_catalog_ids:
            return []
        allowed = set(candidate_catalog_ids)
        vector = self._embedding.embed_text(query_text)
        hits = self._store.search(
            vector,
            user_id=CATALOG_NAMESPACE_USER_ID,
            limit=max(limit * 3, 20),
            record_types=[MemoryRecordType.resource.value],
        )
        ranked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for hit in hits:
            catalog_id = str(
                hit.metadata.get("catalog_id") or hit.source_record_id or ""
            )
            if not catalog_id or catalog_id not in allowed or catalog_id in seen:
                continue
            # Cosine similarity in [-1, 1] -> [0, 1]
            score = max(0.0, min(1.0, (float(hit.similarity_score) + 1.0) / 2.0))
            ranked.append((catalog_id, score))
            seen.add(catalog_id)
            if len(ranked) >= limit:
                break

        # Ensure every candidate has a score (0 if not retrieved)
        for catalog_id in candidate_catalog_ids:
            if catalog_id not in seen:
                ranked.append((catalog_id, 0.0))
                seen.add(catalog_id)
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[: max(limit, len(candidate_catalog_ids))]
