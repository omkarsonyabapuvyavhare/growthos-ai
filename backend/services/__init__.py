"""GrowthOS AI services package."""

from exceptions import (
    AIProviderConfigurationError,
    EmbeddingConfigurationError,
    EmbeddingInvocationError,
    EmbeddingResponseError,
    GeminiConfigurationError,
    GeminiInvocationError,
    GeminiResponseError,
    VectorStoreError,
    VectorStoreValidationError,
    YouTubeConfigurationError,
    YouTubeInvocationError,
    YouTubeResponseError,
)
from services.ai_provider import AIProvider, GeminiProvider, get_ai_provider
from services.database import get_connection, init_db
from services.embedding import GeminiEmbeddingService, get_embedding_service
from services.gemini import GeminiService, get_gemini_service
from services.memory import SemanticMemoryService, get_memory_service
from services.vector_models import MemoryRecordType, VectorMemoryRecord, VectorSearchResult
from services.vector_store import FAISSVectorStore, get_vector_store
from services.youtube import YouTubeService, get_youtube_service

__all__ = [
    "AIProvider",
    "AIProviderConfigurationError",
    "EmbeddingConfigurationError",
    "EmbeddingInvocationError",
    "EmbeddingResponseError",
    "FAISSVectorStore",
    "GeminiConfigurationError",
    "GeminiEmbeddingService",
    "GeminiInvocationError",
    "GeminiProvider",
    "GeminiResponseError",
    "GeminiService",
    "MemoryRecordType",
    "SemanticMemoryService",
    "VectorMemoryRecord",
    "VectorSearchResult",
    "VectorStoreError",
    "VectorStoreValidationError",
    "YouTubeConfigurationError",
    "YouTubeInvocationError",
    "YouTubeResponseError",
    "YouTubeService",
    "get_ai_provider",
    "get_connection",
    "get_embedding_service",
    "get_gemini_service",
    "get_memory_service",
    "get_vector_store",
    "get_youtube_service",
    "init_db",
]
