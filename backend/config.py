"""
Application configuration loaded from environment variables.

Values are read from a local `.env` file when present. Never hardcode API keys.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Backend package root (growthos-ai/backend). Relative paths resolve here,
# not from the current terminal working directory.
BACKEND_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Environment-based settings for the GrowthOS AI backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AI provider selection — gemini is the only active provider for the MVP
    ai_provider: str = "gemini"

    # Google Gemini chat — no usable default for the API key
    # Backward compatible single key; optional numbered keys for quota rotation.
    gemini_api_key: str = ""
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    gemini_api_key_4: str = ""
    # Example model name documented by langchain-google-genai
    gemini_model: str = "gemini-flash-latest"
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    gemini_max_retries: int = Field(default=2, ge=0, le=10)
    gemini_request_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Gemini embeddings — documented example in langchain-google-genai
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # SQLite connection string, e.g. sqlite:///./growthos.db
    database_url: str = "sqlite:///./growthos.db"

    # CORS origins for local Next.js (comma-separated). Same-origin Vercel
    # production browser calls do not need a production frontend URL here.
    frontend_origin: str = "http://localhost:3000"

    # FAISS persistence (relative paths resolve against BACKEND_ROOT)
    faiss_index_path: str = "./data/faiss_index/index.faiss"
    faiss_metadata_path: str = "./data/faiss_index/metadata.json"

    # Dedicated FAISS index for the free-resource catalog (not user memory)
    faiss_catalog_index_path: str = "./data/faiss_catalog/index.faiss"
    faiss_catalog_metadata_path: str = "./data/faiss_catalog/metadata.json"

    # Approved free-resource seed catalog
    resource_catalog_path: str = "./data/sample_resources.json"

    # Optional YouTube Data API discovery (Curator only; never required to start)
    youtube_api_key: str = ""
    youtube_api_enabled: bool = False
    youtube_max_results: int = Field(default=10, ge=1, le=25)
    youtube_request_timeout_seconds: int = Field(default=10, ge=1, le=60)

    def resolve_path(self, configured: str) -> Path:
        """Resolve a configured filesystem path against the backend root."""
        path = Path(configured.strip())
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path.resolve()

    def resolve_sqlite_path(self) -> Path:
        """
        Convert DATABASE_URL into a filesystem Path.

        Supports forms such as:
        - sqlite:///./growthos.db
        - sqlite:///growthos.db
        - sqlite:////C:/absolute/path.db
        - a bare relative/absolute path (fallback)
        """
        url = self.database_url.strip()
        if url.startswith("sqlite:///"):
            raw_path = url.removeprefix("sqlite:///")
        elif url.startswith("sqlite://"):
            raw_path = url.removeprefix("sqlite://")
        else:
            raw_path = url
        return self.resolve_path(raw_path)

    def resolve_faiss_index_path(self) -> Path:
        """Return the FAISS binary index file path."""
        return self.resolve_path(self.faiss_index_path)

    def resolve_faiss_metadata_path(self) -> Path:
        """Return the FAISS metadata JSON file path."""
        return self.resolve_path(self.faiss_metadata_path)

    def resolve_faiss_catalog_index_path(self) -> Path:
        """Return the dedicated resource-catalog FAISS index path."""
        return self.resolve_path(self.faiss_catalog_index_path)

    def resolve_faiss_catalog_metadata_path(self) -> Path:
        """Return the dedicated resource-catalog FAISS metadata path."""
        return self.resolve_path(self.faiss_catalog_metadata_path)

    def resolve_resource_catalog_path(self) -> Path:
        """Return the free-resource catalog JSON path."""
        return self.resolve_path(self.resource_catalog_path)

    def gemini_api_keys(self) -> list[str]:
        """
        Ordered Gemini API keys for chat requests.

        If any of GEMINI_API_KEY_1..4 is set, those non-empty keys are used
        (deduplicated, order preserved). Otherwise GEMINI_API_KEY is used alone
        for backward compatibility.
        """
        numbered = [
            self.gemini_api_key_1.strip(),
            self.gemini_api_key_2.strip(),
            self.gemini_api_key_3.strip(),
            self.gemini_api_key_4.strip(),
        ]
        configured_numbered = [key for key in numbered if key]
        if configured_numbered:
            return list(dict.fromkeys(configured_numbered))
        primary = self.gemini_api_key.strip()
        return [primary] if primary else []

    def primary_gemini_api_key(self) -> str:
        """First configured Gemini key (empty when none configured)."""
        keys = self.gemini_api_keys()
        return keys[0] if keys else ""

    def is_gemini_configured(self) -> bool:
        """Return True when at least one Gemini API key is present."""
        return bool(self.gemini_api_keys())

    def is_youtube_configured(self) -> bool:
        """Return True when a non-empty YouTube Data API key is present."""
        return bool(self.youtube_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
