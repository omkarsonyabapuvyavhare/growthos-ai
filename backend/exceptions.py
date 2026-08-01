"""
Project-specific exceptions for GrowthOS AI.

Keep this hierarchy minimal and frontend-safe (no provider secrets).
"""


class GrowthOSError(Exception):
    """Base exception for GrowthOS AI backend errors."""


class AIProviderConfigurationError(GrowthOSError):
    """Raised when AI_PROVIDER is missing, empty, or unsupported."""


class GeminiConfigurationError(GrowthOSError):
    """Raised when Gemini chat is not configured for use (e.g. missing API key)."""


class GeminiInvocationError(GrowthOSError):
    """Raised when a Gemini chat provider call fails."""


class GeminiResponseError(GrowthOSError):
    """Raised when Gemini chat returns empty, malformed, or unusable output."""


class EmbeddingConfigurationError(GrowthOSError):
    """Raised when Gemini embeddings are not configured (e.g. missing API key)."""


class EmbeddingInvocationError(GrowthOSError):
    """Raised when a Gemini embedding provider call fails."""


class EmbeddingResponseError(GrowthOSError):
    """Raised when an embedding response is empty, malformed, or inconsistent."""


class VectorStoreError(GrowthOSError):
    """Raised for FAISS vector-store operational failures."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when vector-store inputs or persisted data fail validation."""


class ProfileAgentError(GrowthOSError):
    """Raised when the Profile Agent fails during orchestration."""


class ProfilePersistenceError(ProfileAgentError):
    """Raised when onboarding SQLite persistence fails (transaction rolled back)."""


class ProfileMemoryError(ProfileAgentError):
    """
    Raised when SQLite onboarding succeeded but semantic-memory writes failed.

    Structured user/profile/goal records are kept. The partial result is attached
    as ``result`` when available for recovery.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class RoadmapAgentError(GrowthOSError):
    """Raised when the Roadmap Agent fails during orchestration."""


class RoadmapContextError(RoadmapAgentError):
    """Raised when user/profile/goal context is missing or incomplete."""


class RoadmapOwnershipError(RoadmapAgentError):
    """Raised when a goal does not belong to the requested user."""


class RoadmapPersistenceError(RoadmapAgentError):
    """Raised when roadmap SQLite persistence fails (transaction rolled back)."""


class RoadmapMemoryError(RoadmapAgentError):
    """
    Raised when SQLite roadmap persistence succeeded but semantic memory failed.

    The roadmap remains valid. Attach ``result`` for recovery when available.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class ResourceCatalogError(GrowthOSError):
    """Raised when the free-resource catalog fails to load or validate."""


class YouTubeConfigurationError(GrowthOSError):
    """Raised when YouTube Data API is not configured for use."""


class YouTubeInvocationError(GrowthOSError):
    """Raised when a YouTube Data API request fails (timeout, HTTP, quota)."""


class YouTubeResponseError(GrowthOSError):
    """Raised when YouTube Data API returns empty, malformed, or unusable data."""


class CuratorAgentError(GrowthOSError):
    """Raised when the Curator Agent fails during orchestration."""


class CuratorContextError(CuratorAgentError):
    """Raised when user/profile/roadmap/milestone context is missing."""


class CuratorOwnershipError(CuratorAgentError):
    """Raised when roadmap/milestone ownership does not match the user."""


class CuratorRankingError(CuratorAgentError):
    """Raised when Gemini ranking output is invalid or unusable."""


class CuratorPersistenceError(CuratorAgentError):
    """Raised when recommendation SQLite persistence fails (rolled back)."""


class CuratorMemoryError(CuratorAgentError):
    """
    Raised when recommendations were saved but catalog semantic indexing failed.

    Recommendations remain valid. Attach ``result`` when available.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class PlannerAgentError(GrowthOSError):
    """Raised when the Daily Planner Agent fails during orchestration."""


class PlannerContextError(PlannerAgentError):
    """Raised when user/profile/roadmap/milestone context is missing."""


class PlannerOwnershipError(PlannerAgentError):
    """Raised when roadmap/milestone ownership does not match the user."""


class PlannerGenerationError(PlannerAgentError):
    """Raised when Gemini plan generation is invalid or unusable."""


class PlannerBudgetError(PlannerAgentError):
    """Raised when the generated plan exceeds time or capacity constraints."""


class PlannerPersistenceError(PlannerAgentError):
    """Raised when daily-plan SQLite persistence fails (rolled back)."""


class ReflectionAgentError(GrowthOSError):
    """Raised when the Reflection Agent fails during orchestration."""


class ReflectionContextError(ReflectionAgentError):
    """Raised when user/plan/milestone context is missing."""


class ReflectionOwnershipError(ReflectionAgentError):
    """Raised when plan/task ownership does not match the user."""


class ReflectionEvidenceError(ReflectionAgentError):
    """Raised when task/resource evidence is invalid or inconsistent."""


class ReflectionGenerationError(ReflectionAgentError):
    """Raised when Gemini reflection insight generation is invalid or unusable."""


class ReflectionPersistenceError(ReflectionAgentError):
    """Raised when reflection SQLite persistence fails (rolled back)."""


class ReflectionConflictError(ReflectionAgentError):
    """Raised when a reflection already exists for the plan and replacement is refused."""


class ReflectionMemoryError(ReflectionAgentError):
    """
    Raised when the reflection was saved but semantic memory persistence failed.

    The SQLite reflection remains valid. Attach ``result`` when available.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class AdaptationAgentError(GrowthOSError):
    """Raised when the Adaptation Agent fails during orchestration."""


class AdaptationContextError(AdaptationAgentError):
    """Raised when user/reflection/history context is missing."""


class AdaptationOwnershipError(AdaptationAgentError):
    """Raised when reflection ownership does not match the user."""


class AdaptationEvidenceError(AdaptationAgentError):
    """Raised when adaptation evidence or preference values are invalid."""


class AdaptationGenerationError(AdaptationAgentError):
    """Raised when Gemini adaptation output is invalid or unusable."""


class AdaptationPersistenceError(AdaptationAgentError):
    """Raised when adaptation SQLite persistence fails (rolled back)."""


class AdaptationConflictError(AdaptationAgentError):
    """Raised when an adaptation already exists for a reflection and force is false."""


class AdaptationMemoryError(AdaptationAgentError):
    """
    Raised when adaptations were saved but semantic memory persistence failed.

    SQLite adaptations remain valid. Attach ``result`` when available.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class WorkflowExecutionError(GrowthOSError):
    """
    Raised when a LangGraph workflow fails at a named stage.

    Safe for API boundaries: no secrets, SQL, prompts, or private blobs.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        original_type: str,
        partial_result: object | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.original_type = original_type
        self.partial_result = partial_result
