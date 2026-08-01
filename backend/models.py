"""
Shared Pydantic v2 models and enums for GrowthOS AI.

Domain models support onboarding, roadmaps, daily plans, reflections,
and dashboard responses. Arbitrary learning goals remain free-text.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums (frontend-friendly lowercase values)
# ---------------------------------------------------------------------------


class CurrentLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class GoalStatus(str, Enum):
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class RoadmapStatus(str, Enum):
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class MilestoneStatus(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


class PhaseStatus(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


class Mood(str, Enum):
    focused = "focused"
    motivated = "motivated"
    curious = "curious"
    calm = "calm"
    tired = "tired"
    stressed = "stressed"
    distracted = "distracted"
    low_energy = "low_energy"


class EnergyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActivityType(str, Enum):
    watch = "watch"
    read = "read"
    listen = "listen"
    practice = "practice"
    review = "review"
    mixed = "mixed"


class Difficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class CompletionStatus(str, Enum):
    completed = "completed"
    partial = "partial"
    skipped = "skipped"


class PlanStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


class DifficultyFeedback(str, Enum):
    too_easy = "too_easy"
    suitable = "suitable"
    too_difficult = "too_difficult"


class LearningStyle(str, Enum):
    visual = "visual"
    auditory = "auditory"
    reading = "reading"
    kinesthetic = "kinesthetic"
    mixed = "mixed"


class PreferredLearningTime(str, Enum):
    morning = "morning"
    afternoon = "afternoon"
    evening = "evening"
    night = "night"
    flexible = "flexible"


class RecommendationStatus(str, Enum):
    suggested = "suggested"
    accepted = "accepted"
    dismissed = "dismissed"
    completed = "completed"
    archived = "archived"


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _non_blank(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be blank")
    return cleaned


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["GrowthOS AI API"])
    # Non-secret YouTube discovery diagnostics (never call YouTube from /health)
    youtube_enabled: bool = False
    youtube_configured: bool = False


# ---------------------------------------------------------------------------
# Users / onboarding
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        return _non_blank(value)


class UserResponse(BaseModel):
    id: int
    display_name: str
    created_at: datetime
    updated_at: datetime


class OnboardingRequest(BaseModel):
    """
    Full onboarding payload.

    learning_goal stays free-text so users can enter any aspiration.
    """

    display_name: str = Field(..., min_length=1, max_length=120)
    learning_goal: str = Field(..., min_length=1, max_length=500)
    aspiration: str = Field(..., min_length=1, max_length=1000)
    motivation: str = Field(..., min_length=1, max_length=2000)
    current_level: CurrentLevel
    target_outcome: str = Field(..., min_length=1, max_length=1000)
    preferred_formats: list[str] = Field(..., min_length=1)
    learning_style: LearningStyle
    daily_available_minutes: int = Field(..., gt=0, le=24 * 60)
    preferred_session_minutes: int = Field(..., gt=0, le=24 * 60)
    attention_span_minutes: int = Field(..., gt=0, le=24 * 60)
    preferred_learning_time: PreferredLearningTime
    habits: list[str] = Field(default_factory=list)
    distractions: list[str] = Field(default_factory=list)

    @field_validator(
        "display_name",
        "learning_goal",
        "aspiration",
        "motivation",
        "target_outcome",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("preferred_formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        cleaned = [_non_blank(item) for item in value]
        if not cleaned:
            raise ValueError("preferred_formats must contain at least one item")
        return cleaned


class UserProfileResponse(BaseModel):
    id: int
    user_id: int
    aspiration: str
    motivation: str
    current_level: CurrentLevel
    target_outcome: str
    learning_style: LearningStyle
    preferred_formats: list[str]
    daily_available_minutes: int
    preferred_session_minutes: int
    attention_span_minutes: int
    preferred_learning_time: PreferredLearningTime
    habits: list[str]
    distractions: list[str]
    created_at: datetime
    updated_at: datetime


class ProfileInterpretation(BaseModel):
    """Structured Gemini interpretation of onboarding answers."""

    identity_summary: str = Field(..., min_length=1)
    aspiration_summary: str = Field(..., min_length=1)
    motivation_summary: str = Field(..., min_length=1)
    current_state_summary: str = Field(..., min_length=1)
    target_state_summary: str = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list)
    likely_challenges: list[str] = Field(default_factory=list)
    learning_preferences_summary: str = Field(..., min_length=1)
    recommended_pacing: str = Field(..., min_length=1)
    attention_strategy: str = Field(..., min_length=1)
    consistency_strategy: str = Field(..., min_length=1)
    initial_personalization_insights: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Goals / roadmap
# ---------------------------------------------------------------------------


class GoalCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=5000)
    status: GoalStatus = GoalStatus.active

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _non_blank(value)


class GoalResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: str
    status: GoalStatus
    created_at: datetime
    updated_at: datetime


class ProfileAgentResult(BaseModel):
    """Typed result from ProfileAgent.process_onboarding."""

    user: UserResponse
    profile: UserProfileResponse
    goal: GoalResponse
    interpretation: ProfileInterpretation
    memory_ids: list[str] = Field(default_factory=list)
    memories_complete: bool = True
    memory_error: Optional[str] = None
    created_at: datetime


class MilestoneGeneration(BaseModel):
    """Gemini-generated milestone before persistence."""

    sequence_number: int = Field(..., ge=1, le=5)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    skills: list[str] = Field(..., min_length=1)
    suggested_activities: list[str] = Field(..., min_length=1)
    completion_criteria: str = Field(..., min_length=1, max_length=1000)
    estimated_sessions: int = Field(..., ge=1, le=40)
    estimated_minutes: int = Field(..., gt=0, le=24 * 60)
    difficulty: Difficulty


class RoadmapPhaseGeneration(BaseModel):
    """Gemini-generated phase before persistence."""

    sequence_number: int = Field(..., ge=1, le=6)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    expected_outcome: str = Field(..., min_length=1, max_length=2000)
    milestones: list[MilestoneGeneration] = Field(..., min_length=1, max_length=5)

    @model_validator(mode="before")
    @classmethod
    def normalize_milestone_sequences(cls, data: object) -> object:
        """Repair occasional Gemini sequence numbering outside 1..5."""
        if not isinstance(data, dict):
            return data
        milestones = data.get("milestones")
        if isinstance(milestones, list) and milestones:
            trimmed = milestones[:5]
            normalized = []
            for index, item in enumerate(trimmed, start=1):
                if isinstance(item, dict):
                    fixed = dict(item)
                    fixed["sequence_number"] = index
                    normalized.append(fixed)
                else:
                    normalized.append(item)
            data = dict(data)
            data["milestones"] = normalized
        return data


class RoadmapGeneration(BaseModel):
    """Structured Gemini roadmap output."""

    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=4000)
    estimated_duration_weeks: int = Field(..., ge=1, le=52)
    pacing_rationale: str = Field(..., min_length=1, max_length=2000)
    personalization_rationale: str = Field(..., min_length=1, max_length=2000)
    phases: list[RoadmapPhaseGeneration] = Field(..., min_length=2, max_length=6)

    @model_validator(mode="before")
    @classmethod
    def normalize_phase_sequences(cls, data: object) -> object:
        """Repair occasional Gemini phase sequence numbering outside 1..6."""
        if not isinstance(data, dict):
            return data
        phases = data.get("phases")
        if isinstance(phases, list) and phases:
            trimmed = phases[:6]
            normalized = []
            for index, item in enumerate(trimmed, start=1):
                if isinstance(item, dict):
                    fixed = dict(item)
                    fixed["sequence_number"] = index
                    normalized.append(fixed)
                else:
                    normalized.append(item)
            data = dict(data)
            data["phases"] = normalized
        return data


class MilestoneResponse(BaseModel):
    id: int
    phase_id: int
    sequence_number: int
    title: str
    description: str
    skills: list[str] = Field(default_factory=list)
    suggested_activities: list[str] = Field(default_factory=list)
    completion_criteria: str = ""
    estimated_sessions: int = Field(default=1, ge=1)
    estimated_minutes: int = Field(default=30, gt=0)
    difficulty: Difficulty = Difficulty.beginner
    status: MilestoneStatus
    progress_percent: float = Field(..., ge=0, le=100)
    created_at: datetime
    updated_at: datetime


class RoadmapPhaseResponse(BaseModel):
    id: int
    roadmap_id: int
    sequence_number: int
    title: str
    description: str
    expected_outcome: str
    status: PhaseStatus
    milestones: list[MilestoneResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RoadmapResponse(BaseModel):
    id: int
    user_id: int
    goal_id: int
    title: str
    summary: str
    estimated_duration_weeks: int = Field(..., gt=0)
    progress_percent: float = Field(..., ge=0, le=100)
    status: RoadmapStatus
    pacing_rationale: str = ""
    personalization_rationale: str = ""
    phases: list[RoadmapPhaseResponse] = Field(default_factory=list)
    current_active_milestone_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class RoadmapAgentResult(BaseModel):
    """Typed result from RoadmapAgent.generate_roadmap."""

    roadmap: RoadmapResponse
    phases: list[RoadmapPhaseResponse] = Field(default_factory=list)
    milestones: list[MilestoneResponse] = Field(default_factory=list)
    active_milestone: Optional[MilestoneResponse] = None
    pacing_rationale: str
    personalization_rationale: str
    memory_ids: list[str] = Field(default_factory=list)
    memories_complete: bool = True
    memory_error: Optional[str] = None
    reused_existing: bool = False
    created_at: datetime


# ---------------------------------------------------------------------------
# Resources / recommendations / curator
# ---------------------------------------------------------------------------


class ResourceCatalogItem(BaseModel):
    """One validated free resource from the approved seed catalog."""

    catalog_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=300)
    source: str = Field(..., min_length=1, max_length=100)
    resource_type: str = Field(..., min_length=1, max_length=64)
    url: HttpUrl
    description: str = Field(..., min_length=1, max_length=4000)
    topics: list[str] = Field(..., min_length=1)
    skills: list[str] = Field(..., min_length=1)
    difficulty: Difficulty
    estimated_duration_minutes: int = Field(..., gt=0, le=24 * 60)
    supported_formats: list[str] = Field(..., min_length=1)
    language: str = Field(default="en", min_length=2, max_length=16)
    is_free: bool = True
    requires_account: bool = False
    suitable_moods: list[Mood] = Field(default_factory=list)
    suitable_energy_levels: list[EnergyLevel] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @field_validator("catalog_id", "title", "source", "resource_type", "description", "language")
    @classmethod
    def non_blank_catalog_text(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("is_free")
    @classmethod
    def must_be_free(cls, value: bool) -> bool:
        if not value:
            raise ValueError("catalog resources must be free and publicly accessible")
        return value


class ResourceCandidate(BaseModel):
    """Catalog item enriched for curator ranking."""

    catalog_id: str
    resource_id: int
    title: str
    source: str
    resource_type: str
    url: HttpUrl
    description: str
    topics: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    estimated_duration_minutes: int = Field(..., gt=0)
    supported_formats: list[str] = Field(default_factory=list)
    suitable_moods: list[Mood] = Field(default_factory=list)
    suitable_energy_levels: list[EnergyLevel] = Field(default_factory=list)
    semantic_score: float = Field(default=0.0, ge=0, le=1)
    skill_overlap_score: float = Field(default=0.0, ge=0, le=1)
    difficulty_fit_score: float = Field(default=0.0, ge=0, le=1)
    format_fit_score: float = Field(default=0.0, ge=0, le=1)
    duration_fit_score: float = Field(default=0.0, ge=0, le=1)
    mood_fit_score: float = Field(default=0.0, ge=0, le=1)
    energy_fit_score: float = Field(default=0.0, ge=0, le=1)
    deterministic_score: float = Field(default=0.0, ge=0, le=1)


class CuratorRankedItem(BaseModel):
    """Gemini ranking row — IDs only, never URLs."""

    candidate_id: str = Field(..., min_length=1, max_length=64)
    relevance_score: float = Field(..., ge=0, le=1)
    reason: str = Field(..., min_length=1, max_length=1000)
    milestone_fit: str = Field(..., min_length=1, max_length=1000)
    mood_suitability: str = Field(..., min_length=1, max_length=500)
    suggested_use: str = Field(..., min_length=1, max_length=1000)
    estimated_effort: str = Field(..., min_length=1, max_length=200)


class CuratorRankingGeneration(BaseModel):
    """Structured Gemini ranking response."""

    selections: list[CuratorRankedItem] = Field(default_factory=list)


class CuratedRecommendation(BaseModel):
    """One persisted, trusted recommendation."""

    id: int
    user_id: int
    roadmap_id: int
    milestone_id: int
    resource_id: int
    catalog_id: str
    title: str
    source: str
    resource_type: str
    url: HttpUrl
    description: str
    difficulty: Difficulty
    estimated_duration_minutes: int = Field(..., gt=0)
    relevance_score: float = Field(..., ge=0, le=1)
    reason: str
    milestone_fit: str = ""
    mood_suitability: str = ""
    suggested_use: str = ""
    estimated_effort: str = ""
    score_breakdown: dict = Field(default_factory=dict)
    status: RecommendationStatus = RecommendationStatus.suggested
    recommended_at: datetime


class CuratorAgentResult(BaseModel):
    """Typed result from CuratorAgent.recommend_resources."""

    user_id: int
    roadmap_id: int
    milestone_id: int
    recommendations: list[CuratedRecommendation] = Field(default_factory=list)
    candidate_count: int = 0
    used_deterministic_fallback: bool = False
    reused_existing: bool = False
    ranking_notes: str = ""
    created_at: datetime


class ResourceResponse(BaseModel):
    id: int
    title: str
    source: str
    resource_type: str
    url: HttpUrl
    description: str
    difficulty: Difficulty
    estimated_duration_minutes: int = Field(..., gt=0)
    is_free: bool = True
    why_it_matches: Optional[str] = None
    related_milestone_id: Optional[int] = None
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1)
    mood_suitability: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Daily check-in / plan
# ---------------------------------------------------------------------------


class DailyCheckInRequest(BaseModel):
    mood: Mood
    energy_level: EnergyLevel
    focus_level: int = Field(..., ge=1, le=5)
    available_minutes: int = Field(..., gt=0, le=24 * 60)
    preferred_activity: ActivityType
    notes: str = Field(default="", max_length=2000)


class DailyCheckInResponse(BaseModel):
    id: int
    user_id: int
    mood: Mood
    energy_level: EnergyLevel
    focus_level: int = Field(..., ge=1, le=5)
    available_minutes: int = Field(..., gt=0)
    preferred_activity: ActivityType
    notes: str = ""
    created_at: datetime


class TaskCompletionRequest(BaseModel):
    """Update a daily task and optionally record a resource interaction."""

    status: TaskStatus
    completion_percent: Optional[float] = Field(default=None, ge=0, le=100)
    duration_minutes: Optional[int] = Field(default=None, ge=0, le=24 * 60)
    effectiveness_rating: Optional[int] = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)


class ResourceInteractionRequest(BaseModel):
    resource_id: int
    daily_plan_id: Optional[int] = None
    interaction_type: str = Field(..., min_length=1, max_length=64)
    completion_percent: float = Field(default=0, ge=0, le=100)
    effectiveness_rating: Optional[int] = Field(default=None, ge=1, le=5)
    duration_minutes: Optional[int] = Field(default=None, ge=0, le=24 * 60)


class DailyTaskResponse(BaseModel):
    id: int
    daily_plan_id: int
    resource_id: Optional[int] = None
    sequence_number: int
    title: str
    description: str
    activity_type: ActivityType
    estimated_minutes: int = Field(..., gt=0)
    difficulty: Difficulty
    status: TaskStatus
    completed_at: Optional[datetime] = None
    # Explain-every-decision fields (from task metadata + resource join)
    why_selected: str = ""
    milestone_connection: str = ""
    expected_outcome: str = ""
    content_type: str = ""
    mood_rationale: str = ""
    resource_title: Optional[str] = None
    resource_source: Optional[str] = None
    resource_url: Optional[HttpUrl] = None
    resource_thumbnail_url: Optional[str] = None
    resource_channel: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DailyPlanResponse(BaseModel):
    id: int
    user_id: int
    roadmap_id: Optional[int] = None
    milestone_id: Optional[int] = None
    checkin_id: Optional[int] = None
    plan_date: date
    summary: str
    total_estimated_minutes: int = Field(..., ge=0)
    status: PlanStatus
    tasks: list[DailyTaskResponse] = Field(default_factory=list)
    guidance_tone: str = ""
    mood_influence_summary: str = ""
    adaptation_explanation: str = ""
    task_count_rationale: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PlannerTaskGeneration(BaseModel):
    """Gemini-generated daily task before persistence."""

    sequence_number: int = Field(..., ge=1, le=5)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    activity_type: ActivityType
    resource_id: Optional[int] = Field(default=None, gt=0)
    estimated_minutes: int = Field(..., ge=3, le=24 * 60)
    difficulty: Difficulty
    expected_outcome: str = Field(..., min_length=1, max_length=1000)
    why_selected: str = Field(..., min_length=1, max_length=1000)
    milestone_connection: str = Field(..., min_length=1, max_length=1000)
    mood_rationale: str = Field(..., min_length=1, max_length=1000)
    content_type: str = Field(default="", max_length=64)


class DailyPlanGeneration(BaseModel):
    """Structured Gemini daily-plan output."""

    summary: str = Field(..., min_length=1, max_length=2000)
    guidance_tone: str = Field(..., min_length=1, max_length=200)
    mood_influence_summary: str = Field(..., min_length=1, max_length=1000)
    task_count_rationale: str = Field(..., min_length=1, max_length=1000)
    adaptation_explanation: str = Field(..., min_length=1, max_length=2000)
    tasks: list[PlannerTaskGeneration] = Field(..., min_length=1, max_length=5)


class PlannerAgentResult(BaseModel):
    """Typed result from the Daily Planner Agent (Phase B)."""

    checkin: DailyCheckInResponse
    plan: DailyPlanResponse
    goal_title: str
    milestone_title: Optional[str] = None
    reused_existing: bool = False
    created_at: datetime


# ---------------------------------------------------------------------------
# Reflection / adaptation
# ---------------------------------------------------------------------------


class ReflectionTaskUpdate(BaseModel):
    """Task completion evidence submitted with a reflection."""

    task_id: int = Field(..., gt=0)
    update: TaskCompletionRequest


class ReflectionRequest(BaseModel):
    daily_plan_id: int = Field(..., gt=0)
    completion_status: CompletionStatus
    learning_summary: str = Field(default="", max_length=5000)
    focus_rating: int = Field(..., ge=1, le=5)
    resource_effectiveness: int = Field(..., ge=1, le=5)
    difficulty_feedback: DifficultyFeedback
    mood_match: bool
    distractions: list[str] = Field(default_factory=list)
    wants_similar_resources: bool
    mood_after: Mood | str = Field(default="")
    task_updates: list[ReflectionTaskUpdate] = Field(default_factory=list)
    resource_interactions: list[ResourceInteractionRequest] = Field(default_factory=list)
    actual_minutes_spent: Optional[int] = Field(default=None, ge=0, le=24 * 60)


class ReflectionInsightGeneration(BaseModel):
    """Structured Gemini reflection insight (evidence-based only)."""

    insight: str = Field(..., min_length=1, max_length=1000)
    learning_progress_summary: str = Field(..., min_length=1, max_length=1000)
    completion_observation: str = Field(..., min_length=1, max_length=1000)
    focus_observation: str = Field(..., min_length=1, max_length=1000)
    difficulty_observation: str = Field(..., min_length=1, max_length=1000)
    resource_observation: str = Field(..., min_length=1, max_length=1000)
    distraction_observation: str = Field(..., min_length=1, max_length=1000)
    mood_observation: str = Field(..., min_length=1, max_length=1000)
    positive_signals: list[str] = Field(default_factory=list, max_length=8)
    friction_signals: list[str] = Field(default_factory=list, max_length=8)
    evidence_for_adaptation: list[str] = Field(default_factory=list, max_length=10)
    recommended_next_session_adjustments: list[str] = Field(
        default_factory=list,
        max_length=8,
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class ReflectionResponse(BaseModel):
    id: int
    user_id: int
    daily_plan_id: int
    completion_status: CompletionStatus
    learning_summary: str
    focus_rating: int = Field(..., ge=1, le=5)
    resource_effectiveness: int = Field(..., ge=1, le=5)
    difficulty_feedback: DifficultyFeedback
    mood_match: bool
    distractions: list[str]
    wants_similar_resources: bool
    mood_after: str
    insight: Optional[str] = None
    created_at: datetime


class ReflectionAgentResult(BaseModel):
    reflection: ReflectionResponse
    plan_completion_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    milestone_progress_before: float = Field(default=0.0, ge=0.0, le=100.0)
    milestone_progress_after: float = Field(default=0.0, ge=0.0, le=100.0)
    roadmap_progress_before: float = Field(default=0.0, ge=0.0, le=100.0)
    roadmap_progress_after: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_ids: list[str] = Field(default_factory=list)
    memories_complete: bool = True
    memory_error: Optional[str] = None
    reused_existing: bool = False
    created_at: datetime


class PreferenceUpdateGeneration(BaseModel):
    """One Gemini-proposed preference change."""

    preference_key: str = Field(..., min_length=1, max_length=64)
    preference_value: str = Field(..., min_length=1, max_length=255)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    action: Literal["create", "update", "keep"] = "keep"


class AdaptationGeneration(BaseModel):
    """Structured Gemini adaptation interpretation."""

    summary: str = Field(..., min_length=1, max_length=1000)
    detected_patterns: list[str] = Field(default_factory=list, max_length=10)
    next_session_adjustments: list[str] = Field(default_factory=list, max_length=8)
    preference_updates: list[PreferenceUpdateGeneration] = Field(
        default_factory=list,
        max_length=8,
    )
    pacing_adjustment: str = Field(default="", max_length=500)
    difficulty_adjustment: str = Field(default="", max_length=500)
    format_adjustment: str = Field(default="", max_length=500)
    duration_adjustment: str = Field(default="", max_length=500)
    task_count_adjustment: str = Field(default="", max_length=500)
    evidence_summary: str = Field(..., min_length=1, max_length=2000)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    is_early_signal: bool = False
    adaptation_explanation: str = Field(..., min_length=1, max_length=1000)


class AdaptationInsightResponse(BaseModel):
    id: int
    user_id: int
    insight_type: str
    insight: str
    confidence_score: float = Field(..., ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class UserPreferenceResponse(BaseModel):
    id: int
    user_id: int
    preference_key: str
    preference_value: str
    confidence_score: float = Field(..., ge=0, le=1)
    source: str = "system"
    created_at: datetime
    updated_at: datetime


class AdaptationAgentResult(BaseModel):
    insights: list[AdaptationInsightResponse] = Field(default_factory=list)
    preferences: list[UserPreferenceResponse] = Field(default_factory=list)
    adaptation_explanation: str
    detected_patterns: list[str] = Field(default_factory=list)
    goal_unchanged: bool = True
    roadmap_unchanged: bool = True
    milestone_unchanged: bool = True
    is_early_signal: bool = False
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reflection_id: Optional[int] = None
    reused_existing: bool = False
    memory_ids: list[str] = Field(default_factory=list)
    memories_complete: bool = True
    memory_error: Optional[str] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class DashboardResponse(BaseModel):
    user: UserResponse
    active_goal: Optional[GoalResponse] = None
    current_roadmap: Optional[RoadmapResponse] = None
    current_milestone: Optional[MilestoneResponse] = None
    overall_progress_percent: float = Field(default=0, ge=0, le=100)
    today_mood: Optional[Mood] = None
    today_plan: Optional[DailyPlanResponse] = None
    completion_streak: int = Field(default=0, ge=0)
    recent_reflections: list[ReflectionResponse] = Field(default_factory=list)
    preferred_content_type: Optional[str] = None
    preferred_session_minutes: Optional[int] = Field(default=None, ge=0)
    average_session_minutes: Optional[float] = Field(default=None, ge=0)
    resource_effectiveness_avg: Optional[float] = Field(default=None, ge=1, le=5)
    weekly_learning_consistency: Optional[float] = Field(default=None, ge=0, le=1)
    skill_growth: list[str] = Field(default_factory=list)
    detected_patterns: list[str] = Field(default_factory=list)
    growthos_knows_you: list[str] = Field(default_factory=list)
    plan_change_explanation: Optional[str] = None
    ai_insight: Optional[str] = None
    recommended_next_action: Optional[str] = None
    adaptation_insights: list[AdaptationInsightResponse] = Field(default_factory=list)
    completed_sessions: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# LangGraph workflow results
# ---------------------------------------------------------------------------


class OnboardingWorkflowResult(BaseModel):
    """Result of Profile → Roadmap onboarding orchestration."""

    user: UserResponse
    profile: UserProfileResponse
    goal: GoalResponse
    roadmap: RoadmapResponse
    active_milestone: Optional[MilestoneResponse] = None
    profile_result: ProfileAgentResult
    roadmap_result: RoadmapAgentResult
    completed_steps: list[str] = Field(default_factory=list)
    status: str = "completed"
    current_stage: str = "completed"


class DailyPlanningWorkflowResult(BaseModel):
    """Result of daily planning stage (stops at awaiting user completion)."""

    user_id: int
    plan: DailyPlanResponse
    tasks: list[DailyTaskResponse] = Field(default_factory=list)
    checkin: DailyCheckInResponse
    planner_result: PlannerAgentResult
    completed_steps: list[str] = Field(default_factory=list)
    status: str = "awaiting_user_completion"
    current_stage: str = "awaiting_user_completion"
    awaiting_user_completion: bool = True


class DailyPostSessionWorkflowResult(BaseModel):
    """Result of Reflection → Adaptation post-session orchestration."""

    reflection: ReflectionResponse
    adaptation: AdaptationAgentResult
    adaptation_explanation: str
    reflection_result: ReflectionAgentResult
    completed_steps: list[str] = Field(default_factory=list)
    status: str = "completed"
    current_stage: str = "completed"
    goal_unchanged: bool = True


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------


class RoadmapCreateRequest(BaseModel):
    """Optional goal targeting for roadmap generation."""

    goal_id: Optional[int] = Field(default=None, gt=0)
    regenerate: bool = False


class DailyPlanCreateRequest(BaseModel):
    """Create today's plan from a mood check-in."""

    mood: Mood
    energy_level: EnergyLevel
    focus_level: int = Field(..., ge=1, le=5)
    available_minutes: int = Field(..., gt=0, le=24 * 60)
    preferred_activity: ActivityType
    notes: str = Field(default="", max_length=2000)
    plan_date: Optional[date] = None
    refresh: bool = False


class AdaptationRunRequest(BaseModel):
    """Run adaptation for a persisted reflection."""

    reflection_id: int = Field(..., gt=0)
    force: bool = False


class DemoDayLoopRequest(BaseModel):
    """Optional overrides for the Day 1 → Day 2 demo loop."""

    day1_checkin: Optional[DailyCheckInRequest] = None
    day2_checkin: Optional[DailyCheckInRequest] = None
    reflection: Optional[ReflectionRequest] = None


class DemoDayLoopResponse(BaseModel):
    """Judge-facing Day 1 → Day 2 demo result from real workflows."""

    user_id: int
    goal_title: str
    goal_unchanged: bool
    day1_checkin: DailyCheckInResponse
    day1_plan: DailyPlanResponse
    day1_tasks: list[DailyTaskResponse] = Field(default_factory=list)
    reflection: ReflectionResponse
    reflection_insight: Optional[str] = None
    adaptation: AdaptationAgentResult
    adaptation_explanation: str
    detected_patterns: list[str] = Field(default_factory=list)
    is_early_signal: bool = True
    day2_checkin: DailyCheckInResponse
    day2_plan: DailyPlanResponse
    day2_tasks: list[DailyTaskResponse] = Field(default_factory=list)
    recommended_next_action: Optional[str] = None
    completed_steps: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Safe API error body."""

    detail: str
    stage: Optional[str] = None
    error_type: Optional[str] = None
