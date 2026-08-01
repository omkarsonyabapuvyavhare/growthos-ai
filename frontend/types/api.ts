/**
 * Typed contracts mirrored from the GrowthOS AI FastAPI backend.
 */

export type CurrentLevel = "beginner" | "intermediate" | "advanced";
export type LearningStyle =
  | "visual"
  | "auditory"
  | "reading"
  | "kinesthetic"
  | "mixed";
export type PreferredLearningTime =
  | "morning"
  | "afternoon"
  | "evening"
  | "night"
  | "flexible";
export type Mood =
  | "focused"
  | "motivated"
  | "curious"
  | "calm"
  | "tired"
  | "stressed"
  | "distracted"
  | "low_energy";
export type EnergyLevel = "low" | "medium" | "high";
export type ActivityType =
  | "watch"
  | "read"
  | "listen"
  | "practice"
  | "review"
  | "mixed";
export type Difficulty = "beginner" | "intermediate" | "advanced";
export type TaskStatus = "pending" | "in_progress" | "completed" | "skipped";
export type PlanStatus = "pending" | "in_progress" | "completed" | "skipped";
export type CompletionStatus = "completed" | "partial" | "skipped";
export type DifficultyFeedback = "too_easy" | "suitable" | "too_difficult";
export type MilestoneStatus =
  | "not_started"
  | "in_progress"
  | "completed"
  | "skipped";
export type PhaseStatus =
  | "not_started"
  | "in_progress"
  | "completed"
  | "skipped";
export type RoadmapStatus = "active" | "paused" | "completed" | "archived";
export type GoalStatus = "active" | "paused" | "completed" | "archived";

export interface ApiErrorBody {
  detail: string;
  stage?: string | null;
  error_type?: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface UserResponse {
  id: number;
  display_name: string;
  created_at: string;
  updated_at: string;
}

export interface OnboardingRequest {
  display_name: string;
  learning_goal: string;
  aspiration: string;
  motivation: string;
  current_level: CurrentLevel;
  target_outcome: string;
  preferred_formats: string[];
  learning_style: LearningStyle;
  daily_available_minutes: number;
  preferred_session_minutes: number;
  attention_span_minutes: number;
  preferred_learning_time: PreferredLearningTime;
  habits: string[];
  distractions: string[];
}

export interface UserProfileResponse {
  id: number;
  user_id: number;
  aspiration: string;
  motivation: string;
  current_level: CurrentLevel;
  target_outcome: string;
  learning_style: LearningStyle;
  preferred_formats: string[];
  daily_available_minutes: number;
  preferred_session_minutes: number;
  attention_span_minutes: number;
  preferred_learning_time: PreferredLearningTime;
  habits: string[];
  distractions: string[];
  created_at: string;
  updated_at: string;
}

export interface GoalResponse {
  id: number;
  user_id: number;
  title: string;
  description: string;
  status: GoalStatus;
  created_at: string;
  updated_at: string;
}

export interface MilestoneResponse {
  id: number;
  phase_id: number;
  sequence_number: number;
  title: string;
  description: string;
  skills: string[];
  suggested_activities: string[];
  completion_criteria: string;
  estimated_sessions: number;
  estimated_minutes: number;
  difficulty: Difficulty;
  status: MilestoneStatus;
  progress_percent: number;
  created_at: string;
  updated_at: string;
}

export interface RoadmapPhaseResponse {
  id: number;
  roadmap_id: number;
  sequence_number: number;
  title: string;
  description: string;
  expected_outcome: string;
  status: PhaseStatus;
  milestones: MilestoneResponse[];
  created_at: string;
  updated_at: string;
}

export interface RoadmapResponse {
  id: number;
  user_id: number;
  goal_id: number;
  title: string;
  summary: string;
  estimated_duration_weeks: number;
  progress_percent: number;
  status: RoadmapStatus;
  pacing_rationale: string;
  personalization_rationale: string;
  phases: RoadmapPhaseResponse[];
  current_active_milestone_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface ProfileInterpretation {
  identity_summary: string;
  aspiration_summary: string;
  motivation_summary: string;
  current_state_summary: string;
  target_state_summary: string;
  strengths: string[];
  likely_challenges: string[];
  learning_preferences_summary: string;
  recommended_pacing: string;
  attention_strategy: string;
  consistency_strategy: string;
  initial_personalization_insights: string[];
}

export interface ProfileAgentResult {
  user: UserResponse;
  profile: UserProfileResponse;
  goal: GoalResponse;
  interpretation: ProfileInterpretation;
  memory_ids: string[];
  memories_complete: boolean;
  memory_error: string | null;
  created_at: string;
}

export interface RoadmapAgentResult {
  roadmap: RoadmapResponse;
  phases: RoadmapPhaseResponse[];
  milestones: MilestoneResponse[];
  active_milestone: MilestoneResponse | null;
  pacing_rationale: string;
  personalization_rationale: string;
  memory_ids: string[];
  memories_complete: boolean;
  memory_error: string | null;
  reused_existing: boolean;
  created_at: string;
}

export interface OnboardingWorkflowResult {
  user: UserResponse;
  profile: UserProfileResponse;
  goal: GoalResponse;
  roadmap: RoadmapResponse;
  active_milestone: MilestoneResponse | null;
  profile_result: ProfileAgentResult;
  roadmap_result: RoadmapAgentResult;
  completed_steps: string[];
  status: string;
  current_stage: string;
}

export interface RoadmapCreateRequest {
  goal_id?: number | null;
  regenerate?: boolean;
}

export interface DailyCheckInRequest {
  mood: Mood;
  energy_level: EnergyLevel;
  focus_level: number;
  available_minutes: number;
  preferred_activity: ActivityType;
  notes?: string;
}

export interface DailyCheckInResponse {
  id: number;
  user_id: number;
  mood: Mood;
  energy_level: EnergyLevel;
  focus_level: number;
  available_minutes: number;
  preferred_activity: ActivityType;
  notes: string;
  created_at: string;
}

export interface DailyPlanCreateRequest {
  mood: Mood;
  energy_level: EnergyLevel;
  focus_level: number;
  available_minutes: number;
  preferred_activity: ActivityType;
  notes?: string;
  plan_date?: string | null;
  refresh?: boolean;
}

export interface DailyTaskResponse {
  id: number;
  daily_plan_id: number;
  resource_id: number | null;
  sequence_number: number;
  title: string;
  description: string;
  activity_type: ActivityType;
  estimated_minutes: number;
  difficulty: Difficulty;
  status: TaskStatus;
  completed_at: string | null;
  why_selected: string;
  milestone_connection: string;
  expected_outcome: string;
  content_type: string;
  mood_rationale: string;
  resource_title: string | null;
  resource_source: string | null;
  resource_url: string | null;
  resource_thumbnail_url?: string | null;
  resource_channel?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface DailyPlanResponse {
  id: number;
  user_id: number;
  roadmap_id: number | null;
  milestone_id: number | null;
  checkin_id: number | null;
  plan_date: string;
  summary: string;
  total_estimated_minutes: number;
  status: PlanStatus;
  tasks: DailyTaskResponse[];
  guidance_tone: string;
  mood_influence_summary: string;
  adaptation_explanation: string;
  task_count_rationale: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PlannerAgentResult {
  checkin: DailyCheckInResponse;
  plan: DailyPlanResponse;
  goal_title: string;
  milestone_title: string | null;
  reused_existing: boolean;
  created_at: string;
}

export interface DailyPlanningWorkflowResult {
  user_id: number;
  plan: DailyPlanResponse;
  tasks: DailyTaskResponse[];
  checkin: DailyCheckInResponse;
  planner_result: PlannerAgentResult;
  completed_steps: string[];
  status: string;
  current_stage: string;
  awaiting_user_completion: boolean;
}

export interface TaskCompletionRequest {
  status: TaskStatus;
  completion_percent?: number | null;
  duration_minutes?: number | null;
  effectiveness_rating?: number | null;
  notes?: string;
}

export interface ReflectionTaskUpdate {
  task_id: number;
  update: TaskCompletionRequest;
}

export interface ReflectionRequest {
  daily_plan_id: number;
  completion_status: CompletionStatus;
  learning_summary?: string;
  focus_rating: number;
  resource_effectiveness: number;
  difficulty_feedback: DifficultyFeedback;
  mood_match: boolean;
  distractions?: string[];
  wants_similar_resources: boolean;
  mood_after?: Mood | string;
  task_updates?: ReflectionTaskUpdate[];
  actual_minutes_spent?: number | null;
}

export interface ReflectionResponse {
  id: number;
  user_id: number;
  daily_plan_id: number;
  completion_status: CompletionStatus;
  learning_summary: string;
  focus_rating: number;
  resource_effectiveness: number;
  difficulty_feedback: DifficultyFeedback;
  mood_match: boolean;
  distractions: string[];
  wants_similar_resources: boolean;
  mood_after: string;
  insight: string | null;
  created_at: string;
}

export interface ReflectionAgentResult {
  reflection: ReflectionResponse;
  plan_completion_percent: number;
  milestone_progress_before: number;
  milestone_progress_after: number;
  roadmap_progress_before: number;
  roadmap_progress_after: number;
  memory_ids: string[];
  memories_complete: boolean;
  memory_error: string | null;
  reused_existing: boolean;
  created_at: string;
}

export interface AdaptationInsightResponse {
  id: number;
  user_id: number;
  insight_type: string;
  insight: string;
  confidence_score: number;
  evidence: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserPreferenceResponse {
  id: number;
  user_id: number;
  preference_key: string;
  preference_value: string;
  confidence_score: number;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface AdaptationAgentResult {
  insights: AdaptationInsightResponse[];
  preferences: UserPreferenceResponse[];
  adaptation_explanation: string;
  detected_patterns: string[];
  goal_unchanged: boolean;
  roadmap_unchanged: boolean;
  milestone_unchanged: boolean;
  is_early_signal: boolean;
  confidence_score: number;
  reflection_id: number | null;
  reused_existing: boolean;
  memory_ids: string[];
  memories_complete: boolean;
  memory_error: string | null;
  created_at: string;
}

export interface AdaptationRunRequest {
  reflection_id: number;
  force?: boolean;
}

export interface DailyPostSessionWorkflowResult {
  reflection: ReflectionResponse;
  adaptation: AdaptationAgentResult;
  adaptation_explanation: string;
  reflection_result: ReflectionAgentResult;
  completed_steps: string[];
  status: string;
  current_stage: string;
  goal_unchanged: boolean;
}

export interface DashboardResponse {
  user: UserResponse;
  active_goal: GoalResponse | null;
  current_roadmap: RoadmapResponse | null;
  current_milestone: MilestoneResponse | null;
  overall_progress_percent: number;
  today_mood: Mood | null;
  today_plan: DailyPlanResponse | null;
  completion_streak: number;
  recent_reflections: ReflectionResponse[];
  preferred_content_type: string | null;
  preferred_session_minutes: number | null;
  average_session_minutes: number | null;
  resource_effectiveness_avg: number | null;
  weekly_learning_consistency: number | null;
  skill_growth: string[];
  detected_patterns: string[];
  growthos_knows_you: string[];
  plan_change_explanation: string | null;
  ai_insight: string | null;
  recommended_next_action: string | null;
  adaptation_insights: AdaptationInsightResponse[];
  completed_sessions: number;
}

export interface DemoDayLoopRequest {
  day1_checkin?: DailyCheckInRequest | null;
  day2_checkin?: DailyCheckInRequest | null;
  reflection?: ReflectionRequest | null;
}

export interface DemoDayLoopResponse {
  user_id: number;
  goal_title: string;
  goal_unchanged: boolean;
  day1_checkin: DailyCheckInResponse;
  day1_plan: DailyPlanResponse;
  day1_tasks: DailyTaskResponse[];
  reflection: ReflectionResponse;
  reflection_insight: string | null;
  adaptation: AdaptationAgentResult;
  adaptation_explanation: string;
  detected_patterns: string[];
  is_early_signal: boolean;
  day2_checkin: DailyCheckInResponse;
  day2_plan: DailyPlanResponse;
  day2_tasks: DailyTaskResponse[];
  recommended_next_action: string | null;
  completed_steps: string[];
}

/** Lightweight local session summary for Day 1 vs Day 2 UI. */
export interface SessionDaySummary {
  plan_date: string;
  mood: Mood | string;
  energy?: EnergyLevel | string;
  available_minutes: number;
  task_count: number;
  practice_task_count?: number;
  completion_status?: CompletionStatus | string;
  plan_summary?: string;
  activity_mix?: string;
}
