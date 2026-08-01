/**
 * Typed GrowthOS AI API client.
 * Base URL comes from NEXT_PUBLIC_API_BASE_URL — never hardcode in pages.
 *
 * Local:  NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
 * Vercel: NEXT_PUBLIC_API_BASE_URL=/api/backend
 */

import type {
  AdaptationAgentResult,
  AdaptationRunRequest,
  DailyCheckInRequest,
  DailyCheckInResponse,
  DailyPlanCreateRequest,
  DailyPlanResponse,
  DailyPlanningWorkflowResult,
  DailyPostSessionWorkflowResult,
  DailyTaskResponse,
  DashboardResponse,
  DemoDayLoopRequest,
  DemoDayLoopResponse,
  HealthResponse,
  OnboardingRequest,
  OnboardingWorkflowResult,
  ReflectionRequest,
  RoadmapAgentResult,
  RoadmapCreateRequest,
  RoadmapResponse,
  TaskCompletionRequest,
} from "@/types";

export class ApiClientError extends Error {
  readonly status: number;
  readonly stage?: string | null;
  readonly errorType?: string | null;

  constructor(
    message: string,
    status: number,
    options?: { stage?: string | null; errorType?: string | null },
  ) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.stage = options?.stage;
    this.errorType = options?.errorType;
  }
}

export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!raw) {
    throw new ApiClientError(
      "API base URL is not configured. Set NEXT_PUBLIC_API_BASE_URL in .env.local " +
        "(local: http://localhost:8080) or Vercel (production: /api/backend).",
      0,
    );
  }
  // Absolute local URL or same-origin relative production path.
  return raw.replace(/\/$/, "");
}

function friendlyMessage(status: number, detail: string): string {
  const lower = detail.toLowerCase();
  if (status === 0 || lower.includes("failed to fetch") || lower.includes("network")) {
    return "GrowthOS can’t reach the server. Check that the backend is running.";
  }
  if (status === 404) {
    if (lower.includes("roadmap")) return "No active roadmap yet. Finish onboarding first.";
    if (lower.includes("plan")) return "No plan for today yet. Start with a check-in.";
    if (lower.includes("user")) return "We couldn’t find your growth profile. Start a new journey.";
    return detail || "We couldn’t find what you asked for.";
  }
  if (status === 403) {
    return "That action doesn’t belong to your growth profile.";
  }
  if (status === 409) {
    return detail || "This session was already saved.";
  }
  if (status === 422) {
    return detail || "Please check your answers and try again.";
  }
  if (status === 502 || lower.includes("gemini") || lower.includes("configuration")) {
    return "GrowthOS couldn’t generate a response right now. Gemini may be unavailable or not configured.";
  }
  if (status >= 500) {
    return detail || "Something went wrong on the server. Please try again.";
  }
  return detail || `Request failed (${status}).`;
}

async function parseError(response: Response): Promise<ApiClientError> {
  let detail = `Request failed with status ${response.status}`;
  let stage: string | null | undefined;
  let errorType: string | null | undefined;
  try {
    const body = (await response.json()) as {
      detail?: unknown;
      stage?: string | null;
      error_type?: string | null;
    };
    if (typeof body.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body.detail)) {
      detail = body.detail
        .map((item: unknown) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return String(item);
        })
        .join("; ");
    }
    stage = body.stage;
    errorType = body.error_type;
  } catch {
    // keep default detail
  }
  return new ApiClientError(friendlyMessage(response.status, detail), response.status, {
    stage,
    errorType,
  });
}

export type RequestOptions = {
  signal?: AbortSignal;
  method?: string;
  body?: unknown;
};

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  let baseUrl: string;
  try {
    baseUrl = getApiBaseUrl();
  } catch (error) {
    if (error instanceof ApiClientError) throw error;
    throw new ApiClientError("API base URL is not configured.", 0);
  }

  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: options.method ?? (options.body !== undefined ? "POST" : "GET"),
      headers: {
        Accept: "application/json",
        ...(options.body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      cache: "no-store",
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiClientError(
      friendlyMessage(0, error instanceof Error ? error.message : "network"),
      0,
    );
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health", { signal });
}

export function onboardUser(
  payload: OnboardingRequest,
  signal?: AbortSignal,
): Promise<OnboardingWorkflowResult> {
  return apiRequest<OnboardingWorkflowResult>("/onboarding", {
    method: "POST",
    body: payload,
    signal,
  });
}

export function createRoadmap(
  userId: number,
  payload: RoadmapCreateRequest = {},
  signal?: AbortSignal,
): Promise<RoadmapAgentResult> {
  return apiRequest<RoadmapAgentResult>(`/users/${userId}/roadmaps`, {
    method: "POST",
    body: payload,
    signal,
  });
}

export function getActiveRoadmap(
  userId: number,
  signal?: AbortSignal,
): Promise<RoadmapResponse> {
  return apiRequest<RoadmapResponse>(`/users/${userId}/roadmap`, { signal });
}

export function createCheckIn(
  userId: number,
  payload: DailyCheckInRequest,
  signal?: AbortSignal,
): Promise<DailyCheckInResponse> {
  return apiRequest<DailyCheckInResponse>(`/users/${userId}/checkins`, {
    method: "POST",
    body: payload,
    signal,
  });
}

export function createDailyPlan(
  userId: number,
  payload: DailyPlanCreateRequest,
  signal?: AbortSignal,
): Promise<DailyPlanningWorkflowResult> {
  return apiRequest<DailyPlanningWorkflowResult>(`/users/${userId}/daily-plans`, {
    method: "POST",
    body: payload,
    signal,
  });
}

export function getTodayPlan(
  userId: number,
  signal?: AbortSignal,
): Promise<DailyPlanResponse> {
  return apiRequest<DailyPlanResponse>(`/users/${userId}/daily-plans/today`, {
    signal,
  });
}

export function updateTask(
  userId: number,
  taskId: number,
  payload: TaskCompletionRequest,
  signal?: AbortSignal,
): Promise<DailyTaskResponse> {
  return apiRequest<DailyTaskResponse>(`/users/${userId}/tasks/${taskId}`, {
    method: "PATCH",
    body: payload,
    signal,
  });
}

export function submitReflection(
  userId: number,
  payload: ReflectionRequest,
  signal?: AbortSignal,
): Promise<DailyPostSessionWorkflowResult> {
  return apiRequest<DailyPostSessionWorkflowResult>(`/users/${userId}/reflections`, {
    method: "POST",
    body: payload,
    signal,
  });
}

export function runAdaptation(
  userId: number,
  payload: AdaptationRunRequest,
  signal?: AbortSignal,
): Promise<AdaptationAgentResult> {
  return apiRequest<AdaptationAgentResult>(`/users/${userId}/adaptations/run`, {
    method: "POST",
    body: payload,
    signal,
  });
}

export function getDashboard(
  userId: number,
  signal?: AbortSignal,
): Promise<DashboardResponse> {
  return apiRequest<DashboardResponse>(`/users/${userId}/dashboard`, { signal });
}

export function runDemoDayLoop(
  userId: number,
  payload: DemoDayLoopRequest = {},
  signal?: AbortSignal,
): Promise<DemoDayLoopResponse> {
  return apiRequest<DemoDayLoopResponse>(`/users/${userId}/demo/day-loop`, {
    method: "POST",
    body: payload,
    signal,
  });
}
