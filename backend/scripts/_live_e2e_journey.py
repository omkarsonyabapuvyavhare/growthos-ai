"""One-shot live E2E journey against the running API. Not part of unittest suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

BASE = "http://127.0.0.1:8080"
GOAL = "Learn practical SQL for product analytics"


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=180.0)
    health = client.get("/health")
    health.raise_for_status()
    print("HEALTH", health.json())

    onboard = client.post(
        "/onboarding",
        json={
            "display_name": "Live E2E Judge",
            "learning_goal": GOAL,
            "aspiration": "Analyze product funnels confidently",
            "motivation": "Make better product decisions with data",
            "current_level": "beginner",
            "target_outcome": "Write useful SELECT queries for metrics",
            "preferred_formats": ["video", "practice"],
            "learning_style": "mixed",
            "daily_available_minutes": 40,
            "preferred_session_minutes": 20,
            "attention_span_minutes": 12,
            "preferred_learning_time": "evening",
            "habits": ["evening review"],
            "distractions": ["phone"],
        },
    )
    print("ONBOARDING", onboard.status_code)
    if onboard.status_code >= 400:
        print(onboard.text[:1200])
        return 1
    body = onboard.json()
    user_id = body["user"]["id"]
    assert body["goal"]["title"] == GOAL, body["goal"]["title"]
    print("GOAL", body["goal"]["title"])
    print("ROADMAP", body["roadmap"]["title"], "progress", body["roadmap"]["progress_percent"])

    roadmap = client.get(f"/users/{user_id}/roadmap")
    roadmap.raise_for_status()
    print("ROADMAP_GET phases", len(roadmap.json().get("phases") or []))

    plan = client.post(
        f"/users/{user_id}/daily-plans",
        json={
            "mood": "tired",
            "energy_level": "low",
            "focus_level": 2,
            "available_minutes": 15,
            "preferred_activity": "watch",
            "notes": "Live e2e tired check-in",
            "refresh": True,
        },
    )
    print("PLAN", plan.status_code)
    if plan.status_code >= 400:
        print(plan.text[:1200])
        return 1
    plan_body = plan.json()
    tasks = plan_body["tasks"]
    print(
        "PLAN_SUMMARY",
        plan_body["plan"]["summary"],
        "tasks",
        len(tasks),
        "minutes",
        plan_body["plan"]["total_estimated_minutes"],
    )
    assert 1 <= len(tasks) <= 5
    assert plan_body["plan"]["total_estimated_minutes"] <= 15
    for task in tasks:
        if task.get("resource_url"):
            print("RESOURCE", task.get("resource_title"), task.get("resource_url"))

    today = client.get(f"/users/{user_id}/daily-plans/today")
    print("TODAY", today.status_code, "tasks", len(today.json().get("tasks") or []))

    if tasks:
        patched = client.patch(
            f"/users/{user_id}/tasks/{tasks[0]['id']}",
            json={
                "status": "completed",
                "completion_percent": 70,
                "duration_minutes": min(12, tasks[0]["estimated_minutes"]),
                "effectiveness_rating": 2,
            },
        )
        print("TASK_UPDATE", patched.status_code, patched.json().get("status"))

    updates = []
    for index, task in enumerate(tasks):
        if index == 0:
            updates.append(
                {
                    "task_id": task["id"],
                    "update": {
                        "status": "completed",
                        "completion_percent": 70,
                        "duration_minutes": min(12, task["estimated_minutes"]),
                        "effectiveness_rating": 2,
                    },
                }
            )
        elif task["activity_type"] == "practice":
            updates.append(
                {
                    "task_id": task["id"],
                    "update": {
                        "status": "completed",
                        "completion_percent": 100,
                        "duration_minutes": task["estimated_minutes"],
                        "effectiveness_rating": 5,
                    },
                }
            )

    reflection = client.post(
        f"/users/{user_id}/reflections",
        json={
            "daily_plan_id": plan_body["plan"]["id"],
            "completion_status": "partial",
            "learning_summary": "Practice helped; longer resource drained focus.",
            "focus_rating": 2,
            "resource_effectiveness": 2,
            "difficulty_feedback": "suitable",
            "mood_match": False,
            "distractions": ["phone"],
            "wants_similar_resources": False,
            "mood_after": "tired",
            "task_updates": updates,
            "actual_minutes_spent": 14,
        },
    )
    print("REFLECTION", reflection.status_code)
    if reflection.status_code >= 400:
        print(reflection.text[:1200])
        return 1
    ref = reflection.json()
    print("INSIGHT", (ref.get("reflection") or {}).get("insight"))
    print("ADAPT_EXPLAIN", ref.get("adaptation_explanation"))
    print("GOAL_UNCHANGED", ref.get("goal_unchanged"))
    print("PATTERNS", (ref.get("adaptation") or {}).get("detected_patterns"))

    dashboard = client.get(f"/users/{user_id}/dashboard")
    print("DASHBOARD", dashboard.status_code)
    if dashboard.status_code >= 400:
        print(dashboard.text[:1200])
        return 1
    dash = dashboard.json()
    print("DASH_GOAL", (dash.get("active_goal") or {}).get("title"))
    print("KNOWS_YOU", dash.get("growthos_knows_you"))
    print("DETECTED", dash.get("detected_patterns"))
    print("WHY_CHANGED", dash.get("plan_change_explanation"))
    assert (dash.get("active_goal") or {}).get("title") == GOAL

    demo = client.post(f"/users/{user_id}/demo/day-loop", json={})
    print("DEMO", demo.status_code)
    if demo.status_code >= 400:
        print(demo.text[:1200])
        return 1
    demo_body = demo.json()
    print(
        json.dumps(
            {
                "goal_title": demo_body.get("goal_title"),
                "goal_unchanged": demo_body.get("goal_unchanged"),
                "day1_mood": (demo_body.get("day1_checkin") or {}).get("mood"),
                "day1_minutes": (demo_body.get("day1_checkin") or {}).get("available_minutes"),
                "day1_tasks": len(demo_body.get("day1_tasks") or []),
                "day2_mood": (demo_body.get("day2_checkin") or {}).get("mood"),
                "day2_minutes": (demo_body.get("day2_checkin") or {}).get("available_minutes"),
                "day2_tasks": len(demo_body.get("day2_tasks") or []),
                "is_early_signal": demo_body.get("is_early_signal"),
                "adaptation_explanation": demo_body.get("adaptation_explanation"),
                "detected_patterns": demo_body.get("detected_patterns"),
            },
            indent=2,
        )
    )
    assert demo_body.get("goal_title") == GOAL
    assert demo_body.get("goal_unchanged") is True
    print("LIVE_E2E_OK user_id=", user_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
