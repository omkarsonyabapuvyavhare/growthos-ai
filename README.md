# GrowthOS AI

Agentic AI Growth Curator — learn anything you want, in a way that fits who you are and how you feel today.

GrowthOS optimizes for **growth, not attention**. Mood changes today’s learning method, duration, and task mix — **not** your long-term goal. Progress is based on completion, usefulness, focus, and difficulty fit.

## Approved Technology Stack

### Frontend
- Next.js (App Router), React, TypeScript, Tailwind CSS, shadcn/ui

### Backend
- Python 3.11, FastAPI, Pydantic, SQLite

### AI and orchestration
- Google Gemini, LangChain, LangGraph, Gemini embeddings, FAISS

## Prerequisites

- Python **3.11**
- Node.js 18+
- A valid **Google Gemini** API key for live AI (optional for health + unit tests)

## Environment Variables

### Backend (`backend/.env`)

```bash
cd backend
cp .env.example .env
```

| Variable | Description | Local demo example |
|----------|-------------|--------------------|
| `AI_PROVIDER` | Mandatory AI provider | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | *(required for live AI)* |
| `YOUTUBE_API_ENABLED` | Optional YouTube discovery for Curator | `false` (set `true` to enable) |
| `YOUTUBE_API_KEY` | YouTube Data API key (backend only) | *(optional)* |
| `YOUTUBE_MAX_RESULTS` | Max YouTube search results | `10` |
| `YOUTUBE_REQUEST_TIMEOUT_SECONDS` | YouTube HTTP timeout | `10` |
| `DATABASE_URL` | SQLite URL | `sqlite:///./growthos.db` |
| `FRONTEND_ORIGIN` | CORS origin(s), comma-separated | `http://localhost:3002` |
| `FAISS_INDEX_PATH` | FAISS index path | `./data/faiss_index/index.faiss` |

### Optional YouTube discovery

YouTube Data API is **optional** and used only by the Curator Agent to discover real public video candidates. It does **not** replace Gemini, the validated free-resource catalog, or FAISS ranking.

1. Create a YouTube Data API key in Google Cloud.
2. Put it only in `backend/.env` as `YOUTUBE_API_KEY=...` and set `YOUTUBE_API_ENABLED=true`.
3. Restart the backend.

When YouTube is disabled, missing a key, rate-limited, or otherwise unavailable, Curator continues with the local validated catalog. Never put `YOUTUBE_API_KEY` in frontend env files or any `NEXT_PUBLIC_*` variable. YouTube free tier quotas are limited — treat discovery as best-effort.

### Frontend (`frontend/.env.local`)

```bash
cd frontend
cp .env.local.example .env.local
```

| Variable | Description | Local demo example |
|----------|-------------|--------------------|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI base URL | `http://localhost:8080` |

Do not put Gemini or YouTube keys in the frontend.

## Local ports (judge / MVP demo)

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3002 |
| Backend | http://localhost:8080 |
| Swagger | http://localhost:8080/docs |
| Health | http://localhost:8080/health |

To change ports:

1. Start uvicorn with `--port <port>`.
2. Set `NEXT_PUBLIC_API_BASE_URL` to match.
3. Set `FRONTEND_ORIGIN` to the Next.js origin (comma-separate multiple).
4. Restart both apps.

**Note:** Another project may already use port `8000`. Prefer `8080` for GrowthOS unless you intentionally free `8000`.

## Backend Setup

```bash
cd growthos-ai/backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
cp .env.example .env
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

## Frontend Setup

```bash
cd growthos-ai/frontend
cp .env.local.example .env.local
# Ensure NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
npm install
npm run dev -- --port 3002
```

## Judge Demo

### Prerequisites

- Backend on **8080**, frontend on **3002**
- `frontend/.env.local` → `NEXT_PUBLIC_API_BASE_URL=http://localhost:8080`
- Backend CORS includes `http://localhost:3002`
- Valid Gemini key for live generation (unit tests use fakes and do not need one)

### Commands

```bash
# Terminal 1 — API
cd growthos-ai/backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8080

# Terminal 2 — UI
cd growthos-ai/frontend
npm run dev -- --port 3002
```

### Browser URLs

- App: http://localhost:3002
- API docs: http://localhost:8080/docs
- Health: http://localhost:8080/health

### Normal product journey

1. Open http://localhost:3002
2. **Start My Growth Journey**
3. Enter any free-text learning goal and complete onboarding
4. Review the roadmap (goal text should match exactly)
5. Check in (try **tired** / low energy / 15 minutes)
6. Open **Today’s plan** (1–5 focused tasks, each with why-selected)
7. Mark tasks complete / partial / skipped
8. Reflect and review “What GrowthOS learned”
9. Open **Dashboard** for progress and explanations
10. Next day (or second check-in): use **focused** / 30 minutes and confirm the plan shape changed while the goal did not

### Fast demo journey

1. Complete onboarding once (any free-text goal)
2. Open **Dashboard**
3. Click **Run Day 1 to Day 2 Demo**
4. Read the side-by-side comparison (labeled demo data)

### Day 1 → Day 2 story

**Day 1:** tired · low energy · low focus · 15 minutes · short low-pressure plan · partial completion · longer resource less useful · practice useful  

**Adaptation:** early signal · shorter resources · more practice · **goal unchanged**  

**Day 2:** focused · higher energy · 30 minutes · changed task mix · more practice · adjusted duration/difficulty · explanation grounded in Day 1 evidence  

### Expected adaptation result

- Early signal flagged
- Patterns such as focus dropping on longer resources / practice being more useful
- Dashboard shows “What GrowthOS learned” and “Why your plan changed”
- Original goal text preserved exactly

### Troubleshooting

**Invalid Gemini key**

- Keys must be Google Gemini credentials (typically `AIza…`), not Groq (`gsk_…`) or other providers.
- Symptoms: friendly UI errors mentioning Gemini unavailable / generation failed; HTTP 502 from planning endpoints.
- Unit tests still pass with fake Gemini services.
- Fix: set a valid `GEMINI_API_KEY` in `backend/.env` and restart uvicorn.

**Occupied ports**

- If `3002` or `8080` is busy, pick free ports and update `NEXT_PUBLIC_API_BASE_URL` + `FRONTEND_ORIGIN`.
- Do not stop unrelated apps (for example MentorMind on `8000`) unless you intend to.

**Missing user / empty dashboard**

- Local storage holds the MVP user id. Start onboarding again if state was cleared.

## Testing

```bash
# Backend
cd growthos-ai/backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Frontend
cd growthos-ai/frontend
npx tsc --noEmit
npm run lint
npm run build
```

## Project Structure

```text
growthos-ai/
├── backend/
│   ├── agents/
│   ├── api/
│   ├── services/
│   ├── workflows/
│   ├── tests/
│   ├── scripts/
│   └── main.py
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── types/
└── README.md
```

## Product principles

- Growth over attention — no infinite feeds, no autoplay rabbit holes
- Mood shapes **today’s** plan, never the long-term goal
- Every resource/task explains why it was selected
- Progress comes from completion and reflection evidence
