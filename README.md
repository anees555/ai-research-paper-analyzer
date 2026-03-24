# AI Research Paper Analyzer

AI Research Paper Analyzer is a full-stack application for uploading academic PDFs, generating structured summaries, extracting figures, and running paper-scoped Q&A over indexed content.

## Current Implementation Summary

### Core capabilities

- PDF ingestion and parsing using GROBID and custom section extraction.
- Multi-mode analysis pipeline:
  - `fast`: quick summary path.
  - `enhanced`: structured, section-oriented analysis with figure extraction.
  - `interactive`: TOC/diagram-oriented output for interactive navigation.
- Background job processing with status polling.
- Authentication (register/login with JWT) and user history APIs.
- Chat system with retrieval-augmented generation (RAG):
  - Paper content chunking and indexing.
  - Vector search over indexed chunks.
  - Context-grounded answer generation with optional Groq LLM enhancement.
- Frontend analysis pages, interactive navigation, and integrated paper chat.
- Local reverse-proxy + public tunnel helpers for sharing one public URL.

### Technology stack

- Backend: FastAPI, SQLAlchemy, Pydantic, Uvicorn
- Frontend: Next.js (App Router), React, TypeScript, Tailwind CSS, Axios
- NLP/ML: Transformers, sentence-transformers
- Vector store: ChromaDB
- PDF/structure extraction: GROBID

## Architecture

### Backend

- API entry: `backend/app/main.py`
- Service layer: `backend/app/services`
- Routers: `backend/app/api/v1/routers`
- Data models/schemas: `backend/app/data_models`
- Default API base path: `/api/v1`

### Frontend

- App entry: `frontend/src/app`
- API client: `frontend/src/lib/api.ts`
- The frontend API client now supports:
  - direct backend URL when accessed from localhost
  - proxy-relative API calls (`/api/v1`) for tunneled/remote access

### Chat (RAG) flow

1. A paper is analyzed and represented as sections/summary content.
2. Content is chunked and indexed in Chroma.
3. On question requests, relevant chunks are retrieved by similarity.
4. Answer generation uses retrieved context, with source attribution and confidence.

## Repository Layout

```text
research_summary_project/
├── backend/
│   ├── app/
│   │   ├── api/v1/routers/
│   │   ├── core/
│   │   ├── data_models/
│   │   ├── interactive_navigation/
│   │   └── services/
│   ├── data/
│   │   ├── chroma_chat_db/
│   │   ├── figures/
│   │   └── uploads/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── contexts/
│   │   ├── lib/
│   │   └── types/
│   └── package.json
├── infra/
│   └── local-share/
│       ├── Caddyfile
│       ├── start-proxy.ps1
│       ├── start-cloudflare.ps1
│       ├── start-ngrok.ps1
│       └── README.md
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for GROBID)
- PostgreSQL-compatible database reachable by `DATABASE_URL`
- Windows PowerShell (for helper scripts in `infra/local-share`)

## Environment Configuration

Create `.env` from `.env.example` in project root and update values:

- `DATABASE_URL`
- `GROQ_API_KEY` (optional but recommended for better chat responses)
- `SECRET_KEY`
- `GROBID_URL` (default: `http://localhost:8070`)

Example:

```env
DATABASE_URL=postgresql://username:password@host:5432/database
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=replace_with_secure_random_value
ACCESS_TOKEN_EXPIRE_MINUTES=30
GROBID_URL=http://localhost:8070
```

## Local Setup

### 1) Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### 2) Frontend

```powershell
Set-Location frontend
npm install
Set-Location ..
```

### 3) Start GROBID

```powershell
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
```

If container already exists:

```powershell
docker start grobid
```

## Running the Application

### Terminal 1: Backend API (port 8003)

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location backend
python main.py
```

### Terminal 2: Frontend (port 3000)

```powershell
Set-Location frontend
npm run dev
```

### Local URLs

- Frontend: `http://localhost:3000`
- Backend API docs: `http://localhost:8003/docs`
- Health: `http://localhost:8003/api/v1/health`

## Public Sharing (One URL for Frontend + Backend)

This project includes isolated helper scripts in `infra/local-share`.

### 1) Start local reverse proxy

```powershell
Set-Location infra/local-share
./start-proxy.ps1
```

Proxy URL:

- `http://localhost:8080`

### 2) Expose proxy publicly

Cloudflare quick tunnel:

```powershell
Set-Location infra/local-share
./start-cloudflare.ps1
```

or ngrok:

```powershell
Set-Location infra/local-share
./start-ngrok.ps1
```

Notes:

- Keep backend, frontend, proxy, and tunnel processes running.
- Quick tunnel URLs are temporary and change when restarted.

## API Endpoints (Current)

### Health

- `GET /api/v1/health`

### Authentication

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login/access-token`

### User

- `GET /api/v1/users/me`
- `GET /api/v1/users/me/history`

### Analysis

- `POST /api/v1/analysis/upload?mode=fast|enhanced|interactive`
- `GET /api/v1/analysis/status/{job_id}`
- `POST /api/v1/analysis/analyze-instant?mode=fast|enhanced`
- `GET /api/v1/analysis/modes`
- `POST /api/v1/analysis/enhanced`
- `POST /api/v1/analysis/professional`

### Chat

- `POST /api/v1/chat/ask`
- `POST /api/v1/chat/index/{job_id}`
- `GET /api/v1/chat/status/{job_id}`

## Processing Modes

- `fast`: short turnaround summary flow.
- `enhanced`: deeper section-level analysis with figure extraction and richer output.
- `interactive`: output designed for interactive TOC/diagram visualization.

## Data Storage

- Uploaded files: `backend/data/uploads`
- Extracted figures: `backend/data/figures`
- Chat vector index: `backend/data/chroma_chat_db`
- Processed output artifacts: `backend/output`

## Operational Notes

- Chat quality depends on successful indexing after analysis completion.
- If `GROQ_API_KEY` is not set, chat falls back to a simpler context-based response path.
- Some legacy root-level batch scripts reference older ports; the current implementation uses backend port `8003` and frontend port `3000`.

## Troubleshooting

### Registration or login fails on mobile via tunnel

- Ensure frontend is using proxy-relative API path for remote hosts.
- Keep proxy and tunnel running.
- Verify health through the same public domain:
  - `https://<your-public-domain>/api/v1/health`

### Chat returns low-context answers

- Confirm paper indexing completed:
  - `GET /api/v1/chat/status/{job_id}`
- If needed, trigger indexing manually:
  - `POST /api/v1/chat/index/{job_id}`

### GROBID unavailable

- Check container status:

```powershell
docker ps
```

- Verify endpoint:

```powershell
curl http://localhost:8070/api/isalive
```

## License

MIT License.
