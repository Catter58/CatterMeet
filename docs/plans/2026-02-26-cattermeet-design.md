# CatterMeet — Design Document

**Date:** 2026-02-26
**Status:** Approved

## Overview

Web service for transcription and diarization of long audio/video files (up to 4 hours), focused on Russian and English. Runs in a single Docker container with ≤3–4 GB RAM.

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, asyncio.Queue
- **Database:** SQLite3 + FTS5
- **ML:** faster-whisper (large-v3-turbo, int8), silero-vad (ONNX), speechbrain ECAPA-TDNN, scikit-learn KMeans
- **Frontend:** React (Vite), Tailwind CSS, shadcn/ui
- **System:** ffmpeg, Docker (multi-stage build)
- **Deploy:** docker-compose.yml → `docker compose up`

## Project Structure

```
CatterMeet/
├── backend/
│   ├── main.py           # FastAPI app + static mount
│   ├── database.py       # SQLite + FTS5
│   ├── worker.py         # asyncio.Queue + background worker
│   ├── processor.py      # AudioProcessor (VAD → STT → Diarization)
│   ├── routers/
│   │   └── api.py        # API endpoints
│   ├── static/           # Built frontend (copied from stage 1)
│   ├── uploads/          # Incoming files (volume-mounted)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       ├── DropZone.tsx
│   │       ├── TasksTable.tsx
│   │       └── ResultPage.tsx
│   └── package.json
├── Dockerfile             # Multi-stage: node → python:3.11-slim
├── docker-compose.yml
└── README.md
```

## ML Pipeline (strict sequential, 1 model in RAM at a time)

1. **ffmpeg** → convert to 16kHz mono WAV
2. **silero-vad (ONNX)** → find speech segments → `del` + `gc.collect()`
3. **faster-whisper large-v3-turbo int8** → transcribe by VAD timestamps → `del` + `gc.collect()`
4. **speechbrain ECAPA-TDNN** → extract speaker embeddings → `del` + `gc.collect()`
5. **sklearn KMeans** → assign speakers → save to SQLite → delete temp files

## Database Schema

```sql
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    speaker TEXT,
    text TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE VIRTUAL TABLE transcripts_fts USING fts5(
    text, task_id UNINDEXED
);
```

## API Endpoints

- `POST /api/upload` → `{task_id}`
- `GET /api/status/{task_id}` → `{status, filename, created_at}`
- `GET /api/transcript/{task_id}` → `[{start_time, end_time, speaker, text}]`
- `GET /api/search?q=&task_id=` → FTS5 results

## Frontend Screens

1. **DropZone** — Drag & Drop file upload
2. **TasksTable** — Polling every 5s, task status display
3. **ResultPage** — Audio player + speaker-divided transcript + search with highlight + click-to-seek

## Deployment

```bash
docker compose up --build
# Service available at http://localhost:8000
```
