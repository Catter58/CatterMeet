# CatterMeet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-container web service for audio/video transcription and speaker diarization (Russian + English).

**Architecture:** FastAPI backend serves a built React frontend from `/static`. An `asyncio.Queue` serializes processing so only one file is in RAM at a time. ML pipeline: ffmpeg → silero-vad → faster-whisper → ECAPA-TDNN → KMeans, with strict model eviction between stages.

**Tech Stack:** Python 3.11, FastAPI, SQLite+FTS5, faster-whisper, silero-vad, speechbrain, scikit-learn; React+Vite+Tailwind+shadcn/ui; Docker multi-stage build.

---

## Task 1: Monorepo Skeleton

**Files:**
- Create: `backend/` (directory)
- Create: `frontend/` (directory)
- Create: `backend/static/.gitkeep`
- Create: `backend/uploads/.gitkeep`
- Create: `.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p backend/routers backend/static backend/uploads frontend
touch backend/static/.gitkeep backend/uploads/.gitkeep
```

**Step 2: Create .gitignore**

```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
*.db

# Node
node_modules/
dist/

# App
backend/uploads/*
!backend/uploads/.gitkeep
backend/static/*
!backend/static/.gitkeep
```

**Step 3: Commit**

```bash
git init
git add .
git commit -m "chore: initialize monorepo structure"
```

---

## Task 2: Backend requirements.txt

**Files:**
- Create: `backend/requirements.txt`

**Step 1: Write requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.20
aiofiles==24.1.0
faster-whisper==1.1.0
silero-vad==5.1.2
speechbrain==1.0.2
scikit-learn==1.6.1
numpy==1.26.4
torch==2.5.1
torchaudio==2.5.1
onnxruntime==1.20.1
```

> Note: torch CPU-only build is pulled; no CUDA required. speechbrain pulls huggingface-hub automatically.

**Step 2: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add Python requirements"
```

---

## Task 3: Dockerfile (multi-stage)

**Files:**
- Create: `Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (layer cache)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy built frontend into static folder
COPY --from=frontend-builder /app/frontend/dist ./static

# Create upload dir (volume mount will overlay this)
RUN mkdir -p uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "chore: add multi-stage Dockerfile"
```

---

## Task 4: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

**Step 1: Write docker-compose.yml**

```yaml
services:
  cattermeet:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/db:/app/db
    environment:
      - DB_PATH=/app/db/cattermeet.db
      - UPLOAD_DIR=/app/uploads
    restart: unless-stopped
```

> `data/` directory is created automatically by Docker on first run.

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add docker-compose.yml"
```

---

## Task 5: Database module

**Files:**
- Create: `backend/database.py`
- Create: `backend/tests/test_database.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_database.py
import asyncio, os, tempfile, pytest
os.environ["DB_PATH"] = ":memory:"

from database import init_db, create_task, get_task, update_task_status, save_transcript, search_transcripts

@pytest.fixture(autouse=True)
def setup():
    asyncio.run(init_db())

def test_create_and_get_task():
    task_id = asyncio.run(create_task("audio.mp3"))
    task = asyncio.run(get_task(task_id))
    assert task["filename"] == "audio.mp3"
    assert task["status"] == "pending"

def test_update_status():
    task_id = asyncio.run(create_task("f.mp3"))
    asyncio.run(update_task_status(task_id, "processing"))
    task = asyncio.run(get_task(task_id))
    assert task["status"] == "processing"

def test_save_and_search_transcript():
    task_id = asyncio.run(create_task("f.mp3"))
    segments = [{"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_0", "text": "Привет мир"}]
    asyncio.run(save_transcript(task_id, segments))
    results = asyncio.run(search_transcripts("Привет", task_id))
    assert len(results) == 1
    assert "Привет" in results[0]["text"]
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_database.py -v
```
Expected: ImportError — `database` module not found.

**Step 3: Write database.py**

```python
# backend/database.py
import os, sqlite3, uuid, asyncio
from typing import Optional
from contextlib import asynccontextmanager

DB_PATH = os.environ.get("DB_PATH", "cattermeet.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


async def init_db() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_db_sync)


def _init_db_sync() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                speaker TEXT,
                text TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                text,
                task_id UNINDEXED
            );
        """)


async def create_task(filename: str) -> str:
    task_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _create_task_sync, task_id, filename)
    return task_id


def _create_task_sync(task_id: str, filename: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (id, filename) VALUES (?, ?)",
            (task_id, filename)
        )


async def get_task(task_id: str) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_task_sync, task_id)


def _get_task_sync(task_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None


async def update_task_status(task_id: str, status: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _update_status_sync, task_id, status)


def _update_status_sync(task_id: str, status: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, task_id)
        )


async def save_transcript(task_id: str, segments: list[dict]) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_transcript_sync, task_id, segments)


def _save_transcript_sync(task_id: str, segments: list[dict]) -> None:
    with _get_conn() as conn:
        for seg in segments:
            conn.execute(
                "INSERT INTO transcripts (task_id, start_time, end_time, speaker, text) VALUES (?,?,?,?,?)",
                (task_id, seg["start_time"], seg["end_time"], seg["speaker"], seg["text"])
            )
            conn.execute(
                "INSERT INTO transcripts_fts (text, task_id) VALUES (?, ?)",
                (seg["text"], task_id)
            )


async def get_transcript(task_id: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_transcript_sync, task_id)


def _get_transcript_sync(task_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT start_time, end_time, speaker, text FROM transcripts WHERE task_id = ? ORDER BY start_time",
            (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]


async def search_transcripts(query: str, task_id: Optional[str] = None) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_sync, query, task_id)


def _search_sync(query: str, task_id: Optional[str]) -> list[dict]:
    with _get_conn() as conn:
        if task_id:
            rows = conn.execute(
                """SELECT t.task_id, t.start_time, t.end_time, t.speaker, fts.text
                   FROM transcripts_fts fts
                   JOIN transcripts t ON t.text = fts.text AND t.task_id = fts.task_id
                   WHERE transcripts_fts MATCH ? AND fts.task_id = ?
                   ORDER BY rank""",
                (query, task_id)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT t.task_id, t.start_time, t.end_time, t.speaker, fts.text
                   FROM transcripts_fts fts
                   JOIN transcripts t ON t.text = fts.text AND t.task_id = fts.task_id
                   WHERE transcripts_fts MATCH ?
                   ORDER BY rank""",
                (query,)
            ).fetchall()
        return [dict(r) for r in rows]
```

**Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_database.py -v
```
Expected: 3 PASS.

**Step 5: Commit**

```bash
git add backend/database.py backend/tests/
git commit -m "feat: add SQLite + FTS5 database module with tests"
```

---

## Task 6: AudioProcessor (ML Pipeline)

**Files:**
- Create: `backend/processor.py`

> This module is hard to unit-test without real audio files. Write it directly; integration is tested in Task 9.

**Step 1: Write processor.py**

```python
# backend/processor.py
import gc
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any audio/video to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def _run_vad(wav_path: str) -> list[dict]:
    """Run silero-vad and return speech segments. Model evicted after use."""
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    model = load_silero_vad()
    audio = read_audio(wav_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)

    del model
    del audio
    gc.collect()

    return [{"start": ts["start"], "end": ts["end"]} for ts in timestamps]


def _run_stt(wav_path: str, vad_segments: list[dict]) -> list[dict]:
    """Transcribe using faster-whisper, only over VAD segments. Model evicted after use."""
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

    results = []
    for seg in vad_segments:
        segments_iter, _ = model.transcribe(
            wav_path,
            language=None,  # auto-detect (ru/en)
            beam_size=5,
            clip_timestamps=[seg["start"], seg["end"]],
            word_timestamps=False,
        )
        for s in segments_iter:
            results.append({
                "start_time": float(s.start),
                "end_time": float(s.end),
                "text": s.text.strip(),
            })

    del model
    gc.collect()

    return results


def _run_diarization(wav_path: str, segments: list[dict]) -> list[dict]:
    """Extract speaker embeddings with ECAPA-TDNN, cluster with KMeans. Model evicted after use."""
    import torch
    import torchaudio
    from speechbrain.pretrained import SpeakerRecognition

    spk_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    embeddings = []
    valid_indices = []
    for i, seg in enumerate(segments):
        start_sample = int(seg["start_time"] * 16000)
        end_sample = int(seg["end_time"] * 16000)
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] < 1600:  # skip segments < 0.1s
            continue
        with torch.no_grad():
            emb = spk_model.encode_batch(chunk)
        embeddings.append(emb.squeeze().numpy())
        valid_indices.append(i)

    del spk_model, waveform
    gc.collect()

    if not embeddings:
        for seg in segments:
            seg["speaker"] = "SPEAKER_0"
        return segments

    from sklearn.cluster import KMeans
    X = np.array(embeddings)
    n_speakers = min(2, len(embeddings))
    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Assign speakers to valid segments; others default to SPEAKER_0
    labeled_segments = []
    label_ptr = 0
    for i, seg in enumerate(segments):
        if label_ptr < len(valid_indices) and valid_indices[label_ptr] == i:
            seg["speaker"] = f"SPEAKER_{labels[label_ptr]}"
            label_ptr += 1
        else:
            seg["speaker"] = "SPEAKER_0"
        labeled_segments.append(seg)

    return labeled_segments


def process_file(file_path: str) -> list[dict]:
    """
    Full pipeline: convert → VAD → STT → Diarization.
    Returns list of {start_time, end_time, speaker, text}.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        logger.info(f"Converting {file_path} to WAV...")
        _convert_to_wav(file_path, wav_path)

        logger.info("Running VAD...")
        vad_segments = _run_vad(wav_path)
        logger.info(f"VAD found {len(vad_segments)} speech segments")

        if not vad_segments:
            return []

        logger.info("Running STT...")
        transcript_segments = _run_stt(wav_path, vad_segments)
        logger.info(f"STT produced {len(transcript_segments)} segments")

        if not transcript_segments:
            return []

        logger.info("Running diarization...")
        diarized = _run_diarization(wav_path, transcript_segments)
        logger.info("Diarization complete")

        return diarized

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
```

**Step 2: Commit**

```bash
git add backend/processor.py
git commit -m "feat: add AudioProcessor ML pipeline (VAD → STT → Diarization)"
```

---

## Task 7: Background Worker

**Files:**
- Create: `backend/worker.py`

**Step 1: Write worker.py**

```python
# backend/worker.py
import asyncio
import logging
import os
from pathlib import Path

from database import update_task_status, save_transcript, get_task
from processor import process_file

logger = logging.getLogger(__name__)

task_queue: asyncio.Queue = asyncio.Queue()
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


async def process_worker() -> None:
    """Reads task_ids from queue, processes one at a time."""
    while True:
        task_id = await task_queue.get()
        try:
            task = await get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found in DB")
                continue

            file_path = str(Path(UPLOAD_DIR) / task["filename"])
            logger.info(f"Processing task {task_id}: {file_path}")

            await update_task_status(task_id, "processing")

            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(None, process_file, file_path)

            await save_transcript(task_id, segments)
            await update_task_status(task_id, "completed")
            logger.info(f"Task {task_id} completed with {len(segments)} segments")

        except Exception as exc:
            logger.exception(f"Task {task_id} failed: {exc}")
            try:
                await update_task_status(task_id, "failed")
            except Exception:
                pass
        finally:
            task_queue.task_done()
```

**Step 2: Commit**

```bash
git add backend/worker.py
git commit -m "feat: add asyncio background worker"
```

---

## Task 8: FastAPI Application + API Router

**Files:**
- Create: `backend/main.py`
- Create: `backend/routers/api.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/tests/test_api.py`

**Step 1: Write failing API tests**

```python
# backend/tests/test_api.py
import os, asyncio
os.environ["DB_PATH"] = ":memory:"
os.environ["UPLOAD_DIR"] = "/tmp/test_uploads"
os.makedirs("/tmp/test_uploads", exist_ok=True)

import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from database import init_db

@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()

@pytest.mark.asyncio
async def test_upload_returns_task_id(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fakecontent")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(audio, "rb") as f:
            resp = await client.post("/api/upload", files={"file": ("test.mp3", f, "audio/mpeg")})
    assert resp.status_code == 200
    assert "task_id" in resp.json()

@pytest.mark.asyncio
async def test_status_unknown_task():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status/nonexistent-id")
    assert resp.status_code == 404

@pytest.mark.asyncio
async def test_transcript_unknown_task():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/transcript/nonexistent-id")
    assert resp.status_code == 404
```

**Step 2: Run tests, verify they fail**

```bash
cd backend && pip install pytest pytest-asyncio httpx
python -m pytest tests/test_api.py -v
```
Expected: ImportError or connection error.

**Step 3: Write routers/api.py**

```python
# backend/routers/api.py
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

import database
from worker import task_queue

router = APIRouter(prefix="/api")

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".bin"
    safe_name = f"{uuid.uuid4()}{ext}"
    dest = Path(UPLOAD_DIR) / safe_name

    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)

    task_id = await database.create_task(safe_name)
    await task_queue.put(task_id)

    return {"task_id": task_id}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task = await database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/transcript/{task_id}")
async def get_transcript(task_id: str):
    task = await database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    segments = await database.get_transcript(task_id)
    return {"task_id": task_id, "segments": segments}


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    task_id: str = Query(None),
):
    results = await database.search_transcripts(q, task_id)
    return {"results": results}
```

**Step 4: Write main.py**

```python
# backend/main.py
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import database
from routers.api import router as api_router
from worker import process_worker

logging.basicConfig(level=logging.INFO)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    worker_task = asyncio.create_task(process_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)

static_dir = Path(__file__).parent / "static"
if static_dir.exists() and any(static_dir.iterdir()):
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
```

**Step 5: Write routers/__init__.py**

```python
# backend/routers/__init__.py
```

**Step 6: Run tests**

```bash
cd backend && python -m pytest tests/test_api.py -v
```
Expected: 3 PASS.

**Step 7: Commit**

```bash
git add backend/main.py backend/routers/ backend/tests/test_api.py
git commit -m "feat: add FastAPI app, API router, and API tests"
```

---

## Task 9: Frontend — Vite + React + Tailwind + shadcn/ui

**Files:**
- Create: `frontend/` (via npm)

**Step 1: Initialize Vite + React + TypeScript**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
```

**Step 2: Install Tailwind CSS**

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Edit `tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

Edit `src/index.css` (replace all content):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 3: Install shadcn/ui**

```bash
npx shadcn@latest init
```
When prompted:
- Style: Default
- Base color: Neutral
- CSS variables: Yes

Then add components:

```bash
npx shadcn@latest add button input table progress scroll-area card badge
```

**Step 4: Configure Vite API proxy for development**

Edit `vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

**Step 5: Commit**

```bash
cd ..
git add frontend/
git commit -m "feat: initialize frontend with Vite + React + Tailwind + shadcn/ui"
```

---

## Task 10: Frontend Components

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/components/DropZone.tsx`
- Create: `frontend/src/components/TasksTable.tsx`
- Create: `frontend/src/components/ResultPage.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Write types.ts**

```ts
// frontend/src/types.ts
export interface Task {
  id: string;
  filename: string;
  status: "pending" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
}

export interface Segment {
  start_time: number;
  end_time: number;
  speaker: string;
  text: string;
}
```

**Step 2: Write api.ts**

```ts
// frontend/src/api.ts
import { Task, Segment } from "./types";

const BASE = "/api";

export async function uploadFile(file: File): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(taskId: string): Promise<Task> {
  const res = await fetch(`${BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTranscript(taskId: string): Promise<{ segments: Segment[] }> {
  const res = await fetch(`${BASE}/transcript/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function searchTranscripts(q: string, taskId?: string): Promise<{ results: Segment[] }> {
  const params = new URLSearchParams({ q });
  if (taskId) params.append("task_id", taskId);
  const res = await fetch(`${BASE}/search?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

**Step 3: Write DropZone.tsx**

```tsx
// frontend/src/components/DropZone.tsx
import { useCallback, useState } from "react";
import { uploadFile } from "@/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Props {
  onUploaded: (taskId: string) => void;
}

export function DropZone({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const { task_id } = await uploadFile(file);
      onUploaded(task_id);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }, [onUploaded]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <Card
      className={`p-12 border-2 border-dashed cursor-pointer text-center transition-colors ${
        dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {uploading ? (
        <p className="text-gray-500">Загрузка...</p>
      ) : (
        <>
          <p className="text-lg font-medium text-gray-700 mb-2">
            Перетащите аудио или видео файл сюда
          </p>
          <p className="text-sm text-gray-400 mb-4">или</p>
          <label>
            <Button variant="outline" asChild>
              <span>Выберите файл</span>
            </Button>
            <input
              type="file"
              className="hidden"
              accept="audio/*,video/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </label>
        </>
      )}
      {error && <p className="text-red-500 mt-2 text-sm">{error}</p>}
    </Card>
  );
}
```

**Step 4: Write TasksTable.tsx**

```tsx
// frontend/src/components/TasksTable.tsx
import { useEffect, useState } from "react";
import { getStatus } from "@/api";
import { Task } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

const STATUS_COLOR: Record<string, string> = {
  pending: "secondary",
  processing: "default",
  completed: "outline",
  failed: "destructive",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "В очереди",
  processing: "Обрабатывается",
  completed: "Готово",
  failed: "Ошибка",
};

interface Props {
  taskIds: string[];
  onViewResult: (taskId: string) => void;
}

export function TasksTable({ taskIds, onViewResult }: Props) {
  const [tasks, setTasks] = useState<Record<string, Task>>({});

  useEffect(() => {
    if (taskIds.length === 0) return;

    const poll = async () => {
      for (const id of taskIds) {
        try {
          const task = await getStatus(id);
          setTasks((prev) => ({ ...prev, [id]: task }));
        } catch {}
      }
    };

    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [taskIds]);

  if (taskIds.length === 0) return null;

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold mb-4">Задачи</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Файл</TableHead>
            <TableHead>Статус</TableHead>
            <TableHead>Создано</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {taskIds.map((id) => {
            const task = tasks[id];
            if (!task) return (
              <TableRow key={id}>
                <TableCell colSpan={4} className="text-gray-400 text-sm">{id} — загрузка...</TableCell>
              </TableRow>
            );
            return (
              <TableRow key={id}>
                <TableCell className="font-mono text-sm">{task.filename}</TableCell>
                <TableCell>
                  <Badge variant={STATUS_COLOR[task.status] as any}>
                    {STATUS_LABEL[task.status] ?? task.status}
                  </Badge>
                </TableCell>
                <TableCell className="text-sm text-gray-500">
                  {new Date(task.created_at).toLocaleString("ru")}
                </TableCell>
                <TableCell>
                  {task.status === "completed" && (
                    <Button size="sm" onClick={() => onViewResult(id)}>
                      Открыть
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
```

**Step 5: Write ResultPage.tsx**

```tsx
// frontend/src/components/ResultPage.tsx
import { useEffect, useRef, useState } from "react";
import { getTranscript, searchTranscripts } from "@/api";
import { Segment } from "@/types";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Props {
  taskId: string;
  onBack: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function highlight(text: string, query: string): string {
  if (!query) return text;
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return text.replace(new RegExp(`(${escaped})`, "gi"), "<mark>$1</mark>");
}

const SPEAKER_COLORS = ["text-blue-700", "text-emerald-700", "text-purple-700", "text-orange-700"];

export function ResultPage({ taskId, onBack }: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [query, setQuery] = useState("");
  const [filtered, setFiltered] = useState<Segment[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    getTranscript(taskId).then(({ segments }) => {
      setSegments(segments);
      setFiltered(segments);
    });
  }, [taskId]);

  useEffect(() => {
    if (!query) {
      setFiltered(segments);
      return;
    }
    const lower = query.toLowerCase();
    setFiltered(segments.filter((s) => s.text.toLowerCase().includes(lower)));
  }, [query, segments]);

  const seek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      audioRef.current.play();
    }
  };

  const speakerIndex = (speaker: string) =>
    parseInt(speaker.replace(/\D/g, "")) % SPEAKER_COLORS.length;

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={onBack}>← Назад</Button>
        <h2 className="text-xl font-semibold">Транскрипт</h2>
      </div>

      <Card className="p-4">
        <audio
          ref={audioRef}
          controls
          className="w-full"
          src={`/api/audio/${taskId}`}
        />
      </Card>

      <Input
        placeholder="Поиск по тексту..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <ScrollArea className="flex-1 border rounded-md p-4">
        {filtered.length === 0 && (
          <p className="text-gray-400 text-sm">Ничего не найдено</p>
        )}
        {filtered.map((seg, i) => (
          <div
            key={i}
            className="mb-3 cursor-pointer hover:bg-gray-50 rounded p-2 transition-colors"
            onClick={() => seek(seg.start_time)}
          >
            <div className="flex items-center gap-2 mb-1">
              <Badge variant="outline" className={`text-xs ${SPEAKER_COLORS[speakerIndex(seg.speaker)]}`}>
                {seg.speaker}
              </Badge>
              <span className="text-xs text-gray-400 font-mono">
                {formatTime(seg.start_time)} – {formatTime(seg.end_time)}
              </span>
            </div>
            <p
              className="text-sm text-gray-800 leading-relaxed"
              dangerouslySetInnerHTML={{ __html: highlight(seg.text, query) }}
            />
          </div>
        ))}
      </ScrollArea>
    </div>
  );
}
```

**Step 6: Write App.tsx**

```tsx
// frontend/src/App.tsx
import { useState } from "react";
import { DropZone } from "@/components/DropZone";
import { TasksTable } from "@/components/TasksTable";
import { ResultPage } from "@/components/ResultPage";

export default function App() {
  const [taskIds, setTaskIds] = useState<string[]>([]);
  const [viewingTaskId, setViewingTaskId] = useState<string | null>(null);

  const handleUploaded = (taskId: string) => {
    setTaskIds((prev) => [taskId, ...prev]);
  };

  if (viewingTaskId) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto h-[calc(100vh-3rem)] flex flex-col">
          <ResultPage taskId={viewingTaskId} onBack={() => setViewingTaskId(null)} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">CatterMeet</h1>
        <p className="text-gray-500 mb-8">Транскрибация и диаризация аудио/видео</p>
        <DropZone onUploaded={handleUploaded} />
        <TasksTable taskIds={taskIds} onViewResult={setViewingTaskId} />
      </div>
    </div>
  );
}
```

**Step 7: Commit**

```bash
cd frontend
git add src/
git commit -m "feat: add all frontend components (DropZone, TasksTable, ResultPage)"
```

---

## Task 11: Audio File Serving Endpoint

The ResultPage references `/api/audio/{task_id}`. Add this endpoint to serve the original uploaded file.

**Files:**
- Modify: `backend/routers/api.py`

**Step 1: Add audio endpoint**

Add to `backend/routers/api.py`:

```python
from fastapi.responses import FileResponse

@router.get("/audio/{task_id}")
async def get_audio(task_id: str):
    task = await database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    file_path = Path(UPLOAD_DIR) / task["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))
```

**Step 2: Commit**

```bash
git add backend/routers/api.py
git commit -m "feat: add audio file serving endpoint"
```

---

## Task 12: README.md

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
# CatterMeet

Транскрибация и диаризация аудио/видео файлов (до 4 часов) с поддержкой русского и английского языков.

## Требования

- Docker и Docker Compose

## Запуск

```bash
git clone <repo>
cd CatterMeet
docker compose up --build
```

Откройте http://localhost:8000 в браузере.

## Использование

1. Перетащите аудио или видео файл в зону загрузки (или нажмите кнопку).
2. Дождитесь обработки — статус обновляется автоматически каждые 5 секунд.
3. После завершения нажмите **Открыть** для просмотра транскрипта.
4. Используйте строку поиска для фильтрации реплик.
5. Нажмите на реплику — аудиоплеер перемотает к нужному моменту.

## Ограничения

- Максимальный размер файла: ограничен доступной RAM (рекомендуется ≤ 4 ГБ).
- Обработка идёт строго последовательно (одна задача за раз).
- CPU-only режим; GPU не используется.

## Данные

Файлы и база данных хранятся в папке `./data/` на хосте.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick-start instructions"
```

---

## Final Verification

```bash
# Build and start
docker compose up --build

# In another terminal — smoke test
curl -s http://localhost:8000/api/status/nonexistent | python3 -m json.tool
# Expected: {"detail": "Task not found"}

# Upload a test file
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/audio.mp3" | python3 -m json.tool
# Expected: {"task_id": "..."}
```

Open http://localhost:8000 — the React UI should load.
