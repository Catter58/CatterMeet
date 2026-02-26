# backend/routers/api.py
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import database
from worker import task_queue

router = APIRouter(prefix="/api")

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "audio.bin").suffix or ".bin"
    safe_name = f"{uuid.uuid4()}{ext}"
    dest = Path(UPLOAD_DIR) / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(str(dest), "wb") as f:
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


@router.get("/audio/{task_id}")
async def get_audio(task_id: str):
    task = await database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    file_path = Path(UPLOAD_DIR) / task["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(file_path))
