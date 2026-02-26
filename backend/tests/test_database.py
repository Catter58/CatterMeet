# backend/tests/test_database.py
import asyncio
import os
import pytest

os.environ["DB_PATH"] = ":memory:"

from database import (
    init_db,
    create_task,
    get_task,
    update_task_status,
    save_transcript,
    get_transcript,
    search_transcripts,
)


@pytest.fixture(autouse=True)
def setup():
    asyncio.run(init_db())


def test_create_and_get_task():
    task_id = asyncio.run(create_task("audio.mp3"))
    assert task_id  # non-empty string
    task = asyncio.run(get_task(task_id))
    assert task is not None
    assert task["filename"] == "audio.mp3"
    assert task["status"] == "pending"


def test_update_status():
    task_id = asyncio.run(create_task("f.mp3"))
    asyncio.run(update_task_status(task_id, "processing"))
    task = asyncio.run(get_task(task_id))
    assert task["status"] == "processing"


def test_get_unknown_task_returns_none():
    result = asyncio.run(get_task("nonexistent-id"))
    assert result is None


def test_save_and_get_transcript():
    task_id = asyncio.run(create_task("f.mp3"))
    segments = [
        {"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_0", "text": "Привет мир"},
        {"start_time": 3.0, "end_time": 5.0, "speaker": "SPEAKER_1", "text": "Hello world"},
    ]
    asyncio.run(save_transcript(task_id, segments))
    result = asyncio.run(get_transcript(task_id))
    assert len(result) == 2
    assert result[0]["text"] == "Привет мир"
    assert result[0]["speaker"] == "SPEAKER_0"


def test_search_transcripts():
    task_id = asyncio.run(create_task("f.mp3"))
    segments = [
        {"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_0", "text": "Привет мир"},
    ]
    asyncio.run(save_transcript(task_id, segments))
    results = asyncio.run(search_transcripts("Привет", task_id))
    assert len(results) >= 1
    assert "Привет" in results[0]["text"]


def test_search_no_results():
    task_id = asyncio.run(create_task("f.mp3"))
    results = asyncio.run(search_transcripts("nonexistentword", task_id))
    assert results == []
