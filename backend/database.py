# backend/database.py
import asyncio
import os
import sqlite3
import uuid
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "cattermeet.db")

# Shared connection used only for in-memory databases (":memory:"),
# because each new sqlite3.connect(":memory:") call creates a completely
# separate, empty database. For file-based DBs we create connections normally.
_shared_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _shared_conn
    if DB_PATH == ":memory:":
        if _shared_conn is None:
            _shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            _shared_conn.row_factory = sqlite3.Row
        return _shared_conn
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


async def init_db() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_db_sync)


def _init_db_sync() -> None:
    global _shared_conn
    if DB_PATH == ":memory:":
        # Reset the shared connection so each test starts with a fresh database.
        if _shared_conn is not None:
            _shared_conn.close()
        _shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
        _shared_conn.row_factory = sqlite3.Row
    conn = _get_conn()
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
                task_id UNINDEXED,
                transcript_id UNINDEXED
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
            (task_id, filename),
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
            (status, task_id),
        )


async def save_transcript(task_id: str, segments: list) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_transcript_sync, task_id, segments)


def _save_transcript_sync(task_id: str, segments: list) -> None:
    with _get_conn() as conn:
        for seg in segments:
            cursor = conn.execute(
                "INSERT INTO transcripts (task_id, start_time, end_time, speaker, text) VALUES (?,?,?,?,?)",
                (task_id, seg["start_time"], seg["end_time"], seg["speaker"], seg["text"]),
            )
            transcript_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO transcripts_fts (text, task_id, transcript_id) VALUES (?, ?, ?)",
                (seg["text"], task_id, transcript_id),
            )


async def get_transcript(task_id: str) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_transcript_sync, task_id)


def _get_transcript_sync(task_id: str) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT start_time, end_time, speaker, text FROM transcripts WHERE task_id = ? ORDER BY start_time",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]


async def search_transcripts(query: str, task_id: Optional[str] = None) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_sync, query, task_id)


def _search_sync(query: str, task_id: Optional[str]) -> list:
    with _get_conn() as conn:
        if task_id:
            rows = conn.execute(
                """SELECT t.task_id, t.start_time, t.end_time, t.speaker, t.text
                   FROM transcripts_fts fts
                   JOIN transcripts t ON t.id = fts.transcript_id
                   WHERE transcripts_fts MATCH ? AND fts.task_id = ?
                   ORDER BY rank""",
                (query, task_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT t.task_id, t.start_time, t.end_time, t.speaker, t.text
                   FROM transcripts_fts fts
                   JOIN transcripts t ON t.id = fts.transcript_id
                   WHERE transcripts_fts MATCH ?
                   ORDER BY rank""",
                (query,),
            ).fetchall()
        return [dict(r) for r in rows]
