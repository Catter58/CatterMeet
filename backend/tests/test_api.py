# backend/tests/test_api.py
import asyncio
import os

# Must be set before importing app modules
os.environ["DB_PATH"] = ":memory:"
os.environ["UPLOAD_DIR"] = "/tmp/cattermeet_test_uploads"

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Initialize the DB before importing main (which starts the worker)
from database import init_db

# Import app after env vars set
from main import app


@pytest.fixture(autouse=True)
def setup_db():
    asyncio.run(init_db())
    os.makedirs("/tmp/cattermeet_test_uploads", exist_ok=True)


@pytest.mark.asyncio
async def test_upload_returns_task_id(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fakecontent")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(str(audio), "rb") as f:
            resp = await client.post(
                "/api/upload",
                files={"file": ("test.mp3", f, "audio/mpeg")},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0


@pytest.mark.asyncio
async def test_status_returns_task():
    # First upload a file
    import io
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
        )
        task_id = resp.json()["task_id"]

        # Now get status
        resp2 = await client.get(f"/api/status/{task_id}")

    assert resp2.status_code == 200
    data = resp2.json()
    assert data["id"] == task_id
    assert data["status"] in ("pending", "processing", "completed", "failed")


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


@pytest.mark.asyncio
async def test_search_returns_results():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/search?q=hello")
    assert resp.status_code == 200
    assert "results" in resp.json()
