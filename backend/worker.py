# backend/worker.py
import asyncio
import logging
import os
from pathlib import Path

import database
from processor import process_file

logger = logging.getLogger(__name__)

task_queue: asyncio.Queue = asyncio.Queue()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


async def process_worker() -> None:
    """
    Reads task_ids from the queue indefinitely.
    Processes one file at a time to stay within RAM limits.
    """
    while True:
        task_id = await task_queue.get()
        try:
            task = await database.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found in DB, skipping")
                continue

            file_path = str(Path(UPLOAD_DIR) / task["filename"])
            logger.info(f"Starting processing of task {task_id}: {file_path}")

            await database.update_task_status(task_id, "processing")

            # Run blocking ML processing in executor (does not block event loop)
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(None, process_file, file_path)

            await database.save_transcript(task_id, segments)
            await database.update_task_status(task_id, "completed")
            logger.info(f"Task {task_id} completed with {len(segments)} segments")

        except Exception as exc:
            logger.exception(f"Task {task_id} failed: {exc}")
            try:
                await database.update_task_status(task_id, "failed")
            except Exception:
                pass
        finally:
            task_queue.task_done()
