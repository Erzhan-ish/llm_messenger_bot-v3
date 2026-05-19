# app/worker.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "..env")

import asyncio

from app.config import settings
from app.logging import logger
from app.storage.repositories.jobs_repo import (
    fetch_and_lock_jobs,
    mark_done,
    retry_or_give_up,
    requeue_stale_running_jobs,
)
from app.processing.simple_message_processor import process_message, JobDeferred
from app.services.transcription_service import transcribe_audio
from app.services.whatsapp_media import download_whatsapp_media

WORKER_ID = os.getenv("WORKER_ID", "worker-unknown")


def _require(payload: dict, key: str) -> str:
    val = payload.get(key)
    if not val:
        raise ValueError(f"Job payload missing required field: {key}")
    return str(val)


async def _handle_inbound(payload: dict) -> None:
    await process_message(payload)


async def _handle_stt(payload: dict) -> None:
    channel = payload.get("channel")
    if channel != "whatsapp":
        raise ValueError("STT job supported only for whatsapp")

    external_user_id = _require(payload, "external_user_id")
    message_id = _require(payload, "message_id")
    media_id = _require(payload, "media_id")

    audio_path, mime_type = await download_whatsapp_media(
        media_id=media_id,
        user_id=external_user_id,
        message_id=message_id,
    )

    text = await transcribe_audio(audio_path)
    text = (text or "").strip()

    payload["audio_path"] = audio_path
    payload["mime_type"] = mime_type

    if not text:
        payload["message_type"] = "text"
        payload["text"] = "Не смог распознать голосовое. Напишите, пожалуйста, текстом."
        await process_message(payload)
        return

    payload["message_type"] = "text"
    payload["text"] = text
    await process_message(payload)


async def handle_job(job) -> None:
    payload = dict(job.payload or {})
    job_type = (job.job_type or "").strip()

    if job_type == "inbound":
        payload["_job_id"] = job.id
        await _handle_inbound(payload)
        return

    if job_type == "stt":
        await _handle_stt(payload)
        return

    raise ValueError(f"Unknown job_type: {job_type}")


async def worker_loop(poll_interval: float | None = None, batch_size: int | None = None):
    _poll = poll_interval if poll_interval is not None else settings.WORKER_POLL_INTERVAL
    _batch = batch_size if batch_size is not None else settings.WORKER_BATCH_SIZE

    try:
        from app.storage.db import engine, Base
        import app.storage.models  # noqa: F401 — register all ORM models
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Worker DB schema verified")
    except Exception:
        logger.exception("Worker DB init failed — worker will not start")
        raise

    requeued = await requeue_stale_running_jobs(stale_after_seconds=settings.JOB_STALE_LOCK_TTL_SECONDS)
    from app.storage.repositories.conversation_locks_repo import delete_expired_conversation_locks
    await delete_expired_conversation_locks()

    logger.info(
        "Worker started | worker_id={} | poll_interval={} | batch_size={}",
        WORKER_ID, _poll, _batch,
    )

    while True:
        jobs = await fetch_and_lock_jobs(limit=_batch, job_types=["inbound", "stt"])

        if not jobs:
            await asyncio.sleep(_poll)
            continue

        for job in jobs:
            try:
                logger.info(
                    "Job running | worker_id={} | id={} | type={}",
                    WORKER_ID, job.id, job.job_type,
                )
                await handle_job(job)
                await mark_done(job.id)
                logger.info("Job done | worker_id={} | id={}", WORKER_ID, job.id)
            except JobDeferred:
                logger.info("Job deferred | worker_id={} | id={}", WORKER_ID, job.id)
            except Exception as e:
                logger.exception("Job failed | worker_id={} | id={}", WORKER_ID, job.id)
                await retry_or_give_up(job.id, str(e), backoff_seconds=10)

        await asyncio.sleep(0)  # yield


def main():
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
