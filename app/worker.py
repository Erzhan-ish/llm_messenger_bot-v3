# app/worker.py
from __future__ import annotations

import asyncio

from app.logging import logger
from app.storage.repositories.jobs_repo import (
    fetch_and_lock_jobs,
    mark_done,
    retry_or_give_up,
)
from app.processing.message_processor import process_message
from app.services.transcription_service import transcribe_audio
from app.services.whatsapp_media import download_whatsapp_media


def _require(payload: dict, key: str) -> str:
    val = payload.get(key)
    if not val:
        raise ValueError(f"Job payload missing required field: {key}")
    return str(val)


async def _handle_inbound(payload: dict) -> None:
    # payload = UnifiedMessage dict
    await process_message(payload)


async def _handle_stt(payload: dict) -> None:
    # payload = UnifiedMessage dict (message_type=audio, media_id set)
    channel = payload.get("channel")
    if channel != "whatsapp":
        raise ValueError("STT job supported only for whatsapp")

    external_user_id = _require(payload, "external_user_id")
    message_id = _require(payload, "message_id")
    media_id = _require(payload, "media_id")

    # 1) download media -> audio_path
    audio_path, mime_type = await download_whatsapp_media(
        media_id=media_id,
        user_id=external_user_id,
        message_id=message_id,
    )

    # 2) transcribe
    text = await transcribe_audio(audio_path)
    text = (text or "").strip()

    # 3) если STT не дал текста — отправляем fallback через тот же пайплайн
    payload["audio_path"] = audio_path
    payload["mime_type"] = mime_type

    if not text:
        payload["message_type"] = "text"
        payload["text"] = "Не смог распознать голосовое. Напишите, пожалуйста, текстом."
        await process_message(payload)
        return

    # 4) дальше как text-сообщение
    payload["message_type"] = "text"
    payload["text"] = text
    await process_message(payload)


async def handle_job(job) -> None:
    payload = dict(job.payload or {})
    job_type = (job.job_type or "").strip()

    if job_type == "inbound":
        await _handle_inbound(payload)
        return

    if job_type == "stt":
        await _handle_stt(payload)
        return

    raise ValueError(f"Unknown job_type: {job_type}")


async def worker_loop(poll_interval: float = 0.7, batch_size: int = 10):
    logger.info("Worker started")

    while True:
        jobs = await fetch_and_lock_jobs(limit=batch_size)

        if not jobs:
            await asyncio.sleep(poll_interval)
            continue

        for job in jobs:
            try:
                logger.info("Job running | id={} | type={}", job.id, job.job_type)
                await handle_job(job)
                await mark_done(job.id)
                logger.info("Job done | id={}", job.id)
            except Exception as e:
                logger.exception("Job failed | id={}", job.id)
                await retry_or_give_up(job.id, str(e), backoff_seconds=10)

        await asyncio.sleep(0)  # yield


def main():
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()

