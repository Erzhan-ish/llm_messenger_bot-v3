from __future__ import annotations

from typing import Any

from app.channels.base import UnifiedMessage
from app.logging import logger
from app.storage.repositories.jobs_repo import enqueue_job


def _to_payload(msg: UnifiedMessage) -> dict[str, Any]:
    # UnifiedMessage -> dict, который worker сможет восстановить
    return {
        "channel": msg.channel,
        "external_user_id": msg.external_user_id,
        "message_id": msg.message_id,
        "message_type": msg.message_type,
        "media_id": getattr(msg, "media_id", None),
        "mime_type": getattr(msg, "mime_type", None),
        "text": msg.text,
        "audio_path": getattr(msg, "audio_path", None),
        "chat_type": getattr(msg, "chat_type", None),
        "wazzup_channel_id": getattr(msg, "wazzup_channel_id", None),
        "created_at": msg.created_at.isoformat() if getattr(msg, "created_at", None) else None,
    }


async def enqueue_inbound_message_job(msg: UnifiedMessage) -> int:
    payload = _to_payload(msg)

    job_id = await enqueue_job(
        job_type="inbound",
        payload=payload,
        run_after=None,
        max_attempts=5,
    )

    logger.info(
        "Inbound enqueued | job_id={} | channel={} | external_user_id={}",
        job_id,
        msg.channel,
        msg.external_user_id,
    )
    return job_id


async def enqueue_stt_job(msg: UnifiedMessage) -> int:
    payload = _to_payload(msg)

    if msg.channel != "whatsapp":
        raise ValueError("STT job supported only for whatsapp")
    if not getattr(msg, "media_id", None):
        raise ValueError("STT job requires media_id")

    job_id = await enqueue_job(
        job_type="stt",
        payload=payload,
        run_after=None,
        max_attempts=5,
    )

    logger.info(
        "STT enqueued | job_id={} | external_user_id={} | media_id={}",
        job_id,
        msg.external_user_id,
        getattr(msg, "media_id", None),
    )
    return job_id

