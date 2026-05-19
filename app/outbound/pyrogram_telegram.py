from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.storage.repositories.jobs_repo import enqueue_job
from app.logging import logger


async def send_pyrogram_telegram(user_id: str, text: str) -> None:
    """Enqueue a pyrogram_outbound_{session} job for the correct Pyrogram runner."""
    session_name = await _resolve_session(user_id)
    job_type = f"pyrogram_outbound_{session_name}"

    await enqueue_job(
        job_type=job_type,
        payload={"user_id": user_id, "text": text},
        max_attempts=3,
    )
    logger.info(
        "Pyrogram outbound enqueued | session={} | user_id={}",
        session_name, user_id,
    )


async def send_pyrogram_typing(user_id: str) -> None:
    """Enqueue a pyrogram_typing_{session} job (fire-and-forget, stale jobs are skipped)."""
    session_name = await _resolve_session(user_id)
    job_type = f"pyrogram_typing_{session_name}"

    await enqueue_job(
        job_type=job_type,
        payload={"user_id": user_id, "enqueued_at": datetime.now(timezone.utc).isoformat()},
        max_attempts=1,
    )


async def _resolve_session(user_id: str) -> str:
    """Return the session name this user belongs to (stored in wazzup_channel_id)."""
    try:
        from app.storage.repositories.users_repo import get_user
        user = await get_user("telegram_personal", user_id)
        if user and user.wazzup_channel_id:
            return user.wazzup_channel_id
    except Exception:
        pass
    return settings.PYROGRAM_SESSION_NAME
