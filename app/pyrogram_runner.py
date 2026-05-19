"""Pyrogram personal account runner.

Runs as a separate process (spawned by run.py when PYROGRAM_ENABLED=True).

Responsibilities:
  - Receive incoming messages from the personal Telegram account → enqueue inbound jobs
  - Poll pyrogram_outbound jobs → send replies via the personal account
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from app.config import settings
from app.logging import logger
from app.channels.base import UnifiedMessage
from app.jobs.enqueue import enqueue_inbound_message_job
from app.storage.repositories.jobs_repo import (
    fetch_and_lock_jobs,
    mark_done,
    retry_or_give_up,
    requeue_stale_running_jobs,
)
from app.storage.repositories.conversation_locks_repo import delete_expired_conversation_locks


async def _send_with_peer_resolve(client, user_id: int, text: str) -> None:
    """Send message, refreshing dialog cache on PeerIdInvalid."""
    from pyrogram.errors import PeerIdInvalid
    try:
        await client.send_message(user_id, text)
    except PeerIdInvalid:
        logger.warning("PeerIdInvalid for user_id={} — refreshing dialogs cache", user_id)
        async for _ in client.get_dialogs(limit=200):
            pass
        await client.send_message(user_id, text)


async def _outbound_loop(client, session_name: str) -> None:
    """Poll pyrogram_outbound_{session_name} and pyrogram_typing_{session_name} jobs."""
    from pyrogram.enums import ChatAction

    outbound_type = f"pyrogram_outbound_{session_name}"
    typing_type = f"pyrogram_typing_{session_name}"

    while True:
        try:
            jobs = await fetch_and_lock_jobs(limit=10, job_types=[outbound_type, typing_type])
            for job in jobs:
                try:
                    user_id = job.payload.get("user_id")
                    if not user_id:
                        await mark_done(job.id)
                        continue

                    if job.job_type == typing_type:
                        # Skip stale typing jobs (older than 8 seconds)
                        from datetime import timezone
                        now = datetime.now(timezone.utc)
                        enqueued_at_str = job.payload.get("enqueued_at")
                        if enqueued_at_str:
                            enqueued_at = datetime.fromisoformat(enqueued_at_str)
                            age = (now - enqueued_at).total_seconds()
                        else:
                            age = 0
                        if age < 8:
                            await client.send_chat_action(int(user_id), ChatAction.TYPING)
                            logger.debug(
                                "Pyrogram typing sent | user_id={} | age={:.1f}s",
                                user_id, age,
                            )
                        await mark_done(job.id)

                    else:  # outbound_type
                        text = job.payload.get("text", "")
                        if text:
                            await _send_with_peer_resolve(client, int(user_id), text)
                            logger.info(
                                "Pyrogram outbound sent | user_id={} | job_id={}",
                                user_id, job.id,
                            )
                        await mark_done(job.id)

                except Exception as e:
                    logger.exception(
                        "Pyrogram outbound failed | job_id={}", job.id
                    )
                    await retry_or_give_up(job.id, str(e), backoff_seconds=5)
        except Exception:
            logger.exception("Pyrogram outbound loop error")
        await asyncio.sleep(0.5)


async def _run() -> None:
    SESSION_NAME = os.getenv("PYROGRAM_SESSION_NAME", settings.PYROGRAM_SESSION_NAME)

    try:
        from pyrogram import Client, filters
        from pyrogram.types import Message as PyroMessage
    except ImportError:
        raise RuntimeError("pyrogram not installed. Run: pip install pyrogram")

    if not settings.PYROGRAM_API_ID or not settings.PYROGRAM_API_HASH:
        raise RuntimeError(
            "PYROGRAM_API_ID and PYROGRAM_API_HASH must be set. "
            "Get them at https://my.telegram.org"
        )

    # DB init + stale cleanup
    from app.storage.db import engine, Base
    import app.storage.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await requeue_stale_running_jobs(stale_after_seconds=settings.JOB_STALE_LOCK_TTL_SECONDS)
    await delete_expired_conversation_locks()
    logger.info("Pyrogram runner | DB ready")

    client = Client(
        name=settings.PYROGRAM_SESSION_NAME,
        api_id=settings.PYROGRAM_API_ID,
        api_hash=settings.PYROGRAM_API_HASH,
    )

    async def _tag_user_session(user_id_str: str) -> None:
        """Store which session this user belongs to for outbound routing."""
        from app.storage.repositories.users_repo import get_or_create_user, update_wazzup_meta
        user = await get_or_create_user("telegram_personal", user_id_str)
        if user.wazzup_channel_id != SESSION_NAME:
            await update_wazzup_meta(user.id, SESSION_NAME, None)

    @client.on_message(filters.private & filters.incoming & filters.text)
    async def on_text(_, message: PyroMessage) -> None:
        if not message.from_user:
            return

        user_id_str = str(message.from_user.id)
        await _tag_user_session(user_id_str)

        try:
            await client.read_chat_history(int(user_id_str))
        except Exception:
            pass

        um = UnifiedMessage(
            channel="telegram_personal",
            external_user_id=user_id_str,
            message_id=str(message.id),
            message_type="text",
            text=message.text or "",
            created_at=datetime.utcnow(),
        )

        logger.info(
            "Pyrogram inbound | session={} | user_id={} | message_id={} | text={!r}",
            SESSION_NAME, um.external_user_id, um.message_id, (um.text or "")[:60],
        )

        await enqueue_inbound_message_job(um)

    @client.on_message(filters.private & filters.incoming & filters.voice)
    async def on_voice(_, message: PyroMessage) -> None:
        if not message.from_user:
            return

        user_id_str = str(message.from_user.id)
        await _tag_user_session(user_id_str)

        try:
            await client.read_chat_history(int(user_id_str))
        except Exception:
            pass

        um = UnifiedMessage(
            channel="telegram_personal",
            external_user_id=user_id_str,
            message_id=str(message.id),
            message_type="text",
            text="[голосовое сообщение — напишите текстом]",
            created_at=datetime.utcnow(),
        )

        logger.info(
            "Pyrogram inbound voice | session={} | user_id={} | message_id={}",
            SESSION_NAME, um.external_user_id, um.message_id,
        )

        await enqueue_inbound_message_job(um)

    await client.start()

    me = await client.get_me()
    logger.info(
        "Pyrogram runner started | account=@{} | session={}",
        me.username or me.first_name,
        settings.PYROGRAM_SESSION_NAME,
    )

    asyncio.create_task(_outbound_loop(client, SESSION_NAME))

    # Keep running until interrupted
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
