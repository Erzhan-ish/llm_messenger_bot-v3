from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError

from app.storage.db import AsyncSessionLocal
from app.storage.models import ConversationLock
from app.logging import logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def try_acquire_conversation_lock(
    *,
    conversation_key: str,
    worker_id: str,
    ttl_seconds: int,
) -> bool:
    now = _utcnow()
    expires_at = now + timedelta(seconds=ttl_seconds)

    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(ConversationLock).where(
                ConversationLock.conversation_key == conversation_key,
                ConversationLock.expires_at < now,
            )
        )

        try:
            session.add(
                ConversationLock(
                    conversation_key=conversation_key,
                    locked_by=worker_id,
                    expires_at=expires_at,
                )
            )
            await session.commit()
            logger.info(
                "ConversationLock | action=acquired | worker_id={} | conversation_key={}",
                worker_id, conversation_key,
            )
            return True
        except IntegrityError:
            await session.rollback()
            logger.info(
                "ConversationLock | action=deferred | worker_id={} | conversation_key={} | reason=already_locked",
                worker_id, conversation_key,
            )
            return False


async def release_conversation_lock(
    *,
    conversation_key: str,
    worker_id: str,
) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(ConversationLock).where(
                ConversationLock.conversation_key == conversation_key,
                ConversationLock.locked_by == worker_id,
            )
        )
        await session.commit()
    logger.info(
        "ConversationLock | action=released | worker_id={} | conversation_key={}",
        worker_id, conversation_key,
    )


async def refresh_conversation_lock(
    *,
    conversation_key: str,
    worker_id: str,
    ttl_seconds: int,
) -> None:
    expires_at = _utcnow() + timedelta(seconds=ttl_seconds)
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(ConversationLock)
            .where(
                ConversationLock.conversation_key == conversation_key,
                ConversationLock.locked_by == worker_id,
            )
            .values(expires_at=expires_at)
        )
        await session.commit()


async def delete_expired_conversation_locks() -> int:
    now = _utcnow()
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(ConversationLock).where(ConversationLock.expires_at < now)
        )
        await session.commit()
        deleted = result.rowcount or 0

    if deleted:
        logger.info("StaleConversationLocksCleanup | deleted={}", deleted)
    return deleted
