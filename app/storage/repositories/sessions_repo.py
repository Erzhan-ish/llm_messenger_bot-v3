from __future__ import annotations

from datetime import datetime
from sqlalchemy import select, update, desc
from app.storage.db import async_session
from app.storage.models import Session, User



async def close_active_session(user_id: int) -> None:
    """
    Закрывает активную сессию пользователя (если есть)
    """
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(
                Session.user_id == user_id,
                Session.status == "active",
            )
            .values(
                status="closed",
                last_activity_at=datetime.utcnow(),
            )
        )
        await session.commit()


async def create_new_session(user_id: int) -> Session:
    """
    Создаёт новую активную сессию
    """
    async with async_session() as session:
        new_session = Session(
            user_id=user_id,
            status="active",
            last_activity_at=datetime.utcnow(),
        )
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session


async def get_last_inbound_time(channel: str, external_user_id: str) -> datetime | None:
    """
    Время последней inbound-активности пользователя по (channel, external_user_id).
    Нужно для WhatsApp 24h окна.
    """
    async with async_session() as session:
        res = await session.execute(
            select(Session.last_activity_at)
            .join(User, User.id == Session.user_id)
            .where(
                User.channel == channel,
                User.external_user_id == external_user_id,
            )
            .order_by(desc(Session.last_activity_at))
            .limit(1)
        )
        return res.scalar_one_or_none()

async def touch_session_activity(session_id: int) -> None:
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(last_activity_at=datetime.utcnow())
        )
        await session.commit()

async def update_session_status(session_id: int, status: str) -> None:
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=status, last_activity_at=datetime.utcnow())
        )
        await session.commit()