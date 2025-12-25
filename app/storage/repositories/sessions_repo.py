from __future__ import annotations

from datetime import datetime
from sqlalchemy import select, update, desc

from app.storage.db import async_session
from app.storage.models import Session


async def get_active_session(user_id: int) -> Session | None:
    async with async_session() as session:
        res = await session.execute(
            select(Session)
            .where(Session.user_id == user_id, Session.status != "closed")
            .order_by(desc(Session.last_activity_at))
            .limit(1)
        )
        return res.scalar_one_or_none()


async def create_new_session(user_id: int, status: str = "active") -> Session:
    async with async_session() as session:
        s = Session(
            user_id=user_id,
            status=status,
            last_activity_at=datetime.utcnow(),
        )
        session.add(s)
        await session.flush()
        return s


async def close_active_session(user_id: int) -> None:
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.user_id == user_id, Session.status != "closed")
            .values(status="closed", closed_at=datetime.utcnow())
        )


async def touch_session_activity(session_id: int) -> None:
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(last_activity_at=datetime.utcnow())
        )


async def update_session_status(session_id: int, status: str) -> None:
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=status, last_activity_at=datetime.utcnow())
        )


async def get_last_inbound_time(user_id: int) -> datetime | None:
    """
    Время последней активности (inbound) пользователя.
    Для WhatsApp 24h окна опираемся на last_activity_at активной/последней сессии.
    """
    async with async_session() as session:
        res = await session.execute(
            select(Session.last_activity_at)
            .where(Session.user_id == user_id)
            .order_by(desc(Session.last_activity_at))
            .limit(1)
        )
        return res.scalar_one_or_none()
