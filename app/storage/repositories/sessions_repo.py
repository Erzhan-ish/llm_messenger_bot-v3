from __future__ import annotations

from datetime import datetime
from sqlalchemy import select, update, desc
from app.storage.db import async_session
from app.storage.models import Session, User


async def get_client_need(session_id: int) -> str | None:
    async with async_session() as db:
        res = await db.execute(
            select(Session.client_need).where(Session.id == session_id)
        )
        return res.scalar_one_or_none()


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
            collected_data={},
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


async def set_dialog_state(session_id: int, dialog_state: str) -> None:
    async with async_session() as db:
        await db.execute(
            update(Session).where(Session.id == session_id).values(dialog_state=dialog_state)
        )
        await db.commit()

async def set_negative_handled(session_id: int, value: bool = True) -> None:
    async with async_session() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(negative_handled=value)
        )
        await db.commit()

async def mark_escalated(session_id: int) -> None:
    async with async_session() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                escalated_at=datetime.utcnow(),
                status="escalated",
            )
        )
        await db.commit()

async def set_client_need(session_id: int, need: str) -> None:
    async with async_session() as db:
        await db.execute(
            update(Session)
            .where(
                Session.id == session_id,
                Session.client_need.is_(None),
            )
            .values(client_need=need)
        )
        await db.commit()


async def get_client_need(session_id: int) -> str | None:
    async with async_session() as db:
        res = await db.execute(
            select(Session.client_need).where(Session.id == session_id)
        )
        return res.scalar_one_or_none()


async def is_escalated(session_id: int) -> bool:
    async with async_session() as db:
        res = await db.execute(
            select(Session.escalated_at)
            .where(Session.id == session_id)
        )
        return res.scalar_one_or_none() is not None


async def get_session_by_id(session_id: int) -> Session | None:
    async with async_session() as db:
        res = await db.execute(
            select(Session).where(Session.id == session_id)
        )
        return res.scalar_one_or_none()

async def get_slots(session_id: int) -> dict:
    session = await get_session_by_id(session_id)
    if not session or not session.collected_data:
        return {}
    return dict(session.collected_data)


async def set_slots(session_id: int, slots: dict) -> None:
    async with async_session() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(collected_data=slots)
        )
        await db.commit()
        await db.commit()


async def get_user_last_escalation(user_id: int) -> datetime | None:
    async with async_session() as db:
        res = await db.execute(
            select(Session.escalated_at)
            .where(
                Session.user_id == user_id,
                Session.escalated_at.is_not(None)
            )
            .order_by(Session.escalated_at.desc())
            .limit(1)
        )
        return res.scalar_one_or_none()
