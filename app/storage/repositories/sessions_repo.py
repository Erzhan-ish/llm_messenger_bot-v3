from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.storage.models import Session
from app.storage.db import async_session


SESSION_TTL = timedelta(hours=settings.SESSION_TTL_HOURS)


async def close_active_session(user_id: int) -> None:
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
    async with async_session() as session:
        new_session = Session(
            user_id=user_id,
            status="active",
        )
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session


async def update_session_payload(session_id: int, payload: str):
    async with async_session() as session:
        await session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(start_payload=payload)
        )
        await session.commit()


async def get_last_session(user_id: int) -> Session | None:
    async with async_session() as session:  # type: AsyncSession
        result = await session.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.last_activity_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


async def create_session(user_id: int) -> Session:
    async with async_session() as session:
        new_session = Session(
            user_id=user_id,
            status="new",
            last_activity_at=datetime.utcnow(),
        )
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session


async def touch_session(session_id: int) -> None:
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
            .values(status=status)
        )
        await session.commit()


def is_session_expired(session: Session) -> bool:
    return datetime.utcnow() - session.last_activity_at > SESSION_TTL
