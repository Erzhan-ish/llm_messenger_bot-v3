from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import select, update, desc
from sqlalchemy.orm import load_only

from app.storage.db import async_session
from app.storage.models import Session
from app.storage.repositories.users_repo import get_or_create_user
from app.logging import logger


SESSION_TTL_HOURS = 24

# Приведи к ЕДИНОМУ стандарту по всему проекту
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_CLOSED = "closed"

DEFAULT_DIALOG_STATE = "new"

async def reset_session(channel: str, external_user_id: str) -> Session:
    """
    Закрывает ВСЕ активные сессии пользователя и создаёт новую активную.
    """
    user = await get_or_create_user(channel, external_user_id)
    now = datetime.utcnow()

    async with async_session() as db:
        # Закрываем все активные
        await db.execute(
            update(Session)
            .where(
                Session.user_id == user.id,
                Session.status == SESSION_STATUS_ACTIVE,
            )
            .values(
                status=SESSION_STATUS_CLOSED,
                last_activity_at=now,
            )
        )

        # 🔹 СОЗДАЁМ НОВУЮ СЕССИЮ С dialog_state
        new_session = Session(
            user_id=user.id,
            status=SESSION_STATUS_ACTIVE,
            dialog_state=DEFAULT_DIALOG_STATE,   # ← КЛЮЧЕВО
            negative_handled=False,
            last_activity_at=now,
        )

        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        logger.info(
            "Session reset | channel={} | external_user_id={} | new_session_id={}",
            channel,
            external_user_id,
            new_session.id,
        )

        return new_session


async def get_or_create_session(channel: str, external_user_id: str) -> Session:
    user = await get_or_create_user(channel, external_user_id)

    now = datetime.utcnow()
    ttl_border = now - timedelta(hours=SESSION_TTL_HOURS)

    async with async_session() as db:
        stmt = (
            select(Session)
            .where(
                Session.user_id == user.id,
                Session.status == SESSION_STATUS_ACTIVE,
            )
            .order_by(Session.last_activity_at.desc(), Session.id.desc())
        )

        res = await db.execute(stmt)
        active_sessions = list(res.scalars().all())
        active_session = active_sessions[0] if active_sessions else None

        # 1️⃣ если активных несколько — закрываем лишние
        if len(active_sessions) > 1:
            extra_ids = [s.id for s in active_sessions[1:]]
            await db.execute(
                update(Session)
                .where(Session.id.in_(extra_ids))
                .values(status=SESSION_STATUS_CLOSED, last_activity_at=now)
            )
            await db.commit()
            logger.warning(
                "Multiple active sessions detected | user_id={} | kept={} | closed_ids={}",
                user.id,
                active_session.id if active_session else None,
                extra_ids,
            )

        # 2️⃣ если есть активная
        if active_session:
            # TTL истёк → закрываем и создаём новую
            if active_session.last_activity_at and active_session.last_activity_at < ttl_border:
                await db.execute(
                    update(Session)
                    .where(Session.id == active_session.id)
                    .values(status=SESSION_STATUS_CLOSED, last_activity_at=now)
                )

                new_session = Session(
                    user_id=user.id,
                    status=SESSION_STATUS_ACTIVE,
                    dialog_state=DEFAULT_DIALOG_STATE,
                    last_activity_at=now,
                )
                db.add(new_session)
                await db.commit()
                await db.refresh(new_session)
                return new_session

            # 🔒 АКТИВНАЯ И ЖИВАЯ — ТРОГАЕМ И ВОЗВРАЩАЕМ
            await db.execute(
                update(Session)
                .where(Session.id == active_session.id)
                .values(last_activity_at=now)
            )
            await db.commit()

            # dialog_state защита
            if not active_session.dialog_state:
                await db.execute(
                    update(Session)
                    .where(Session.id == active_session.id)
                    .values(dialog_state=DEFAULT_DIALOG_STATE)
                )
                await db.commit()
                active_session.dialog_state = DEFAULT_DIALOG_STATE

            active_session.last_activity_at = now
            return active_session  # ⬅⬅⬅ КЛЮЧЕВОЕ

        # 3️⃣ активной НЕТ вообще → создаём
        new_session = Session(
            user_id=user.id,
            status=SESSION_STATUS_ACTIVE,
            dialog_state=DEFAULT_DIALOG_STATE,
            last_activity_at=now,
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session
