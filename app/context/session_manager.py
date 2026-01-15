from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import select, update
from sqlalchemy.orm import load_only

from app.storage.db import async_session
from app.storage.models import Session
from app.storage.repositories.users_repo import get_or_create_user
from app.logging import logger


SESSION_TTL_HOURS = 24

# Приведи к ЕДИНОМУ стандарту по всему проекту
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_CLOSED = "closed"


async def reset_session(channel: str, external_user_id: str) -> Session:
    """
    Закрывает ВСЕ активные сессии пользователя и создаёт новую активную.
    """
    user = await get_or_create_user(channel, external_user_id)
    now = datetime.utcnow()

    async with async_session() as db:
        # Закрываем все активные (если их несколько — тоже)
        await db.execute(
            update(Session)
            .where(Session.user_id == user.id, Session.status == SESSION_STATUS_ACTIVE)
            .values(status=SESSION_STATUS_CLOSED, last_activity_at=now)
        )

        new_session = Session(
            user_id=user.id,
            status=SESSION_STATUS_ACTIVE,
            last_activity_at=now,
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        logger.info(
            "Session reset | channel={} | external_user_id={} | new_session_id={}",
            channel, external_user_id, new_session.id
        )
        return new_session


async def get_or_create_session(channel: str, external_user_id: str) -> Session:
    """
    Возвращает активную сессию.
    Если активная сессия старше TTL — закрывает её и создаёт новую.

    ВАЖНО:
    - если активных сессий несколько (из-за гонки/старых багов) —
      оставляем самую свежую, остальные закрываем.
    - берём активную сессию всегда с сортировкой (последняя по last_activity_at).
    - используем SELECT ... FOR UPDATE, чтобы снизить вероятность гонки на PostgreSQL.
      На SQLite with_for_update() игнорируется, но логика закрытия дублей всё равно спасает.
    """
    user = await get_or_create_user(channel, external_user_id)

    now = datetime.utcnow()
    ttl_border = now - timedelta(hours=SESSION_TTL_HOURS)

    async with async_session() as db:
        # В идеале это работает на Postgres, на SQLite просто игнорируется.
        stmt = (
            select(Session)
            .options(load_only(Session.id, Session.status, Session.last_activity_at, Session.user_id))
            .where(Session.user_id == user.id, Session.status == SESSION_STATUS_ACTIVE)
            .order_by(Session.last_activity_at.desc(), Session.id.desc())
            .with_for_update()
        )
        res = await db.execute(stmt)
        active_sessions = list(res.scalars().all())

        # Если активных несколько — оставляем самую свежую, остальные закрываем
        active_session = active_sessions[0] if active_sessions else None
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
                user.id, active_session.id if active_session else None, extra_ids
            )

        # Если есть активная — проверяем TTL
        if active_session:
            if active_session.last_activity_at and active_session.last_activity_at < ttl_border:
                # Закрываем старую
                await db.execute(
                    update(Session)
                    .where(Session.id == active_session.id)
                    .values(status=SESSION_STATUS_CLOSED, last_activity_at=now)
                )

                # Создаём новую
                new_session = Session(
                    user_id=user.id,
                    status=SESSION_STATUS_ACTIVE,
                    last_activity_at=now,
                )
                db.add(new_session)
                await db.commit()
                await db.refresh(new_session)
                return new_session

            # Живая — touch
            await db.execute(
                update(Session)
                .where(Session.id == active_session.id)
                .values(last_activity_at=now)
            )
            await db.commit()

            # Обновим объект минимально
            active_session.last_activity_at = now
            return active_session

        # Активной нет — создаём
        new_session = Session(
            user_id=user.id,
            status=SESSION_STATUS_ACTIVE,
            last_activity_at=now,
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session
