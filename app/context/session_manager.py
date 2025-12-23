from app.storage.repositories.sessions_repo import (
    close_active_session,
    create_new_session,
)
from datetime import datetime, timedelta
from sqlalchemy import select, update
from app.storage.db import async_session
from app.storage.models import Session
from app.storage.repositories.users_repo import get_or_create_user


async def reset_session(channel: str, external_user_id: str):
    user = await get_or_create_user(channel, external_user_id)

    await close_active_session(user.id)

    return await create_new_session(user.id)


SESSION_TTL_HOURS = 24


async def get_or_create_session(
    channel: str,
    external_user_id: str,
) -> Session:
    """
    Возвращает активную сессию.
    Если активная сессия старше TTL — закрывает её и создаёт новую.
    """

    user = await get_or_create_user(channel, external_user_id)

    now = datetime.utcnow()
    ttl_border = now - timedelta(hours=SESSION_TTL_HOURS)

    async with async_session() as session:
        result = await session.execute(
            select(Session).where(
                Session.user_id == user.id,
                Session.status == "active",
            )
        )
        active_session = result.scalar_one_or_none()

        # Если есть активная сессия
        if active_session:
            # Проверяем TTL
            if active_session.last_activity_at < ttl_border:
                # Закрываем старую
                await session.execute(
                    update(Session)
                    .where(Session.id == active_session.id)
                    .values(
                        status="closed",
                        last_activity_at=now,
                    )
                )
                await session.commit()

                # Создаём новую
                new_session = Session(
                    user_id=user.id,
                    status="active",
                    last_activity_at=now,
                )
                session.add(new_session)
                await session.commit()
                await session.refresh(new_session)
                return new_session

            # Сессия живая — обновляем last_activity
            active_session.last_activity_at = now
            await session.commit()
            return active_session

        # Активной сессии нет — создаём
        new_session = Session(
            user_id=user.id,
            status="active",
            last_activity_at=now,
        )
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session
