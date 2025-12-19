from datetime import datetime, timedelta
from sqlalchemy import select
from app.storage.db import AsyncSessionLocal
from app.storage.models import Session
from app.config import settings
from app.logging import logger


async def get_or_create_session(user_id: int) -> Session:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.last_activity_at.desc())
        )
        s = result.scalars().first()

        if s:
            if datetime.utcnow() - s.last_activity_at < timedelta(
                hours=settings.SESSION_TTL_HOURS
            ):
                s.last_activity_at = datetime.utcnow()
                await session.commit()
                return s

            logger.info("Session expired | session_id={}", s.id)

        new_session = Session(user_id=user_id)
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session
