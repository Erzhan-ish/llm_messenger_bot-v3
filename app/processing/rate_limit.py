from datetime import datetime, timedelta
from sqlalchemy import select, func

from app.storage.db import async_session
from app.storage.models import Message


class RateLimitExceeded(Exception):
    pass


async def check_rate_limit(
    channel: str,
    external_user_id: str,
    limit: int = 5,
    window_seconds: int = 10,
) -> None:
    """
    Проверяет, не превышен ли лимит сообщений
    """
    since = datetime.utcnow() - timedelta(seconds=window_seconds)

    async with async_session() as session:
        result = await session.execute(
            select(func.count(Message.id)).where(
                Message.channel == channel,
                Message.created_at >= since,
                Message.role == "user",
            )
        )
        count = result.scalar() or 0

        if count >= limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {count}/{limit} in {window_seconds}s"
            )
