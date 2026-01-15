from sqlalchemy import select
from app.storage.db import async_session
from app.storage.models import Message


async def is_duplicate_message(channel: str, external_message_id: str) -> bool:
    async with async_session() as session:
        result = await session.execute(
            select(Message.id).where(
                Message.channel == channel,
                Message.external_message_id == external_message_id,
            )
        )
        return result.scalar_one_or_none() is not None