from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.models import Message
from app.storage.db import async_session


async def save_message(
    session_id: int,
    role: str,
    text: str | None,
    channel: str,
    external_message_id: str | None = None,
) -> Message:
    async with async_session() as session:
        message = Message(
            session_id=session_id,
            role=role,
            text=text,
            channel=channel,
            external_message_id=external_message_id,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message



async def get_last_messages(
    session_id: int,
    limit: int,
) -> list[Message]:
    async with async_session() as session:
        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))

