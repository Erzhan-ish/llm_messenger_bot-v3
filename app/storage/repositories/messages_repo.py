from datetime import datetime, timedelta
from sqlalchemy import select

from app.storage.db import async_session
from app.storage.models import Message, Session

async def get_followup_candidates(
    hours: int = 24,
):
    """
    Возвращает bot-сообщения, на которые не ответили
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with async_session() as session:
        result = await session.execute(
            select(Message)
            .join(Session)
            .where(
                Message.role == "bot",
                Message.followup_sent.is_(False),
                Message.created_at <= since,
                Session.status == "active",
            )
        )
        return result.scalars().all()


async def mark_followup_sent(message_id: int):
    async with async_session() as session:
        msg = await session.get(Message, message_id)
        if msg:
            msg.followup_sent = True
            await session.commit()


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

async def get_messages_by_session(session_id: int) -> list[dict]:
    async with async_session() as session:
        res = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        messages = res.scalars().all()

    return [
        {
            "role": m.role,
            "text": m.text,
            "created_at": m.created_at,
        }
        for m in messages
    ]