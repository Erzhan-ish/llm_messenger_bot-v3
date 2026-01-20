from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.storage.db import async_session
from app.storage.models import Message, Session, User


async def is_duplicate_message(channel: str, external_user_id: str, external_message_id: str) -> bool:
    if not external_message_id:
        return False

    async with async_session() as session:
        q = (
            select(Message.id)
            .join(Session, Session.id == Message.session_id)
            .join(User, User.id == Session.user_id)
            .where(
                Message.channel == channel,
                Message.external_message_id == external_message_id,
                User.external_user_id == external_user_id,
                User.channel == channel,
            )
            .limit(1)
        )
        res = await session.execute(q)
        return res.scalar_one_or_none() is not None
