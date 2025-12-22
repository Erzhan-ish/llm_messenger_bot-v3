from sqlalchemy import select
from app.storage.db import AsyncSessionLocal
from app.storage.models import Message
from app.config import settings
from app.logging import logger


async def build_context(session_id: int) -> str | None:
    """
    Возвращает текстовый контекст диалога для LLM
    (последние N сообщений)
    """

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(settings.CONTEXT_MESSAGES_LIMIT)
        )

        messages = list(reversed(result.scalars().all()))

    if not messages:
        logger.debug(
            "Context empty | session_id={}",
            session_id,
        )
        return None

    context_lines: list[str] = []

    for msg in messages:
        if not msg.text:
            continue

        role = msg.role.upper()

        # ВАЖНО: человекочитаемый формат
        context_lines.append(f"{role}: {msg.text}")

    context = "\n".join(context_lines)

    logger.debug(
        "Context built | session_id={} | messages_count={}",
        session_id,
        len(context_lines),
    )

    return context
