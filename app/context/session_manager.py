from datetime import datetime, timedelta
from app.config import settings
from app.logging import logger
from app.storage.repositories.users_repo import get_or_create_user
from app.storage.repositories.sessions_repo import (
    get_last_session,
    create_session,
    touch_session,
    is_session_expired,
    update_session_payload,
    close_active_session,
    create_new_session,
)


async def reset_session(channel: str, external_user_id: str):
    user = await get_or_create_user(channel, external_user_id)

    await close_active_session(user.id)

    return await create_new_session(user.id)


async def get_or_create_session(
    channel: str,
    external_user_id: str,
    start_payload: str | None = None,
):
    user = await get_or_create_user(channel, external_user_id)

    session = await get_last_session(user.id)

    if not session or is_session_expired(session):
        session = await create_session(user.id)

    if start_payload and not session.start_payload:
        session.start_payload = start_payload
        await update_session_payload(session.id, start_payload)

    await touch_session(session.id)
    return session
