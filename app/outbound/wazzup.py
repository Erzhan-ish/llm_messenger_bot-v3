from __future__ import annotations

import httpx

from app.config import settings
from app.logging import logger
from app.storage.repositories.users_repo import get_user


async def send_wazzup(external_user_id: str, text: str) -> None:
    user = await get_user("wazzup", external_user_id)
    if not user or not user.wazzup_channel_id or not user.wazzup_chat_type:
        logger.warning(
            "Wazzup outbound missing meta | external_user_id={}",
            external_user_id,
        )
        return

    url = f"{settings.WAZZUP_API_URL.rstrip('/')}/message"
    headers = {"Authorization": f"Bearer {settings.WAZZUP_API_KEY}"}
    payload = {
        "channelId": user.wazzup_channel_id,
        "chatType": user.wazzup_chat_type,
        "chatId": external_user_id,
        "text": text,
    }

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("Wazzup send failed | status={} | body='{}'", r.status_code, r.text)
            r.raise_for_status()


# ✅ НОВОЕ — typing
async def send_wazzup_typing(external_user_id: str) -> None:
    """
    Показывает 'печатает...' через Wazzup (если канал поддерживает).
    Никогда не ломает основной поток.
    """
    user = await get_user("wazzup", external_user_id)
    if not user or not user.wazzup_channel_id or not user.wazzup_chat_type:
        return

    url = f"{settings.WAZZUP_API_URL.rstrip('/')}/typing"
    headers = {"Authorization": f"Bearer {settings.WAZZUP_API_KEY}"}

    payload = {
        "channelId": user.wazzup_channel_id,
        "chatType": user.wazzup_chat_type,
        "chatId": external_user_id,
        "typing": True,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            r = await client.post(url, json=payload, headers=headers)

        if r.status_code >= 400:
            logger.debug(
                "Wazzup typing not supported | status={} | body='{}'",
                r.status_code,
                r.text,
            )
    except Exception:
        # typing не должен ломать диалог
        logger.debug("Wazzup typing silently ignored")