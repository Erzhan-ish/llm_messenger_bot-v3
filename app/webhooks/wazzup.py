from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.channels.base import UnifiedMessage
from app.jobs.enqueue import enqueue_inbound_message_job
from app.logging import logger
from app.storage.repositories.users_repo import get_or_create_user, update_wazzup_meta


router = APIRouter(tags=["wazzup"])


@router.post("")
async def wazzup_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raw = await request.body()
        logger.error("Wazzup webhook | invalid json | raw={}", raw[:500])
        raise HTTPException(status_code=400, detail="Invalid JSON")

    items = payload.get("messagesAndStatuses")
    if items is None:
        items = payload.get("messages")
    items = items or []

    logger.info(
        "Wazzup webhook received | keys={} | items={}",
        list(payload.keys()),
        len(items),
    )

    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="Invalid payload")

    enqueued = 0

    for item in items:
        logger.info(
            "Wazzup item | type={} | status={} | isEcho={} | chatType={} | chatId={} | messageId={}",
            item.get("type"),
            item.get("status"),
            item.get("isEcho"),
            item.get("chatType"),
            item.get("chatId"),
            item.get("messageId"),
        )
        item_type = item.get("type")
        if item_type not in ("message", "text", "image", "audio", "file", "document"):
            continue
        status = item.get("status")
        if status and status != "inbound":
            continue
        if item.get("isEcho") is True:
            continue

        chat_id = item.get("chatId")
        message_id = item.get("messageId")
        chat_type = item.get("chatType")
        channel_id = item.get("channelId")
        text = item.get("text")

        if not chat_id or not message_id:
            continue
        if not text:
            # Пока обрабатываем только текст
            continue

        user = await get_or_create_user("wazzup", str(chat_id))
        await update_wazzup_meta(user.id, channel_id, chat_type)
        logger.info(
            "Wazzup meta saved | user_id={} | channel_id={} | chat_type={}",
            user.id,
            channel_id,
            chat_type,
        )

        um = UnifiedMessage(
            channel="wazzup",
            external_user_id=str(chat_id),
            message_id=str(message_id),
            message_type="text",
            text=text,
        )
        um.chat_type = chat_type
        um.wazzup_channel_id = channel_id

        logger.info(
            "Wazzup inbound | chat_type={} | external_user_id={} | message_id={}",
            chat_type,
            chat_id,
            message_id,
        )

        asyncio.create_task(enqueue_inbound_message_job(um))
        enqueued += 1

    return JSONResponse({"ok": True, "enqueued": enqueued})
