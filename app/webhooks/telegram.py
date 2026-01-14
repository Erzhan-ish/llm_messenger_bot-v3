# app/webhooks/telegram.py
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.logging import logger
from app.channels.telegram_adapter import TelegramAdapter
from app.jobs.enqueue import enqueue_inbound_message_job

router = APIRouter(tags=["telegram"])


@router.post("")
async def telegram_webhook(request: Request):
    raw = await request.body()
    if not raw:
        logger.warning("Telegram webhook | empty body")
        raise HTTPException(status_code=400, detail="Empty body")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        ct = request.headers.get("content-type")
        logger.error(
            "Telegram webhook | invalid json | content-type={} | raw={}",
            ct,
            raw[:200],
        )
        raise HTTPException(status_code=400, detail="Invalid JSON")

    um = await TelegramAdapter.from_payload(payload)

    logger.info(
        "Telegram inbound | external_user_id={} | message_id={}",
        um.external_user_id,
        um.message_id,
    )

    # enqueue быстро, обработку делает worker
    asyncio.create_task(enqueue_inbound_message_job(um))

    return JSONResponse({"ok": True})
