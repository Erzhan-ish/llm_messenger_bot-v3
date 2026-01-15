# app/debug/whatsapp.py
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.logging import logger
from app.jobs.enqueue import enqueue_inbound_message_job
from app.channels.base import UnifiedMessage

debug_router_wh = APIRouter(prefix="/debug", tags=["debug"])


class DebugWhatsAppIn(BaseModel):
    external_user_id: str = Field(..., examples=["79990001122"])
    text: str = Field(..., examples=["Привет, это тест WhatsApp"])
    message_id: Optional[str] = Field(default=None, examples=["dbg-wa-1"])
    message_type: Literal["text", "audio"] = "text"

    # если хочешь тестировать stt-пайплайн без Meta:
    audio_path: Optional[str] = None
    media_id: Optional[str] = None
    mime_type: Optional[str] = None


@debug_router_wh.post("/whatsapp")
async def debug_whatsapp(payload: DebugWhatsAppIn):
    """
    Локальный debug endpoint:
    - создаёт UnifiedMessage как будто он пришёл из WhatsApp
    - enqueue в jobs
    - возвращает 200 и job_id
    """
    msg_id = payload.message_id or f"dbg-wa-{int(datetime.utcnow().timestamp())}"

    um = UnifiedMessage(
        channel="whatsapp",
        external_user_id=payload.external_user_id,
        message_id=msg_id,
        message_type=payload.message_type,
        text=payload.text if payload.message_type == "text" else None,
        audio_path=payload.audio_path,
        media_id=payload.media_id,
        mime_type=payload.mime_type,
        created_at=datetime.utcnow(),
    )

    job_id = await enqueue_inbound_message_job(um)

    logger.info(
        "DEBUG WhatsApp enqueued | job_id={} | external_user_id={} | msg_id={} | type={}",
        job_id,
        um.external_user_id,
        um.message_id,
        um.message_type,
    )

    return {"ok": True, "job_id": job_id, "message_id": um.message_id}
