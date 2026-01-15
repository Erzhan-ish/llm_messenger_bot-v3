# app/webhooks/debug
from fastapi import APIRouter
from app.channels.base import UnifiedMessage
from app.jobs.enqueue import enqueue_inbound_message_job
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime

debug_router_tg = APIRouter(prefix="/debug", tags=["debug"])

class DebugTelegramIn(BaseModel):
    user_id: str
    text: str
    message_id: str | None = None  # можно передать вручную

@debug_router_tg.post("/telegram")
async def debug_telegram(inp: DebugTelegramIn):
    msg_id = inp.message_id or f"dbg-tg-{uuid4().hex}"

    um = UnifiedMessage(
        channel="telegram",
        external_user_id=inp.user_id,
        message_id=msg_id,
        message_type="text",
        text=inp.text,
        created_at=datetime.utcnow(),
    )
    job_id = await enqueue_inbound_message_job(um)
    return {"ok": True, "job_id": job_id, "message_id": msg_id}
