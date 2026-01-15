import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.logging import logger
from app.channels.whatsapp_adapter import WhatsAppAdapter
from app.security.whatsapp_signature import verify_whatsapp_signature

from app.jobs.enqueue import enqueue_inbound_message_job, enqueue_stt_job


router = APIRouter(tags=["whatsapp"])


@router.get("")
async def whatsapp_verify(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN and challenge:
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("")
async def whatsapp_inbound(request: Request):
    raw = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_whatsapp_signature(raw_body=raw, header_value=signature):
        logger.warning("Invalid WhatsApp signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = await request.json()
    unified_messages = WhatsAppAdapter.extract_messages(payload)

    logger.info("WhatsApp inbound | messages={}", len(unified_messages))

    # Важно быстро отдать 200 → enqueue в фоне
    for um in unified_messages:
        if um.message_type == "audio":
            asyncio.create_task(enqueue_stt_job(um))
        else:
            asyncio.create_task(enqueue_inbound_message_job(um))

    return {"ok": True}
