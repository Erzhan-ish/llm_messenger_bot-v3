import asyncio
import json
from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.security.whatsapp_signature import verify_whatsapp_signature
from app.channels.whatsapp_adapter import WhatsAppAdapter
from app.processing.message_processor import process_message
from app.logging import logger

router = APIRouter(prefix="/webhook/whatsapp", tags=["webhooks-whatsapp"])


@router.get("")
async def whatsapp_verify(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN and challenge:
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=challenge)

    return Response(status_code=200)


@router.post("")
async def whatsapp_inbound(request: Request):
    raw = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_whatsapp_signature(raw_body=raw, header_value=signature):
        logger.warning("Invalid WhatsApp signature")
        return Response(status_code=200)

    payload = json.loads(raw)
    messages = WhatsAppAdapter.extract_messages(payload)

    logger.info("WhatsApp inbound | messages={}", len(messages))

    for msg in messages:
        asyncio.create_task(process_message(msg))

    return Response(status_code=200)
