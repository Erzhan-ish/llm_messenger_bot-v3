import asyncio
import json
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.logging import logger
from app.security.whatsapp_signature import verify_whatsapp_signature
from app.channels.whatsapp_adapter import WhatsAppAdapter
from app.processing.message_processor import process_message
from app.services.whatsapp_media import download_whatsapp_media


router = APIRouter(prefix="/webhook/whatsapp")


@router.get("")
async def whatsapp_verify(request: Request):
    """
    Meta Webhook verification.
    Нужно вернуть hub.challenge, если verify_token совпал.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN and challenge:
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("WhatsApp webhook verification failed")
    raise HTTPException(status_code=403, detail="Webhook verification failed")


async def _handle_whatsapp_message(um):
    """
    Для audio: скачиваем media -> проставляем audio_path -> process_message()
    Для text: сразу process_message()
    """
    try:
        if um.message_type  == "audio" and um.media_id:
            path, mime_type = await download_whatsapp_media(
                media_id=um.media_id,
                user_id=um.user_id,
                message_id=um.message_id,
            )
            um.audio_path = path
            um.mime_type = mime_type

        await process_message(um)

    except Exception:
        logger.exception("WhatsApp message processing failed | user_id={} | message_id={}", um.user_id, um.message_id)


@router.post("")
async def whatsapp_inbound(request: Request):
    """
    Приём inbound сообщений от WhatsApp (Meta Cloud API).
    Обязательно:
    - проверить X-Hub-Signature-256
    - быстро вернуть 200
    """
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_whatsapp_signature(
        raw_body=raw_body,
        header_value=signature,
    ):
        logger.warning("Invalid WhatsApp signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = json.loads(raw_body)
    messages = WhatsAppAdapter.extract_messages(payload)

    logger.info("WhatsApp inbound | messages={}", len(messages))

    # НЕ блокируем webhook
    for msg in messages:
        asyncio.create_task(_handle_whatsapp_message(msg))

    return Response(status_code=200)
