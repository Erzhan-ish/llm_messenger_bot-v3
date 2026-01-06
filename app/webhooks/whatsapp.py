import asyncio
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.security.whatsapp_signature import verify_whatsapp_signature
from app.channels.whatsapp_adapter import WhatsAppAdapter
from app.processing.message_processor import process_message
from app.logging import logger

router = APIRouter(prefix="/webhook/whatsapp", tags=["webhooks-whatsapp"])


@router.get("")
async def whatsapp_verify(
    request: Request,
):
    """
    Meta verification:
    /webhook?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...
    Нужно вернуть hub.challenge, если verify_token совпал.
    :contentReference[oaicite:3]{index=3}
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN and challenge:
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("")
async def whatsapp_inbound(request: Request):
    """
    Приём webhook событий.
    Важно: быстро вернуть 200.
    Рекомендуется валидировать X-Hub-Signature-256.
    :contentReference[oaicite:4]{index=4}
    """
    raw = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_whatsapp_signature(raw_body=raw, header_value=signature):
        logger.warning("Invalid WhatsApp signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = await request.json()
    unified_messages = WhatsAppAdapter.extract_messages(payload)

    logger.info("WhatsApp inbound | messages={}", len(unified_messages))

    # Не блокируем webhook: запускаем обработку параллельно
    for um in unified_messages:
        asyncio.create_task(process_message(um))

    return Response(status_code=200)
