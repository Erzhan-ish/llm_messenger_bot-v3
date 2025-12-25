from fastapi import APIRouter
from app.schemas.incoming import TgUpdate
from app.channels.telegram_adapter import TelegramAdapter
from app.logging import logger

router = APIRouter()


@router.post("/")
async def telegram_webhook(update: TgUpdate):
    if not update.message or not update.message.text:
        return {"ok": True}

    text = update.message.text.strip()
    user_id = str(update.message.from_.id)

    logger.info(
        "Telegram update | user_id={} | text={}",
        user_id,
        text,
    )

    # /start и deep-link
    if text.startswith("/start"):
        payload = text.replace("/start", "").strip() or None
        await TelegramAdapter.handle_start(
            user_id=user_id,
            payload=payload,
        )
        return {"ok": True}

    # обычные сообщения
    await TelegramAdapter.handle(update)
    return {"ok": True}
