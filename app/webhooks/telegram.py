from fastapi import APIRouter
from app.schemas.incoming import TgUpdate
from app.channels.telegram_adapter import TelegramAdapter

router = APIRouter()

@router.post("/")
async def telegram_webhook(update: TgUpdate):
    if not update.message or not update.message.text:
        return {"status": "ignored"}

    adapter = TelegramAdapter()
    await adapter.handle(update)

    return {"status": "ok"}
