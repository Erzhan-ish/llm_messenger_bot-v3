import httpx
from app.config import settings
from loguru import logger

async def send_telegram_message(user_id: str, text: str):
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            url,
            json={
                "chat_id": int(user_id),
                "text": text,
            }
        )

    if response.status_code != 200:
        logger.error(
            "Telegram send failed | status={} | body={}",
            response.status_code,
            response.text,
        )
    else:
        logger.info("Telegram message sent | user_id={}", user_id)

