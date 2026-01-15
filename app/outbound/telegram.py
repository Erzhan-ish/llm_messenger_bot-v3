import httpx
from app.config import settings
from app.logging import logger

BASE_URL = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}"


async def send_telegram(user_id: str, text: str):
    payload = {
        "chat_id": user_id,
        "text": text,
        "parse_mode": "HTML",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{BASE_URL}/sendMessage",
            json=payload,
        )

    if resp.status_code != 200:
        logger.error(
            "Telegram send failed | user_id={} | status={} | body={}",
            user_id,
            resp.status_code,
            resp.text,
        )
        raise RuntimeError("Telegram sendMessage failed")

    logger.info("Telegram message sent | user_id={}", user_id)
