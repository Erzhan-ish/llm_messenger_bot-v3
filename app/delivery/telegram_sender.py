import httpx
from app.config import settings

async def send_telegram_message(user_id: str, text: str):
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "chat_id": user_id,
            "text": text
        })
