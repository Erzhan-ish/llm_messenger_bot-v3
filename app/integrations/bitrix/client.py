import httpx
from app.config import settings


class BitrixClient:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.rstrip("/")

    async def call(self, method: str, params):
        url = f"{self.webhook_url}/{method}.json"

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=params)
            if r.status_code != 200:
                print(r.text)
                raise RuntimeError(f"Bitrix error: {r.text}")
            data = r.json()

        if "error" in data:
            raise RuntimeError(f"Bitrix error: {data}")

        return data.get("result")


bitrix = BitrixClient(settings.BITRIX_WEBHOOK_URL)
