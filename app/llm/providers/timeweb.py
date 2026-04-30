# app/llm/providers/timeweb.py
import httpx
from app.config import settings
from app.logging import logger


class TimewebProvider:
    def __init__(self, model: str | None = None):
        self.base_url = (settings.TIMEWEB_AI_BASE_URL or "").rstrip("/")
        self.model = model or settings.TIMEWEB_AI_MODEL
        self.token = settings.TIMEWEB_AI_TOKEN

    async def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/chat/completions"

        # OpenAI-compatible APIs require the last message to be from "user".
        # Some callers send only a system message — append a minimal user turn.
        normalized = list(messages)
        if not normalized or normalized[-1].get("role") != "user":
            normalized.append({"role": "user", "content": "Ответь"})

        payload = {
            "model": self.model,
            "messages": normalized,
            "max_tokens": max_tokens if max_tokens is not None else 450,
            "temperature": 0.55,
            "top_p": 0.90,
        }

        token = self.token.strip()
        auth_value = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        headers = {
            "Authorization": auth_value,
            "Content-Type": "application/json",
        }

        logger.info("Sending request to Timeweb AI | model={}", self.model)

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except Exception:
            logger.error("Timeweb AI request failed", exc_info=True)
            raise

        data = response.json()
        return data["choices"][0]["message"]["content"]


async def ask_timeweb(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = TimewebProvider(model=model)
    return await provider.generate(messages, max_tokens=max_tokens)
