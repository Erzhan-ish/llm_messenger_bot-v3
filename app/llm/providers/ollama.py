# app/llm/providers/ollama.py
import httpx
from app.config import settings
from app.logging import logger


class OllamaProvider:
    def __init__(self, model: str | None = None):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL

    async def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.55,
                "top_p": 0.90,
                "num_ctx": 4096,
                "num_predict": max_tokens if max_tokens is not None else 450,
                "repeat_penalty": 1.15,
            }
        }

        logger.info("Sending request to Ollama | model={}", self.model)

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except Exception:
            logger.error("Ollama request failed", exc_info=True)
            raise

        data = response.json()
        return data["message"]["content"]


# === единая точка вызова ===
async def ask_ollama(messages: list[dict], model: str | None = None, max_tokens: int | None = None) -> str:
    provider = OllamaProvider(model=model)
    return await provider.generate(messages, max_tokens=max_tokens)

