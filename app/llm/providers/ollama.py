# app/llm/providers/ollama.py
import httpx
from app.config import settings
from app.logging import logger


class OllamaProvider:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def generate(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,   # Низкая температура для уменьшения галлюцинаций
                "top_p": 0.9,
                "num_ctx": 2048,      # Ограничение контекста для ускорения обработки (снижает нагрузку на CPU/GPU)
                "num_predict": 250,   # Ограничение максимальной длины ответа (ускоряет генерацию, предотвращая зацикливание)
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
async def ask_ollama(messages: list[dict]) -> str:
    provider = OllamaProvider()
    return await provider.generate(messages)

