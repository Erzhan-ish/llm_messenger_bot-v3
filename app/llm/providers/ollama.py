# app/llm/providers/ollama.py
import httpx
from app.config import settings
from app.logging import logger

_OLLAMA_NUM_PREDICT_HARD_MAX = 2048


class OllamaProvider:
    def __init__(self, model: str | None = None):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL

    async def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/api/chat"

        requested = max_tokens if max_tokens is not None else settings.OLLAMA_RENDER_MAX_TOKENS
        cap_applied = requested > _OLLAMA_NUM_PREDICT_HARD_MAX
        num_predict = min(requested, _OLLAMA_NUM_PREDICT_HARD_MAX)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.55,
                "top_p": 0.90,
                "num_ctx": settings.OLLAMA_NUM_CTX,
                "num_predict": num_predict,
                "repeat_penalty": 1.15,
            }
        }

        logger.info(
            "Ollama request | model={} | num_predict={}{}",
            self.model, num_predict,
            f" | cap_reason=hard_max_{_OLLAMA_NUM_PREDICT_HARD_MAX}" if cap_applied else "",
        )

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                response = await client.post(url, json=payload)
        except Exception:
            logger.error("Ollama network error", exc_info=True)
            return ""

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(
                "Ollama HTTP error | status={} | body={}",
                response.status_code,
                response.text[:1000],
            )
            return ""

        data = response.json()
        content = (data.get("message") or {}).get("content") or ""
        return content


async def ask_ollama(messages: list[dict], model: str | None = None, max_tokens: int | None = None) -> str:
    provider = OllamaProvider(model=model)
    return await provider.generate(messages, max_tokens=max_tokens)
