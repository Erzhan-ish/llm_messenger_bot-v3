"""Fireworks AI provider — OpenAI-compatible chat completions."""
from __future__ import annotations

import asyncio
import httpx

from app.config import settings
from app.logging import logger


def _normalize_model(model: str | None) -> str:
    """Ensure the model name uses the full Fireworks accounts/ prefix."""
    m = (model or settings.FIREWORKS_MODEL or "llama-v3p3-70b-instruct").strip()
    if not m.startswith("accounts/"):
        m = f"accounts/fireworks/models/{m}"
    return m


def _chat_completions_url() -> str:
    base = (settings.FIREWORKS_BASE_URL or "https://api.fireworks.ai/inference/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class FireworksProvider:
    def __init__(self, model: str | None = None):
        self.model = _normalize_model(model)
        self.api_key = settings.FIREWORKS_API_KEY or ""

    async def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        url = _chat_completions_url()
        limit = max_tokens if max_tokens is not None else 1000

        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": limit,
            "temperature": 0.55,
            "top_p": 0.90,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "Fireworks request | model={} | msgs={} | max_tokens={}",
            self.model, len(messages), limit,
        )

        response = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                    response = await client.post(url, json=payload, headers=headers)
            except Exception:
                logger.error("Fireworks network error (attempt {})", attempt + 1, exc_info=True)
                if attempt == 0:
                    await asyncio.sleep(0.8)
                    continue
                raise

            if response.status_code in (502, 503, 504):
                if attempt == 0:
                    logger.warning(
                        "Fireworks {} — retrying | model={}",
                        response.status_code, self.model,
                    )
                    await asyncio.sleep(0.8)
                    continue
                logger.error(
                    "Fireworks {} on 2nd attempt — giving up | model={}",
                    response.status_code, self.model,
                )
                return ""

            break

        logger.info("Fireworks response | status={}", response.status_code)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(
                "Fireworks HTTP error | status={} | body={}",
                response.status_code, response.text[:300],
            )
            return ""

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        content = (content or "").strip()

        if not content:
            logger.warning(
                "Fireworks empty content | status={} | finish_reason={} | usage={}",
                response.status_code,
                choice.get("finish_reason"),
                data.get("usage"),
            )
            return ""

        return content


async def ask_fireworks(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = FireworksProvider(model=model)
    return await provider.generate(messages, max_tokens=max_tokens)
