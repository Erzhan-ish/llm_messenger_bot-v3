# app/llm/providers/timeweb.py
import asyncio
import httpx
from app.config import settings
from app.logging import logger


def _is_gpt5_model(model: str | None) -> bool:
    return bool(model and model.lower().startswith("gpt-5"))


class TimewebProvider:
    def __init__(self, model: str | None = None):
        self.base_url = (settings.TIMEWEB_AI_BASE_URL or "").rstrip("/")
        self.model = model or settings.TIMEWEB_AI_MODEL
        self.token = settings.TIMEWEB_AI_TOKEN

    async def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/chat/completions"

        normalized = list(messages)
        if not normalized or normalized[-1].get("role") != "user":
            normalized.append({"role": "user", "content": "Ответь"})

        limit = max_tokens if max_tokens is not None else 450
        is_gpt5 = _is_gpt5_model(self.model)
        token_field = "max_completion_tokens" if is_gpt5 else "max_tokens"

        payload: dict = {
            "model": self.model,
            "messages": normalized,
            "stream": False,
        }
        if is_gpt5:
            payload["max_completion_tokens"] = limit
            reasoning_effort = (settings.TIMEWEB_REASONING_EFFORT or "").strip()
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
        else:
            payload["max_tokens"] = limit
            payload["temperature"] = 0.55
            payload["top_p"] = 0.90

        token = self.token.strip()
        auth_value = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        headers = {
            "Authorization": auth_value,
            "Content-Type": "application/json",
        }

        logger.info(
            "Timeweb request | model={} | msgs={} | {}={}",
            self.model, len(normalized), token_field, limit,
        )

        response = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                    response = await client.post(url, json=payload, headers=headers)
            except Exception:
                logger.error("Timeweb network error (attempt {})", attempt + 1, exc_info=True)
                if attempt == 0:
                    await asyncio.sleep(0.8)
                    continue
                raise

            # reasoning_effort not supported by this endpoint — strip and inline-retry
            if response.status_code == 400 and payload.pop("reasoning_effort", None):
                logger.warning("Timeweb 400 — retrying without reasoning_effort | model={}", self.model)
                try:
                    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as _c:
                        response = await _c.post(url, json=payload, headers=headers)
                except Exception:
                    logger.error("Timeweb reasoning_effort retry failed", exc_info=True)
                    return ""

            if response.status_code in (502, 503, 504):
                if attempt == 0:
                    logger.warning(
                        "Timeweb {} — retrying | model={}",
                        response.status_code, self.model,
                    )
                    await asyncio.sleep(0.8)
                    continue
                logger.error(
                    "Timeweb {} on 2nd attempt — giving up | model={}",
                    response.status_code, self.model,
                )
                return ""

            break  # not a transient gateway error

        logger.info("Timeweb response | status={}", response.status_code)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error("Timeweb HTTP error | status={}", response.status_code)
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
                "Timeweb empty content | status={} | finish_reason={} | usage={} | message={!r} | raw={}",
                response.status_code,
                choice.get("finish_reason"),
                data.get("usage"),
                message,
                data,
            )
            return ""

        return content


async def ask_timeweb(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = TimewebProvider(model=model)
    return await provider.generate(messages, max_tokens=max_tokens)
