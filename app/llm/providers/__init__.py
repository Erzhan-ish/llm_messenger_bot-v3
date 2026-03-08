from __future__ import annotations

from typing import Any

from app.config import settings
from app.logging import logger

from app.llm.providers.stub import ask_stub
from app.llm.providers.ollama import ask_ollama


async def ask_llm(messages: list[dict[str, Any]], model: str | None = None) -> str:
    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").strip().lower()

    if provider == "ollama":
        return await ask_ollama(messages, model=model)

    # default = stub
    return await ask_stub(messages)