from __future__ import annotations

from typing import Any

from app.config import settings
from app.logging import logger

from app.llm.providers.stub import ask_stub
from app.llm.providers.ollama import ask_ollama


def _timeweb_model(model: str | None) -> str | None:
    """Map Ollama model names to Timeweb equivalents when switching providers."""
    if not model:
        return settings.TIMEWEB_AI_MODEL or None
    # If caller passed the Ollama analyzer model, route to Timeweb analyzer model
    if model == settings.OLLAMA_ANALYZER_MODEL:
        return settings.TIMEWEB_AI_ANALYZER_MODEL or settings.TIMEWEB_AI_MODEL or None
    # Any other explicit model falls back to the main Timeweb model
    return settings.TIMEWEB_AI_MODEL or None


async def ask_llm(
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").strip().lower()

    if provider == "ollama":
        return await ask_ollama(messages, model=model, max_tokens=max_tokens)

    if provider == "timeweb":
        from app.llm.providers.timeweb import ask_timeweb
        return await ask_timeweb(messages, model=_timeweb_model(model), max_tokens=max_tokens)

    # default = stub
    return await ask_stub(messages)