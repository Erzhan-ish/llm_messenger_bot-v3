from __future__ import annotations

import time
from typing import Any, Optional

from app.config import settings
from app.logging import logger

from app.llm.providers.stub import ask_stub
from app.llm.providers.ollama import ask_ollama


def _timeweb_model(model: str | None) -> str | None:
    """Map Ollama model names to Timeweb equivalents when switching providers."""
    if not model:
        return settings.TIMEWEB_AI_MODEL or None
    if model == settings.OLLAMA_ANALYZER_MODEL:
        return settings.TIMEWEB_AI_ANALYZER_MODEL or settings.TIMEWEB_AI_MODEL or None
    return settings.TIMEWEB_AI_MODEL or None


async def ask_llm(
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int | None = None,
    trace_ctx: Optional[dict] = None,
) -> str:
    """Call the configured LLM provider, optionally writing a full trace record."""
    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").strip().lower()
    trace_ctx = trace_ctx or {}
    trace_id = trace_ctx.get("trace_id", "")

    # Log every call with trace_id so normal logs link to JSONL traces
    _model = model or (settings.TIMEWEB_AI_MODEL if provider == "timeweb" else settings.OLLAMA_MODEL)
    logger.info(
        "LLMCall | trace_id={} | provider={} | model={} | phase={} | msgs={}",
        trace_id, provider, _model, trace_ctx.get("phase", "unknown"), len(messages),
    )

    t0 = time.monotonic()
    error_str: Optional[str] = None
    raw = ""

    try:
        if provider == "ollama":
            raw = await ask_ollama(messages, model=model, max_tokens=max_tokens)
        elif provider == "timeweb":
            from app.llm.providers.timeweb import ask_timeweb
            raw = await ask_timeweb(messages, model=_timeweb_model(model), max_tokens=max_tokens)
        else:
            raw = await ask_stub(messages)
    except Exception as exc:
        error_str = str(exc)
        raise
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Write JSONL trace if enabled
        try:
            from app.services.llm_trace import get_tracer
            tracer = get_tracer()
            if tracer.enabled or not tracer._loaded:
                tracer._load()
            if tracer.enabled:
                from app.services.conversation_brain import load_brain_prompt
                try:
                    _, sp_path, sp_hash = load_brain_prompt()
                except Exception:
                    sp_path, sp_hash = None, None

                tracer.write(
                    trace_id=trace_id or "unknown",
                    session_id=trace_ctx.get("session_id"),
                    job_id=trace_ctx.get("job_id"),
                    channel=trace_ctx.get("channel"),
                    external_user_id=str(trace_ctx.get("external_user_id") or ""),
                    phase=trace_ctx.get("phase", "unknown"),
                    provider=provider,
                    model=_model,
                    temperature=trace_ctx.get("temperature"),
                    max_tokens=max_tokens,
                    system_prompt_path=sp_path,
                    system_prompt_hash=sp_hash,
                    messages=messages,
                    fact_pack=trace_ctx.get("fact_pack"),
                    rag=trace_ctx.get("rag"),
                    dialog_policy=trace_ctx.get("dialog_policy"),
                    raw_response=raw,
                    parsed_response=None,
                    latency_ms=latency_ms,
                    error=error_str,
                )
        except Exception:
            pass  # tracing must never crash the main flow

    return raw