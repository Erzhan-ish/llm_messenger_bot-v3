"""Centralized LLM call tracer — writes JSONL for every LLM call.

Environment flags (set in .env or environment):
  LLM_TRACE_ENABLED=true          — master switch (default false)
  LLM_TRACE_FULL_PROMPT=true      — log full messages[] array (default false → metadata only)
  LLM_TRACE_DIR=logs/llm_traces   — output directory
  LLM_TRACE_REDACT=false          — mask phone/INN/email
  LLM_TRACE_INCLUDE_RESPONSE=true — include raw_response and parsed_response

Output: logs/llm_traces/YYYY-MM-DD.jsonl
Each line is one JSON record per LLM call.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# PII masking (applied when LLM_TRACE_REDACT=true)
# ---------------------------------------------------------------------------

_PII_RE = re.compile(
    r"(\+?7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"   # RU phone
    r"|\b\d{10,12}\b"                                                     # INN / phone digits
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",              # email
    re.U,
)


def _redact_str(text: str) -> str:
    return _PII_RE.sub("[REDACTED]", text or "")


def _redact_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": m.get("role", ""), "content": _redact_str(str(m.get("content", "")))}
        for m in (messages or [])
    ]


# ---------------------------------------------------------------------------
# trace_id helper
# ---------------------------------------------------------------------------

def make_trace_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"llm_{ts}_{uid}"


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class LLMTracer:
    """Thread-safe JSONL tracer. Configured lazily on first write."""

    def __init__(self) -> None:
        self._loaded = False
        self.enabled = False
        self.full_prompt = False
        self.trace_dir = "logs/llm_traces"
        self.do_redact = False
        self.include_response = True

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from app.config import settings
            self.enabled = bool(getattr(settings, "LLM_TRACE_ENABLED", False))
            self.full_prompt = bool(getattr(settings, "LLM_TRACE_FULL_PROMPT", False))
            self.trace_dir = str(getattr(settings, "LLM_TRACE_DIR", "logs/llm_traces"))
            self.do_redact = bool(getattr(settings, "LLM_TRACE_REDACT", False))
            self.include_response = bool(getattr(settings, "LLM_TRACE_INCLUDE_RESPONSE", True))
        except Exception:
            pass

    def _path(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        p = Path(self.trace_dir) / f"{date_str}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(
        self,
        *,
        trace_id: str,
        session_id: Optional[int] = None,
        job_id: Optional[int] = None,
        channel: Optional[str] = None,
        external_user_id: Optional[str] = None,
        phase: str = "brain",
        provider: str = "",
        model: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt_path: Optional[str] = None,
        system_prompt_hash: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        fact_pack: Optional[dict] = None,
        rag: Optional[dict] = None,
        dialog_policy: Optional[dict] = None,
        raw_response: Optional[str] = None,
        parsed_response: Optional[dict] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self._load()
        if not self.enabled:
            return

        ts = datetime.now(tz=timezone.utc).isoformat()
        record: dict[str, Any] = {
            "ts": ts,
            "trace_id": trace_id,
            "session_id": session_id,
            "job_id": job_id,
            "channel": channel,
            "external_user_id": external_user_id,
            "phase": phase,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt_path": system_prompt_path,
            "system_prompt_hash": system_prompt_hash,
            "latency_ms": latency_ms,
            "error": error,
        }

        if self.full_prompt and messages:
            msgs = _redact_messages(messages) if self.do_redact else list(messages)
            record["messages"] = msgs
        else:
            record["messages_count"] = len(messages or [])
            record["messages_total_chars"] = sum(
                len(str(m.get("content", ""))) for m in (messages or [])
            )

        if fact_pack:
            try:
                record["fact_pack_keys"] = sorted(fact_pack.keys())
                active_scen = fact_pack.get("_active_scenario")
                if active_scen:
                    record["active_scenario"] = active_scen
            except Exception:
                pass

        if rag:
            record["rag"] = rag

        if dialog_policy:
            record["dialog_policy"] = dialog_policy

        if self.include_response:
            resp = _redact_str(raw_response or "") if self.do_redact else raw_response
            record["raw_response"] = resp
            record["parsed_response"] = parsed_response

        try:
            with open(self._path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str))
                f.write("\n")
        except Exception:
            pass  # tracing must never crash the main flow

    def read_recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent trace records from today's JSONL file."""
        self._load()
        try:
            path = self._path()
            if not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()
            records = []
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
                if len(records) >= limit:
                    break
            return list(reversed(records))
        except Exception:
            return []

    def get_by_id(self, trace_id: str) -> Optional[dict]:
        """Find a single trace record by trace_id."""
        self._load()
        try:
            path = self._path()
            if not path.exists():
                return None
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("trace_id") == trace_id:
                        return rec
                except Exception:
                    pass
        except Exception:
            pass
        return None


# Singleton
_tracer = LLMTracer()


def get_tracer() -> LLMTracer:
    return _tracer
