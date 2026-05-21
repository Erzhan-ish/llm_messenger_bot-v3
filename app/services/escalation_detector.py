# app/services/escalation_detector.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, TypedDict, Literal

from app.llm.providers import ask_llm
from app.config import settings, llm_token_budget as _budget


EscReason = Literal[
    "pricing",
    "callback",
    "ready_to_open",
    "documents",
    "complex_case",
    "unknown_kb",
    "angry",
    "human_request",
    "other",
]

NextStep = Literal["none", "ask_clarify", "handoff_manager"]

ALLOWED_REASONS: set[str] = {
    "pricing",
    "callback",
    "ready_to_open",
    "documents",
    "complex_case",
    "unknown_kb",
    "angry",
    "human_request",
    "other",
}

ALLOWED_NEXT: set[str] = {"none", "ask_clarify", "handoff_manager"}

ALLOWED_NEEDS = {
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "CONDITIONS",
    "DOCUMENTS",
    "CONSULTATION",
    "SUPPORT",
    "UNKNOWN",
}

_ESCALATION_PROMPT_CACHE: Optional[str] = None

_FALLBACK_ESCALATION_PROMPT = (
    "Ты — классификатор эскалации. Реши по диалогу, нужно ли подключить менеджера.\n"
    "Верни строго JSON: {\"escalate\": true|false, \"reason\": \"other\", "
    "\"interest_score\": 0, \"confidence\": 0.0, \"next_step\": \"none\", "
    "\"client_need\": \"UNKNOWN\", \"reasons\": []}"
)


def _load_escalation_prompt() -> str:
    global _ESCALATION_PROMPT_CACHE
    if _ESCALATION_PROMPT_CACHE is not None:
        return _ESCALATION_PROMPT_CACHE
    prompt_path = Path(settings.ESCALATION_PROMPT_PATH)
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).resolve().parents[2] / prompt_path
    try:
        _ESCALATION_PROMPT_CACHE = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        _ESCALATION_PROMPT_CACHE = _FALLBACK_ESCALATION_PROMPT
    return _ESCALATION_PROMPT_CACHE


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class EscalationSignal(TypedDict):
    escalate: bool
    reason: str
    interest_score: int
    confidence: float
    next_step: str
    client_need: str
    reasons: list[str]


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _clamp_int(v: object, lo: int, hi: int, default: int) -> int:
    try:
        x = int(float(v))  # type: ignore[arg-type]
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
    except Exception:
        return default


def _clamp_float(v: object, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
    except Exception:
        return default


def _norm_need(v: Optional[str]) -> str:
    if not v:
        return "UNKNOWN"
    n = str(v).strip().upper().replace(" ", "_").replace("-", "_")
    return n if n in ALLOWED_NEEDS else "UNKNOWN"


def _norm_reason(v: Optional[str]) -> str:
    if not v:
        return "other"
    r = str(v).strip().lower()
    return r if r in ALLOWED_REASONS else "other"


def _norm_next(v: Optional[str]) -> str:
    if not v:
        return "none"
    n = str(v).strip().lower()
    return n if n in ALLOWED_NEXT else "none"


async def detect_escalation_signal(
    dialog_text: str,
    *,
    had_unknown_kb: bool = False,
) -> EscalationSignal:
    user_payload = dialog_text or ""
    if had_unknown_kb:
        user_payload += "\n\n[CONTEXT] had_unknown_kb=true (в диалоге были вопросы без ответа из базы знаний)."

    messages = [
        {"role": "system", "content": _load_escalation_prompt()},
        {"role": "user", "content": user_payload},
    ]

    try:
        raw = await ask_llm(messages, model=settings.OLLAMA_ANALYZER_MODEL, max_tokens=_budget("ESCALATION"))
        data = _extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")

        next_step = _norm_next(data.get("next_step"))
        escalate = bool(data.get("escalate", False))

        # жёсткое правило: если не handoff_manager — не эскалируем
        if next_step != "handoff_manager":
            escalate = False

        reason = _norm_reason(data.get("reason"))
        interest_score = _clamp_int(data.get("interest_score"), 0, 100, 0)
        confidence = _clamp_float(data.get("confidence"), 0.0, 1.0, 0.0)
        client_need = _norm_need(data.get("client_need"))

        rs = data.get("reasons")
        reasons: list[str] = []
        if isinstance(rs, list):
            reasons = [str(x)[:80] for x in rs if x is not None][:8]

        if had_unknown_kb and reason == "other" and not escalate:
            reason = "unknown_kb"

        return {
            "escalate": escalate,
            "reason": reason,
            "interest_score": interest_score,
            "confidence": confidence,
            "next_step": next_step,
            "client_need": client_need,
            "reasons": reasons,
        }
    except Exception:
        return {
            "escalate": False,
            "reason": "other",
            "interest_score": 0,
            "confidence": 0.0,
            "next_step": "none",
            "client_need": "UNKNOWN",
            "reasons": [],
        }