# app/services/escalation_detector.py
from __future__ import annotations

import json
import re
from typing import Optional, TypedDict, Literal

from app.llm.providers import ask_llm
from app.config import settings


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

SYSTEM_PROMPT = """
Ты — классификатор эскалации. Ты НЕ пишешь клиенту.
Твоя задача: по диалогу решить, нужно ли подключить менеджера прямо сейчас.

ВАЖНО:
- Вопросы про тарифы/комиссии/проценты/условия САМИ ПО СЕБЕ НЕ являются причиной эскалации.
  Если клиент просто уточняет тарифы/условия без явной готовности к действию — ставь:
  - escalate=false
  - reason="pricing"
  - interest_score <= 45
  - next_step="ask_clarify" или "none"

Эскалация НУЖНА (escalate=true, next_step="handoff_manager", interest_score>=85, confidence>=0.85), если:
1) Клиент явно готов к следующему шагу, выражает прямое намерение открыть счет, выбрал банк или просит помочь:
   "оформляем", "давайте начнем", "что дальше?", "куда оплатить?", "готов", "открывайте", "мне нужен счет", "хочу открыть счет", "поможете?", "сможете помочь?", "давайте [название банка]", "давайте росбанк".
   -> reason="ready_to_open", client_need="OPEN_ACCOUNT" или "CONDITIONS" (по смыслу)
2) Клиент просит человека/менеджера/звонок/контакт:
   "позвоните", "перезвоните", "дайте номер", "подключите менеджера", "оператор".
   -> reason="human_request" (или "callback" если прямо про звонок), client_need="CONSULTATION"
3) Конфликт/жалоба/агрессия/исправление:
   Клиент недоволен, ругается ИЛИ прямо исправляет бота (например, "я же сказал", "нет, не умерший", "я не ИП", "Вы не поняли").
   -> reason="complex_case" или "unknown_kb", client_need="SUPPORT"
4) Сложный/нестандартный случай или тупик:
   много уточнений, нет данных в KB, had_unknown_kb=true несколько раз, либо бот ходит по кругу.
   -> reason="complex_case" или "unknown_kb", client_need="CONSULTATION"

ЖЁСТКОЕ ПРАВИЛО:
Если next_step не "handoff_manager", то escalate должен быть false.

Верни строго JSON:
{
  "escalate": true|false,
  "reason": "<pricing|callback|ready_to_open|documents|complex_case|unknown_kb|angry|human_request|other>",
  "interest_score": 0..100,
  "confidence": 0..1,
  "next_step": "none|ask_clarify|handoff_manager",
  "client_need": "<OPEN_ACCOUNT|OPEN_SPECIAL_ACCOUNT|CONDITIONS|DOCUMENTS|CONSULTATION|SUPPORT|UNKNOWN>",
  "reasons": ["короткие причины (опционально)"]
}
""".strip()

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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    try:
        raw = await ask_llm(messages, model=settings.OLLAMA_ANALYZER_MODEL)
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