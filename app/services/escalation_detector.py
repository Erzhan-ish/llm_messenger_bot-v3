# app/services/escalation_detector.py
from __future__ import annotations

import json
import re
from typing import Optional, TypedDict, Literal

from app.llm.providers import ask_llm


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

SYSTEM_PROMPT = """
Ты — классификатор эскалации для чат-бота банка. Ты НЕ отвечаешь клиенту.
Твоя задача — по диалогу определить, нужно ли подключать реального менеджера.

Эскалация нужна, если:
- Клиент явно хочет купить/подключить/оформить/открыть счёт ("готов", "оформляем", "подключайте").
- Клиент просит контакт/созвон/номер/связаться.
- Клиент спрашивает цену/тариф/стоимость/комиссии и видно намерение продолжить.
- Случай сложный/нестандартный (не удаётся корректно ответить по базе, много уточнений).
- Клиент злится/жалуется/конфликтует.
- Клиент прямо просит "позовите человека/менеджера".

Если клиент просто задаёт общий вопрос и можно ответить без вмешательства — эскалация НЕ нужна.

Верни строго JSON (без лишнего текста) по схеме:

{
  "escalate": true|false,
  "reason": "<one of: pricing|callback|ready_to_open|documents|complex_case|unknown_kb|angry|human_request|other>",
  "interest_score": 0..100,
  "confidence": 0..1,
  "next_step": "none|ask_clarify|handoff_manager",
  "client_need": "<OPEN_ACCOUNT|OPEN_SPECIAL_ACCOUNT|CONDITIONS|DOCUMENTS|CONSULTATION|SUPPORT|UNKNOWN>",
  "reasons": ["короткие причины (опционально)"]
}

Правила:
- interest_score — насколько клиент близок к действию (созвон/оформление/оплата).
- confidence — твоя уверенность.
- next_step:
    - "handoff_manager" если точно нужно подключать менеджера,
    - "ask_clarify" если близко к этому, но нужно подтверждение ("удобно передать менеджеру?"),
    - "none" если эскалация не нужна.
""".strip()

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

ALLOWED_NEEDS = {
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "CONDITIONS",
    "DOCUMENTS",
    "CONSULTATION",
    "SUPPORT",
    "UNKNOWN",
}


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
    """
    Возвращает сигнал для эскалации (строго машинный JSON).
    had_unknown_kb=True — если в текущем ответе были "нет инфы в базе" (это повышает шанс complex/unknown_kb).
    """
    user_payload = dialog_text or ""
    if had_unknown_kb:
        user_payload += "\n\n[CONTEXT] В ответе были вопросы без информации в базе знаний (had_unknown_kb=true)."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    try:
        raw = await ask_llm(messages)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")

        escalate = bool(data.get("escalate", False))
        reason = _norm_reason(data.get("reason"))
        interest_score = _clamp_int(data.get("interest_score"), 0, 100, 0)
        confidence = _clamp_float(data.get("confidence"), 0.0, 1.0, 0.0)
        next_step = _norm_next(data.get("next_step"))
        client_need = _norm_need(data.get("client_need"))

        rs = data.get("reasons")
        reasons: list[str] = []
        if isinstance(rs, list):
            reasons = [str(x)[:80] for x in rs if x is not None][:8]

        # если KB unknown и модель не указала — подстрахуемся
        if had_unknown_kb and reason == "other" and not escalate:
            # не заставляем эскалировать, но помечаем как unknown_kb, чтобы код мог поднять score
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
        # безопасный дефолт: не эскалируем, не ломаем обработку
        return {
            "escalate": False,
            "reason": "other",
            "interest_score": 0,
            "confidence": 0.0,
            "next_step": "none",
            "client_need": "UNKNOWN",
            "reasons": [],
        }
