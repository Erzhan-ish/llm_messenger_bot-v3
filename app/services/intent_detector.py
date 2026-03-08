# app/services/intent_detector.py
from __future__ import annotations

import json
import re
from typing import Literal, TypedDict, Optional

from app.llm.providers import ask_llm
from app.config import settings


IntentType = Literal[
    "PRICING",
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "DOCUMENTS",
    "CONSULTATION",
    "END_DIALOG",
    "OTHER",
]


class IntentSignal(TypedDict):
    intent: IntentType
    confidence: float


SYSTEM_PROMPT = """
Ты — классификатор намерения клиента. Ты НЕ отвечаешь клиенту.

Определи АКТИВНОЕ намерение клиента прямо сейчас (последние сообщения важнее старых).

Верни строго JSON:
{
  "intent": "PRICING|OPEN_ACCOUNT|OPEN_SPECIAL_ACCOUNT|DOCUMENTS|CONSULTATION|END_DIALOG|OTHER",
  "confidence": 0..1
}

Правила:
- END_DIALOG: клиент завершает общение ("до свидания", "пока", "спасибо, понятно", "закрываю чат", "всё, понял") или явно просит закончить.
- PRICING: тарифы/комиссии/проценты/условия/стоимость/сколько стоит
- OPEN_SPECIAL_ACCOUNT: "спецсчёт", "спец счет", "открыть спецсчет", "нужен спецсчёт"
- OPEN_ACCOUNT: "открыть счёт", "оформить счёт", "завести заявку", "давайте откроем", "хочу открыть основной счет"
- DOCUMENTS: "какие документы", "пакет документов", "что нужно для открытия"
- CONSULTATION: просит человека/менеджера/звонок/контакт ("позвоните", "перезвоните", "позовите менеджера", "оператор")
- OTHER: всё остальное

Если одновременно "спасибо" и новый вопрос — НЕ END_DIALOG.
""".strip()

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


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


async def detect_intent(dialog_text: str) -> IntentSignal:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dialog_text or ""},
    ]
    try:
        raw = await ask_llm(messages, model=settings.OLLAMA_ANALYZER_MODEL)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")

        intent = str(data.get("intent") or "OTHER").strip().upper()
        confidence = _clamp_float(data.get("confidence"), 0.0, 1.0, 0.0)

        allowed = {
            "PRICING",
            "OPEN_ACCOUNT",
            "OPEN_SPECIAL_ACCOUNT",
            "DOCUMENTS",
            "CONSULTATION",
            "END_DIALOG",
            "OTHER",
        }
        if intent not in allowed:
            intent = "OTHER"

        return {"intent": intent, "confidence": confidence}  # type: ignore[return-value]
    except Exception:
        return {"intent": "OTHER", "confidence": 0.0}