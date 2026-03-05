from __future__ import annotations

import json
import re
from typing import Optional, TypedDict, Literal, List

from app.llm.providers import ask_llm
from app.logging import logger

IntentType = Literal[
    "PRICING",
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "DOCUMENTS",
    "CONSULTATION",
    "END_DIALOG",
    "OTHER",
]

NextStep = Literal["none", "ask_clarify", "handoff_manager"]

class AnalysisSignal(TypedDict):
    intent: IntentType
    intent_confidence: float
    escalate: bool
    escalate_reason: str
    interest_score: int
    escalate_confidence: float
    next_step: NextStep
    client_need: str

SYSTEM_PROMPT = """
Ты — эксперт-аналитик диалогов для отдела продаж. Твоя задача — проанализировать текущее состояние диалога и вернуть JSON с параметрами.

### ПАРАМЕТРЫ ДЛЯ ОПРЕДЕЛЕНИЯ:

1. **intent**: Активное намерение клиента прямо сейчас.
   - PRICING: тарифы/комиссии/стоимость.
   - OPEN_SPECIAL_ACCOUNT: спецсчёт/залоговый/задатковый.
   - OPEN_ACCOUNT: обычный расчетный счет.
   - DOCUMENTS: какие документы нужны.
   - CONSULTATION: зовет человека/менеджера.
   - END_DIALOG: клиент прощается (пока, до свидания, досвидания), говорит "на этом все", "на сегодня всё", "спасибо это все" или "больше вопросов нет". Формулировки могут быть с ошибками, обращай внимание на смысл завершения.
   - OTHER: остальное.

2. **escalate**: Нужно ли ПРЯМО СЕЙЧАС передать диалог живому менеджеру или закрыть сессию с передачей истории.
   - true: если клиент явно готов к открытию или просит человека. ОЧЕНЬ ВАЖНО: Если клиент завершает диалог/прощается (intent = END_DIALOG), ТЫ ОБЯЗАН поставить escalate: true!
   - false: во всех остальных случаях, включая просто вопросы по тарифам.

3. **interest_score**: Оценка интереса клиента (0-100).
4. **next_step**:
   - handoff_manager: если escalate=true.
   - ask_clarify: если вопрос неполный.
   - none: если диалог идет в штатном режиме.

### ТРЕБОВАНИЯ К ФОРМАТУ:
Верни СТРОГО JSON:
{
  "intent": "...",
  "intent_confidence": 0..1,
  "escalate": true|false,
  "escalate_reason": "ready_to_open|callback|human_request|complex_case|pricing|other",
  "interest_score": 0..100,
  "escalate_confidence": 0..1,
  "next_step": "none|ask_clarify|handoff_manager",
  "client_need": "OPEN_ACCOUNT|CONDITIONS|DOCUMENTS|CONSULTATION|SUPPORT|UNKNOWN"
}

ВЫВЕДИ ТОЛЬКО РЕЗУЛЬТИРУЮЩИЙ JSON. НЕ ПИШИ НИКАКИХ ПОЯСНЕНИЙ ИЛИ ТЕКСТА ДО И ПОСЛЕ JSON!
""".strip()

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _clamp_float(v: object, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
        if x < lo: return lo
        if x > hi: return hi
        return x
    except Exception:
        return default

def _clamp_int(v: object, lo: int, hi: int, default: int) -> int:
    try:
        x = int(float(v))  # type: ignore[arg-type]
        if x < lo: return lo
        if x > hi: return hi
        return x
    except Exception:
        return default

async def analyze_dialog(dialog_text: str, had_unknown_kb: bool = False) -> AnalysisSignal:
    user_payload = dialog_text or ""
    if had_unknown_kb:
        user_payload += "\n\n[NOTICE] Бот не нашел ответа в базе знаний на последний вопрос."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    default_signal: AnalysisSignal = {
        "intent": "OTHER",
        "intent_confidence": 0.0,
        "escalate": False,
        "escalate_reason": "other",
        "interest_score": 0,
        "escalate_confidence": 0.0,
        "next_step": "none",
        "client_need": "UNKNOWN",
    }

    try:
        raw = await ask_llm(messages)
        # Try to find the JSON block. We look for { ... } using non-greedy approach first, 
        # but take the longest possible valid-looking block.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            logger.warning(f"No JSON found in LLM response: {raw}")
            return default_signal
        
        json_str = m.group(0).strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt a quick fix for common LLM issues (like trailing commas or unescaped quotes)
            # but simpler is better: let's just log and return default if it's too broken.
            logger.error(f"JSONDecodeError in dialog_analyzer. Raw: {raw}")
            return default_signal
        
        # Mapping and validation
        res: AnalysisSignal = {
            "intent": str(data.get("intent") or "OTHER").upper(),
            "intent_confidence": _clamp_float(data.get("intent_confidence"), 0.0, 1.0, 0.0),
            "escalate": bool(data.get("escalate", False)),
            "escalate_reason": str(data.get("escalate_reason") or "other"),
            "interest_score": _clamp_int(data.get("interest_score"), 0, 100, 0),
            "escalate_confidence": _clamp_float(data.get("escalate_confidence"), 0.0, 1.0, 0.0),
            "next_step": data.get("next_step", "none"),
            "client_need": str(data.get("client_need") or "UNKNOWN"),
        }
        
        # Force consistency
        if res["escalate"]:
            res["next_step"] = "handoff_manager"
        elif res["next_step"] == "handoff_manager":
            res["escalate"] = True
            
        return res
    except Exception:
        logger.exception("analyze_dialog failed")
        return default_signal
