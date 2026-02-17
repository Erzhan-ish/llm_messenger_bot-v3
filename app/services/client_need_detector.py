# app/service/client_need_detector.py
from __future__ import annotations

import json
import re
from typing import Optional

from app.llm.providers import ask_llm

ALLOWED_NEEDS = {
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "CONDITIONS",
    "DOCUMENTS",
    "CONSULTATION",
    "SUPPORT",
    "UNKNOWN",
}

NEED_LABELS = {
    "OPEN_ACCOUNT": "Открытие счёта",
    "OPEN_SPECIAL_ACCOUNT": "Открытие спецсчёта",
    "CONDITIONS": "Условия",
    "DOCUMENTS": "Документы",
    "CONSULTATION": "Консультация",
    "SUPPORT": "Поддержка",
    "UNKNOWN": "Консультация",
}

SYSTEM_PROMPT = """
Ты классификатор клиентской потребности.
Определи основную потребность клиента по диалогу.

Выбери ТОЛЬКО ОДНО значение из списка:
OPEN_ACCOUNT
OPEN_SPECIAL_ACCOUNT
CONDITIONS
DOCUMENTS
CONSULTATION
SUPPORT
UNKNOWN

Правила:
- DOCUMENTS ставь ТОЛЬКО если клиент сам явно просит список документов/требований или спрашивает "какие документы".
- Если клиент спрашивает про банки/условия/комиссии/сроки — это CONDITIONS (если он не сказал "хочу открыть").
- Если клиент говорит "хочу открыть" / "нужно открыть" / "открыть счёт" — это OPEN_ACCOUNT или OPEN_SPECIAL_ACCOUNT (если явно "спецсчёт/задатковый/залоговый/специальный").
- Если непонятно — UNKNOWN.

Ответ строго в JSON без лишнего текста:
{"client_need": "<VALUE>"}
""".strip()

# вытащить первый JSON-объект из ответа модели
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None

    raw = raw.strip()

    # 1) пробуем как есть
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) вытащим первый {...} кусок
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None

    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _normalize_need(value: Optional[str]) -> str:
    if not value:
        return "UNKNOWN"

    need = str(value).strip().upper()

    # частые варианты “OPEN ACCOUNT”, “OPEN-ACCOUNT”
    need = need.replace(" ", "_").replace("-", "_")

    if need in ALLOWED_NEEDS:
        return need
    return "UNKNOWN"


async def detect_client_need(dialog_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dialog_text or ""},
    ]

    try:
        raw = await ask_llm(messages)
        data = _extract_json(raw)
        if isinstance(data, dict):
            norm = _normalize_need(data.get("client_need"))
            return NEED_LABELS.get(norm, "Консультация")
        # fall through to heuristic
    except Exception:
        return "Консультация"

    # Fallback heuristic if model didn't return JSON
    t = (dialog_text or "").lower()
    if "спец" in t or "задат" in t or "залог" in t:
        return NEED_LABELS["OPEN_SPECIAL_ACCOUNT"]
    if "открыть" in t or "открытие" in t or "нужно открыть" in t:
        return NEED_LABELS["OPEN_ACCOUNT"]
    if "документ" in t:
        return NEED_LABELS["DOCUMENTS"]
    if "услов" in t or "комис" in t or "срок" in t:
        return NEED_LABELS["CONDITIONS"]
    if "поддерж" in t or "помог" in t:
        return NEED_LABELS["SUPPORT"]
    return NEED_LABELS["CONSULTATION"]
