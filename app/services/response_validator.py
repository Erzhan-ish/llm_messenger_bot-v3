"""Валидатор ответа conversation_brain.

Проверяет корректность ответа перед отправкой клиенту.
НЕ классифицирует — только проверяет факты и бизнес-правила.
"""
from __future__ import annotations

import re
from typing import Optional

_STOP_WORDS = {
    "и", "а", "но", "что", "это", "как", "в", "на", "по", "для", "у", "я", "мы",
    "вы", "с", "к", "из", "же", "то", "так", "уже", "ещё", "при", "без",
}

# Мягкие фразы намерения открыть счёт (не попадают в hard handoff guard)
_OPEN_ACCOUNT_SOFT_RE = re.compile(
    r"(счет\s+откройте|откройте\s+(?:мне\s+)?счет"
    r"|как\s+открыть\s+счет"
    r"|давайте\s+открыть"
    r"|откроем\s*[?!]"
    r"|открываем\s*[?!]?"
    r"|оформим\s*[?!]?"
    r"|что\s+нужно\s+для\s+открытия)",
    re.I | re.U,
)

# Недопустимые обещания действия без handoff
_PROMISE_ACTION_RE = re.compile(
    r"\b(откроем|давайте\s+откроем|открываем|сделаем\s+счет|оформляем\s+счет"
    r"|мы\s+откроем|я\s+откр[ою]\w*|сейчас\s+откроем)\b",
    re.I | re.U,
)

# Активные задачи, при которых "открыть" = продолжение сравнения, не намерение
_COMPARISON_TASK_TYPES = {"transfer_fee_quote", "compare", "bank_selection", "pricing"}


def _keywords(text: str) -> set[str]:
    toks = re.findall(r"[а-яёa-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOP_WORDS and len(t) >= 3}


def _is_near_duplicate(a: str, b: str) -> bool:
    ak = _keywords(a)
    bk = _keywords(b)
    if not ak or not bk:
        return False
    inter = len(ak & bk)
    union = len(ak | bk)
    return (inter / max(1, union)) >= 0.80


def _extract_numbers_from_text(text: str) -> set[int]:
    nums: set[int] = set()
    for m in re.finditer(r"\d[\d\s]*", text):
        raw = m.group(0).replace(" ", "")
        try:
            nums.add(int(raw))
        except ValueError:
            pass
    return nums


def validate_reply(
    reply: str,
    brain_result: dict,
    current_entities: dict,
    slots: dict,
    tool_results: Optional[dict] = None,
    user_text: str = "",
) -> dict:
    """
    Проверить ответ brain.

    Returns dict: {"is_valid": bool, "reason": str|None}
    """
    if not reply or not reply.strip():
        return {"is_valid": False, "reason": "empty_reply"}

    # Повтор предыдущего ответа
    prev_text = slots.get("_last_bot_text") or ""
    if prev_text and _is_near_duplicate(reply, prev_text):
        return {"is_valid": False, "reason": "near_duplicate_of_previous"}

    # Комиссия из tool_result должна быть в ответе
    if tool_results:
        fee_result = tool_results.get("calculate_transfer_fee") or {}
        fee = fee_result.get("calculated_fee")
        if fee is not None and fee > 0:
            reply_nums = _extract_numbers_from_text(reply)
            if fee not in reply_nums:
                total = fee_result.get("total_fee")
                if total is None or total not in reply_nums:
                    return {"is_valid": False, "reason": f"fee_{fee}_missing_from_reply"}

    # Банк в ответе должен совпадать с упомянутым/ожидаемым
    mentioned_bank = current_entities.get("mentioned_bank")
    active_task = (slots.get("_active_task") or {})
    expected_bank = mentioned_bank or active_task.get("bank_name") or active_task.get("bank")
    if expected_bank:
        all_banks = ["Альфа-Банк", "ТКБ", "Уралсиб", "Т-Банк", "МКБ", "Росбанк"]
        other_banks = [b for b in all_banks if b != expected_bank]
        reply_lower = reply.lower()
        for other in other_banks:
            if other.lower() in reply_lower and expected_bank.lower() not in reply_lower:
                return {"is_valid": False, "reason": f"wrong_bank_{other}_instead_of_{expected_bank}"}

    # Handoff требует явного намерения
    handoff = (brain_result.get("handoff") or {})
    action = brain_result.get("action") or "answer"
    if handoff.get("needed") and action not in ("handoff", "request_data"):
        if not slots.get("_had_consent"):
            return {"is_valid": False, "reason": "handoff_without_explicit_consent"}

    # Бот не должен обещать действие без handoff/request_data
    if _PROMISE_ACTION_RE.search(reply):
        if not handoff.get("needed") and action not in ("handoff", "request_data"):
            return {"is_valid": False, "reason": "promised_action_without_handoff"}

    # Фраза намерения открыть → должен быть handoff или request_data
    if user_text and _OPEN_ACCOUNT_SOFT_RE.search(user_text):
        active_task_type = active_task.get("type") or active_task.get("intent") or ""
        is_comparison = active_task_type in _COMPARISON_TASK_TYPES
        if not is_comparison and not handoff.get("needed") and action not in ("handoff", "request_data"):
            return {"is_valid": False, "reason": "open_account_without_handoff_or_request_data"}

    return {"is_valid": True, "reason": None}
