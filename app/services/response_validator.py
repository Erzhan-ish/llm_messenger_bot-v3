"""Валидатор ответа conversation_brain.

Проверяет корректность ответа перед отправкой клиенту.
НЕ классифицирует — только проверяет факты.
"""
from __future__ import annotations

import re
from typing import Optional

_HANDOFF_EXPLICIT_RE = re.compile(
    r"(оформляем|хочу\s+открыть|готов\s+начать|подключите\s+менеджера"
    r"|позовите\s+(человека|менеджера)|куда\s+оплатить|выставляй(те)?\s+счёт"
    r"|пришлю\s+документы|отправлю\s+документы|подходит.*что\s+дальше"
    r"|готов\s+к\s+оформлению)",
    re.I | re.U,
)

_STOP_WORDS = {
    "и", "а", "но", "что", "это", "как", "в", "на", "по", "для", "у", "я", "мы",
    "вы", "с", "к", "из", "же", "то", "так", "уже", "ещё", "при", "без",
}


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


def _bank_name_in_text(text: str, bank: str) -> bool:
    if not bank or not text:
        return True
    return bank.lower() in text.lower()


def validate_reply(
    reply: str,
    brain_result: dict,
    current_entities: dict,
    slots: dict,
    tool_results: Optional[dict] = None,
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

    # Если есть tool_result с комиссией — проверить что число есть в ответе
    if tool_results:
        fee_result = tool_results.get("calculate_transfer_fee") or {}
        fee = fee_result.get("calculated_fee")
        if fee is not None and fee > 0:
            reply_nums = _extract_numbers_from_text(reply)
            if fee not in reply_nums:
                # Допускаем total_fee как альтернативу
                total = fee_result.get("total_fee")
                if total is None or total not in reply_nums:
                    return {"is_valid": False, "reason": f"fee_{fee}_missing_from_reply"}

    # Если клиент упомянул банк — проверить что ответ не про другой банк
    mentioned_bank = current_entities.get("mentioned_bank")
    active_task = (slots.get("_active_task") or {})
    expected_bank = mentioned_bank or active_task.get("bank_name") or active_task.get("bank")
    if expected_bank:
        # Получить все известные банки
        all_banks = ["Альфа-Банк", "ТКБ", "Уралсиб", "Т-Банк", "МКБ", "Росбанк"]
        other_banks = [b for b in all_banks if b != expected_bank]
        reply_lower = reply.lower()
        for other in other_banks:
            if other.lower() in reply_lower and expected_bank.lower() not in reply_lower:
                return {"is_valid": False, "reason": f"wrong_bank_{other}_instead_of_{expected_bank}"}

    # handoff требует явного намерения в тексте пользователя
    handoff = (brain_result.get("handoff") or {})
    if handoff.get("needed"):
        # Проверить через slots._had_consent или явную фразу
        if not slots.get("_had_consent"):
            return {"is_valid": False, "reason": "handoff_without_explicit_consent"}

    return {"is_valid": True, "reason": None}
