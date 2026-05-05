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

# Паттерн тарифных цифр (800, 2800, 3500, 2090, 1600) в контексте стоимости
_TARIFF_NUMBERS_RE = re.compile(r"\b(800|2800|3500|2090|1600|1500)\s*руб", re.I | re.U)

# Паттерн слов про наличные
_CASH_WORDS_RE = re.compile(r"\b(наличн\w*|судебное\s+решение\s+о\s+выдаче\s+наличными)\b", re.I | re.U)

# "почему?" паттерн
_WHY_RE = re.compile(r"^\s*(почему|а\s+почему|почему\s+нельзя|в\s+чём\s+причина|в\s+чем\s+причина)\s*[?!]?\s*$", re.I | re.U)

# CJK и нерусские символы в ответе
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ＀-￯]")

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
    r"|мы\s+откроем|я\s+откр[ою]\w*|сейчас\s+откроем"
    r"|приступим\s+к\s+открытию|начнём\s+открытие|начнем\s+открытие)\b",
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
    answer_contract: Optional[dict] = None,
    scenario_facts: Optional[dict] = None,
) -> dict:
    """
    Проверить ответ brain.

    Returns dict: {"is_valid": bool, "reason": str|None}
    """
    if not reply or not reply.strip():
        return {"is_valid": False, "reason": "empty_reply"}

    reply_lower = reply.lower()

    # Нерусский/китайский текст в ответе
    if _CJK_RE.search(reply):
        return {"is_valid": False, "reason": "non_russian_output"}

    # Повтор предыдущего ответа
    prev_text = slots.get("_last_bot_text") or ""
    if prev_text and _is_near_duplicate(reply, prev_text):
        # "почему?" после дубликата — особый случай
        if user_text and _WHY_RE.match(user_text):
            return {"is_valid": False, "reason": "did_not_explain_reason"}
        return {"is_valid": False, "reason": "near_duplicate_of_previous"}

    # Повторное приветствие или представление
    if slots.get("_introduced") and (
        reply_lower.startswith("здравствуйте")
        or reply_lower.startswith("я алексей")
        or reply_lower.startswith("я менеджер алексей")
    ):
        return {"is_valid": False, "reason": "repeated_intro"}

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

    # Answer contract checks
    if answer_contract:
        topic = answer_contract.get("topic") or ""
        do_not_include = answer_contract.get("do_not_include") or []

        # topic=debtor_card: запрещены слова про наличные
        if topic == "debtor_card" and _CASH_WORDS_RE.search(reply):
            return {"is_valid": False, "reason": "wrong_topic_fact"}

        # topic=partner_banks: запрещены тарифные цифры
        if topic == "partner_banks" and _TARIFF_NUMBERS_RE.search(reply):
            return {"is_valid": False, "reason": "answered_tariffs_when_asked_bank_list"}

        # topic=bank_selection_fl: ТКБ должен быть в ответе
        if topic == "bank_selection_fl" and "ткб" not in reply_lower:
            return {"is_valid": False, "reason": "missing_primary_fact"}

        # Общая проверка do_not_include
        for phrase in do_not_include:
            if phrase.lower() in reply_lower:
                return {"is_valid": False, "reason": f"forbidden_phrase_{phrase[:30]}"}

    # Scenario facts completeness check
    if scenario_facts:
        _GENERIC_PHRASES = [
            "как я могу помочь", "чем могу помочь",
            "секунду уточняю информацию",
            "что вас интересует", "задайте вопрос",
        ]
        if any(ph in reply_lower for ph in _GENERIC_PHRASES):
            return {"is_valid": False, "reason": "generic_reply_despite_scenario_facts"}

        # debtor_card_realization: required facts
        if "debtor_card_realization" in scenario_facts:
            if answer_contract and answer_contract.get("topic") == "debtor_card":
                required = ["реализация", "финансовый управляющий"]
                if not all(r in reply_lower for r in required):
                    return {"is_valid": False, "reason": "missing_required_card_facts"}

        # partner_banks: if topic=partner_banks, all 3 active banks must appear
        if "partner_banks" in scenario_facts and answer_contract and answer_contract.get("topic") == "partner_banks":
            required_banks = ["альфа-банк", "ткб", "уралсиб"]
            if not all(b in reply_lower for b in required_banks):
                return {"is_valid": False, "reason": "missing_required_partner_banks"}

    return {"is_valid": True, "reason": None}
