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

# Manager/handoff reference required when deal_stage=ready_to_open
_MANAGER_REF_RE = re.compile(
    r"(менеджер|передам|передаём|передаем|возьм[её]м\s+в\s+работу|берём\s+в\s+работу"
    r"|берем\s+в\s+работу|кейс|проверим|помо[жг]\w*|оформим|запустим"
    r"|начн[её]м|подключ\w*\s+менеджера|escalat|handoff)",
    re.I | re.U,
)

# Replies that are ONLY tariff/pricing content — forbidden when ready_to_open
_TARIFF_ONLY_RE = re.compile(
    r"^[^.!?\n]*(?:открытие|ведение|тариф|руб\.?\s*\/?\s*мес|₽)[^.!?\n]*[.!?]?\s*$",
    re.I | re.U,
)

# Docs / documents words for "что требуется" validation
_DOCS_WORDS_RE = re.compile(
    r"(документ|инн|данные|ау|управляющ|судебный\s+акт|определение\s+суда"
    r"|процедур|подготовить|прислать|передать|требуется|нужн)",
    re.I | re.U,
)

# Паттерн тарифных цифр (800, 2800, 3500, 2090, 1600) в контексте стоимости
_TARIFF_NUMBERS_RE = re.compile(r"\b(800|2800|3500|2090|1600|1500)\s*руб", re.I | re.U)

# "Могу сравнить?" / "хотите сравнить?" — запрещено когда пользователь уже попросил
_ASKS_PERMISSION_RE = re.compile(
    r"(могу\s+сравнить|хотите\s+сравнить\??|сравнить\s+тарифы\??)",
    re.I | re.U,
)

# Явный запрос тарифов/условий
_TARIFF_EXPLICIT_REQUEST_RE = re.compile(
    r"(какие\s+(тарифы|условия)|сравни\s+(тарифы|условия)|покажи\s+(тарифы|условия)"
    r"|условия\s+и\s+тарифы|тарифы\s+у\s+(них|него)|условия\s+у\s+(них|него))",
    re.I | re.U,
)

# Обобщённый откат — "уточните вопрос" при ясном намерении
_GENERIC_FALLBACK_RE = re.compile(
    r"(уточните\s+вопрос|секунду[\s,]+уточняю|уточните[\s,]+пожалуйста\s+вопрос)",
    re.I | re.U,
)

# Ясные бизнес-намерения, при которых generic fallback запрещён
_CLEAR_INTENT_RE = re.compile(
    r"("
    r"выгод\w+\s+(через|с\s+вами|от\s+вас)"
    r"|зачем\s+(через|с\s+вами)"
    r"|напрямую\s+в\s+банк"
    r"|какие\s+(тарифы|условия|банки)"
    r"|нужен\s+банк|подобрать\s+банк"
    r"|подешевле|дешев\w+\s+банк"
    r"|какие\s+документы|что\s+нужно\s+для\s+открытия"
    r"|стадии?\s+наблюден\w*"
    r"|карт\w+\s+должник\w*"
    r")",
    re.I | re.U,
)

# Паттерн слов про наличные
_CASH_WORDS_RE = re.compile(r"\b(наличн\w*|судебное\s+решение\s+о\s+выдаче\s+наличными)\b", re.I | re.U)

# "почему?" паттерн
_WHY_RE = re.compile(r"^\s*(почему|а\s+почему|почему\s+нельзя|в\s+чём\s+причина|в\s+чем\s+причина)\s*[?!]?\s*$", re.I | re.U)

# CJK и нерусские символы в ответе
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ＀-￯]")

# Ответ заканчивается вопросом
_ENDS_WITH_QUESTION_RE = re.compile(r"\?\s*$", re.U)

# Явный запрос данных (альтернатива вопросу при question_policy=required)
_DATA_REQUEST_RE = re.compile(
    r"\b(укажите|пришлите|нужен|нужна|необходим\w*|отправьте|предоставьте|сообщите|назовите)\b",
    re.I | re.U,
)

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

# Вопрос "юрлицо или физлицо?" когда тип должника уже известен
_DEBTOR_TYPE_QUESTION_RE = re.compile(
    r"("
    r"для\s+юр\s*лица?\s+или\s+физ\w*"
    r"|юр\s*лиц\w*\s+или\s+физ\s*лиц\w*"
    r"|счёт\s+подбираем\s+для\s+юр"
    r"|должник\s+—?\s+юрлицо\s+или\s+физлицо"
    r")",
    re.I | re.U,
)

# Bank/pricing scenarios — active scenario falls back to allowed_stages
_BANK_PRICING_SCENARIOS = frozenset({
    "bank_selection_yul", "bank_pricing_yul", "bank_selection_fl",
    "uralsib_yul_conditions", "uralsib_fl_conditions",
    "alfabank_yul_conditions", "tkb_yul_conditions", "tkb_fl_conditions",
    "tbank_yul_conditions", "mkb_yul_conditions", "rosbank_yul_conditions",
})

_OBSERVATION_STAGE_RE = re.compile(
    r"\b(стади[яеи]\s+наблюден\w*|на\s+стадии\s+наблюден\w*|в\s+стадии\s+наблюден\w*)\b",
    re.I | re.U,
)

# Банки-условия: сценарий → ключевое слово банка в нижнем регистре
_BANK_CONDITIONS_SCENARIO_KW: dict[str, str] = {
    "alfabank_yul_conditions":  "альфа",
    "tkb_yul_conditions":       "ткб",
    "tkb_fl_conditions":        "ткб",
    "uralsib_yul_conditions":   "уралсиб",
    "uralsib_fl_conditions":    "уралсиб",
    "tbank_yul_conditions":     "т-банк",
    "mkb_yul_conditions":       "мкб",
    "rosbank_yul_conditions":   "росбанк",
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

    # DealState: ready_to_open — reply must reference manager/handoff
    _deal_state = slots.get("_deal_state") or {}
    if _deal_state.get("deal_stage") == "ready_to_open" and not _deal_state.get("handoff_needed") is False:
        if not _MANAGER_REF_RE.search(reply):
            return {"is_valid": False, "reason": "ready_to_open_no_manager_reference"}

    # DealState: docs_intent — reply must contain docs/data info
    if _deal_state.get("next_manager_move") == "explain_documents":
        if not _DOCS_WORDS_RE.search(reply):
            return {"is_valid": False, "reason": "docs_intent_no_docs_in_reply"}

    # Scenario playbook: forbidden phrases from _forbidden_actions slot
    for _fp in (slots.get("_forbidden_actions") or []):
        if _fp.lower() in reply_lower:
            return {"is_valid": False, "reason": f"forbidden_action:{_fp[:40]}"}

    # Reject "Могу сравнить?" when user explicitly asked for tariff comparison
    if user_text and _TARIFF_EXPLICIT_REQUEST_RE.search(user_text):
        if _ASKS_PERMISSION_RE.search(reply):
            return {"is_valid": False, "reason": "asks_permission_instead_of_answer"}

    # Reject generic fallback when user intent is clear
    if user_text and _CLEAR_INTENT_RE.search(user_text):
        if _GENERIC_FALLBACK_RE.search(reply):
            return {"is_valid": False, "reason": "generic_fallback_for_clear_intent"}

    # Scenario playbook: realization already confirmed → reject contradictory advice
    _known = slots.get("_known_slots") or {}
    if _known.get("realization_started") is True:
        if re.search(r"дождитесь\s+(введения|судебного)", reply_lower):
            return {"is_valid": False, "reason": "contradicts_realization_started"}

    # Reject "юрлицо или физлицо?" when debtor type is already known
    if slots.get("client_type") in ("ЮЛ", "ФЛ", "ИП") or slots.get("debtor_type") in ("ЮЛ", "ФЛ", "ИП"):
        if _DEBTOR_TYPE_QUESTION_RE.search(reply):
            return {"is_valid": False, "reason": "repeated_known_debtor_type_question"}

    # Reject reply that ignores the focused bank when a bank-conditions scenario is active
    _active_scen = slots.get("_active_scenario") or ""
    _expected_bank_kw = _BANK_CONDITIONS_SCENARIO_KW.get(_active_scen)
    if _expected_bank_kw and len(reply) > 60 and _expected_bank_kw not in reply_lower:
        return {"is_valid": False, "reason": f"bank_focus_not_in_reply:{_expected_bank_kw}"}

    # Reject if bank/pricing scenario falls back to allowed_stages content
    if _active_scen in _BANK_PRICING_SCENARIOS and len(reply) > 80:
        if _OBSERVATION_STAGE_RE.search(reply):
            return {"is_valid": False, "reason": "bank_scenario_falls_back_to_allowed_stages"}

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

    # question_policy checks
    if answer_contract:
        question_policy = answer_contract.get("question_policy") or "optional"
        if question_policy == "forbidden":
            if _ENDS_WITH_QUESTION_RE.search(reply.strip()):
                return {"is_valid": False, "reason": "unnecessary_question"}
        elif question_policy == "required":
            pass  # A natural reply does not have to end with a question

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

        # topic=bank_selection_fl: ТКБ и хотя бы одна цифра тарифа
        if topic == "bank_selection_fl":
            if "ткб" not in reply_lower:
                return {"is_valid": False, "reason": "missing_primary_fact"}
            # If user asked for tariffs, must include at least one number
            _TARIFF_Q_RE = re.compile(r"(тариф|условия|стоимость|сколько|открытие|ведение)", re.I | re.U)
            if user_text and _TARIFF_Q_RE.search(user_text):
                if not re.search(r"\b(1500|0\s*руб|бесплатно)", reply_lower):
                    return {"is_valid": False, "reason": "missing_fl_tariff_details"}

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

    # Role confusion — bot echoes nonsense
    _ROLE_CONFUSION_PHRASES = ["о чем я?", "о чём я?", "как я могу вам помочь?", "чем могу помочь?"]
    if any(reply_lower.strip() == ph for ph in _ROLE_CONFUSION_PHRASES):
        return {"is_valid": False, "reason": "role_confusion"}

    # Generic greeting reply to specific question — check before too_short
    _SPECIFIC_QUESTION_RE = re.compile(
        r"(карт|счет|банк|тариф|документ|условия|перевод|задатков|залогов|комисси|наличн|открыт|разниц)",
        re.I | re.U,
    )
    _GREETING_REPLY_START = re.compile(
        r"^(здравствуйте|добрый день|привет)[!,.\s]*(я\s+алексей|как\s+я\s+могу|чем\s+могу)",
        re.I | re.U,
    )
    if user_text and _SPECIFIC_QUESTION_RE.search(user_text) and _GREETING_REPLY_START.match(reply_lower):
        return {"is_valid": False, "reason": "generic_greeting_reply_to_specific_question"}

    # Identity question — reply must contain role markers
    if answer_contract and answer_contract.get("topic") == "identity":
        required = ["в плюсе", "счет", "должник", "банкрот"]
        if not any(r in reply_lower for r in required):
            return {"is_valid": False, "reason": "identity_question_wrong_answer"}

    # Too short for a specific question (not a greeting/follow-up)
    if user_text and _SPECIFIC_QUESTION_RE.search(user_text) and len(reply) < 40:
        return {"is_valid": False, "reason": "too_short_for_specific_question"}

    return {"is_valid": True, "reason": None}
