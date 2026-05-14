"""Валидатор ответа conversation_brain.

Проверяет корректность ответа перед отправкой клиенту.
НЕ классифицирует — только проверяет факты и бизнес-правила.
"""
from __future__ import annotations

import re
from typing import Optional

from app.logging import logger

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

# Иностранные артефакты: Latin Extended (включая ö, ü, ä, é, ê, benötujete и пр.)
# U+00C0-U+024F = Latin-1 Supplement + Extended-A (все акцентированные латинские буквы)
# U+1E00-U+1EFF = Latin Extended Additional (вьетнамские диакритики и пр.)
_FOREIGN_ARTIFACT_RE = re.compile(r"[À-ɏḀ-ỿ]", re.U)

# Неожиданные латинские слова (5+ букв) в русском тексте — для обнаружения benötujete-подобных
_LATIN_ALLOWLIST = frozenset({
    "crm", "bitrix", "api", "url", "pdf", "sms", "ok", "telegram",
    "whatsapp", "google", "excel", "word",
})
_SUSPICIOUS_LATIN_WORD_RE = re.compile(r"\b[a-zA-Z]{5,}\b")

# Фразы-заглушки в конце ответа (plan §1)
_FILLER_TAIL_RE = re.compile(
    r"(всё\s+понял\s*\?"
    r"|окей\s*\?"
    r"|ок\s*\?"
    r"|погнали\s*(?:оформлять)?\s*\?"
    r"|всё\s+ясно\s*\?"
    r"|всё\s+понятно\s*\?"
    r"|договорились\s*\?"
    r"|продолжаем\s*\?)\s*$",
    re.I | re.U,
)

# Handoff-claim слова — запрещены без handoff.needed=true (plan §3, §4)
_HANDOFF_CLAIM_RE = re.compile(
    r"\b(передам\s+менеджеру|передаю\s+менеджеру|подключу\s+менеджера"
    r"|передаём\s+менеджеру|передаем\s+менеджеру"
    r"|передаю\s+кейс(?:\s+менеджеру)?|передам\s+кейс"
    r"|подключаю\s+менеджера|свяжется\s+менеджер"
    r"|вам\s+напишет\s+менеджер|вас\s+возьм[её]т\s+менеджер"
    r"|направлю\s+менеджеру|передам\s+заявку\s+менеджеру"
    r"|менеджер\s+(?:скоро\s+)?свяжется"
    r"|передам\s+(?:\w+\s+){0,3}менеджеру"
    r"|передаю\s+(?:\w+\s+){0,3}менеджеру)\b",
    re.I | re.U,
)

# Неформальное обращение — запрещено в деловом стиле (plan §7)
_INFORMAL_ADDRESS_RE = re.compile(
    r"(?<![а-яёА-ЯЁ])(?:"
    r"\bты\b|\bтебе\b|\bтебя\b"
    r"|\bтвой\b|\bтвоя\b|\bтвои\b|\bтвоё\b|\bтвоего\b|\bтвоему\b|\bтвоей\b|\bтвоих\b|\bтвоим\b"
    r")",
    re.I | re.U,
)

# Пользователь прямо поправил: "я про условия, а не тарифы" (plan §6)
_CORRECTION_NOT_TARIFFS_RE = re.compile(
    r"("
    r"я\s+про\s+условия"
    r"|не\s+тарифы,?\s*а\s+условия"
    r"|условия,?\s*а\s+не\s+тарифы"
    r"|не\s+про\s+тарифы"
    r"|речь\s+(?:шла\s+)?про\s+условия"
    r")",
    re.I | re.U,
)

# Ответ содержит только ценовые данные (тарифы без условий)
_TARIFF_ONLY_PRICE_RE = re.compile(
    r"(?:открытие|ведение)\s*\d|\d+\s*(?:₽|руб)",
    re.I | re.U,
)

# §4 — Валютный счёт: вопрос о расчётах + ответ без ограничений — противоречие домену
_CURRENCY_SETTLEMENT_QUESTION_RE = re.compile(
    r"\b(валютн\w+\s+сч[её]т|расч[её]ты?\s+(?:с\s+)?(?:валют|кредитор)|платить?\s+кредитор"
    r"|кредитор.*?платить?|можно\s+ли\s+делать\s+расч[её]ты?)\b",
    re.I | re.U,
)
_CURRENCY_ALLOWED_CLAIM_RE = re.compile(
    r"\b(расч[её]ты?\s+(?:с\s+кредиторами\s+)?(?:возможны|разрешены|доступны)"
    r"|можно\s+делать\s+расч[её]ты?\s+с\s+кредиторами?"
    r"|можно\s+платить\s+кредиторам?)\b",
    re.I | re.U,
)

# Таймлайн-вопрос (plan §12)
_TIMELINE_INTENT_RE = re.compile(
    r"\b(как\s+долго|сколько\s+времени|сколько\s+дней|сколько\s+ждать"
    r"|как\s+быстро|срок\w*|когда\s+будет|долго\s+ли|за\s+сколько\s+дней"
    r"|как\s+скоро|время\s+открытия|открытие\s+займ[её]т)\b",
    re.I | re.U,
)

# Вопрос о бонусах/процентах (plan §12)
_BONUS_INTEREST_INTENT_RE = re.compile(
    r"\b(бонус\w*|процент\w*\s*(?:годовых|на\s+остаток)?|годовых\b"
    r"|доходность\b|начисляет\s+процент|ставк[аиу]\s+по)\b",
    re.I | re.U,
)

# Ответ содержит признак timeline (сроки)
_HAS_TIMELINE_RE = re.compile(
    r"\b(дн[яей]|недел\w+|час\w+|бизнес[\-\s]дн\w+|рабочих\s+дн\w+|суток|календарных)\b",
    re.I | re.U,
)

# Ответ содержит признак bonus/interest
_HAS_BONUS_RE = re.compile(
    r"\b(бонус\w*|процент\w*|годовых|доходност\w*|начисл\w*|ставк\w*)\b",
    re.I | re.U,
)

# ЮЛ-специфичные документы для открытия (plan §5) — запрещены при неизвестном debtor_type
_YUL_SPECIFIC_DOCS_RE = re.compile(
    r"\b(огрн|устав\s+(?:организации|компании|общества|ооо|оао|зао|ао)"
    r"|учредительн\w+"
    r"|выписка\s+(?:из\s+)?егрюл"
    r"|протокол\s+(?:общего\s+)?собрания)\b",
    re.I | re.U,
)

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

# TASK 4 — Timing questions must not trigger tariff-related validators
_TIMING_QUESTION_RE = re.compile(
    r"\b(как\s+долго|сколько\s+времени|сколько\s+дней|сколько\s+ждать"
    r"|как\s+быстро|срок\w*|когда\s+будет|долго\s+ли|за\s+сколько\s+дней"
    r"|как\s+скоро|время\s+открытия|открытие\s+займёт|займёт\s+сколько"
    r"|как\s+долго\s+ждать)\b",
    re.I | re.U,
)

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

# FIX 3 — Условия vs. документы/открытие
# Пользователь спрашивает об условиях (не документах, не открытии)
_CONDITIONS_REQUEST_RE = re.compile(
    r"\b(какие\s+условия|что\s+по\s+условиям|условия\s+(?:в|у|по|там)"
    r"|какие\s+там\s+условия|расскажи\s+(?:об?\s+)?условиях"
    r"|условия\s+работы|условия\s+счёта|условия\s+счета"
    r"|какие\s+условия\s+там|что\s+за\s+условия)\b",
    re.I | re.U,
)

# Ответ дрейфует в документы / открытие счёта — forbidden when user asked about conditions
_DOCS_OPENING_DRIFT_RE = re.compile(
    r"(?:для\s+открытия\s+нужн|документы\s+(?:для|которые)|нужны\s+документы"
    r"|перечень\s+документов|пакет\s+документов"
    r"|поможем\s+оформить|помогаем\s+с\s+открытием"
    r"|давайте\s+(?:соберём|собрать)\s+документы"
    r"|сначала\s+(?:соберём|подготовим)\s+документы)",
    re.I | re.U,
)


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

    # Иностранные артефакты (акцентированные латинские буквы: ö, ü, ä, é, и пр.)
    if _FOREIGN_ARTIFACT_RE.search(reply):
        # Find the offending token for logging
        _bad_token = (_FOREIGN_ARTIFACT_RE.search(reply) or type("", (), {"group": lambda s: "?"})()).group(0)
        logger.warning("Validation | issue=foreign_language_artifact | token={!r}", _bad_token)
        return {"is_valid": False, "reason": "non_russian_output"}

    # Неожиданные длинные латинские слова в русском тексте (benötujete без акцентов)
    _cyrillic_count = len(re.findall(r"[а-яёА-ЯЁ]", reply))
    if _cyrillic_count > 10:  # only for substantive Russian replies
        for _lat_match in _SUSPICIOUS_LATIN_WORD_RE.finditer(reply):
            _lat_word = _lat_match.group(0).lower()
            if _lat_word not in _LATIN_ALLOWLIST:
                logger.warning("Validation | issue=foreign_language_artifact | token={!r}", _lat_match.group(0))
                return {"is_valid": False, "reason": "non_russian_output"}

    # FIX 3 — условия: ответ не должен дрейфовать в документы/открытие
    if user_text and _CONDITIONS_REQUEST_RE.search(user_text):
        if _DOCS_OPENING_DRIFT_RE.search(reply):
            logger.warning("Validation | issue=conditions_reply_drifted_to_documents")
            return {"is_valid": False, "reason": "conditions_reply_drifted_to_documents"}

    # Фраза-заглушка в конце ответа — plan §1
    if _FILLER_TAIL_RE.search(reply.strip()):
        return {"is_valid": False, "reason": "filler_tail_ending"}

    # Неформальное обращение — план §7
    if _INFORMAL_ADDRESS_RE.search(reply):
        # Не считаем ошибкой если это цитата пользователя (маловероятно в ответе бота)
        return {"is_valid": False, "reason": "informal_address_tu_tebe"}

    # Пользователь поправил: "я про условия, не тарифы" → ответ с только ценами недопустим (plan §6)
    if user_text and _CORRECTION_NOT_TARIFFS_RE.search(user_text):
        if _TARIFF_ONLY_PRICE_RE.search(reply) and not re.search(
            r"\b(перевод|комисс|ограничен|разрешен|работ|дистанц|подпис|договор|открыти|услов)\b",
            reply_lower, re.I | re.U,
        ):
            return {"is_valid": False, "reason": "repeated_tariffs_after_conditions_correction"}

    # Таймлайн-вопрос → ответ должен содержать временные данные (plan §12)
    if user_text and _TIMELINE_INTENT_RE.search(user_text):
        if not _HAS_TIMELINE_RE.search(reply) and len(reply) > 40:
            # Не блокируем если ответ содержит объяснение почему срок неизвестен
            if not re.search(r"\b(уточн\w+|зависит|индивидуальн\w+|от\s+банка|конкретн)\b", reply_lower, re.I | re.U):
                return {"is_valid": False, "reason": "timeline_question_no_timeline_in_reply"}

    # Бонус/процент-вопрос → ответ должен содержать info о бонусах (plan §12)
    if user_text and _BONUS_INTEREST_INTENT_RE.search(user_text):
        if not _HAS_BONUS_RE.search(reply) and len(reply) > 40:
            return {"is_valid": False, "reason": "bonus_question_no_bonus_in_reply"}

    # Handoff-claim без handoff.needed — plan §3, §4
    handoff_for_claim = (brain_result.get("handoff") or {})
    action_for_claim = brain_result.get("action") or "answer"
    if _HANDOFF_CLAIM_RE.search(reply):
        if not handoff_for_claim.get("needed") and action_for_claim == "answer":
            return {"is_valid": False, "reason": "handoff_claim_without_handoff_flag"}

    # §4 — Валютный счёт: ответ не должен утверждать что расчёты возможны без ограничений
    if user_text and _CURRENCY_SETTLEMENT_QUESTION_RE.search(user_text):
        if _CURRENCY_ALLOWED_CLAIM_RE.search(reply):
            return {"is_valid": False, "reason": "currency_settlement_claim_without_restriction"}

    # ЮЛ-специфичные документы при неизвестном debtor_type — plan §5
    _debtor_type = slots.get("debtor_type") or slots.get("client_type") or ""
    if not _debtor_type and _YUL_SPECIFIC_DOCS_RE.search(reply):
        return {"is_valid": False, "reason": "debtor_type_unknown_yul_specific_docs"}

    # Повтор предыдущего ответа — проверяем по истории (plan §5)
    prev_text = slots.get("_last_bot_text") or ""
    if prev_text and _is_near_duplicate(reply, prev_text):
        if user_text and _WHY_RE.match(user_text):
            return {"is_valid": False, "reason": "did_not_explain_reason"}
        return {"is_valid": False, "reason": "near_duplicate_of_previous"}

    # Проверка против расширенной истории (последние 3 ответа) — plan §5
    _reply_history: list = list(slots.get("_bot_reply_history") or [])
    if len(_reply_history) >= 2:
        for _hist_entry in _reply_history[:-1]:  # skip last (=prev_text already checked)
            # History entries may be dicts {"text": ..., ...} or plain strings
            _hist_text = _hist_entry.get("text") if isinstance(_hist_entry, dict) else _hist_entry
            if _hist_text and _is_near_duplicate(reply, _hist_text):
                return {"is_valid": False, "reason": "near_duplicate_of_recent_history"}

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
        # TASK 4: skip bank/tariff requirements for timing questions
        if topic == "bank_selection_fl":
            _is_timing_q = bool(user_text and _TIMING_QUESTION_RE.search(user_text))
            if not _is_timing_q:
                if "ткб" not in reply_lower:
                    return {"is_valid": False, "reason": "missing_primary_fact"}
                # If user asked for tariffs, must include at least one number
                _TARIFF_Q_RE = re.compile(r"(тариф|условия|стоимость|открытие|ведение)", re.I | re.U)
                if user_text and _TARIFF_Q_RE.search(user_text) and not _TIMING_QUESTION_RE.search(user_text):
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
