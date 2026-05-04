"""Response plan builders — fully deterministic, no LLM calls."""
from __future__ import annotations

import re

from app.processing.utils import _FALLBACK_TEXT
from app.processing.domain_guard import out_of_scope_reply, constraint_fallback
from app.services.sales_policy import OBJECTION_TYPE_MAP, STAGE_NEXT_STEP

# ---------------------------------------------------------------------------
# Follow-up detection regex (used by message_processor for short-reply routing)
# ---------------------------------------------------------------------------
_FOLLOWUP_RE = re.compile(
    r"^\s*(ещё|еще|что\s+(ещё|еще)|ну|и|дальше|далее|подробнее|поподробнее"
    r"|я\s+(уже\s+)?(сказал|написал|говорил|указал)|я\s+же\s+сказал"
    r")\s*[?!\.]?\s*$",
    re.I | re.U,
)

_TIMING_REPLY = (
    "Счёт обычно открывают за 1–3 рабочих дня с момента подачи документов. "
    "В некоторых банках — быстрее, если все документы в порядке. "
    "Хотите узнать, что нужно подготовить?"
)

_REFERENCE_EXAMPLES = {
    "pricing": [
        "ДАННЫЕ: Открытие: 3500 руб., Ведение: 1600 руб./мес., Платежи: от 35 руб.\n"
        "ПЛОХО: «Условия: открытие — 3 500 руб., обслуживание — 1 600 руб.»\n"
        "ХОРОШО: «открытие обойдется в 3500 рублей и потом 1600 в месяц. оформляем или посмотрим другие варианты?»"
    ],
    "docs": [
        "ДАННЫЕ: Для открытия счета в Уралсибе по ЮЛ сканы документов от АУ не нужны, достаточно названия или ИНН должника и перечня счетов.\n"
        "ПЛОХО: «Документы: сканы не требуются. Нужен ИНН и перечень счетов.»\n"
        "ХОРОШО: «тут все просто, сканы собирать не придется. нужен только инн должника и список счетов. готовы прислать?»"
    ],
    "bonus": [
        "ДАННЫЕ: Бонус на остаток: до 10 млн руб. — 4%; 10–50 млн — 4.5%; 50–250 млн — 5.5%; 250–500 млн — 6%; ... свыше 1 млрд руб. — 8.5%.\n"
        "ПЛОХО: «Процент на остаток: от 4% до 8.5% годовых.»\n"
        "ХОРОШО: «там еще банк начисляет хороший процент на остаток, от 4 до 8.5 годовых выходит в зависимости от суммы. рассматриваем этот вариант?»"
    ],
    "compare": [
        "ДАННЫЕ: ТКБ (открытие 2800 руб., ведение 2090 руб./мес.), Уралсиб (открытие 3500 руб., ведение 1600 руб./мес.).\n"
        "ПЛОХО: «Сравнение: ТКБ (открытие 2800) и Уралсиб (открытие 3500).»\n"
        "ХОРОШО: «сейчас есть два хороших варианта. в ткб дешевле открыть сам счет за 2800, зато в уралсибе выгоднее его потом вести по 1600 в месяц. вам что важнее сэкономить на старте или на ведении?»"
    ],
    "selection_opening": [
        "ДАННЫЕ: ТКБ (открытие 2800 руб., ведение 2090 руб./мес.), Уралсиб (открытие 3500 руб., ведение 1600 руб./мес.).\n"
        "ПЛОХО: «Сравнение: ТКБ (открытие 2800) и Уралсиб (открытие 3500).»\n"
        "ХОРОШО: «сейчас есть два хороших варианта. в ткб дешевле открыть сам счет за 2800, зато в уралсибе выгоднее его потом вести по 1600 в месяц. вам что важнее сэкономить на старте или на ведении?»"
    ],
    "specific_bank": [
        "ДАННЫЕ: Открытие: 3500 руб., Ведение: 1600 руб./мес., Платежи: от 35 руб.\n"
        "ПЛОХО: «Условия: открытие — 3 500 руб., обслуживание — 1 600 руб.»\n"
        "ХОРОШО: «открытие обойдется в 3500 рублей и потом 1600 в месяц. оформляем или посмотрим другие варианты?»"
    ],
    "clarify": [
        "ДАННЫЕ: Нужно узнать тип клиента (ИП, ООО, ФЛ).\n"
        "ПЛОХО: «Уточните: ИП, ООО, ФЛ.»\n"
        "ХОРОШО: «подскажите, счет нужен для ип, ооо или физлица?»"
    ],
    "conditions": [
        "ДАННЫЕ: ТКБ, ЮЛ, открытие 2800 руб., ведение 2090 руб./мес., платежи 33 руб., бонус на остаток.\n"
        "ПЛОХО: «Условия по ТКБ: открытие — 2 800, ведение — 2 090.»\n"
        "ХОРОШО: «по ткб для юрлица условия такие: открытие 2800 рублей, ведение 2090 в месяц, платежи на юрлица — 33 рубля электронно. плюс есть бонус на остаток. что разобрать подробнее — платежи, документы или сроки?»"
    ],
    "constraint": [
        "ДАННЫЕ: Открыть только специальный счет без открытия основного счета нельзя.\n"
        "ПЛОХО: «Можно, нужны документы.»\n"
        "ХОРОШО: «нет, отдельно спецсчёт без основного открыть нельзя. сначала открывается основной, потом уже спецсчёт. счёт для юрлица?»",
        "ДАННЫЕ: Карту при реализации имущества оформляет и подписывает финансовый управляющий.\n"
        "ХОРОШО: «если введена реализация имущества, карту оформляет и подписывает финансовый управляющий. реализация уже введена?»"
    ],
}

# ---------------------------------------------------------------------------
# Objection detection
# ---------------------------------------------------------------------------
_OBJECTION_EXPENSIVE_RE = re.compile(
    r"\b(дорого|слишком\s+дорого|дороговато|не\s+дёшево|дорогой\s+тариф"
    r"|цена\s+высокая|высокая\s+цена|дороже\s+чем|дорог[ао]вато)\b",
    re.I | re.U,
)
_OBJECTION_THINK_RE = re.compile(
    r"\b(подумаю|надо\s+подумать|нужно\s+подумать|подумать\s+надо"
    r"|посоветуюсь|не\s+готов\s+сейчас|не\s+сейчас|пока\s+не\s+готов)\b",
    re.I | re.U,
)
_OBJECTION_COMPETITOR_RE = re.compile(
    r"\b(у\s+других\s+дешевле|в\s+другом\s+месте|конкурент\w*|другой\s+вариант"
    r"|нашел\s+дешевле|есть\s+дешевле|там\s+дешевле)\b",
    re.I | re.U,
)
_OBJECTION_UNSURE_RE = re.compile(
    r"\b(не\s+уверен|сомнева\w+|не\s+знаю\s+стоит\s+ли|не\s+знаю\s+нужно\s+ли"
    r"|может\s+не\s+стоит|пожалуй\s+нет)\b",
    re.I | re.U,
)
_OBJECTION_NO_VISIT_RE = re.compile(
    r"\b(нужно\s+ли\s+(ехать|идти|приходить|приезжать)"
    r"|надо\s+ли\s+(ехать|идти|приходить|приезжать)"
    r"|без\s+(посещения|визита|поездки)\s+(банк\w*|офис\w*)?"
    r"|можно\s+ли\s+(без\s+визита|удалённо|дистанционно|онлайн)"
    r"|онлайн\s+(открыть|оформить)|удалённо\s+(открыть|оформить)"
    r"|не\s+нужно\s+(ехать|приходить|посещать)"
    r"|в\s+офис\s+(надо|нужно|идти|ехать))\b",
    re.I | re.U,
)

_OBJECTION_REPLIES: dict[str, list[str]] = {
    "expensive": [
        "Понимаю. Здесь есть интересный момент: банки начисляют АУ бонус на остаток — до 9% годовых. На практике это перекрывает стоимость ведения счёта уже при умеренном остатке. Что именно кажется дорогим — открытие или ежемесячное ведение?",
        "Давайте разберём: открытие — разовая трата, а ведение часто компенсируется бонусом на остаток. Что из этого вызывает вопрос?",
    ],
    "no_visit": [
        "Нет, ехать в банк не нужно. Документы мы собираем сами, счёт открывается дистанционно. Вам достаточно подтвердить данные — и всё. Это займёт 1–2 рабочих дня.",
        "Всё делается удалённо: документы, подписание, открытие. В банк ехать не нужно. Что ещё хотите уточнить?",
    ],
    "think": [
        "Конечно, не торопитесь. Могу пока зафиксировать условия по банку — чтобы при возвращении не искать заново. Что для вас сейчас важнее понять перед решением?",
        "Хорошо. Если есть конкретный вопрос который мешает решить — готов разобрать. Или зафиксирую условия пока актуальны?",
    ],
    "competitor": [
        "Интересно. Обычно разница в деталях — комиссии за переводы, скорость открытия, работа с процедурами. Что именно предлагают дешевле и по какому банку?",
        "Уточните, что именно дешевле — само открытие или ведение? Готов сравнить по конкретным условиям.",
    ],
    "unsure": [
        "Что именно вызывает сомнение? Цена, банк или сами условия? Разберём конкретный момент.",
        "Понимаю. Давайте я уточню: что именно не устраивает или непонятно? Постараюсь помочь с конкретикой.",
    ],
}


def _detect_objection(text: str) -> str | None:
    """Returns objection type key or None."""
    if _OBJECTION_EXPENSIVE_RE.search(text):
        return "expensive"
    if _OBJECTION_THINK_RE.search(text):
        return "think"
    if _OBJECTION_COMPETITOR_RE.search(text):
        return "competitor"
    if _OBJECTION_UNSURE_RE.search(text):
        return "unsure"
    if _OBJECTION_NO_VISIT_RE.search(text):
        return "no_visit"
    return None


def _objection_reply(objection_type: str, seed: str = "") -> str:
    import random
    variants = _OBJECTION_REPLIES.get(objection_type, [])
    if not variants:
        return _FALLBACK_TEXT
    idx = hash(seed) % len(variants) if seed else 0
    return variants[abs(idx)]


# ---------------------------------------------------------------------------
# Base plan factory
# ---------------------------------------------------------------------------
def _make_base(client_type=None) -> dict:
    return {
        "action":          "service",
        "intent":          "other",
        "bank":            None,
        "client_type":     client_type,
        "items":           [],
        "candidates":      [],
        "docs":            [],
        "constraints":     [],
        "status":          None,
        "question_to_ask": None,
        "handoff_reason":  None,
        "tone":            "manager",
    }


# ---------------------------------------------------------------------------
# Plan sub-builders (one per route)
# ---------------------------------------------------------------------------
def _plan_service(base: dict, stage: str) -> dict:
    intent_map = {"GREETING": "greeting", "ACK": "ack", "THANKS": "thanks"}
    base["intent"] = intent_map.get(stage, "service")
    return base


def _plan_handoff(base: dict, qmode: str, reason: str) -> dict:
    base["action"]         = "handoff"
    base["intent"]         = qmode
    base["handoff_reason"] = reason
    return base


def _plan_clarify(base: dict, qmode: str, question: str, slots: dict) -> dict:
    base["action"]          = "clarify"
    base["intent"]          = qmode
    base["question_to_ask"] = question
    slots["_pending_question_type"] = question
    return base


def _plan_partner_banks(base: dict, slots: dict) -> dict:
    client_type = slots.get("client_type") or slots.get("_last_client_type")
    base["action"]      = "partner_banks"
    base["intent"]      = "partner_banks"
    base["client_type"] = client_type
    return base


def _plan_selection_opening(base: dict, candidates: list, slots: dict,
                             client_type, priority) -> dict:
    """
    First-contact bank selection response.
    Shows top 1–2 candidates warmly + one focused next-step question.
    Used when sales_stage transitions to SELECT for the first time.
    """
    top = candidates[:2]
    base["action"]      = "selection_opening"
    base["intent"]      = "bank_selection"
    base["candidates"]  = top
    base["client_type"] = client_type

    if len(top) == 1:
        base["bank"] = top[0]["bank"]

    points: list[str] = []
    for c in top:
        bank = c.get("bank", "")
        of   = c.get("opening_fee")
        mf   = c.get("monthly_fee")
        br   = c.get("bonus_rate")
        main = c.get("main_feature") or ""
        if of is not None:
            points.append(f"{bank}: открытие {int(of)} руб.")
        if mf is not None:
            points.append(f"{bank}: ведение {int(mf)} руб./мес.")
        if br:
            points.append(f"{bank}: бонус АУ — {br}")
        elif main:
            points.append(f"{bank}: {main}")
    base["allowed_points"] = points

    if not client_type:
        base["question_to_ask"] = "client_type"
        slots["_pending_question_type"] = "client_type"
    elif not priority:
        base["question_to_ask"] = "priority"
        slots["_pending_question_type"] = "priority"
    else:
        slots.pop("_pending_question_type", None)

    slots["last_offer_bank"] = top[0]["bank"] if top else None
    slots["last_offer_type"] = "selection_opening"
    return base


def _candidate_points(candidates: list) -> list[str]:
    """Build allowed_points list from candidate dicts."""
    points: list[str] = []
    for c in candidates:
        bank = c.get("bank", "")
        of   = c.get("opening_fee")
        mf   = c.get("monthly_fee")
        br   = c.get("bonus_rate")
        main = c.get("main_feature") or ""
        if of is not None:
            points.append(f"{bank}: открытие {int(of)} руб.")
        if mf is not None:
            points.append(f"{bank}: ведение {int(mf)} руб./мес.")
        if br:
            points.append(f"{bank}: бонус АУ — {br}")
        elif main:
            points.append(f"{bank}: {main}")
    return points


def _plan_bank_selection(base: dict, facts: dict, slots: dict, decision: dict,
                          client_type, priority) -> dict:
    all_banks  = facts.get("all_found_banks") or []
    candidates = [c for c in all_banks if c.get("status") == "ACTIVE" and c.get("rank_score", 0) > 0]
    constraints = facts.get("constraints") or []

    if not candidates:
        if not client_type:
            return _plan_clarify(base, "bank_selection", "client_type", slots)
        base["intent"] = "no_candidates"
        return base

    candidates = sorted(candidates, key=lambda x: x.get("rank_score", 0.0), reverse=True)

    sales_stage = slots.get("sales_stage")
    is_first_selection = sales_stage in (None, "QUALIFY", "SELECT") and not slots.get("_last_candidates")
    if decision.get("multi_intent") and decision.get("is_first_turn"):
        is_first_selection = True

    if len(candidates) == 1:
        c = candidates[0]
        base["action"]      = "selection_opening"
        base["intent"]      = "bank_selection"
        base["bank"]        = c["bank"]
        base["client_type"] = c.get("client_type") or client_type
        base["candidates"]  = candidates
        base["constraints"] = constraints
        base["allowed_points"] = _candidate_points([c])
        if not client_type:
            base["question_to_ask"] = "client_type"
            slots["_pending_question_type"] = "client_type"
        else:
            slots.pop("_pending_question_type", None)
        slots["last_offer_bank"] = c["bank"]
        slots["last_offer_type"] = "selection_opening"
        return base

    top = candidates[:3]

    if is_first_selection:
        plan = _plan_selection_opening(base, top, slots, client_type, priority)
        plan["constraints"] = constraints
        return plan

    # Repeat / explicit comparison request → compare
    base["action"]     = "compare"
    base["intent"]     = "bank_selection"
    base["candidates"] = top
    base["constraints"] = constraints
    if not client_type:
        base["question_to_ask"] = "client_type"
        slots["_pending_question_type"] = "client_type"
    elif not priority:
        base["question_to_ask"] = "priority"
        slots["_pending_question_type"] = "priority"
    else:
        slots.pop("_pending_question_type", None)
    return base


def _plan_timing(base: dict, qmode: str, facts: dict, slots: dict, client_type) -> dict:
    """Plan for timing / timing_docs queries — no pricing, no bonus."""
    bank_profile = facts.get("bank_profile") or {}
    bank = (
        bank_profile.get("bank")
        or facts.get("bank")
        or slots.get("_last_bank")
        or slots.get("bank_name")
    )
    timing_text = (
        facts.get("timing_text")
        or bank_profile.get("timing_text")
        or facts.get("opening_time")
    )
    docs = bank_profile.get("docs") or facts.get("docs") or slots.get("_last_docs") or []

    base["action"]      = "answer"
    base["intent"]      = qmode
    base["bank"]        = bank
    base["client_type"] = bank_profile.get("client_type") or client_type
    base["timing_text"] = timing_text
    base["items"]       = []  # timing never shows pricing

    if qmode == "timing_docs":
        base["docs"] = docs
        base["question_to_ask"] = None
    else:
        base["docs"]       = []
        base["funnel_next"] = "docs"

    if bank:
        slots["last_offer_bank"] = bank
        slots["last_offer_type"] = qmode
    return base



def _plan_out_of_scope(base: dict, slots: dict, decision: dict) -> dict:
    base["action"] = "answer"
    base["intent"] = "out_of_scope"
    base["must_use_facts"] = []
    base["answer_text"] = out_of_scope_reply()
    base["question_to_ask"] = None
    return base


# ---------------------------------------------------------------------------
# Transfer fee quote helpers
# ---------------------------------------------------------------------------
_ALFA_TRANSFER_FL_SCALE = [
    (150_000, 0.0,   "до 150 000 руб. — без комиссии"),
    (None,    0.005, "свыше 150 000 руб. — 0.5%"),
]

_URALSIB_EXTRA_FEES = {
    "control_fee": ("150 руб.", "за контроль каждой банкротной операции"),
    "transfer_ul": ("35 руб.", "перевод на юрлицо"),
    "transfer_fl": ("без комиссии до 100 000 руб.", "перевод на физлицо"),
}

_BANK_TRANSFER_FL_SCALE: dict[str, list] = {
    "Альфа-Банк": _ALFA_TRANSFER_FL_SCALE,
}

_LARGE_TRANSFER_THRESHOLD = 30_000_000


def _calc_transfer_fee(bank: str, amount: int | None, target: str | None) -> dict:
    """Return fee calculation result for transfer_fee_quote."""
    if not amount or not bank:
        return {}
    scale = _BANK_TRANSFER_FL_SCALE.get(bank)
    if not scale or target != "ФЛ":
        return {}
    for threshold, rate, label in scale:
        if threshold is None or amount <= threshold:
            fee = int(amount * rate)
            return {
                "bank": bank,
                "amount": amount,
                "transfer_target": target,
                "fee": fee,
                "rate": rate,
                "rate_label": label,
                "calculated_fee": fee,
            }
    return {}


def _plan_transfer_fee_quote(base: dict, facts: dict, slots: dict, decision: dict, client_type) -> dict:
    planner = decision.get("planner") or {}
    bank = (
        slots.get("_current_bank_mention")
        or decision.get("bank_name")
        or planner.get("bank_name")
        or slots.get("bank_name")
        or slots.get("_last_bank")
        or (facts.get("bank_profile") or {}).get("bank")
        or facts.get("bank")
    )
    amount = (
        slots.get("_transfer_amount")
        or decision.get("amount")
        or planner.get("amount")
    )
    target = (
        slots.get("_transfer_target")
        or decision.get("transfer_target")
        or planner.get("transfer_target")
    )

    fee_calc = _calc_transfer_fee(bank, amount, target) if bank else {}
    is_large = bool(amount and amount > _LARGE_TRANSFER_THRESHOLD)

    base["action"]           = "answer"
    base["intent"]           = "transfer_fee_quote"
    base["bank"]             = bank
    base["client_type"]      = client_type
    base["amount"]           = amount
    base["transfer_target"]  = target
    base["fee_details"]      = fee_calc
    base["is_large_transfer"] = is_large
    base["items"]            = []

    if is_large:
        base["constraint_topic"] = "large_transfer_requires_2_day_notice_and_creditor_registry"
        from app.processing.domain_guard import constraint_fallback
        base["large_transfer_note"] = constraint_fallback("large_transfer_requires_2_day_notice_and_creditor_registry")

    if not bank:
        base["question_to_ask"] = "bank_name"
        slots["_pending_question_type"] = "bank_name"
    else:
        slots.pop("_pending_question_type", None)
        if bank:
            slots["_last_bank"] = bank

    return base


def _plan_extra_fees(base: dict, facts: dict, slots: dict, decision: dict, client_type) -> dict:
    planner = decision.get("planner") or {}
    bank = (
        slots.get("_current_bank_mention")
        or decision.get("bank_name")
        or planner.get("bank_name")
        or slots.get("bank_name")
        or slots.get("_last_bank")
        or (facts.get("bank_profile") or {}).get("bank")
        or facts.get("bank")
    )
    bank_profile = facts.get("bank_profile") or {}
    constraints  = bank_profile.get("constraints") or facts.get("constraints") or []
    source_chunks = facts.get("source_chunks") or []

    # Known extra fees for Уралсиб
    extra_fee_details: dict = {}
    if bank == "Уралсиб":
        extra_fee_details = _URALSIB_EXTRA_FEES.copy()

    base["action"]           = "answer"
    base["intent"]           = "extra_fees"
    base["bank"]             = bank
    base["client_type"]      = client_type
    base["extra_fee_details"] = extra_fee_details
    base["constraints"]      = constraints
    base["items"]            = []
    base["source_chunks"]    = source_chunks[:4]

    if not bank:
        base["question_to_ask"] = "bank_name"
        slots["_pending_question_type"] = "bank_name"
    else:
        slots.pop("_pending_question_type", None)
        if bank:
            slots["_last_bank"] = bank

    return base


def _plan_constraint(base: dict, facts: dict, slots: dict, decision: dict, client_type) -> dict:
    planner = decision.get("planner") or {}
    topic = (
        slots.get("_constraint_topic")
        or decision.get("constraint_topic")
        or planner.get("constraint_topic")
        or planner.get("answer_focus")
    )
    scenario_topic = decision.get("scenario_topic") or planner.get("scenario_topic")
    next_action = decision.get("next_action") or planner.get("next_action")

    # Use NEXT_AFTER_CONSTRAINT map for deterministic routing
    if topic and not next_action:
        from app.domain.scenario_catalog import NEXT_AFTER_CONSTRAINT
        next_action = NEXT_AFTER_CONSTRAINT.get(topic)

    bank_profile = facts.get("bank_profile") or {}
    constraints = bank_profile.get("constraints") or facts.get("constraints") or []
    bank = (
        slots.get("_current_bank_mention")
        or bank_profile.get("bank")
        or facts.get("bank")
        or slots.get("bank_name")
        or slots.get("_last_bank")
    )

    # High-risk constraints get deterministic wording even if KB retrieval is sparse.
    must_use = list(constraints[:3])
    resolved_client = bank_profile.get("client_type") or facts.get("client_type") or client_type
    fallback_text = constraint_fallback(topic, bank=bank, client_type=resolved_client)
    if not must_use:
        must_use = [fallback_text]

    base["action"]         = "answer"
    base["intent"]         = "constraint"
    base["bank"]           = bank
    base["client_type"]    = resolved_client
    base["constraints"]    = constraints
    base["must_use_facts"] = must_use
    base["constraint_topic"] = topic
    base["scenario_topic"] = scenario_topic
    base["next_action"]    = next_action
    base["answer_text"]    = fallback_text
    base["items"]          = []
    base["docs"]           = []

    # Route post-constraint flow by topic
    resolved_ct = base.get("client_type") or client_type
    if topic == "special_account_without_main":
        if not resolved_ct:
            slots["_pending_question_type"] = "client_type_after_constraint"
            base["question_to_ask"] = "client_type_after_constraint"
        else:
            slots["_pending_question_type"] = "confirm_client_type_after_constraint"
            slots["_suggested_client_type"] = resolved_ct
            base["question_to_ask"] = "priority"
    elif topic in ("fl_realization_card_signed_by_financial_manager", "debtor_card_realization"):
        # Do NOT ask ЮЛ/ИП/ФЛ — ask about realization status instead
        slots["_pending_question_type"] = "realization_status"
        slots["_last_constraint_topic"] = topic
        base["question_to_ask"] = "realization_status"
    else:
        base["question_to_ask"] = None
    return base

def _plan_conditions(base: dict, facts: dict, slots: dict, client_type) -> dict:
    """Plan for 'conditions' queries — broad overview: pricing + bonus + docs + constraints."""
    bank_profile = facts.get("bank_profile") or {}
    bank = (
        bank_profile.get("bank")
        or facts.get("bank")
        or slots.get("_last_bank")
        or slots.get("bank_name")
    )

    of = bank_profile.get("opening_fee")
    mf = bank_profile.get("monthly_fee")
    tf = bank_profile.get("transfer_fee")
    br = bank_profile.get("bonus_rate")
    docs        = bank_profile.get("docs")        or facts.get("docs")        or []
    constraints = bank_profile.get("constraints") or facts.get("constraints") or []

    items: list = []
    if of is not None:
        items.append({"label": "Открытие счёта", "value": f"{int(of)} руб."})
    if mf is not None:
        items.append({"label": "Ведение счёта",  "value": f"{int(mf)} руб./мес."})
    if tf is not None:
        items.append({"label": "Платежи",        "value": f"{int(tf)} руб."})
    if br:
        items.append({"label": "Бонус АУ",       "value": str(br)})

    base["action"]      = "answer"
    base["intent"]      = "conditions"
    base["bank"]        = bank
    base["client_type"] = bank_profile.get("client_type") or client_type
    base["items"]       = items
    base["docs"]        = docs[:3]
    base["constraints"] = constraints[:2]
    base["bonus_rate"]  = br
    base["opening_fee"] = of
    base["monthly_fee"] = mf
    base["transfer_fee"] = tf

    if bank:
        slots["last_offer_bank"] = bank
        slots["last_offer_type"] = "conditions"
    return base


def _plan_factual(base: dict, qmode: str, facts_result: dict, facts: dict,
                   slots: dict, decision: dict, client_type, confidence: float) -> dict:
    """Plan for specific_bank / pricing / docs / process."""
    if facts_result.get("retrieval_reason") == "conflict":
        return _plan_handoff(base, qmode, "data_conflict")

    bank_profile = facts.get("bank_profile") or {}
    bank         = bank_profile.get("bank") or facts.get("bank")

    # Process/constraint query: answer directly from constraints, no bank required
    if qmode == "process":
        constraints = bank_profile.get("constraints") or facts.get("constraints") or []
        if constraints:
            base["action"]      = "answer"
            base["intent"]      = "process"
            base["constraints"] = constraints
            base["bank"]        = bank
            base["client_type"] = bank_profile.get("client_type") or client_type
            if not bank:
                q = "bank_name" if client_type else "client_type"
                base["question_to_ask"] = q
                slots["_pending_question_type"] = q
            return base
        if client_type:
            return _plan_clarify(base, qmode, "bank_name", slots)
        return _plan_clarify(base, qmode, "other", slots)

    # Generic pricing/docs query with no known client_type and no explicit bank → clarify first
    if not client_type and qmode in ("pricing", "docs") and not slots.get("bank_name"):
        return _plan_clarify(base, qmode, "client_type", slots)

    if confidence < 0.25 and not bank:
        q = "bank_name" if not slots.get("bank_name") else "client_type"
        return _plan_clarify(base, qmode, q, slots)

    if confidence < 0.25:
        return _plan_handoff(base, qmode, "low_confidence")

    if bank_profile.get("status") == "PAUSE":
        base["intent"] = "no_candidates"
        base["bank"] = bank
        return base

    of = bank_profile.get("opening_fee")
    mf = bank_profile.get("monthly_fee")
    br = bank_profile.get("bonus_rate")

    # docs mode: don't include pricing items — keep the response focused
    if qmode == "docs":
        items: list = []
    else:
        items: list = []
        if of is not None:
            items.append({"label": "Открытие счёта", "value": f"{int(of)} руб."})
        if mf is not None:
            items.append({"label": "Ведение счёта",  "value": f"{int(mf)} руб./мес."})
        if br:
            items.append({"label": "Бонус АУ", "value": str(br)})

    if not items and bank and slots.get("_last_bank") == bank and slots.get("_last_items"):
        items = list(slots["_last_items"])

    docs        = bank_profile.get("docs")        or facts.get("docs")        or []
    constraints = bank_profile.get("constraints") or facts.get("constraints") or []

    base["action"]      = "answer"
    base["intent"]      = qmode
    base["bank"]        = bank
    base["client_type"] = bank_profile.get("client_type") or facts.get("client_type") or client_type
    base["items"]       = items
    base["docs"]        = docs
    base["constraints"] = constraints
    base["status"]      = bank_profile.get("status") or facts.get("status")
    # Normalized fields for direct renderer access (no label parsing needed)
    base["opening_fee"] = of
    base["monthly_fee"] = mf
    base["bonus_rate"]  = br

    sales_stage = slots.get("sales_stage") or "QUALIFY"
    stage_q = STAGE_NEXT_STEP.get(sales_stage)

    if not client_type:
        base["question_to_ask"] = "client_type"
        slots["_pending_question_type"] = "client_type"
    elif decision.get("action") == "CLARIFY":
        base["question_to_ask"] = "other"
        slots["_pending_question_type"] = "other"
    elif stage_q:
        if not slots.get(stage_q.replace("_criteria", "")):
            base["question_to_ask"] = stage_q
            slots["_pending_question_type"] = stage_q
        else:
            slots.pop("_pending_question_type", None)
    else:
        slots.pop("_pending_question_type", None)

    if items and not base.get("question_to_ask"):
        base["funnel_next"] = "docs" if qmode == "pricing" else "pricing"

    if bank:
        slots["last_offer_bank"] = bank
        slots["last_offer_type"] = qmode

    return base


# ---------------------------------------------------------------------------
# Response plan builder
# ---------------------------------------------------------------------------
def build_response_plan(
    user_text: str,
    slots: dict,
    decision: dict,
    facts_result: dict,
) -> dict:
    """Deterministic router — no LLM. Delegates to sub-builders per route."""
    qmode       = decision.get("query_mode", "smalltalk")
    stage       = decision.get("stage", "OTHER")
    client_type = slots.get("client_type")
    priority    = slots.get("priority_criteria")
    confidence  = facts_result.get("confidence", 0.0)
    facts       = facts_result.get("facts", {})
    # Attach source_chunks so sub-builders can access them
    facts["source_chunks"] = facts_result.get("source_chunks") or []
    base        = _make_base(client_type)

    # Propagate new planner fields into the plan base
    for field in ("scenario_topic", "constraint_topic", "next_action", "amount", "transfer_target"):
        val = decision.get(field)
        if val is not None:
            base[field] = val

    if qmode == "out_of_scope":
        return _plan_out_of_scope(base, slots, decision)
    if qmode == "constraint":
        plan = _plan_constraint(base, facts, slots, decision, client_type)
        plan["reference_examples"] = _REFERENCE_EXAMPLES.get("constraint", [])
        return plan
    if qmode == "service":
        return _plan_service(base, stage)
    if qmode in ("intro", "smalltalk"):
        base["intent"] = qmode
        return base
    if decision.get("action") == "HANDOFF":
        return _plan_handoff(base, qmode, decision.get("handoff_reason") or "early_handoff")
    if qmode == "transfer_fee_quote":
        plan = _plan_transfer_fee_quote(base, facts, slots, decision, client_type)
    elif qmode == "extra_fees":
        plan = _plan_extra_fees(base, facts, slots, decision, client_type)
    elif qmode == "bank_selection":
        plan = _plan_bank_selection(base, facts, slots, decision, client_type, priority)
    elif qmode == "partner_banks":
        plan = _plan_partner_banks(base, slots)
    elif qmode in ("timing", "timing_docs"):
        plan = _plan_timing(base, qmode, facts, slots, client_type)
    elif qmode == "conditions":
        plan = _plan_conditions(base, facts, slots, client_type)
    else:
        plan = _plan_factual(base, qmode, facts_result, facts, slots, decision, client_type, confidence)

    intent = plan.get("intent")
    action = plan.get("action")
    if intent in ("pricing", "docs", "specific_bank", "bonus", "timing", "timing_docs",
                  "conditions", "transfer_fee_quote", "extra_fees"):
        ref_key = intent
    elif action in ("compare", "selection_opening", "clarify"):
        ref_key = action
    else:
        ref_key = intent or action

    plan["reference_examples"] = _REFERENCE_EXAMPLES.get(ref_key, [])

    return plan
