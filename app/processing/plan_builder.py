"""Response plan builders — fully deterministic, no LLM calls."""
from __future__ import annotations

import re

from app.processing.utils import _FALLBACK_TEXT
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
    ]
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
    base        = _make_base(client_type)

    if qmode == "service":
        return _plan_service(base, stage)
    if qmode in ("intro", "smalltalk"):
        base["intent"] = qmode
        return base
    if decision.get("action") == "HANDOFF":
        return _plan_handoff(base, qmode, decision.get("handoff_reason") or "early_handoff")
    if qmode == "bank_selection":
        plan = _plan_bank_selection(base, facts, slots, decision, client_type, priority)
    elif qmode == "partner_banks":
        plan = _plan_partner_banks(base, slots)
    else:
        plan = _plan_factual(base, qmode, facts_result, facts, slots, decision, client_type, confidence)

    intent = plan.get("intent")
    action = plan.get("action")
    if intent in ("pricing", "docs", "specific_bank", "bonus"):
        ref_key = intent
    elif action in ("compare", "selection_opening", "clarify"):
        ref_key = action
    else:
        ref_key = intent or action
    
    plan["reference_examples"] = _REFERENCE_EXAMPLES.get(ref_key, [])

    return plan
