"""Response plan builders — fully deterministic, no LLM calls."""
from __future__ import annotations

import re

from app.processing.utils import _FALLBACK_TEXT
from app.services.sales_policy import OBJECTION_TYPE_MAP, STAGE_NEXT_STEP

# ---------------------------------------------------------------------------
# Follow-up detection regexes
# ---------------------------------------------------------------------------
_FOLLOWUP_RE = re.compile(
    r"^\s*(ещё|еще|что\s+(ещё|еще)|ну|и|дальше|далее|подробнее|поподробнее"
    r"|я\s+(уже\s+)?(сказал|написал|говорил|указал)|я\s+же\s+сказал"
    r")\s*[?!\.]?\s*$",
    re.I | re.U,
)
_FOLLOWUP_WHERE_RE = re.compile(
    r"^\s*(где|где\s+(это|открыть|находится|у\s+вас)|в\s+каком\s+банке)\s*[?!]?\s*$",
    re.I | re.U,
)
_FOLLOWUP_ONLY_RE = re.compile(
    # "только что" исключаем — это про время, не про единственность варианта
    r"(только\s+(?!что\b)\w+(\s+(есть|доступ\w+))?|других\s+нет|больше\s+нет|других\s+вариантов\s+нет"
    r"|один\s+вариант|нет\s+других|всего\s+один"
    r"|какие\s+еще\s+(есть|варианты|банки)\b|что\s+еще\s+есть|еще\s+варианты\s+есть"
    r"|есть\s+еще\s+(варианты|банки)|другие\s+варианты\s+есть|а\s+другие\s+\w+\s+есть)",
    re.I | re.U,
)
_FOLLOWUP_CONDITIONS_RE = re.compile(
    r"(какие\s+условия|ещ[её]\s+условия|подробнее\s+(об?\s+)?условия|условия\s+счет"
    r"|условия\s+тариф|расскажи\s+условия|что\s+за\s+условия|а\s+условия|а\s+тариф)",
    re.I | re.U,
)
_FOLLOWUP_PARAMS_RE = re.compile(
    r"(по\s+каким\s+(еще\s+)?параметрам|какие\s+параметры|каким\s+параметрам"
    r"|что\s+за\s+параметры|объясни\s+подробнее|почему\s+(именно\s+)?этот)",
    re.I | re.U,
)
_YES_RE = re.compile(
    r"^\s*(да|да\s*,?\s*(хочу|давайте|конечно|пожалуйста|интересно|расскажи|можно)"
    r"|ок\s+давайте|хочу\s+узнать|да\s+пожалуйста|ну\s+да|конечно\s+да"
    r"|расскажите?|давайте|можно|интересно)\s*[.!?]?\s*$",
    re.I | re.U,
)
_EXPAND_BANKS_RE = re.compile(
    r"^(а\s+(в\s+|у\s+)?(уралсиб|альфа|ткб|т-банк|мкб|росбанк|россельхоз)\w*"
    r"|а\s+(другие|остальные|ещ[её])\s+(банки|варианты)"
    r"|а\s+в\s+других\s+банках"
    r"|что\s+по\s+(уралсибу|альфе|ткб|т-банку|мкб|росбанку))",
    re.I | re.U,
)
_TIMING_RE = re.compile(
    r"\b(как\s+долго|сколько\s+(времени|дней?|ждать)|срок\s+открыт|за\s+сколько\s+(откр|дн)|"
    r"когда\s+(будет\s+)?готов|долго\s+ли|быстро\s+ли\s+откр|время\s+открыт)",
    re.I | re.U,
)
_TIMING_REPLY = (
    "Счёт обычно открывают за 1–3 рабочих дня с момента подачи документов. "
    "В некоторых банках — быстрее, если все документы в порядке. "
    "Хотите узнать, что нужно подготовить?"
)
_PARTNER_BANKS_RE = re.compile(
    r"(с\s+кем\s+(вы\s+)?(работаете|сотрудничаете)|какие\s+банки\s+у\s+вас"
    r"|с\s+какими\s+банками|какие\s+(у\s+вас\s+)?партнеры|банки\s*[?!]$"
    r"|список\s+банков|ваши\s+банки|какие\s+есть\s+банки)",
    re.I | re.U,
)

# Follow-up: "а открытие сколько?", "ведение стоит?", "а в месяц?" — компонент цены по тому же банку
_PRICE_COMPONENT_RE = re.compile(
    r"(открытие\s+(сколько|стоит|цена)|а\s+открытие|сколько\s+(стоит\s+)?открыт"
    r"|ведение\s+(сколько|стоит)|а\s+ведение|в\s+месяц\s+сколько|сколько\s+в\s+месяц"
    r"|платёж[а-я]*\s+сколько|переводы?\s+сколько|комисси[а-я]+\s+за"
    r"|а\s+ежемесячн|ежемесячное?\s+(плат|стоит|сколько))",
    re.I | re.U,
)

# Follow-up: "вы же сказали 1600", "только что было другое" — противоречие в цифрах
_CONTRADICTION_RE = re.compile(
    r"(вы\s+же\s+(сказали|говорили)|ты\s+(же\s+)?(сказал|говорил)"
    r"|только\s+что\s+(было|сказал|написал|говорил|сказали)"
    r"|сначала\s+(одно|другое|было|сказал|сказали)"
    r"|раньше\s+(было|сказал|написал|сказали)|недавно\s+(сказал|написал|сказали)"
    r"|теперь\s+(говоришь|пишешь|говорите|пишете)\s+(другое|иначе)"
    r"|противоречи|сначала\s+\w+\s+а\s+теперь|сначала\s+написал)",
    re.I | re.U,
)

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


def _plan_bank_selection(base: dict, facts: dict, slots: dict,
                          client_type, priority) -> dict:
    all_banks  = facts.get("all_found_banks") or []
    candidates = [c for c in all_banks if c.get("status") == "ACTIVE" and c.get("rank_score", 0) > 0]

    if not candidates:
        if not client_type:
            return _plan_clarify(base, "bank_selection", "client_type", slots)
        base["intent"] = "no_candidates"
        return base

    candidates = sorted(candidates, key=lambda x: x.get("rank_score", 0.0), reverse=True)

    sales_stage = slots.get("sales_stage")
    is_first_selection = sales_stage in (None, "QUALIFY", "SELECT") and not slots.get("_last_candidates")

    if len(candidates) == 1:
        c = candidates[0]
        base["action"]      = "selection_opening"
        base["intent"]      = "bank_selection"
        base["bank"]        = c["bank"]
        base["client_type"] = c.get("client_type") or client_type
        base["candidates"]  = candidates
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
        return _plan_selection_opening(base, top, slots, client_type, priority)

    # Repeat / explicit comparison request → compare
    base["action"]     = "compare"
    base["intent"]     = "bank_selection"
    base["candidates"] = top
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
    """Plan for specific_bank / pricing / docs."""
    if facts_result.get("retrieval_reason") == "conflict":
        return _plan_handoff(base, qmode, "data_conflict")

    bank_profile = facts.get("bank_profile") or {}
    bank         = bank_profile.get("bank") or facts.get("bank")

    if confidence < 0.25 and not bank:
        q = "bank_name" if not slots.get("bank_name") else "client_type"
        return _plan_clarify(base, qmode, q, slots)

    if confidence < 0.25:
        return _plan_handoff(base, qmode, "low_confidence")

    if bank_profile.get("status") == "PAUSE":
        base["intent"] = "no_candidates"
        return base

    items: list = []
    of = bank_profile.get("opening_fee")
    mf = bank_profile.get("monthly_fee")
    if of is not None:
        items.append({"label": "Открытие счёта", "value": f"{int(of)} руб."})
    if mf is not None:
        items.append({"label": "Ведение счёта",  "value": f"{int(mf)} руб./мес."})
    br = bank_profile.get("bonus_rate")
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
# Follow-up plan builders
# ---------------------------------------------------------------------------
def _plan_selection_explain(base: dict, slots: dict) -> dict:
    """'только ТКБ?' / 'других нет?' — объяснить границы выбора."""
    candidates  = slots.get("_last_candidates") or []
    client_type = slots.get("client_type") or slots.get("_last_client_type")
    base["action"]      = "selection_explain"
    base["intent"]      = "availability_scope"
    base["candidates"]  = candidates
    base["client_type"] = client_type

    n = len(candidates)
    ct_label = client_type or "данного типа клиента"
    if n == 1:
        bank_name = candidates[0].get("bank", "") if candidates else ""
        base["allowed_points"] = [
            f"для {ct_label} сейчас доступен именно {bank_name}",
            "список актуальный — другие варианты либо на паузе, либо недоступны",
            "готов разобрать этот вариант подробнее",
        ]
    else:
        names = ", ".join(c.get("bank", "") for c in candidates if c.get("bank"))
        base["allowed_points"] = [
            f"для {ct_label} сейчас доступны: {names}",
            "это текущий рабочий список — другие варианты временно недоступны",
            "предложу выбрать один для детального разбора",
        ]
    return base


def _plan_pricing_expand(base: dict, slots: dict) -> dict:
    """'еще?' / 'тарифы условия' после ответа по цене — раскрыть детали."""
    base["action"]   = "pricing_expand"
    base["intent"]   = "pricing"
    base["bank"]     = slots.get("_last_bank") or slots.get("bank_name")
    items = slots.get("_last_items") or []
    base["items"]    = items
    base["client_type"] = slots.get("client_type") or slots.get("_last_client_type")
    base["allowed_points"] = [
        f"{i['label']}: {i['value']}" for i in items if i.get("label") and i.get("value")
    ]
    return base


def _plan_where_answer(base: dict, slots: dict) -> dict:
    """'где?' после ответа с банком — уточнить банк из контекста."""
    bank = slots.get("_last_bank") or slots.get("bank_name")
    base["action"] = "where_answer"
    base["intent"] = "location"
    base["bank"]   = bank
    return base


def _plan_partner_banks(base: dict, slots: dict) -> dict:
    """'с кем работаете?' — прямой список банков-партнёров."""
    client_type = slots.get("client_type") or slots.get("_last_client_type")
    base["action"]      = "partner_banks"
    base["intent"]      = "partner_banks"
    base["client_type"] = client_type
    return base


def _plan_params_explain(base: dict, slots: dict) -> dict:
    """'по каким параметрам?' / 'почему этот банк?' — объяснение критериев выбора."""
    base["action"]     = "params_explain"
    base["intent"]     = "selection_explain"
    base["bank"]       = slots.get("_last_bank") or slots.get("bank_name")
    candidates = slots.get("_last_candidates") or []
    items      = slots.get("_last_items") or []
    base["candidates"] = candidates
    base["items"]      = items
    base["client_type"] = slots.get("client_type") or slots.get("_last_client_type")
    points: list[str] = []
    for i in items[:3]:
        if i.get("label") and i.get("value"):
            points.append(f"{i['label']}: {i['value']}")
    for c in candidates[:2]:
        if c.get("main_feature"):
            points.append(c["main_feature"])
    base["allowed_points"] = points
    return base


def _plan_pricing_same_bank(base: dict, slots: dict) -> dict:
    """'а открытие сколько?' — раскрыть недостающий компонент цены по тому же банку.

    Не идёт в retriever — использует только сохранённые _last_items/_last_bank_anchor.
    """
    bank  = slots.get("_last_bank_anchor") or slots.get("_last_bank") or slots.get("bank_name")
    items = slots.get("_last_items") or []
    base["action"]        = "pricing_expand"
    base["intent"]        = "pricing"
    base["bank"]          = bank
    base["items"]         = items
    base["client_type"]   = slots.get("client_type") or slots.get("_last_client_type")
    base["allowed_points"] = [
        f"{i['label']}: {i['value']}" for i in items if i.get("label") and i.get("value")
    ]
    return base


def _plan_contradiction_repair(base: dict, slots: dict) -> dict:
    """'вы же сказали 1600' — признать расхождение, дать только подтверждённые факты.

    Статический ответ — LLM не вызывается, чтобы не добавить новых галлюцинаций.
    """
    bank  = slots.get("_last_bank_anchor") or slots.get("_last_bank") or slots.get("bank_name")
    items = slots.get("_last_items") or []
    base["action"]        = "contradiction_repair"
    base["intent"]        = "pricing"
    base["bank"]          = bank
    base["items"]         = items
    base["client_type"]   = slots.get("client_type") or slots.get("_last_client_type")
    base["allowed_points"] = [
        f"{i['label']}: {i['value']}" for i in items if i.get("label") and i.get("value")
    ]
    return base


# ---------------------------------------------------------------------------
# Semantic follow-up interpreter
# ---------------------------------------------------------------------------
def try_build_followup_plan(user_text: str, slots: dict) -> dict | None:
    """
    Перехватывает короткие follow-up реплики, опираясь на _last_* из slots.
    Возвращает готовый plan или None (→ обычный pipeline).
    """
    last_mode   = slots.get("_last_mode")
    last_action = slots.get("_last_action")
    has_history = bool(last_mode or slots.get("_last_bank") or slots.get("_last_candidates"))

    base = _make_base(slots.get("client_type") or slots.get("_last_client_type"))

    if _TIMING_RE.search(user_text):
        base["action"] = "timing"
        base["intent"] = "timing"
        return base

    if not has_history:
        return None

    # "вы же сказали 1600" — клиент указывает на противоречие в цифрах
    # Перехватываем ПЕРВЫМ — до _FOLLOWUP_ONLY_RE, чтобы "только что" не попало в selection_explain
    if _CONTRADICTION_RE.search(user_text):
        if slots.get("_last_bank") or slots.get("_last_items"):
            return _plan_contradiction_repair(base, slots)

    # "а открытие сколько?", "ведение стоит?" — компонент цены по тому же банку
    if _PRICE_COMPONENT_RE.search(user_text) and (
        slots.get("_last_bank_anchor") or slots.get("_last_bank")
    ):
        return _plan_pricing_same_bank(base, slots)

    if _YES_RE.match(user_text):
        expected     = slots.get("_last_expected_followup")
        offered_turn = slots.get("_last_expected_followup_turn", 0)
        current_turn = slots.get("_turn_count", 0)
        followup_fresh = (current_turn - offered_turn) <= 1

        if expected and followup_fresh:
            slots.pop("_last_expected_followup", None)
            slots.pop("_last_expected_followup_turn", None)

            if expected == "docs" and slots.get("_last_bank"):
                base["action"] = "answer"
                base["intent"] = "docs"
                base["bank"]   = slots.get("_last_bank")
                base["client_type"] = slots.get("client_type") or slots.get("_last_client_type")
                base["docs"] = slots.get("_last_docs") or []
                base["allowed_points"] = [f"документ: {d}" for d in (base["docs"] or [])]
                return base

            if expected == "pricing" and slots.get("_last_bank") and slots.get("_last_items"):
                return _plan_pricing_expand(base, slots)

    if _EXPAND_BANKS_RE.match(user_text):
        had_shortlist = bool(slots.get("_last_candidates")) or slots.get("_last_plan_type") in (
            "partner_banks", "selection_opening", "compare"
        )
        if had_shortlist:
            return _plan_partner_banks(base, slots)

    if _PARTNER_BANKS_RE.search(user_text):
        return _plan_partner_banks(base, slots)

    if _FOLLOWUP_WHERE_RE.match(user_text) and slots.get("_last_bank"):
        return _plan_where_answer(base, slots)

    if _FOLLOWUP_ONLY_RE.search(user_text):
        return _plan_selection_explain(base, slots)

    if _FOLLOWUP_PARAMS_RE.search(user_text):
        return _plan_params_explain(base, slots)

    if _FOLLOWUP_CONDITIONS_RE.search(user_text) and slots.get("_last_bank"):
        if slots.get("_last_items"):
            return _plan_pricing_expand(base, slots)

    last_plan_type = slots.get("_last_plan_type")
    if _FOLLOWUP_RE.match(user_text) and (
        last_mode in ("pricing", "specific_bank")
        or last_plan_type in ("selection_opening", "answer")
    ):
        if slots.get("_last_items"):
            return _plan_pricing_expand(base, slots)

    if last_plan_type == "selection_opening" and _FOLLOWUP_ONLY_RE.search(user_text):
        return _plan_selection_explain(base, slots)

    return None


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
        return _plan_bank_selection(base, facts, slots, client_type, priority)
    if qmode == "partner_banks":
        return _plan_partner_banks(base, slots)
    return _plan_factual(base, qmode, facts_result, facts, slots, decision, client_type, confidence)
