# app/processing/slots.py
from __future__ import annotations

import re
from typing import Dict, Optional

"""Session slots — runtime extraction and normalization."""

DEFAULT_SLOTS: Dict[str, Optional[object]] = {
    # Debtor context
    "procedure_type":         None,  # "restructuring" / "liquidation" / "external_management" / "competitive"
    "debtor_status":          None,  # "deceased" / "non_resident" / "liquidated"
    # Internal tracking
    "_introduced":            False,
    "_escalation_sent":       False,
    "_escalation_reason":     None,
    "_interest_scores":       [],
    "_interest_score_last":   None,
    "_escalation_signal_last": None,
    # Sales funnel state
    "sales_stage":            None,       # QUALIFY / SELECT / PRESENT / OBJECTION / READY / HANDOFF
    "lead_temperature":       "info_only",# info_only / interested / warm / ready
    "ready_to_open":          False,
    "last_offer_bank":        None,
    "last_offer_type":        None,
    "objection_type":         None,       # price / offline_visit / trust / speed / other
    "partner_scope_shown":    False,
    "_turn_count":            0,
    "_callback_requested":    False,
    "_last_plan_type":        None,
    "_last_bank_anchor":      None,   # банк, явно названный в последнем ответе бота
    # Active task — carries current dialog intent across follow-up messages.
    "_active_task":           None,
}

_EMAIL_RE      = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
_INN_RE        = re.compile(r"\b(\d{10}|\d{12})\b")
_INN_LABELED_RE = re.compile(r"(?i)\bинн[:\s]*([0-9]{10}|[0-9]{12})\b")

# Amount extraction: "200 000", "200000", "200 тыс", "30 млн", "2 млрд"
_AMOUNT_RE = re.compile(
    r"(\d[\d\s]*(?:\.\d+)?)\s*(млрд|млн|тыс(?:яч)?\.?)?(?:\s*руб\.?)?",
    re.I | re.U,
)
_TRANSFER_TO_FL_RE = re.compile(
    r"\b(физ\s*лиц\w*|фл\b|физик\w*|кредитору[-\s]*физ|гражданин\w*)\b",
    re.I | re.U,
)
_TRANSFER_TO_UL_RE = re.compile(
    r"\b(юр\s*лиц\w*|юл\b|ооо\b|компани\w*|организаци\w*)\b",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Client style detection
# ---------------------------------------------------------------------------
_HURRIED_RE = re.compile(
    r"\b(срочно|побыстрее|быстро|некогда|вкратце|кратко|коротко"
    r"|без\s+деталей|скорее|только\s+главное|в\s+двух\s+словах"
    r"|нет\s+времени|сжато)\b",
    re.I | re.U,
)
_DOUBTFUL_RE = re.compile(
    r"\b(а\s+вдруг|не\s+уверен|не\s+уверена|сомнева\w+|рискованно|надёжно\s+ли"
    r"|ненадёжно|гарантии|гарантия|а\s+если|что\s+если|а\s+что\s+если"
    r"|какие\s+минусы|минусы|подводные\s+камни|риски|риск|опасения"
    r"|нет\s+ли|есть\s+ли\s+риск|насколько\s+надёжно|смущает)\b",
    re.I | re.U,
)
_DETAILED_RE = re.compile(
    r"\b(подробнее|поподробнее|расскажите|объясните|как\s+именно|детально"
    r"|все\s+детали|полностью|хочу\s+узнать\s+больше|расскажи|подробно"
    r"|развёрнуто|поподробней|расписать|расписите)\b",
    re.I | re.U,
)


def _update_client_style(text: str, slots: Dict) -> None:
    """Накапливает сигналы стиля клиента и обновляет slots['client_style'].

    Победитель определяется при накоплении ≥2 баллов — защита от случайной смены стиля.
    """
    t = text or ""
    signals: dict = slots.setdefault(
        "_style_signals", {"hurried": 0, "doubtful": 0, "detailed": 0}
    )

    if _HURRIED_RE.search(t):
        signals["hurried"] += 2
    elif slots.get("_last_bot_text") and len(t.split()) <= 4:
        signals["hurried"] += 1

    if _DOUBTFUL_RE.search(t):
        signals["doubtful"] += 2

    if _DETAILED_RE.search(t):
        signals["detailed"] += 2
    elif len(t.split()) >= 20:
        signals["detailed"] += 1

    msg_count = slots.get("_msg_count", 0) + 1
    slots["_msg_count"] = msg_count
    if msg_count % 5 == 0:
        for k in signals:
            signals[k] = max(0, signals[k] - 1)

    best = max(signals, key=lambda k: signals[k])
    if signals[best] >= 2:
        slots["client_style"] = best


def normalize_email(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().replace("!@", "@").replace(" @", "@").replace("@ ", "@")
    m = _EMAIL_RE.search(t)
    return m.group(0) if m else None


# Bank name aliases -> canonical
_BANK_PATTERNS: list[tuple[list[str], str]] = [
    (["альфа-банк", "альфа банк", "альфа", "alfabank", "alfa bank", "alfa"],      "Альфа-Банк"),
    (["ткб", "транскапитал", "транскапиталбанк", "transcapital", "tkb"],           "ТКБ"),
    (["уралсиб", "uralsib"],                                                        "Уралсиб"),
    (["т-банк", "т банк", "тинькофф", "тинькоф", "tbank", "t-bank", "tinkoff"],   "Т-Банк"),
    (["мкб", "московский кредитный"],                                               "МКБ"),
    (["росбанк", "rosbank"],                                                        "Росбанк"),
]


def _detect_bank(t: str) -> Optional[str]:
    for patterns, canonical in _BANK_PATTERNS:
        for p in patterns:
            if p in t:
                return canonical
    return None


_CONTEXT_KEYS = (
    "_last_bank", "_last_bank_anchor", "_last_items", "_last_docs", "_last_candidates",
    "_last_client_type", "_last_plan_type", "_last_mode", "_last_action",
    "_last_intent", "_last_expected_followup", "_last_expected_followup_turn",
)


def _invalidate_last_context(slots: Dict, *, reason: str = "") -> None:
    """Clear stale _last_* context when bank or client_type changes."""
    for k in _CONTEXT_KEYS:
        slots.pop(k, None)


def extract_runtime_slots(text: str, slots: Dict) -> Dict:
    """Early runtime slot extraction.

    Extracts and normalises: client_type, priority_criteria, bank_name, inn, email.
    Resolves pending slot first if _pending_question_type is set.
    """
    t = (text or "").strip().lower()

    # 1. Pending slot resolution — attempt to close before general extraction
    # If user sends a substantive message (>6 words) that doesn't look like a direct answer,
    # the pending question is stale — clear it to avoid misrouting.
    pending = slots.get("_pending_question_type")
    if pending and len(t.split()) > 6:
        slots.pop("_pending_question_type", None)
        pending = None
    if pending == "client_type":
        if any(x in t for x in ["ооо", "юр лицо", "юр.", "юл", "юрлицо", "организация", "компания"]):
            slots["client_type"] = "ЮЛ"
            slots.pop("_pending_question_type", None)
        elif any(x in t for x in ["ип", "индивидуальный предприниматель", "предприниматель"]):
            slots["client_type"] = "ИП"
            slots.pop("_pending_question_type", None)
        elif any(x in t for x in ["физ", "фл", "физлицо", "физическое лицо"]):
            slots["client_type"] = "ФЛ"
            slots.pop("_pending_question_type", None)
    elif pending == "priority":
        if any(x in t for x in ["подешевле", "дешевле", "недорого", "цена", "стоимость", "бюджет"]):
            slots["priority_criteria"] = "price"
            slots.pop("_pending_question_type", None)
        elif any(x in t for x in ["срочно", "быстро", "побыстрее", "скорость", "скорее"]):
            slots["priority_criteria"] = "speed"
            slots.pop("_pending_question_type", None)
    elif pending == "bank_name":
        bank = _detect_bank(t)
        if bank:
            slots["bank_name"] = bank
            slots.pop("_pending_question_type", None)

    # 2. client_type normalization
    _prev_client_type = slots.get("client_type")

    # Priority correction check (Entity Stickiness purge)
    correction_match = re.search(r"(?:нет|не)\s*(?:ип|фл|физ[\.\s]*лиц\w*|юл|ооо|юр[\.\s]*лиц\w*|физик).*(?:а\s+)?(ооо|ип|юл|фл|физ[\.\s]*лиц\w*|юр[\.\s]*лиц\w*|физик)", t)
    if correction_match:
        corrected = correction_match.group(1)
        if any(x in corrected for x in ["ооо", "юр", "юл"]):
            slots["client_type"] = "ЮЛ"
        elif any(x in corrected for x in ["ип"]):
            slots["client_type"] = "ИП"
        elif any(x in corrected for x in ["физ", "фл", "физик"]):
            slots["client_type"] = "ФЛ"

    # First pass: set if unknown
    # NOTE: "должник", "банкрот", "списывают долги" intentionally NOT included —
    # these describe the debtor type, not the client (АУ) organisation type.
    if not slots.get("client_type"):
        if any(x in t for x in ["физ лиц", "фл", "физлицо", "физическое",
                                "физ.", "физику", "физики", "физик"]):
            slots["client_type"] = "ФЛ"
        elif any(x in t for x in ["ооо", "юр лицо", "юл", "юрлицо", "организаци"]):
            slots["client_type"] = "ЮЛ"
        elif any(x in t for x in ["ип ", " ип", "индивидуальный предприниматель", "предприниматель"]):
            slots["client_type"] = "ИП"

    # Explicit override — fires even if client_type already set
    # ФЛ: "у меня физ лицо", "я физ лицо", "физическое лицо", "фл"
    if re.search(r"\b(у\s+меня|я|это|физическое)\s+(физ[\.\s]*лиц\w*|фл)\b"
                 r"|^(фл|физлицо|физическое\s+лицо)\s*[.!?]?\s*$", t):
        slots["client_type"] = "ФЛ"
    elif re.search(r"\b(я\s+)?(ип|предприниматель)\b", t):
        slots["client_type"] = "ИП"
    elif re.search(r"\b(я\s+)?(ооо|юр\.?\s*лицо|юл)\b", t):
        slots["client_type"] = "ЮЛ"

    # Invalidate cached context if client_type changed — prior responses were for wrong type
    if slots.get("client_type") and slots["client_type"] != _prev_client_type and _prev_client_type:
        _invalidate_last_context(slots, reason="client_type_changed")

    # 3. priority_criteria
    if not slots.get("priority_criteria"):
        if any(x in t for x in ["подешевле", "дешевле", "недорого", "минимальная цена", "тариф пониже", "бюджетн"]):
            slots["priority_criteria"] = "price"
        elif any(x in t for x in ["срочно", "побыстрее", "быстро", "скорость", "быстрее"]):
            slots["priority_criteria"] = "speed"

    # 4. bank_name hint (may override if explicitly named)
    _prev_bank = slots.get("bank_name")
    bank_hint = _detect_bank(t)
    if bank_hint:
        # Invalidate cached context if bank changed
        if _prev_bank and bank_hint != _prev_bank:
            _invalidate_last_context(slots, reason="bank_changed")
        slots["bank_name"] = bank_hint

    # 5. product_type
    if not slots.get("product_type"):
        if any(x in t for x in ["задатков", "задатк"]):
            slots["product_type"] = "задатковый"
        elif any(x in t for x in ["залогов", "залог"]):
            slots["product_type"] = "залоговый"
        elif any(x in t for x in ["спецсчет", "спец счет", "специальный счет"]):
            slots["product_type"] = "спецсчет"
        elif any(x in t for x in ["расчетный", "расчётный", "рко", "основной счет"]):
            slots["product_type"] = "расчетный"

    # 6. INN
    m_inn = _INN_LABELED_RE.search(text)
    if m_inn:
        slots["inn"] = m_inn.group(1)
    elif not slots.get("inn"):
        m_inn2 = _INN_RE.search(text)
        if m_inn2:
            slots["inn"] = m_inn2.group(1)

    # 7. Email
    email = normalize_email(text)
    if email:
        slots["email"] = email

    # 8. procedure_type (тип процедуры банкротства)
    if not slots.get("procedure_type"):
        if any(x in t for x in ["реструктуризаци", "реструктур"]):
            slots["procedure_type"] = "restructuring"
        elif any(x in t for x in ["реализаци имущества", "реализаци", "конкурсное производств"]):
            slots["procedure_type"] = "liquidation"
        elif any(x in t for x in ["внешнее управлени", "внешн управлени"]):
            slots["procedure_type"] = "external_management"

    # 9. debtor_status (особый статус должника)
    if not slots.get("debtor_status"):
        if any(x in t for x in ["умер", "покойн", "скончал", "посмертн"]):
            slots["debtor_status"] = "deceased"
        elif any(x in t for x in ["нерезидент", "иностранец", "иностранная компани", "иностранн организаци"]):
            slots["debtor_status"] = "non_resident"
        elif any(x in t for x in ["ликвидирован", "уже ликвидирован", "ликвидаци завершен"]):
            slots["debtor_status"] = "liquidated"

    # 10. Client style (accumulated signals)
    _update_client_style(text, slots)

    # 11. Current bank mention — always reflects the bank named in THIS message.
    # Has higher priority than _last_bank in plan/retrieval decisions.
    bank_in_msg = _detect_bank(t)
    if bank_in_msg:
        slots["_current_bank_mention"] = bank_in_msg
    else:
        slots.pop("_current_bank_mention", None)

    # 12. Transfer amount extraction (for transfer_fee_quote intent)
    _m = _AMOUNT_RE.search(t)
    if _m:
        raw_num = _m.group(1).replace(" ", "")
        try:
            val = float(raw_num)
            suffix = (_m.group(2) or "").lower()
            if "млрд" in suffix:
                val *= 1_000_000_000
            elif "млн" in suffix:
                val *= 1_000_000
            elif "тыс" in suffix:
                val *= 1_000
            slots["_transfer_amount"] = int(val)
        except ValueError:
            pass
    else:
        slots.pop("_transfer_amount", None)

    # 13. Transfer target type (ФЛ/ЮЛ) for transfer_fee_quote
    if _TRANSFER_TO_FL_RE.search(t):
        slots["_transfer_target"] = "ФЛ"
    elif _TRANSFER_TO_UL_RE.search(t):
        slots["_transfer_target"] = "ЮЛ"

    return slots


# ---------------------------------------------------------------------------
# Active task helpers
# ---------------------------------------------------------------------------

def get_active_task(slots: Dict) -> Dict:
    """Return current active_task dict or empty dict."""
    return slots.get("_active_task") or {}


def update_active_task(slots: Dict, decision: Dict) -> None:
    """Persist the planner decision as the active task for follow-up resolution.

    Called after a successful plan+render cycle so the next message knows
    what we were talking about.
    """
    planner = decision.get("planner") or {}
    intent = planner.get("intent") or decision.get("query_mode") or ""

    # These intents carry multi-turn context worth preserving
    trackable = {
        "transfer_fee_quote", "extra_fees", "pricing", "docs", "timing",
        "constraint", "bank_selection", "specific_bank", "conditions", "bonus",
        "access_cards", "operations",
    }

    if intent not in trackable:
        # Don't overwrite a real task with a transient service reply
        return

    task: Dict = {
        "intent":            intent,
        "scenario_topic":    decision.get("scenario_topic") or planner.get("scenario_topic"),
        "constraint_topic":  decision.get("constraint_topic") or planner.get("constraint_topic"),
        "bank_name": (
            decision.get("bank_name")
            or planner.get("bank_name")
            or slots.get("_current_bank_mention")
            or slots.get("bank_name")
            or slots.get("_last_bank")
        ),
        "client_type":       planner.get("client_type") or slots.get("client_type"),
        "amount":            decision.get("amount") or planner.get("amount") or slots.get("_transfer_amount"),
        "transfer_target":   decision.get("transfer_target") or planner.get("transfer_target") or slots.get("_transfer_target"),
        "pending_question":  planner.get("next_question"),
        "next_action":       decision.get("next_action") or planner.get("next_action"),
        "last_answer_summary": None,
    }
    slots["_active_task"] = task


def clear_active_task(slots: Dict) -> None:
    slots["_active_task"] = None
