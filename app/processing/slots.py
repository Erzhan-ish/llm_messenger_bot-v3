# app/processing/slots.py
from __future__ import annotations

import re
from typing import Dict, Optional

"""Session slots — runtime extraction and normalization."""

DEFAULT_SLOTS: Dict[str, Optional[object]] = {
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
}

_EMAIL_RE      = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
_INN_RE        = re.compile(r"\b(\d{10}|\d{12})\b")
_INN_LABELED_RE = re.compile(r"(?i)\bинн[:\s]*([0-9]{10}|[0-9]{12})\b")


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
    pending = slots.get("_pending_question_type")
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

    # First pass: set if unknown
    if not slots.get("client_type"):
        if any(x in t for x in ["физ лицо", "фл", "физлицо", "физическое лицо"]):
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

    return slots
