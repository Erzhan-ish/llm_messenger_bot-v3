"""Contextual KB query expansion.

Expands short or referential user messages using dialog state (slots, recent dialog)
so the KB retriever receives a richer query.

Examples:
  "какие тарифы там?" + last_bank=Уралсиб → "какие тарифы там Уралсиб ЮЛ"
  "а условия какие?" + active_scenario=bank_pricing_yul → "условия тарифы Альфа ТКБ Уралсиб ЮЛ"
  "да" after tariff offer → "тарифы активные банки ЮЛ"
"""
from __future__ import annotations

import re

# Short referential words that point back to something mentioned earlier
_REFERENTIAL_RE = re.compile(
    r"\b(там|у\s+них|у\s+него|такие\s+же|такой\s+же|тот\s+же|те\s+же"
    r"|такие|аналогичн\w+|похожие|подобные)\b",
    re.I | re.U,
)

# Tariff/condition indicators
_TARIFF_RE = re.compile(
    r"\b(тариф\w*|условия|условие|стоимость|цен[аы]|почём|сколько|открытие|ведение)\b",
    re.I | re.U,
)

# Short confirmations that should be expanded using last_bot_offer context
_CONFIRM_RE = re.compile(
    r"^\s*(да|ага|угу|ок|окей|хорошо|давайте|ладно|конечно|согласен|принял)\s*[.!]?\s*$",
    re.I | re.U,
)

# Active banks for each debtor type
_ACTIVE_BANKS_YUL = "Альфа-Банк ТКБ Уралсиб"
_ACTIVE_BANKS_FL = "ТКБ Уралсиб"


def expand_kb_query(
    user_text: str,
    slots: dict,
    recent_dialog: list[dict] | None = None,
) -> str:
    """Return expanded query string for KB retrieval.

    Returns the original text if no expansion is needed.
    """
    t = (user_text or "").strip()
    if not t:
        return t

    words = t.split()
    has_referential = bool(_REFERENTIAL_RE.search(t))
    is_short = len(words) <= 5
    is_confirm = bool(_CONFIRM_RE.match(t))

    # Only expand if short or referential or short confirmation
    if not (is_short or has_referential or is_confirm):
        return t

    debtor_type = slots.get("debtor_type") or slots.get("client_type") or ""
    last_bank = slots.get("_last_bank") or slots.get("bank_name") or ""
    active_scenario = slots.get("_active_scenario") or ""
    last_bot_question = slots.get("_last_bot_question") or ""

    parts: list[str] = [t]

    # Short confirmation: expand using last bot offer context
    if is_confirm:
        # "да" after "Могу сравнить тарифы?" → tariff comparison query
        lower_q = (last_bot_question or "").lower()
        if any(w in lower_q for w in ["тариф", "сравнить", "условия", "ценник"]):
            parts = ["тарифы активные банки сравнение"]
            if debtor_type:
                parts.append(debtor_type)
            if last_bank:
                parts.append(last_bank)
            return " ".join(parts)
        # Otherwise let original "да" propagate (will be handled as follow-up)
        return t

    # Add debtor type context
    if debtor_type:
        parts.append(debtor_type)

    # Add last mentioned bank
    if last_bank:
        parts.append(last_bank)
    elif has_referential:
        # "там"/"у них" — try to resolve bank from recent dialog
        resolved = _resolve_bank_from_dialog(recent_dialog or [])
        if resolved:
            parts.append(resolved)

    # Add active bank list for tariff/condition queries
    if _TARIFF_RE.search(t) and not last_bank:
        if debtor_type == "ФЛ":
            parts.append(_ACTIVE_BANKS_FL)
        elif debtor_type in ("ЮЛ", "ИП") or "yul" in active_scenario:
            parts.append(_ACTIVE_BANKS_YUL)

    # DealState: ready_to_open + docs intent → boost document/requirements chunks
    _deal_state = slots.get("_deal_state") or {}
    _deal_stage = _deal_state.get("deal_stage") or ""
    _docs_words = re.compile(
        r"(требуется|нужно|нужен|документ|дальше|прислать|данные|что\s+от\s+меня)", re.I | re.U
    )
    if _deal_stage in ("ready_to_open", "bank_selected", "collecting_documents") and _docs_words.search(t):
        _selected_bank = _deal_state.get("selected_bank") or last_bank
        _debtor_type = _deal_state.get("debtor_type") or debtor_type
        _doc_parts = ["документы требования ИНН судебный акт арбитражный управляющий"]
        if _selected_bank:
            _doc_parts.append(_selected_bank)
        if _debtor_type:
            _doc_parts.append(_debtor_type)
        return " ".join(_doc_parts)

    # Add scenario-based topic keywords
    if "pricing" in active_scenario or "tariff" in active_scenario:
        parts.append("тарифы условия открытие ведение")
    elif "selection" in active_scenario:
        parts.append("банки партнёры список")
    elif "objection" in active_scenario or "benefit" in active_scenario:
        parts.append("выгода льготные условия сопровождение")

    expanded = " ".join(dict.fromkeys(parts))
    return expanded if expanded != t else t


def _resolve_bank_from_dialog(recent_dialog: list[dict]) -> str:
    """Try to find the last mentioned bank in recent dialog messages."""
    _BANK_RE = re.compile(
        r"\b(Альфа-Банк|альфа\b|ТКБ|ткб\b|Уралсиб|уралсиб\b|Т-Банк|МКБ|Росбанк)\b",
        re.I | re.U,
    )
    for m in reversed(recent_dialog or []):
        text = m.get("text") or ""
        match = _BANK_RE.search(text)
        if match:
            return match.group(0)
    return ""
