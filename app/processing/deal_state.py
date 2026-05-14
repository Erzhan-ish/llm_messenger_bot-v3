"""DealState — structured sales-stage memory.

Detects when the client moves from consultation to action, stores the result in
session slots under ``_deal_state``, and exposes helpers used by the pipeline.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Action / ready-to-open intent detection
# ---------------------------------------------------------------------------

_ACTION_INTENT_RE = re.compile(
    r"("
    # Direct account need/want
    r"нужен\s+(?:мне\s+)?сч[её]т"
    r"|сч[её]т\s+нужен"
    r"|хочу\s+открыть"
    r"|давайте\s+(?:открыть|открываем|откроем|оформим|открывать)"
    r"|открыть\s+сч[её]т"
    r"|оформляем|приступаем|запускаем"
    r"|готов\s+(?:открыть|запускать)|готовы\s+открыть"
    # Confirmation/question about opening
    r"|откроете\b|откроем\b"
    r"|берёте\s+в\s+работу|берете\s+в\s+работу"
    r"|возьм[её]те\s+в\s+работу|возьмёте\s+в\s+работу"
    r"|беретесь\b"
    # Handoff / manager
    r"|передайте\s+менеджеру"
    r"|подключите\s+менеджера"
    r"|позовите\s+менеджера"
    # Getting started
    r"|как\s+начать|начинаем"
    r"|как\s+оформить"
    r")",
    re.I | re.U,
)

# Phrases meaning "what docs/data do I need" — context-dependent
_DOCS_INTENT_RE = re.compile(
    r"("
    r"что\s+(?:от\s+меня\s+)?требуется"
    r"|от\s+меня\s+что\s+(?:нужно|требуется|надо)"
    r"|что\s+нужно\s+(?:от\s+меня|мне|сделать|подготовить|прислать|принести)"
    r"|что\s+надо\s+(?:от\s+меня|мне|сделать|принести)"
    r"|что\s+мне\s+(?:прислать|нужно|делать\s+дальше)"
    r"|что\s+дальше|дальше\s+что|следующий\s+шаг"
    r"|как\s+дальше|что\s+прислать|что\s+передать"
    r"|что\s+скинуть|что\s+скидывать|сюда\s+скидывать|сюда\s+скинуть"
    r"|какие\s+документы|какие\s+данные"
    r")",
    re.I | re.U,
)

# Short affirmatives after docs explanation — trigger escalation to ready_to_open
_DOCS_AFFIRM_RE = re.compile(
    r"^\s*(хорошо|ладно|давайте|да|ок|окей|отлично|подходит|принято|годится"
    r"|договорились|понял|ясно|понятно|всё\s+ясно|всё\s+понятно"
    r"|мне\s+подходит|нам\s+подходит|это\s+подходит|подойдёт|подойдет)\s*[.!?]?\s*$",
    re.I | re.U,
)

# §8 — Weak acknowledgements: these must NOT trigger post_docs_consent without a pending question
_WEAK_AFFIRM_RE = re.compile(
    r"^\s*(ну\s+ладно|понятно|понял|ясно|хорошо|ок|окей|угу|ага|ладно)\s*[.!?]?\s*$",
    re.I | re.U,
)

# Informational question phrasing — prevents misclassification as action intent
_QUESTION_PHRASING_RE = re.compile(
    r"\b(где|как\s+быстро|как\s+скоро|как\s+долго|сколько\s+по\s+времени"
    r"|по\s+времени\s+сколько|сколько\s+времени|сколько\s+дней|за\s+сколько"
    r"|быстрее\s+всего|быстрее\s+открыть|где\s+быстрее)\b",
    re.I | re.U,
)

_DEBTOR_DISPLAY: dict[str, str] = {
    "ЮЛ": "ЮЛ", "ИП": "ИП", "ФЛ": "ФЛ",
    "legal_entity": "ЮЛ", "individual": "ФЛ",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_action_intent(user_text: str) -> bool:
    """Return True if the user signals readiness to open / act."""
    t = user_text or ""
    if _QUESTION_PHRASING_RE.search(t):
        return False
    return bool(_ACTION_INTENT_RE.search(t))


def detect_docs_intent(user_text: str, slots: dict) -> bool:
    """Return True if user asks 'what do I need' in a deal context."""
    if not _DOCS_INTENT_RE.search(user_text or ""):
        return False
    ds = slots.get("_deal_state") or {}
    if ds.get("deal_stage") in ("bank_selected", "ready_to_open", "collecting_documents"):
        return True
    if slots.get("_last_bank") or slots.get("bank_name") or slots.get("debtor_type"):
        return True
    return False


def _get_selected_bank(slots: dict) -> Optional[str]:
    for k in ("_last_bank", "bank_name", "_current_bank_mention"):
        v = slots.get(k)
        if v:
            return v
    return None


def _get_debtor_type(slots: dict) -> Optional[str]:
    for k in ("debtor_type", "client_type"):
        v = slots.get(k)
        if v:
            return _DEBTOR_DISPLAY.get(v, v)
    return None


def update_deal_state(
    slots: dict,
    user_text: str,
    intent_signals: Optional[dict] = None,
    hard_only: bool = False,
) -> dict:
    """Update ``slots['_deal_state']`` and return the updated dict.

    hard_only=True: only sync slots, track last reply, and allow ready_to_open
    with full context (bank + debtor). Skips consulting/comparing_banks assignments
    so the LLM Planner can set business state first.
    """
    ds: dict = dict(slots.get("_deal_state") or {})

    ds.setdefault("deal_stage", None)
    ds.setdefault("selected_bank", None)
    ds.setdefault("selected_product", "bankruptcy_account")
    ds.setdefault("debtor_type", None)
    ds.setdefault("procedure_stage", None)
    ds.setdefault("client_intent", None)
    ds.setdefault("next_manager_move", None)
    ds.setdefault("ready_to_open_reason", None)
    ds.setdefault("handoff_needed", False)
    ds.setdefault("last_offer", None)
    ds.setdefault("last_bank_list_context", [])
    ds.setdefault("last_conditions_context", [])
    ds.setdefault("last_bot_reply_hash", None)

    stage_before = ds.get("deal_stage")

    # Sync bank/debtor/procedure from slots
    bank = _get_selected_bank(slots)
    debtor_type = _get_debtor_type(slots)
    if bank:
        ds["selected_bank"] = bank
    if debtor_type:
        ds["debtor_type"] = debtor_type
    procedure = slots.get("procedure_type") or slots.get("procedure_stage")
    if procedure:
        ds["procedure_stage"] = procedure

    # Track last bot reply for anti-echo and offer context
    _last_text = slots.get("_last_bot_text") or ""
    if _last_text:
        _new_hash = hashlib.md5(_last_text.encode("utf-8", errors="replace")).hexdigest()[:8]
        if ds.get("last_bot_reply_hash") != _new_hash:
            ds["last_bot_reply_hash"] = _new_hash
            ds["last_offer"] = _last_text[:200]
            # Extract bank mentions from the last bot reply
            _bank_re = re.compile(
                r"\b(Альфа-Банк|ТКБ|Уралсиб|Т-Банк|МКБ|Росбанк)\b", re.I | re.U
            )
            _mentioned = list(dict.fromkeys(m.group(0) for m in _bank_re.finditer(_last_text)))
            if _mentioned:
                ds["last_bank_list_context"] = _mentioned
            # Track conditions context when the reply contains pricing/conditions info
            if re.search(r"\b(открытие|ведение|тариф|условия|руб)\b", _last_text, re.I | re.U):
                _ctx_bank = ds.get("selected_bank") or slots.get("_last_bank") or ""
                if _ctx_bank:
                    ds["last_conditions_context"] = [f"{_ctx_bank}_conditions"]

    is_action = detect_action_intent(user_text)
    is_docs = detect_docs_intent(user_text, slots)
    intents = list((intent_signals or {}).get("intents") or [])

    if is_action:
        ds["client_intent"] = "open_account"
        if bank and debtor_type:
            # Full context: hard ready_to_open regardless of hard_only mode
            ds["deal_stage"] = "ready_to_open"
            ds["next_manager_move"] = "handoff_to_manager"
            ds["ready_to_open_reason"] = "action_intent_with_context"
            ds["handoff_needed"] = True
        elif not hard_only:
            # Partial context: only set in full mode (Planner handles ambiguous cases)
            if bank or debtor_type:
                ds["deal_stage"] = "bank_selected" if bank else (ds.get("deal_stage") or "consulting")
                ds["next_manager_move"] = "collect_missing_slot"
            else:
                ds["deal_stage"] = ds.get("deal_stage") or "consulting"
                ds["next_manager_move"] = "collect_missing_slot"

    elif (
        ds.get("next_manager_move") == "explain_documents"
        and bank
        and _DOCS_AFFIRM_RE.match(user_text or "")
        and ds.get("deal_stage") not in ("ready_to_open", "collecting_documents", "handoff")
        # §8: weak acknowledgements only count when there was an explicit pending question
        and not (
            _WEAK_AFFIRM_RE.match(user_text or "")
            and not (slots.get("_pending_question") or slots.get("_last_bot_question"))
        )
    ):
        # Post-documents consent: docs were just explained, user explicitly affirms → escalate
        ds["deal_stage"] = "ready_to_open"
        ds["client_intent"] = "open_account"
        ds["next_manager_move"] = "handoff_to_manager"
        ds["ready_to_open_reason"] = "post_docs_consent"
        ds["handoff_needed"] = True

    elif is_docs and not hard_only:
        ds["client_intent"] = "ask_documents"
        ds["next_manager_move"] = "explain_documents"
        if ds.get("deal_stage") not in ("ready_to_open", "collecting_documents", "handoff"):
            ds["deal_stage"] = "bank_selected" if bank else "consulting"

    elif not ds.get("deal_stage") and not hard_only:
        # §1: Planner decides business stage — only set defaults in full mode
        if "compare_banks" in intents or "bank_selection" in intents:
            ds["deal_stage"] = "comparing_banks"
            ds["client_intent"] = "compare"
        else:
            ds["deal_stage"] = "consulting"
            ds["client_intent"] = "ask_info"

    # Monotone: once ready_to_open, keep it (don't revert to consulting on follow-ups)
    if stage_before == "ready_to_open" and ds.get("deal_stage") not in (
        "ready_to_open", "collecting_documents", "handoff"
    ):
        if not is_action:
            ds["deal_stage"] = "ready_to_open"
            ds["handoff_needed"] = True

    slots["_deal_state"] = ds
    return ds
