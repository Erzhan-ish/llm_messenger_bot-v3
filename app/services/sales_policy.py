"""Sales state tracker — no LLM, pure rule-based.

Updates slots: sales_stage, lead_temperature, ready_to_open, objection_type.
Call update_sales_state() after decision is known, before plan building.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_READY_RE = re.compile(
    r"\b(давайте|оформляем|оформить|открывайте|что\s+дальше|куда\s+оплатить"
    r"|готов\s+начать|согласен|начнем|поехали|приступим"
    r"|подходит|сойдет|по\s+рукам|берём|берем"
    r"|пришлите\s+реквизиты|дайте\s+реквизиты)\b",
    re.I | re.U,
)

_CALLBACK_RE = re.compile(
    r"\b(перезвоните|позвоните|позвони|наберите|набери|свяжитесь|свяжись"
    r"|ваш\s+номер|дайте\s+номер|оставьте\s+контакты)\b",
    re.I | re.U,
)

_OBJECTION_SIGNALS_RE = re.compile(
    r"\b(дорого|слишком|не\s+хочу\s+ехать|без\s+визита|почему\s+через\s+вас"
    r"|не\s+уверен|подумаю|у\s+других|нашел\s+дешевле|зачем|сомнева|не\s+готов)\b",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Stage-specific next-step questions
# Used by plan builders to inject a single focused question per stage.
# ---------------------------------------------------------------------------

STAGE_NEXT_STEP: dict[str, str | None] = {
    "QUALIFY":   "client_type",   # → "для кого счёт?"
    "SELECT":    "priority",      # → "цена или скорость?"
    "PRESENT":   None,            # funnel_next handles (docs / pricing)
    "OBJECTION": None,            # objection reply carries its own question
    "READY":     None,            # INN bridge handles
    "HANDOFF":   None,
}

# ---------------------------------------------------------------------------
# Objection type mapping (from _detect_objection keys → sales taxonomy)
# ---------------------------------------------------------------------------

OBJECTION_TYPE_MAP: dict[str, str] = {
    "expensive":    "price",
    "no_visit":     "offline_visit",
    "competitor":   "price",
    "think":        "trust",
    "unsure":       "trust",
}


# ---------------------------------------------------------------------------
# Main updater
# ---------------------------------------------------------------------------

def update_sales_state(
    user_text: str,
    slots: dict,
    decision: dict,
    facts_result: dict | None = None,
) -> dict:
    """
    Determine current sales stage and temperature from user text + decision.
    Mutates and returns slots. Never regresses stage to an earlier one.
    """
    t = (user_text or "").strip()
    qmode = decision.get("query_mode", "smalltalk")
    current_stage = slots.get("sales_stage") or "QUALIFY"

    # Increment turn counter
    slots["_turn_count"] = int(slots.get("_turn_count") or 0) + 1

    # --- READY: client signals consent / "what's next" ---
    if _READY_RE.search(t) or decision.get("handoff_reason") == "ready_to_open":
        slots["sales_stage"]      = "READY"
        slots["lead_temperature"] = "ready"
        slots["ready_to_open"]    = True
        return slots

    # --- OBJECTION ---
    if _OBJECTION_SIGNALS_RE.search(t):
        # Don't regress from READY
        if current_stage != "READY":
            slots["sales_stage"] = "OBJECTION"
        if slots.get("lead_temperature") not in ("warm", "ready"):
            slots["lead_temperature"] = "interested"
        return slots

    # --- PRESENT: discussing specific bank / pricing / docs ---
    _STAGE_ORDER = ["QUALIFY", "SELECT", "PRESENT", "OBJECTION", "READY", "HANDOFF"]
    def _advance(new_stage: str) -> None:
        current_idx = _STAGE_ORDER.index(current_stage) if current_stage in _STAGE_ORDER else 0
        new_idx = _STAGE_ORDER.index(new_stage) if new_stage in _STAGE_ORDER else 0
        if new_idx > current_idx:
            slots["sales_stage"] = new_stage

    if qmode in ("specific_bank", "pricing", "docs"):
        _advance("PRESENT")
        if slots.get("lead_temperature") == "info_only":
            slots["lead_temperature"] = "interested"

    elif qmode == "bank_selection":
        _advance("SELECT")
        if slots.get("lead_temperature") == "info_only":
            slots["lead_temperature"] = "interested"

    elif qmode in ("service", "smalltalk", "intro"):
        if not slots.get("sales_stage"):
            slots["sales_stage"]      = "QUALIFY"
            slots["lead_temperature"] = "info_only"

    # Warm up after several substantive turns
    if slots.get("lead_temperature") == "interested" and slots.get("_turn_count", 0) >= 4:
        slots["lead_temperature"] = "warm"

    # Callback request flag
    if _CALLBACK_RE.search(t):
        slots["_callback_requested"] = True

    # Mark partner scope shown
    if qmode == "partner_banks":
        slots["partner_scope_shown"] = True

    return slots
