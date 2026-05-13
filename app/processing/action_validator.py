"""Action Validator — validates planner output before execution.

Checks:
- action exists in ACTION_CATALOG
- action is in the available_actions list
- required slots are present
- action is not a recent repeat (last 2 actions)
- action constraints are satisfied
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_planner_action(
    planner_result: dict,
    slots: dict,
    available_actions: list[str],
    kb_facts: list[dict] | None = None,
    previous_actions: list[str] | None = None,
) -> dict:
    """Return {'is_valid': bool, 'reason': str, 'fallback': str|None}.

    If invalid, fallback contains a safe replacement action_id.
    """
    from app.processing.action_catalog import ACTION_CATALOG

    action_id = planner_result.get("action") or ""
    previous = list(previous_actions or [])

    # 1. Action must exist in catalog
    if action_id not in ACTION_CATALOG:
        return _invalid(
            f"unknown_action:{action_id}",
            _safe_fallback(slots, available_actions),
        )

    # 2. Action must be in available list (if list provided)
    if available_actions and action_id not in available_actions:
        return _invalid(
            f"action_not_available:{action_id}",
            _safe_fallback(slots, available_actions),
        )

    spec = ACTION_CATALOG[action_id]

    # 3. Required slots must be present
    missing = [s for s in spec.required_slots if not _slot_present(slots, s)]
    if missing:
        return _invalid(
            f"missing_slots:{missing}",
            _slot_to_clarifying_action(missing[0], available_actions),
        )

    # 4. Repetition guard — don't repeat the same action twice in a row
    if (
        action_id in previous[-2:]
        and action_id not in ("handle_repetition", "handoff_to_manager", "close_dialog")
    ):
        return _invalid(
            f"repeated_action:{action_id}",
            _safe_fallback(slots, [a for a in available_actions if a != action_id]),
        )

    # 5. Debtor type constraint — ask_debtor_type is forbidden if type already known
    if action_id == "ask_debtor_type" and (
        slots.get("debtor_type") or slots.get("client_type")
    ):
        return _invalid(
            "debtor_type_already_known",
            _safe_fallback(slots, available_actions),
        )

    # 6. INN constraint — collect_inn is forbidden if INN already provided
    if action_id == "collect_inn" and slots.get("inn"):
        return _invalid(
            "inn_already_known",
            _safe_fallback(slots, available_actions),
        )

    return {"is_valid": True, "reason": "ok", "fallback": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invalid(reason: str, fallback: str | None) -> dict:
    return {"is_valid": False, "reason": reason, "fallback": fallback}


def _slot_present(slots: dict, slot_key: str) -> bool:
    return bool(slots.get(slot_key))


def _slot_to_clarifying_action(slot_key: str, available_actions: list[str]) -> str:
    _map = {
        "_last_bank": "ask_bank_focus",
        "bank_name":  "ask_bank_focus",
        "debtor_type": "ask_debtor_type",
        "client_type": "ask_debtor_type",
        "procedure_type": "ask_procedure_stage",
        "inn": "collect_inn",
    }
    candidate = _map.get(slot_key, "ask_debtor_type")
    if available_actions and candidate not in available_actions:
        return available_actions[0]
    return candidate


def _safe_fallback(slots: dict, available_actions: list[str]) -> str:
    debtor_type = slots.get("debtor_type") or slots.get("client_type")
    preferred = (
        ["compare_tariffs", "list_partner_banks"]
        if debtor_type
        else ["list_partner_banks", "ask_debtor_type"]
    )
    for a in preferred:
        if a in available_actions:
            return a
    return available_actions[0] if available_actions else "list_partner_banks"
