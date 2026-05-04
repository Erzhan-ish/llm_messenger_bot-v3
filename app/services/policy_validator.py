"""Policy checks for planner/render outputs."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict

_FEE_INTENTS = {"transfer_fee_quote", "extra_fees", "pricing", "bonus"}
_FEE_QMODES  = {"transfer_fee_quote", "extra_fees", "pricing", "bonus"}


def text_similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def planner_policy_check(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize unsafe planner decisions before rendering/actions."""
    d = dict(decision or {})
    planner = d.get("planner") or {}

    if planner.get("domain") == "out_of_scope":
        d.update({
            "stage": "OUT_OF_SCOPE", "action": "ANSWER", "query_mode": "out_of_scope",
            "needs_kb": False, "needs_handoff": False,
        })

    if d.get("query_mode") == "constraint":
        d["needs_kb"] = True

    if d.get("needs_handoff"):
        d["action"] = "HANDOFF"

    # Protect fee intents — never let constraint_topic override query_mode to constraint
    if d.get("query_mode") in _FEE_QMODES:
        # constraint_topic stays as metadata but query_mode stays fee-focused
        pass

    # Ensure new modes get needs_kb=True
    if d.get("query_mode") in {"transfer_fee_quote", "extra_fees", "access_cards", "operations", "bonus"}:
        d["needs_kb"] = True

    return d


def render_policy_check(text: str, plan: Dict[str, Any], slots: dict | None = None) -> tuple[bool, str]:
    slots = slots or {}
    intent = plan.get("intent") or plan.get("query_mode")
    low = (text or "").lower()

    if intent == "out_of_scope" and any(x in low for x in ("rtx", "видеокарт", "купить")):
        return False, "out_of_scope_answered_topic"

    if intent == "constraint" and plan.get("must_use_facts"):
        # Constraint answer should not sound like a normal sales offer.
        if any(x in low for x in ("два варианта", "выгоднее", "тариф")) and "нельзя" not in low:
            return False, "constraint_answer_looks_like_sales"

    # Fee intent must not contain opening/monthly pricing as the main content
    if intent == "transfer_fee_quote":
        if any(x in low for x in ("открытие", "ведение", "обслуживание")) and "перевод" not in low:
            return False, "transfer_fee_quote_answered_pricing_instead"

    prev = slots.get("_last_bot_text") or ""
    if prev and text_similarity(text, prev) > 0.72:
        return False, "near_duplicate"

    return True, "ok"
