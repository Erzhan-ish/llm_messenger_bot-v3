"""Policy checks for planner/render outputs.

The validator checks consistency and safety — it does NOT re-do semantic routing.
"""
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
    """Normalize unsafe planner decisions. Does NOT re-route semantics."""
    d = dict(decision or {})
    planner = d.get("planner") or {}

    # 1. out_of_scope → static redirect
    if planner.get("domain") == "out_of_scope":
        d.update({
            "stage": "OUT_OF_SCOPE", "action": "ANSWER", "query_mode": "out_of_scope",
            "needs_kb": False, "needs_handoff": False,
        })

    # 2. constraint always needs KB
    if d.get("query_mode") == "constraint":
        d["needs_kb"] = True

    # 3. handoff overrides action
    if d.get("needs_handoff"):
        d["action"] = "HANDOFF"

    # 4. fee intents — don't override query_mode to constraint because constraint_topic is set
    if d.get("query_mode") in _FEE_QMODES:
        pass  # constraint_topic stays as metadata only

    # 5. New modes need KB
    if d.get("query_mode") in {"transfer_fee_quote", "extra_fees", "access_cards", "operations", "bonus"}:
        d["needs_kb"] = True

    return d


def active_task_consistency_check(
    decision: Dict[str, Any],
    active_task: Dict[str, Any],
) -> tuple[bool, str]:
    """Check if planner respected the active_task context.

    Returns (is_consistent, reason). Only checks obvious cases.
    """
    if not active_task:
        return True, "no_active_task"

    at_intent = active_task.get("intent") or ""
    pl_qmode  = decision.get("query_mode") or ""
    at_bank   = active_task.get("bank_name")
    pl_bank   = decision.get("bank_name") or (decision.get("planner") or {}).get("bank_name")

    # If active_task was a fee calculation and planner returns bank_selection
    # for a short follow-up containing just a number or bank name — likely lost context.
    if at_intent == "transfer_fee_quote" and pl_qmode in ("bank_selection", "smalltalk"):
        return False, "lost_transfer_fee_context"

    return True, "ok"


def render_policy_check(text: str, plan: Dict[str, Any], slots: dict | None = None) -> tuple[bool, str]:
    slots = slots or {}
    intent = plan.get("intent") or plan.get("query_mode")
    low = (text or "").lower()

    # out_of_scope must not answer off-topic
    if intent == "out_of_scope" and any(x in low for x in ("rtx", "видеокарт", "купить")):
        return False, "out_of_scope_answered_topic"

    # constraint must not sound like a sales offer
    if intent == "constraint" and plan.get("must_use_facts"):
        if any(x in low for x in ("два варианта", "выгоднее", "тариф")) and "нельзя" not in low:
            return False, "constraint_answer_looks_like_sales"

    # transfer_fee_quote must mention transfer, not only opening/monthly
    if intent == "transfer_fee_quote":
        if any(x in low for x in ("открытие", "ведение", "обслуживание")) and "перевод" not in low:
            return False, "transfer_fee_quote_answered_pricing_instead"

    # Check calculated fee is preserved (if plan has calculation)
    calc = plan.get("calculation") or {}
    if calc.get("calculated_fee") is not None:
        fee_str = str(calc["calculated_fee"])
        # Allow some formatting variation (e.g. "1 000" vs "1000")
        fee_compact = fee_str.replace(" ", "")
        answer_compact = low.replace(" ", "").replace(" ", "")
        if fee_compact not in answer_compact:
            return False, f"calculated_fee_{fee_str}_missing_from_answer"

    # Near-duplicate guard
    prev = slots.get("_last_bot_text") or ""
    if prev and text_similarity(text, prev) > 0.72:
        return False, "near_duplicate"

    return True, "ok"
