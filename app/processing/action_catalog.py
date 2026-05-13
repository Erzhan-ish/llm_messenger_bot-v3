"""Action Catalog — single source of truth for all dialog actions.

LLM Planner selects one action from this list.
ActionExecutor maps action_id → deterministic text (for is_deterministic=True).
No per-phrase playbooks; each action has constraints LLM must respect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ActionSpec:
    action_id: str
    description: str
    required_slots: List[str] = field(default_factory=list)
    required_kb_tags: List[str] = field(default_factory=list)
    allowed_scenarios: List[str] = field(default_factory=list)  # empty = any scenario
    forbidden_if: List[str] = field(default_factory=list)       # condition names
    is_deterministic: bool = False
    priority: int = 0


ACTION_CATALOG: Dict[str, ActionSpec] = {
    "answer_value_proposition": ActionSpec(
        action_id="answer_value_proposition",
        description="Explain why using the agency is better than going directly to the bank",
        required_kb_tags=["benefits", "direct_bank_objection"],
        forbidden_if=["is_account_request"],
        is_deterministic=True,
        priority=10,
    ),
    "list_partner_banks": ActionSpec(
        action_id="list_partner_banks",
        description="List partner banks (active vs paused) for the debtor type",
        required_kb_tags=["partner_banks", "bank_selection"],
        is_deterministic=True,
        priority=5,
    ),
    "compare_tariffs": ActionSpec(
        action_id="compare_tariffs",
        description="Compare tariffs across active partner banks for the debtor type",
        required_kb_tags=["pricing", "tariffs"],
        is_deterministic=True,
        priority=5,
    ),
    "explain_bank_conditions": ActionSpec(
        action_id="explain_bank_conditions",
        description="Explain conditions or requirements for a specific bank",
        required_slots=["_last_bank"],
        required_kb_tags=["conditions", "pricing"],
        is_deterministic=False,
        priority=4,
    ),
    "explain_documents": ActionSpec(
        action_id="explain_documents",
        description="Explain documents required for account opening",
        required_kb_tags=["docs"],
        is_deterministic=False,
        priority=4,
    ),
    "ask_debtor_type": ActionSpec(
        action_id="ask_debtor_type",
        description="Ask whether the debtor is a legal entity (ЮЛ) or individual (ФЛ)",
        forbidden_if=["debtor_type_known"],
        is_deterministic=True,
        priority=3,
    ),
    "ask_bank_focus": ActionSpec(
        action_id="ask_bank_focus",
        description="Ask which specific bank the user is interested in",
        forbidden_if=["bank_focus_known"],
        is_deterministic=True,
        priority=2,
    ),
    "ask_procedure_stage": ActionSpec(
        action_id="ask_procedure_stage",
        description="Ask about the bankruptcy procedure stage",
        forbidden_if=["procedure_type_known"],
        is_deterministic=True,
        priority=2,
    ),
    "explain_card_rules": ActionSpec(
        action_id="explain_card_rules",
        description="Explain card rules for individual debtor in bankruptcy",
        required_kb_tags=["debtor_card"],
        is_deterministic=True,
        priority=5,
    ),
    "collect_inn": ActionSpec(
        action_id="collect_inn",
        description="Ask the user to provide INN to start account opening",
        forbidden_if=["inn_known"],
        is_deterministic=True,
        priority=3,
    ),
    "handoff_to_manager": ActionSpec(
        action_id="handoff_to_manager",
        description="Transfer to a human manager (user is ready to open account)",
        is_deterministic=True,
        priority=10,
    ),
    "handle_confusion": ActionSpec(
        action_id="handle_confusion",
        description="Acknowledge confusion and offer to clarify",
        is_deterministic=True,
        priority=6,
    ),
    "handle_repetition": ActionSpec(
        action_id="handle_repetition",
        description="Acknowledge bot repetition and give a fresh, different answer",
        is_deterministic=False,
        priority=7,
    ),
    "close_dialog": ActionSpec(
        action_id="close_dialog",
        description="Close the conversation politely when the user is done",
        is_deterministic=True,
        priority=8,
    ),
    "small_talk": ActionSpec(
        action_id="small_talk",
        description="Respond to casual or off-topic messages",
        is_deterministic=True,
        priority=1,
    ),
}


# ---------------------------------------------------------------------------
# Condition evaluators for forbidden_if / available_actions filtering
# ---------------------------------------------------------------------------

_CONDITION_EVALUATORS = {
    "is_account_request":   lambda slots: bool(slots.get("_is_account_request")),
    "debtor_type_known":    lambda slots: bool(slots.get("debtor_type") or slots.get("client_type")),
    "bank_focus_known":     lambda slots: bool(slots.get("_last_bank") or slots.get("bank_name")),
    "procedure_type_known": lambda slots: bool(slots.get("procedure_type")),
    "inn_known":            lambda slots: bool(slots.get("inn")),
}


def _eval_condition(condition: str, slots: dict) -> bool:
    evaluator = _CONDITION_EVALUATORS.get(condition)
    return evaluator(slots) if evaluator else False


def get_action(action_id: str) -> ActionSpec | None:
    return ACTION_CATALOG.get(action_id)


def get_available_actions(
    slots: dict,
    active_scenario: str | None = None,
    forbidden_actions: list[str] | None = None,
) -> list[str]:
    """Return action_ids available given current slots, scenario, and forbidden set.

    Filters out:
    - actions in forbidden_actions
    - actions restricted to other scenarios (allowed_scenarios non-empty and mismatch)
    - actions whose forbidden_if conditions all evaluate to True
    """
    forbidden = set(forbidden_actions or [])
    available: list[str] = []

    for action_id, spec in ACTION_CATALOG.items():
        if action_id in forbidden:
            continue
        if spec.allowed_scenarios and active_scenario not in spec.allowed_scenarios:
            continue
        if any(_eval_condition(cond, slots) for cond in spec.forbidden_if):
            continue
        available.append(action_id)

    return available
