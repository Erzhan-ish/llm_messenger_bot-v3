"""Scenario Catalog — central registry of all dialog scenarios.

Each ScenarioSpec defines:
- scenario_id + description
- trigger_intents: intent signals that activate this scenario
- forced_scenarios: KB scenario IDs to force-fetch for this scenario
- forced_topics: KB topic tags to force-fetch
- required_slots: slots that must be known to answer well
- forbidden_reply_phrases: strings that must NOT appear in any reply
- required_reply_contains: at least one of these must appear (empty = no check)
- allowed_next_scenarios: which scenarios can naturally follow
- deterministic_playbook: whether a deterministic reply exists
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScenarioSpec:
    scenario_id: str
    description: str
    trigger_intents: List[str] = field(default_factory=list)
    forced_scenarios: List[str] = field(default_factory=list)
    forced_topics: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    forbidden_reply_phrases: List[str] = field(default_factory=list)
    required_reply_contains: List[str] = field(default_factory=list)
    allowed_next_scenarios: List[str] = field(default_factory=list)
    deterministic_playbook: bool = False
    # Scoring fields for catalog-driven ScenarioPolicy
    priority: int = 0                                         # higher wins ties
    slot_boosts: Dict[str, float] = field(default_factory=dict)   # slot → score bonus if truthy
    allowed_previous: List[str] = field(default_factory=list)     # scenarios that naturally lead here
    blocked_if_intents: List[str] = field(default_factory=list)   # hard-block this scenario if these intents present


# ---------------------------------------------------------------------------
# Full catalog
# ---------------------------------------------------------------------------

_SPECS: List[ScenarioSpec] = [

    # ── A. Greeting / identity / small talk ──────────────────────────────

    ScenarioSpec(
        scenario_id="greeting",
        description="User greets the bot",
        trigger_intents=["greeting"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="identity_question",
        description="User asks who the bot is or what it does",
        trigger_intents=["identity_question"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="small_talk",
        description="Casual conversation not related to business",
        trigger_intents=["small_talk"],
        forbidden_reply_phrases=["уточните вопрос", "секунду, уточняю"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="gratitude_close",
        description="User expresses gratitude / closes conversation",
        trigger_intents=["gratitude_close"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="user_confusion",
        description="User is confused or says 'I didn't ask about that'",
        trigger_intents=["confusion_or_correction"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="bot_repetition_complaint",
        description="User complains the bot is repeating itself",
        trigger_intents=["repetition_complaint"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="end_dialog",
        description="User ends the conversation",
        trigger_intents=["end_dialog"],
        deterministic_playbook=True,
    ),

    # ── B. Sales / value objection ───────────────────────────────────────

    ScenarioSpec(
        scenario_id="direct_bank_objection",
        description="User asks why use the agency instead of going directly to bank",
        trigger_intents=["direct_bank_objection"],
        forced_scenarios=["direct_bank_objection", "partner_banks"],
        forced_topics=["benefits"],
        forbidden_reply_phrases=["о каких банках хотите узнать", "уточните вопрос", "секунду, уточняю"],
        required_reply_contains=["документ", "финмониторинг", "сопровождение", "льготн"],
        deterministic_playbook=True,
        priority=5,
    ),
    ScenarioSpec(
        scenario_id="value_proposition",
        description="Explaining the benefits of using the service",
        trigger_intents=["direct_bank_objection", "detail_followup"],
        forced_scenarios=["direct_bank_objection", "partner_banks"],
        forced_topics=["benefits"],
        forbidden_reply_phrases=["уточните вопрос"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="value_followup",
        description="Follow-up on value proposition (подробнее?)",
        trigger_intents=["detail_followup"],
        forced_scenarios=["direct_bank_objection", "partner_banks"],
        forced_topics=["benefits"],
        forbidden_reply_phrases=["о каких банках хотите узнать", "уточните вопрос"],
        deterministic_playbook=True,
    ),

    # ── C. Bank selection ────────────────────────────────────────────────

    ScenarioSpec(
        scenario_id="partner_banks",
        description="User asks which banks the agency works with",
        trigger_intents=["bank_selection"],
        forced_scenarios=["partner_banks"],
        required_reply_contains=["Альфа-Банк", "ТКБ", "Уралсиб"],
        allowed_next_scenarios=["bank_selection_yul", "bank_selection_fl", "bank_pricing_yul"],
    ),
    ScenarioSpec(
        scenario_id="bank_selection_yul",
        description="Select bank for legal entity debtor",
        trigger_intents=["bank_selection"],
        forced_scenarios=["bank_selection_yul", "partner_banks"],
        forced_topics=["pricing", "selection"],
        required_slots=["debtor_type"],
        forbidden_reply_phrases=["Счёт подбираем для юрлица или физлица"],
        required_reply_contains=["Альфа-Банк", "ТКБ", "Уралсиб"],
        allowed_next_scenarios=["bank_pricing_yul", "alfabank_yul_conditions", "tkb_yul_conditions", "uralsib_yul_conditions"],
        deterministic_playbook=True,
        priority=2,
        slot_boosts={"debtor_type": 1.5},
    ),
    ScenarioSpec(
        scenario_id="bank_selection_fl",
        description="Select bank for individual debtor",
        trigger_intents=["bank_selection"],
        forced_scenarios=["bank_selection_fl", "tkb_fl_conditions", "uralsib_fl_conditions"],
        forced_topics=["pricing", "selection"],
        required_slots=["debtor_type"],
        forbidden_reply_phrases=["Счёт подбираем для юрлица или физлица"],
        required_reply_contains=["ТКБ"],
        allowed_next_scenarios=["tkb_fl_conditions", "uralsib_fl_conditions"],
        deterministic_playbook=True,
        priority=2,
        slot_boosts={"debtor_type": 1.5},
    ),
    ScenarioSpec(
        scenario_id="bank_selection_yul_low_cost",
        description="Find cheapest bank for legal entity",
        trigger_intents=["bank_selection", "low_cost_requested"],
        forced_scenarios=["bank_selection_yul_low_cost", "bank_selection_yul", "partner_banks"],
        forced_topics=["pricing", "tariffs"],
        forbidden_reply_phrases=["Счёт подбираем для юрлица или физлица"],
        required_reply_contains=["Альфа-Банк", "ТКБ"],
        deterministic_playbook=True,
        priority=3,
        slot_boosts={"debtor_type": 1.5},
    ),
    ScenarioSpec(
        scenario_id="bank_pricing_yul",
        description="YUL bank tariff details",
        trigger_intents=["tariff_comparison_requested"],
        forced_scenarios=["alfabank_yul_conditions", "tkb_yul_conditions", "uralsib_yul_conditions", "bank_selection_yul"],
        forced_topics=["pricing"],
        forbidden_reply_phrases=["Счёт подбираем для юрлица или физлица"],
        deterministic_playbook=True,
        priority=2,
        slot_boosts={"debtor_type": 1.0},
    ),
    ScenarioSpec(
        scenario_id="bank_pricing_fl",
        description="FL bank tariff details",
        trigger_intents=["tariff_comparison_requested"],
        forced_scenarios=["tkb_fl_conditions", "uralsib_fl_conditions"],
        forced_topics=["pricing"],
    ),
    ScenarioSpec(
        scenario_id="bank_tariff_comparison",
        description="Direct comparison of bank tariffs",
        trigger_intents=["tariff_comparison_requested"],
        forced_scenarios=["bank_selection_yul", "alfabank_yul_conditions", "tkb_yul_conditions", "uralsib_yul_conditions"],
        forced_topics=["pricing", "tariffs"],
        forbidden_reply_phrases=["Могу сравнить?", "хотите сравнить?", "уточните вопрос"],
        deterministic_playbook=True,
    ),

    # ── D. Specific banks ────────────────────────────────────────────────

    ScenarioSpec(
        scenario_id="alfabank_yul_conditions",
        description="Alfa-Bank conditions for YUL",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["alfabank_yul_conditions"],
        forced_topics=["pricing", "conditions"],
        forbidden_reply_phrases=["Счёт подбираем для юрлица или физлица"],
        required_reply_contains=["альфа", "Альфа"],
        priority=4,
        slot_boosts={"_last_bank": 2.0},
    ),
    ScenarioSpec(
        scenario_id="tkb_yul_conditions",
        description="TKB conditions for YUL",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["tkb_yul_conditions"],
        forced_topics=["pricing", "conditions"],
        required_reply_contains=["ТКБ", "ткб"],
        priority=4,
        slot_boosts={"_last_bank": 2.0},
    ),
    ScenarioSpec(
        scenario_id="tkb_fl_conditions",
        description="TKB conditions for FL",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["tkb_fl_conditions"],
        forced_topics=["pricing", "conditions"],
        required_reply_contains=["ТКБ", "ткб"],
        priority=4,
        slot_boosts={"_last_bank": 2.0},
    ),
    ScenarioSpec(
        scenario_id="uralsib_yul_conditions",
        description="Uralsib conditions for YUL",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["uralsib_yul_conditions"],
        forced_topics=["pricing", "conditions"],
        required_reply_contains=["Уралсиб", "уралсиб"],
        priority=4,
        slot_boosts={"_last_bank": 2.0},
    ),
    ScenarioSpec(
        scenario_id="uralsib_fl_conditions",
        description="Uralsib conditions for FL",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["uralsib_fl_conditions"],
        forced_topics=["pricing", "conditions"],
        required_reply_contains=["Уралсиб", "уралсиб"],
        priority=4,
        slot_boosts={"_last_bank": 2.0},
    ),
    ScenarioSpec(
        scenario_id="tbank_status",
        description="T-Bank status (paused)",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["partner_banks"],
    ),
    ScenarioSpec(
        scenario_id="mkb_status",
        description="MKB status (paused)",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["partner_banks"],
    ),
    ScenarioSpec(
        scenario_id="rosbank_status",
        description="Rosbank status (paused)",
        trigger_intents=["specific_bank_focus"],
        forced_scenarios=["partner_banks"],
    ),

    # ── E. Bonus / interest / benefits ───────────────────────────────────

    ScenarioSpec(
        scenario_id="interest_on_balance",
        description="Interest rate on special account balance",
        trigger_intents=["interest_or_bonus"],
        forced_scenarios=["partner_banks", "bank_selection_yul"],
        forced_topics=["bonus", "interest", "benefits"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="au_bonus_question",
        description="Personal bonus for the AU (not official bank conditions)",
        trigger_intents=["interest_or_bonus"],
        forbidden_reply_phrases=["уточните вопрос"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="percent_yearly",
        description="Annual interest rate question",
        trigger_intents=["interest_or_bonus"],
        forced_topics=["bonus", "interest"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),

    # ── F. Bankruptcy procedure stages ───────────────────────────────────

    ScenarioSpec(
        scenario_id="allowed_stages",
        description="Account opening at observation / other stages",
        trigger_intents=["procedure_stage"],
        forced_scenarios=["allowed_stages"],
        allowed_next_scenarios=["bank_selection_yul", "bank_selection_fl", "docs_yul", "docs_fl"],
        deterministic_playbook=True,
    ),

    # ── G. Documents ─────────────────────────────────────────────────────

    ScenarioSpec(
        scenario_id="docs_yul",
        description="Documents required for YUL account",
        trigger_intents=["documents_or_requirements"],
        forced_scenarios=["docs_tkb_yul"],
        forced_topics=["docs"],
        required_slots=["debtor_type"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="docs_fl",
        description="Documents required for FL account",
        trigger_intents=["documents_or_requirements"],
        forced_topics=["docs"],
        required_slots=["debtor_type"],
    ),

    # ── H. Special cases ─────────────────────────────────────────────────

    ScenarioSpec(
        scenario_id="red_zone_company",
        description="Legal entity in red zone (high-risk)",
        trigger_intents=["red_zone_company"],
        forced_scenarios=["red_zone_company"],
        required_slots=["debtor_type"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="non_resident",
        description="Non-resident debtor",
        trigger_intents=["non_resident"],
        forced_scenarios=["non_resident"],
        forbidden_reply_phrases=["уточните вопрос"],
    ),
    ScenarioSpec(
        scenario_id="deceased_fl",
        description="Deceased individual debtor",
        trigger_intents=["deceased_debtor"],
        forced_scenarios=["deceased_fl"],
    ),
    ScenarioSpec(
        scenario_id="liquidated_yul",
        description="Liquidated legal entity debtor",
        trigger_intents=["liquidated_entity"],
        forced_scenarios=["liquidated_yul"],
    ),

    # ── I. Card / физлицо ────────────────────────────────────────────────

    ScenarioSpec(
        scenario_id="debtor_card_realization",
        description="Card realization for individual debtor",
        trigger_intents=["debtor_card_realization"],
        forced_scenarios=["debtor_card_realization"],
        required_reply_contains=["реализация", "финансовый управляющий"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="cash_withdrawal_bankruptcy",
        description="Cash withdrawal rules in bankruptcy",
        trigger_intents=["cash_withdrawal"],
        forced_scenarios=["cash_withdrawal_bankruptcy"],
    ),

    # ── J. Operational / escalation ──────────────────────────────────────

    ScenarioSpec(
        scenario_id="handoff_to_manager",
        description="Transfer to human manager",
        trigger_intents=["handoff_request"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="complex_case",
        description="Complex case requiring expert review",
        trigger_intents=["complex_case"],
    ),
    ScenarioSpec(
        scenario_id="ready_to_open",
        description="Client is ready to open the account — handoff to manager",
        trigger_intents=["ready_to_open", "open_account_intent"],
        deterministic_playbook=True,
    ),
    ScenarioSpec(
        scenario_id="currency_account_question",
        description="Questions about currency account operations: settlements, payments to creditors",
        trigger_intents=["currency_account_question"],
    ),
    ScenarioSpec(
        scenario_id="bot_complaint",
        description="User is dissatisfied with the bot or says it's answering wrong",
        trigger_intents=["bot_complaint", "confusion_or_correction"],
    ),

]

# Build lookup dict
CATALOG: Dict[str, ScenarioSpec] = {spec.scenario_id: spec for spec in _SPECS}


def get_spec(scenario_id: str) -> ScenarioSpec | None:
    """Return scenario spec by id, or None if not found."""
    return CATALOG.get(scenario_id)


def forced_kb_for_scenario(scenario_id: str) -> dict:
    """Return {'forced_scenarios': [...], 'forced_topics': [...]} for a scenario."""
    spec = CATALOG.get(scenario_id)
    if not spec:
        return {}
    return {
        "forced_scenarios": list(spec.forced_scenarios),
        "forced_topics": list(spec.forced_topics),
    }
