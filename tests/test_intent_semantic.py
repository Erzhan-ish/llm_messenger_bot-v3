"""Tests for hybrid intent extraction (TASK 8).

Groups:
  1. Semantic paraphrases  — unseen phrasing still resolves to correct intent
  2. Exact/regex priority  — deterministic signals win over semantic
  3. Multi-intent          — several intents in one message
  4. Scenario scoring      — ScenarioPolicy picks the right scenario
  5. Catalog source of truth — forced_scenarios always read from catalog
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Semantic paraphrases
# ---------------------------------------------------------------------------

class TestSemanticParaphrases:
    """Phrasing that regex won't catch, but embeddings should."""

    @pytest.mark.parametrize("text,expected_intent", [
        ("хочу что-то бюджетное",           "low_cost_requested"),
        ("минимальные расходы на открытие", "low_cost_requested"),
        ("почему не открыть самому",        "direct_bank_objection"),
        ("что вы такого делаете",           "direct_bank_objection"),
        ("ты это уже говорил",              "repetition_complaint"),
        ("не о том речь",                   "confusion_or_correction"),
        ("не хочу переплачивать",           "low_cost_requested"),
        ("в чём смысл обращаться к вам",   "direct_bank_objection"),
    ])
    def test_semantic_paraphrase_resolves(self, text, expected_intent):
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, _ = extract_semantic_intents(text)
        intent_names = [m.intent for m in accepted]
        assert expected_intent in intent_names, (
            f"Expected '{expected_intent}' in semantic intents for '{text}', got {intent_names}"
        )

    def test_semantic_match_has_required_fields(self):
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, rejects = extract_semantic_intents("хочу что-то бюджетное")
        assert accepted
        m = accepted[0]
        assert isinstance(m.intent, str)
        assert 0.0 < m.score <= 1.0
        assert m.source == "semantic"
        assert isinstance(m.matched_anchor, str)
        assert isinstance(m.threshold, float)

    def test_near_rejects_logged(self):
        """Near-threshold matches should appear in rejects, not accepted."""
        from app.processing.intent_semantic import extract_semantic_intents, INTENT_THRESHOLDS, _LOG_MARGIN
        # Use a neutral text that shouldn't strongly match anything
        accepted, rejects = extract_semantic_intents("погода сегодня хорошая")
        # All accepted must be above their threshold
        for m in accepted:
            assert m.score >= m.threshold
        # All rejects must be in the log margin
        for m in rejects:
            assert m.score >= m.threshold - _LOG_MARGIN
            assert m.score < m.threshold

    def test_empty_text_returns_empty(self):
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, rejects = extract_semantic_intents("")
        assert accepted == []
        assert rejects == []

    def test_max_3_accepted(self):
        from app.processing.intent_semantic import extract_semantic_intents
        # A long multi-topic text might match many intents
        text = "нужен банк подешевле, какие тарифы, зачем через вас напрямую банк сам откроет"
        accepted, _ = extract_semantic_intents(text)
        assert len(accepted) <= 3


# ---------------------------------------------------------------------------
# 2. Exact / regex still wins for deterministic signals
# ---------------------------------------------------------------------------

class TestExactRegexPriority:
    """Deterministic signals must always be extracted regardless of semantic."""

    def test_reset_command(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("/reset")
        # /reset is handled by message_processor before extractor — this is a safeguard
        # Just confirm extractor doesn't crash on it
        assert "normalized" in sig

    def test_yes_with_pending_slot(self):
        from app.processing.intent_extractor import extract_intent_signals
        slots = {"_pending_slot": "realization_started"}
        sig = extract_intent_signals("да", slots)
        assert sig["is_yes_no"] is True

    def test_debtor_type_fl(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("у меня ФЛ")
        assert sig["debtor_type"] == "individual"

    def test_debtor_type_yul(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("должник — ООО")
        assert sig["debtor_type"] == "legal_entity"

    def test_bank_focus_tkb(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("расскажи про ТКБ")
        assert sig["bank_focus"] == "tkb"

    def test_bank_focus_uralsib(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("что по Уралсибу?")
        assert sig["bank_focus"] == "uralsib"

    def test_regex_intent_in_matches(self):
        """Regex-found intents appear in matches with source='regex'."""
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("нужен банк подешевле")
        regex_match = next(
            (m for m in sig["matches"] if m["source"] == "regex" and m["intent"] == "low_cost_requested"),
            None,
        )
        assert regex_match is not None

    def test_regex_takes_priority_no_duplicate(self):
        """When regex matches an intent, semantic must not duplicate it in merged list."""
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("нужен банк подешевле")
        count = sum(1 for i in sig["intents"] if i == "low_cost_requested")
        assert count == 1, "low_cost_requested should appear exactly once"

    def test_bank_focus_implies_specific_bank_focus_intent(self):
        """bank_focus detection should add specific_bank_focus to intents for catalog scoring."""
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("расскажи про Альфа-Банк")
        assert "specific_bank_focus" in sig["intents"]


# ---------------------------------------------------------------------------
# 3. Multi-intent
# ---------------------------------------------------------------------------

class TestMultiIntent:
    """One message containing multiple distinct intents."""

    def test_bank_selection_and_low_cost_and_tariffs(self):
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("у меня ЮЛ, нужен банк подешевле, какие тарифы")
        intents = sig["intents"]
        assert "bank_selection" in intents or "low_cost_requested" in intents
        assert sig["debtor_type"] == "legal_entity"

    def test_policy_sets_required_next_step_for_tariff(self):
        """When bank_selection_yul_low_cost is active AND tariff_comparison is detected,
        policy should set required_next_step=give_tariff_comparison."""
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": None}
        intent_signals = {
            "intents": ["bank_selection", "low_cost_requested", "tariff_comparison_requested"],
            "matches": [],
            "bank_focus": None,
        }
        result = decide_scenario_policy(
            user_text="у меня ЮЛ, нужен банк подешевле, какие тарифы",
            slots=slots,
            rag_scenarios=[],
            intent_signals=intent_signals,
        )
        # Policy should pick bank_selection_yul_low_cost (2 trigger intents match)
        # AND set required_next_step since tariff_comparison_requested is also present
        if result["active_scenario"] == "bank_selection_yul_low_cost":
            assert result["required_next_step"] == "give_tariff_comparison"

    def test_direct_objection_with_bank_selection_sets_next_step(self):
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": None}
        intent_signals = {
            "intents": ["direct_bank_objection", "bank_selection"],
            "matches": [],
            "bank_focus": None,
        }
        result = decide_scenario_policy(
            user_text="зачем через вас, и вообще с какими банками работаете",
            slots=slots,
            rag_scenarios=[],
            intent_signals=intent_signals,
        )
        assert result["active_scenario"] == "direct_bank_objection"
        assert result.get("required_next_step") == "list_banks"


# ---------------------------------------------------------------------------
# 4. Scenario scoring
# ---------------------------------------------------------------------------

class TestScenarioScoring:
    """ScenarioPolicy must pick the right scenario via scoring."""

    def test_value_objection_always_wins(self):
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": "allowed_stages"}
        result = decide_scenario_policy(
            user_text="зачем через вас если могу напрямую в банк",
            slots=slots,
            rag_scenarios=[{"scenario_id": "allowed_stages", "score": 2.0, "reasons": []}],
            intent_signals={"intents": [], "matches": [], "bank_focus": None},
        )
        assert result["active_scenario"] == "direct_bank_objection"
        assert result["decision"] == "switch"

    def test_low_cost_plus_yul_beats_active_objection(self):
        """Example from plan: direct_bank_objection active, user says 'ЮЛ банк подешевле'."""
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": "direct_bank_objection", "debtor_type": "ЮЛ"}
        intent_signals = {
            "intents": ["bank_selection", "low_cost_requested"],
            "matches": [],
            "bank_focus": None,
        }
        result = decide_scenario_policy(
            user_text="у меня ЮЛ нужен банк подешевле",
            slots=slots,
            rag_scenarios=[],
            intent_signals=intent_signals,
        )
        assert result["active_scenario"] == "bank_selection_yul_low_cost"
        assert result["decision"] == "switch"

    def test_bank_focus_uralsib_switches_scenario(self):
        """Example from plan: bank_selection_yul active, user says 'Уралсиб условия'."""
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": "bank_selection_yul", "client_type": "ЮЛ", "_last_bank": "Уралсиб"}
        intent_signals = {
            "intents": ["tariff_comparison_requested", "specific_bank_focus"],
            "matches": [],
            "bank_focus": "uralsib",
        }
        result = decide_scenario_policy(
            user_text="Уралсиб условия",
            slots=slots,
            rag_scenarios=[],
            intent_signals=intent_signals,
        )
        assert result["active_scenario"] == "uralsib_yul_conditions"

    def test_followup_keeps_active(self):
        """'подробнее' with active scenario → keep active."""
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": "direct_bank_objection"}
        result = decide_scenario_policy(
            user_text="подробнее",
            slots=slots,
            rag_scenarios=[{"scenario_id": "bank_selection_yul", "score": 2.0, "reasons": []}],
            intent_signals={"intents": [], "matches": [], "bank_focus": None},
        )
        assert result["active_scenario"] == "direct_bank_objection"
        assert result["decision"] == "keep_active"

    def test_active_to_allowed_stages_switches_on_bank_selection(self):
        """allowed_stages active + 'с какими банками работаете' → switch away from allowed_stages."""
        from app.processing.scenario_policy import decide_scenario_policy

        slots = {"_active_scenario": "allowed_stages"}
        intent_signals = {
            "intents": ["bank_selection"],
            "matches": [],
            "bank_focus": None,
        }
        result = decide_scenario_policy(
            user_text="с какими банками работаете",
            slots=slots,
            rag_scenarios=[],
            intent_signals=intent_signals,
        )
        assert result["decision"] == "switch"
        # Should route to any bank_selection-related scenario (includes partner_banks)
        bank_scenarios = {"partner_banks", "bank_selection_yul", "bank_selection_fl", "bank_selection_yul_low_cost"}
        assert result["active_scenario"] in bank_scenarios, (
            f"Expected a bank-selection scenario, got: {result['active_scenario']}"
        )

    def test_scores_dict_in_result(self):
        """Policy result must include scores dict for debugging."""
        from app.processing.scenario_policy import decide_scenario_policy

        result = decide_scenario_policy(
            user_text="нужен банк подешевле",
            slots={},
            rag_scenarios=[],
            intent_signals={"intents": ["low_cost_requested"], "matches": [], "bank_focus": None},
        )
        assert "scores" in result
        assert isinstance(result["scores"], dict)


# ---------------------------------------------------------------------------
# 5. Catalog as source of truth
# ---------------------------------------------------------------------------

class TestCatalogSourceOfTruth:
    """forced_scenarios for direct_bank_objection must come from catalog."""

    def test_catalog_has_forced_scenarios_for_objection(self):
        from app.processing.scenario_catalog import forced_kb_for_scenario
        result = forced_kb_for_scenario("direct_bank_objection")
        assert "direct_bank_objection" in result["forced_scenarios"]
        assert "partner_banks" in result["forced_scenarios"]

    def test_forced_scenarios_same_as_hardcoded_was(self):
        """Catalog must return the same list that was previously hardcoded in message_processor."""
        from app.processing.scenario_catalog import forced_kb_for_scenario
        result = forced_kb_for_scenario("direct_bank_objection")
        old_hardcoded = ["direct_bank_objection", "partner_banks"]
        for s in old_hardcoded:
            assert s in result["forced_scenarios"], (
                f"'{s}' was hardcoded in message_processor but is missing from catalog"
            )

    def test_all_deterministic_scenarios_have_forced_kb(self):
        """Every scenario with forced_scenarios in catalog is accessible via forced_kb_for_scenario."""
        from app.processing.scenario_catalog import CATALOG, forced_kb_for_scenario
        for sid, spec in CATALOG.items():
            if spec.forced_scenarios:
                result = forced_kb_for_scenario(sid)
                assert result["forced_scenarios"] == list(spec.forced_scenarios), (
                    f"forced_kb_for_scenario('{sid}') returned wrong list"
                )

    def test_new_scenario_fields_accessible(self):
        """ScenarioSpec must expose priority, slot_boosts, allowed_previous, blocked_if_intents."""
        from app.processing.scenario_catalog import get_spec
        spec = get_spec("bank_selection_yul_low_cost")
        assert spec is not None
        assert hasattr(spec, "priority")
        assert hasattr(spec, "slot_boosts")
        assert hasattr(spec, "allowed_previous")
        assert hasattr(spec, "blocked_if_intents")
        assert spec.priority > 0


# ---------------------------------------------------------------------------
# 6. Regression tests — false positive fixes (plan update)
# ---------------------------------------------------------------------------

class TestFalsePositiveRegression:
    """Guards against the false positive cases observed after adding semantic matching."""

    def test_account_request_not_direct_bank_objection(self):
        """'здравствуйте, у меня ЮЛ должник нужен счет' must NOT trigger direct_bank_objection."""
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("здравствуйте, у меня ЮЛ должник нужен счет")
        # direct_bank_objection must NOT be in accepted intents (may be in rejects)
        assert "direct_bank_objection" not in sig["intents"], (
            f"direct_bank_objection should not be accepted for 'нужен счет'. "
            f"intents={sig['intents']}, rejects={[r['matched_anchor'] for r in sig['semantic_rejects']]}"
        )
        # debtor_type must be detected
        assert sig["debtor_type"] == "legal_entity"

    def test_account_request_bonus_interest_not_fired(self):
        """'нужен счет' must not trigger bonus_interest."""
        from app.processing.intent_extractor import extract_intent_signals
        sig = extract_intent_signals("у меня ЮЛ должник нужен счет")
        assert "bonus_interest" not in sig["intents"], (
            f"bonus_interest falsely accepted. intents={sig['intents']}"
        )

    def test_account_request_selects_bank_selection_scenario(self):
        """Scenario policy must select bank_selection_yul for 'у меня ЮЛ нужен счет'."""
        from app.processing.scenario_policy import decide_scenario_policy
        from app.processing.intent_extractor import extract_intent_signals

        text = "здравствуйте, у меня ЮЛ должник нужен счет"
        sig = extract_intent_signals(text)
        slots = {"_active_scenario": None, "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text=text,
            slots=slots,
            rag_scenarios=[],
            intent_signals=sig,
        )
        assert result["active_scenario"] != "direct_bank_objection", (
            f"direct_bank_objection should not win for 'нужен счет'. "
            f"active={result['active_scenario']}, scores={result['scores']}"
        )
        bank_scenarios = {
            "bank_selection_yul", "bank_selection_fl", "bank_selection_yul_low_cost",
            "partner_banks",
        }
        assert result["active_scenario"] in bank_scenarios, (
            f"Expected a bank-selection scenario, got: {result['active_scenario']} "
            f"scores={result['scores']}"
        )

    def test_short_yes_no_semantic(self):
        """'да' must not produce small_talk, confusion_or_correction, or repetition_complaint."""
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, _ = extract_semantic_intents("да")
        bad_intents = {"small_talk", "confusion_or_correction", "repetition_complaint", "direct_bank_objection"}
        actual = {m.intent for m in accepted}
        assert not actual & bad_intents, (
            f"'да' produced unexpected semantic intents: {actual & bad_intents}"
        )

    def test_short_davajte_no_semantic(self):
        """'давайте' must skip semantic entirely."""
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, _ = extract_semantic_intents("давайте")
        assert accepted == [], f"Expected no semantic matches for 'давайте', got {accepted}"

    def test_topic_switch_after_objection(self):
        """'окей давайте\\nу меня ЮЛ' after direct_bank_objection must switch to bank_selection_yul."""
        from app.processing.scenario_policy import decide_scenario_policy
        from app.processing.intent_extractor import extract_intent_signals

        text = "окей давайте\nу меня ЮЛ"
        sig = extract_intent_signals(text)
        slots = {"_active_scenario": "direct_bank_objection", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text=text,
            slots=slots,
            rag_scenarios=[],
            intent_signals=sig,
        )
        assert result["active_scenario"] != "direct_bank_objection", (
            f"Should switch away from direct_bank_objection, got: {result['active_scenario']} "
            f"scores={result['scores']}"
        )

    def test_actual_objection_still_routes_correctly(self):
        """'а в чем выгода через вас работать? могу напрямую в банк' → direct_bank_objection."""
        from app.processing.scenario_policy import decide_scenario_policy
        from app.processing.intent_extractor import extract_intent_signals

        text = "а в чем выгода через вас работать? я же могу напрямую в банк пойти"
        sig = extract_intent_signals(text)
        slots = {"_active_scenario": None}
        result = decide_scenario_policy(
            user_text=text,
            slots=slots,
            rag_scenarios=[],
            intent_signals=sig,
        )
        assert result["active_scenario"] == "direct_bank_objection", (
            f"Expected direct_bank_objection for actual objection, got: {result['active_scenario']}"
        )

    def test_bonus_interest_accepted_with_required_terms(self):
        """'какой процент годовых?' → bonus_interest accepted (gate: 'годовых' present)."""
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, _ = extract_semantic_intents("какой процент годовых?")
        intents = [m.intent for m in accepted]
        # bonus_interest gate has 'годовых' → should pass
        assert "bonus_interest" in intents, (
            f"bonus_interest should be accepted when 'годовых' is present. intents={intents}"
        )

    def test_direct_objection_gate_blocks_false_positive(self):
        """'нужен счет' → direct_bank_objection rejected (gate: no required terms)."""
        from app.processing.intent_semantic import extract_semantic_intents
        accepted, rejects = extract_semantic_intents("нужен счет")
        objection_accepted = [m for m in accepted if m.intent == "direct_bank_objection"]
        assert not objection_accepted, (
            f"direct_bank_objection should be rejected for 'нужен счет' (missing gate terms)"
        )
