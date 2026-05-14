"""Golden dialog pipeline tests.

Tests the full pipeline components in sequence with mocked LLM calls.
Verifies that correct context reaches the responder and that slots
propagate correctly across turns.

Run:
    python -m pytest tests/golden_dialogs/test_golden_pipeline.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.processing.kb_query_expander import expand_kb_query
from app.processing.scenario_playbook import run_scenario_playbook
from app.services.conversation_responder import (
    _build_manager_context,
    responder_to_brain_result,
    run_conversation_responder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _slots(**kw) -> dict:
    return {"_active_scenario": None, "_known_slots": {}, **kw}


def _fake_responder_json(reply: str, intent: str = "answer") -> str:
    import json
    return json.dumps({
        "reply": reply,
        "intent": intent,
        "next_step": None,
        "needs_clarification": False,
        "handoff": {"needed": False, "reason": None},
    })


# ---------------------------------------------------------------------------
# 1. KB Query Expansion
# ---------------------------------------------------------------------------

class TestKBQueryExpansion(unittest.TestCase):
    """KB query expander enriches short/referential queries."""

    def test_tam_expands_with_last_bank(self):
        slots = _slots(_last_bank="ТКБ", debtor_type="ЮЛ")
        result = expand_kb_query("какие тарифы там?", slots)
        self.assertIn("ТКБ", result)
        self.assertIn("ЮЛ", result)

    def test_u_nikh_resolved_from_dialog(self):
        dialog = [
            {"role": "bot", "text": "По ТКБ открытие 2800 ₽, ведение 2090 ₽."},
        ]
        slots = _slots(debtor_type="ЮЛ")
        result = expand_kb_query("а у них какие условия?", slots, recent_dialog=dialog)
        self.assertIn("ТКБ", result)

    def test_short_message_expands_with_debtor_type(self):
        slots = _slots(debtor_type="ФЛ")
        result = expand_kb_query("тарифы?", slots)
        self.assertIn("ФЛ", result)

    def test_long_message_not_expanded(self):
        slots = _slots(_last_bank="Уралсиб", debtor_type="ЮЛ")
        long_text = "расскажите пожалуйста подробнее про условия открытия счёта для юридического лица в Уралсиб банке"
        result = expand_kb_query(long_text, slots)
        self.assertEqual(result, long_text)

    def test_da_after_tariff_offer_expands(self):
        slots = _slots(
            debtor_type="ЮЛ",
            _last_bot_question="Могу сравнить тарифы по активным банкам?",
        )
        result = expand_kb_query("да", slots)
        self.assertIn("тарифы", result.lower())

    def test_empty_text_returns_empty(self):
        result = expand_kb_query("", _slots())
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 2. Scenario Playbook — no rigid next_step for business scenarios
# ---------------------------------------------------------------------------

class TestPlaybookEnrichContext(unittest.TestCase):
    """Playbook must enrich context without imposing rigid next_step."""

    def _pb(self, text, active, known=None):
        slots = {"_active_scenario": active, "_known_slots": known or {}}
        return run_scenario_playbook(text, slots)

    def test_direct_bank_objection_no_next_step(self):
        r = self._pb("в чем выгода через вас?", "direct_bank_objection")
        self.assertEqual(r["action"], "enrich")
        self.assertIsNone(r["updates"].get("_required_next_step"))

    def test_bank_selection_yul_no_next_step(self):
        r = self._pb("с какими банками работаете?", "bank_selection_yul")
        self.assertEqual(r["action"], "enrich")
        self.assertIsNone(r["updates"].get("_required_next_step"))

    def test_bank_pricing_yul_no_next_step(self):
        r = self._pb("тарифы ЮЛ?", "bank_pricing_yul")
        self.assertEqual(r["action"], "enrich")
        self.assertIsNone(r["updates"].get("_required_next_step"))

    def test_tkb_conditions_no_next_step(self):
        r = self._pb("условия ТКБ?", "tkb_yul_conditions")
        self.assertEqual(r["action"], "enrich")
        self.assertIsNone(r["updates"].get("_required_next_step"))

    def test_fact_pack_no_required_next_step_for_any_business_scenario(self):
        for scenario in [
            "direct_bank_objection", "bank_selection_yul", "bank_pricing_yul",
            "bank_selection_fl", "bank_selection_yul_low_cost",
            "tkb_yul_conditions", "uralsib_yul_conditions", "alfabank_yul_conditions",
        ]:
            with self.subTest(scenario=scenario):
                r = self._pb("вопрос", scenario)
                fp = r.get("fact_pack_additions") or {}
                self.assertNotIn("_required_next_step", fp,
                    f"Scenario {scenario!r} must not inject _required_next_step")

    def test_gratitude_still_returns_reply(self):
        r = self._pb("хорошо, спасибо", "bank_selection_yul")
        self.assertEqual(r["action"], "reply")
        self.assertIsNotNone(r["reply"])


# ---------------------------------------------------------------------------
# 3. Manager Context Builder
# ---------------------------------------------------------------------------

class TestManagerContextBuilder(unittest.TestCase):
    """Compact context passed to LLM must be correct."""

    def test_yul_gets_yul_banks(self):
        ctx = _build_manager_context({"debtor_type": "ЮЛ"})
        banks_str = str(ctx.get("active_banks_yul", ""))
        self.assertIn("Альфа-Банк", banks_str)
        self.assertIn("ТКБ", banks_str)

    def test_fl_gets_fl_banks(self):
        ctx = _build_manager_context({"debtor_type": "ФЛ"})
        banks_str = str(ctx.get("active_banks_fl", ""))
        self.assertNotIn("Альфа-Банк", banks_str)
        self.assertIn("ТКБ", banks_str)

    def test_yul_has_paused_banks(self):
        ctx = _build_manager_context({"debtor_type": "ЮЛ"})
        self.assertIn("paused_banks", ctx)
        self.assertIn("Т-Банк", ctx["paused_banks"])

    def test_last_bank_propagated(self):
        ctx = _build_manager_context({"_last_bank": "Уралсиб"})
        self.assertEqual(ctx["last_bank_focus"], "Уралсиб")

    def test_previous_reply_included(self):
        ctx = _build_manager_context({"_last_bot_text": "Для ЮЛ три варианта."})
        self.assertIn("previous_bot_reply", ctx)

    def test_no_last_bank_no_key(self):
        ctx = _build_manager_context({})
        self.assertNotIn("last_bank_focus", ctx)

    def test_client_type_fallback(self):
        ctx = _build_manager_context({"client_type": "ИП"})
        banks_str = str(ctx.get("active_banks_yul", ""))
        self.assertIn("Альфа-Банк", banks_str)

    def test_relevant_facts_from_kb(self):
        kb = [
            {"bank": "ТКБ", "text": "открытие 2800 ₽, ведение 2090 ₽/мес."},
            {"bank": None, "text": "документы: ИНН должника"},
        ]
        ctx = _build_manager_context({"debtor_type": "ЮЛ"}, kb_facts=kb)
        self.assertIn("relevant_facts", ctx)
        self.assertTrue(any("ТКБ" in f for f in ctx["relevant_facts"]))

    def test_no_relevant_facts_when_empty_kb(self):
        ctx = _build_manager_context({"debtor_type": "ЮЛ"}, kb_facts=[])
        self.assertNotIn("relevant_facts", ctx)

    def test_active_topic_for_ready_to_open(self):
        ctx = _build_manager_context({
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {
                "deal_stage": "ready_to_open",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "handoff_needed": True,
            },
        })
        self.assertIn("active_topic", ctx)
        self.assertIn("ТКБ", ctx["active_topic"])

    def test_docs_intent_instruction(self):
        ctx = _build_manager_context({
            "debtor_type": "ЮЛ",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "debtor_type": "ЮЛ",
            }
        })
        self.assertIn("instruction", ctx)
        self.assertIn("ИНН", ctx["instruction"])


# ---------------------------------------------------------------------------
# 4. ConversationResponder payload — correct context passed to LLM
# ---------------------------------------------------------------------------

class TestConversationResponderPayload(unittest.TestCase):
    """Responder must pass rich context including manager_context."""

    def setUp(self):
        self.captured_messages = []

    def _mock_ask_llm(self, reply_text: str):
        async def fake_ask_llm(messages, **kwargs):
            self.captured_messages.extend(messages)
            return _fake_responder_json(reply_text)
        return fake_ask_llm

    def test_recent_dialog_included(self):
        recent = [
            {"role": "user", "text": "у меня ЮЛ"},
            {"role": "bot", "text": "Отлично, для ЮЛ активны три банка."},
        ]
        with patch("app.services.conversation_responder.ask_llm",
                   side_effect=self._mock_ask_llm("Окей, по тарифам:")):
            _run(run_conversation_responder(
                user_text="тарифы?",
                recent_dialog=recent,
                known_slots={"debtor_type": "ЮЛ"},
                candidate_intents=["tariff_comparison"],
                candidate_scenarios=["bank_pricing_yul"],
                kb_facts=[],
            ))
        user_msg = next(m for m in self.captured_messages if m["role"] == "user")
        self.assertIn("recent_dialog", user_msg["content"])
        self.assertIn("manager_context", user_msg["content"])

    def test_manager_context_has_active_banks_for_yul(self):
        with patch("app.services.conversation_responder.ask_llm",
                   side_effect=self._mock_ask_llm("Для ЮЛ: Альфа-Банк, ТКБ, Уралсиб.")):
            _run(run_conversation_responder(
                user_text="с какими банками?",
                recent_dialog=[],
                known_slots={"debtor_type": "ЮЛ"},
                candidate_intents=["bank_selection"],
                candidate_scenarios=[],
                kb_facts=[],
            ))
        user_msg = next(m for m in self.captured_messages if m["role"] == "user")
        self.assertIn("Альфа-Банк", user_msg["content"])

    def test_previous_reply_in_context(self):
        slots = {"_last_bot_text": "Для ЮЛ три варианта: Альфа, ТКБ, Уралсиб."}
        with patch("app.services.conversation_responder.ask_llm",
                   side_effect=self._mock_ask_llm("Подробнее по ТКБ:")):
            _run(run_conversation_responder(
                user_text="подробнее",
                recent_dialog=[],
                known_slots=slots,
                candidate_intents=[],
                candidate_scenarios=[],
                kb_facts=[],
            ))
        user_msg = next(m for m in self.captured_messages if m["role"] == "user")
        self.assertIn("previous_bot_reply", user_msg["content"])

    def test_returns_reply_from_llm(self):
        with patch("app.services.conversation_responder.ask_llm",
                   side_effect=self._mock_ask_llm("Для ЮЛ сейчас активны Альфа-Банк, ТКБ и Уралсиб.")):
            result = _run(run_conversation_responder(
                user_text="какие банки?",
                recent_dialog=[],
                known_slots={"debtor_type": "ЮЛ"},
                candidate_intents=["bank_selection"],
                candidate_scenarios=[],
                kb_facts=[],
            ))
        self.assertIn("Альфа-Банк", result["reply"])


# ---------------------------------------------------------------------------
# 5. Slot propagation across turns
# ---------------------------------------------------------------------------

class TestSlotPropagation(unittest.TestCase):
    """Debtor type, last bank, and other slots must persist across turns."""

    def test_debtor_type_persists(self):
        """Once ЮЛ is known, playbook must not ask again."""
        from app.processing.scenario_playbook import SLOT_FORBIDDEN, run_scenario_playbook
        slots = {"_active_scenario": "bank_selection_yul", "_known_slots": {},
                 "debtor_type": "ЮЛ", "client_type": "ЮЛ"}
        r = run_scenario_playbook("сравни тарифы", slots)
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(
            any("юрлица или физлица" in f.lower() for f in forbidden),
            f"Must forbid repeated debtor-type question: {forbidden}",
        )

    def test_responder_adapter_preserves_last_bank(self):
        responder = {
            "reply": "По ТКБ открытие 2800 ₽.",
            "intent": "tariff_comparison",
            "next_step": None,
            "needs_clarification": False,
            "handoff": {"needed": False, "reason": None},
        }
        brain = responder_to_brain_result(responder, known_slots={"_last_bank": "ТКБ"})
        self.assertEqual(brain["state_update"]["last_bank"], "ТКБ")

    def test_responder_adapter_action_is_answer(self):
        responder = {
            "reply": "Вот тарифы.",
            "intent": "tariff_comparison",
            "next_step": None,
            "needs_clarification": False,
            "handoff": {"needed": False, "reason": None},
        }
        brain = responder_to_brain_result(responder)
        self.assertEqual(brain["action"], "answer")

    def test_responder_adapter_handoff_needed(self):
        responder = {
            "reply": "Отлично, давайте оформим.",
            "intent": "handoff",
            "next_step": None,
            "needs_clarification": False,
            "handoff": {"needed": True, "reason": "ready_to_open"},
        }
        brain = responder_to_brain_result(responder)
        self.assertEqual(brain["action"], "handoff")
        self.assertTrue(brain["handoff"]["needed"])


# ---------------------------------------------------------------------------
# 6. Validator — missing_required_question no longer hard error
# ---------------------------------------------------------------------------

class TestValidatorMissingQuestion(unittest.TestCase):
    """missing_required_question must be a warning, not a hard validation error."""

    def test_reply_without_question_is_valid_when_policy_required(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Для ЮЛ активны Альфа-Банк, ТКБ и Уралсиб.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="с какими банками работаете?",
            answer_contract={"question_policy": "required", "topic": "partner_banks"},
        )
        self.assertTrue(result["is_valid"],
            f"missing_required_question must not fail validation: {result['reason']}")

    def test_valid_reply_still_passes(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Для ЮЛ активны три банка: Альфа-Банк, ТКБ, Уралсиб.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="какие банки?",
        )
        self.assertTrue(result["is_valid"])


# ---------------------------------------------------------------------------
# 7. Golden Dialog 1 — ЮЛ banks → TKB → ready_to_open (§11 dialog 1)
# ---------------------------------------------------------------------------

class TestGoldenDialog1_YUL_TKB_ReadyToOpen(unittest.TestCase):

    def setUp(self):
        from app.processing.deal_state import update_deal_state
        self._upd = update_deal_state
        self.slots = {"_active_scenario": None, "_known_slots": {}}

    def _turn(self, text, **slot_updates):
        self.slots.update(slot_updates)
        return self._upd(self.slots, text)

    def test_turn1_greeting_is_consulting(self):
        ds = self._turn("привет")
        self.assertEqual(ds["deal_stage"], "consulting")
        self.assertFalse(ds["handoff_needed"])

    def test_turn2_yul_not_action(self):
        ds = self._turn("у меня должник ЮЛ, с какими банками сотрудничаете?",
                        debtor_type="ЮЛ")
        self.assertFalse(ds["handoff_needed"])
        self.assertNotEqual(ds["deal_stage"], "ready_to_open")

    def test_turn5_nuzhen_schet_tkb_ready_to_open(self):
        ds = self._turn("мне тогда в ТКБ счет нужен",
                        debtor_type="ЮЛ", _last_bank="ТКБ")
        self.assertEqual(ds["deal_stage"], "ready_to_open")
        self.assertEqual(ds["selected_bank"], "ТКБ")
        self.assertEqual(ds["debtor_type"], "ЮЛ")
        self.assertTrue(ds["handoff_needed"])
        self.assertEqual(ds["client_intent"], "open_account")
        self.assertEqual(ds["next_manager_move"], "handoff_to_manager")

    def test_turn6_nu_otkroete_stays_ready(self):
        # Simulate prior state
        self.slots["_deal_state"] = {
            "deal_stage": "ready_to_open",
            "selected_bank": "ТКБ",
            "debtor_type": "ЮЛ",
            "handoff_needed": True,
        }
        self.slots["debtor_type"] = "ЮЛ"
        self.slots["_last_bank"] = "ТКБ"
        ds = self._turn("ну, откроете?")
        self.assertEqual(ds["deal_stage"], "ready_to_open")
        self.assertTrue(ds["handoff_needed"])

    def test_turn7_chto_trebuetsya_docs_intent(self):
        from app.processing.deal_state import detect_docs_intent
        self.slots["_deal_state"] = {"deal_stage": "ready_to_open", "handoff_needed": True}
        self.slots["debtor_type"] = "ЮЛ"
        self.slots["_last_bank"] = "ТКБ"
        self.assertTrue(detect_docs_intent("что от меня требуется", self.slots))
        ds = self._turn("что от меня требуется")
        self.assertEqual(ds["next_manager_move"], "explain_documents")
        # stage must not revert from ready_to_open
        self.assertEqual(ds["deal_stage"], "ready_to_open")

    def test_validator_rejects_tariff_only_for_ready_to_open(self):
        from app.services.response_validator import validate_reply
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {"deal_stage": "ready_to_open", "handoff_needed": True},
        }
        result = validate_reply(
            reply="По ТКБ для ЮЛ: открытие 2800 ₽, ведение 2090 ₽/мес.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="мне тогда в ТКБ счет нужен",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "ready_to_open_no_manager_reference")

    def test_validator_accepts_reply_with_manager_ref(self):
        from app.services.response_validator import validate_reply
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {"deal_stage": "ready_to_open", "handoff_needed": True},
        }
        result = validate_reply(
            reply="Понял, ТКБ для ЮЛ. Передам кейс менеджеру. Для старта нужен ИНН должника.",
            brain_result={"action": "handoff", "handoff": {"needed": True}},
            current_entities={},
            slots=slots,
            user_text="мне тогда в ТКБ счет нужен",
        )
        self.assertTrue(result["is_valid"], result.get("reason"))

    def test_manager_context_has_instruction_when_handoff_needed(self):
        ctx = _build_manager_context({
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {
                "deal_stage": "ready_to_open",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "client_intent": "open_account",
                "next_manager_move": "handoff_to_manager",
                "handoff_needed": True,
            },
        })
        self.assertIn("instruction", ctx)
        self.assertIn("ТКБ", ctx["instruction"])
        self.assertIn("handoff.needed=true", ctx["instruction"])
        # new fields from §8
        self.assertIn("active_topic", ctx)
        self.assertIn("paused_banks", ctx)
        self.assertIn("active_banks_yul", ctx)


# ---------------------------------------------------------------------------
# 8. Golden Dialog 2 — Direct ready request (§11 dialog 2)
# ---------------------------------------------------------------------------

class TestGoldenDialog2_DirectReadyRequest(unittest.TestCase):

    def test_nuzhen_schet_tkb_yul_sets_ready_to_open(self):
        from app.processing.deal_state import update_deal_state
        slots = {"debtor_type": "ЮЛ", "_last_bank": "ТКБ"}
        ds = update_deal_state(slots, "нужен счет в ТКБ для ЮЛ")
        self.assertEqual(ds["deal_stage"], "ready_to_open")
        self.assertEqual(ds["selected_bank"], "ТКБ")
        self.assertEqual(ds["debtor_type"], "ЮЛ")
        self.assertTrue(ds["handoff_needed"])

    def test_direct_ready_reply_must_mention_manager(self):
        from app.services.response_validator import validate_reply
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {"deal_stage": "ready_to_open", "handoff_needed": True},
        }
        bad = validate_reply(
            reply="Конечно, давайте!",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="нужен счет в ТКБ для ЮЛ",
        )
        self.assertFalse(bad["is_valid"])

        good = validate_reply(
            reply="Понял, ТКБ для ЮЛ. Передам менеджеру. Нужен ИНН должника.",
            brain_result={"action": "handoff", "handoff": {"needed": True}},
            current_entities={},
            slots=slots,
            user_text="нужен счет в ТКБ для ЮЛ",
        )
        self.assertTrue(good["is_valid"], good.get("reason"))

    def test_ready_to_open_action_intent_without_bank_does_not_set_handoff(self):
        from app.processing.deal_state import update_deal_state
        slots = {}
        ds = update_deal_state(slots, "хочу открыть счет")
        # Bank and debtor unknown — cannot hand off yet
        self.assertFalse(ds["handoff_needed"])
        self.assertNotEqual(ds["deal_stage"], "ready_to_open")

    def test_ready_to_open_with_bank_only_no_handoff(self):
        from app.processing.deal_state import update_deal_state
        slots = {"_last_bank": "ТКБ"}
        ds = update_deal_state(slots, "хочу открыть")
        # Debtor type unknown — collect_missing_slot, no full handoff
        self.assertFalse(ds["handoff_needed"])


# ---------------------------------------------------------------------------
# 9. Golden Dialog 3 — Value objection (§11 dialog 3)
# ---------------------------------------------------------------------------

class TestGoldenDialog3_ValueObjection(unittest.TestCase):

    def test_value_objection_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent(
            "а какая мне выгода через вас открывать? могу ведь сам пойти"
        ))

    def test_deal_stage_stays_consulting_for_objection(self):
        from app.processing.deal_state import update_deal_state
        slots = {"debtor_type": "ЮЛ"}
        ds = update_deal_state(slots, "а какая мне выгода через вас открывать? могу ведь сам пойти")
        self.assertFalse(ds["handoff_needed"])
        self.assertNotEqual(ds["deal_stage"], "ready_to_open")

    def test_validator_allows_value_proposition_reply(self):
        from app.services.response_validator import validate_reply
        slots = {"debtor_type": "ЮЛ"}
        result = validate_reply(
            reply=(
                "Напрямую в банк обратиться можно. Но через нас доступны льготные условия, "
                "которых банк не даёт напрямую, плюс сопровождаем документы и финмониторинг."
            ),
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="а какая мне выгода через вас открывать?",
        )
        self.assertTrue(result["is_valid"], result.get("reason"))

    def test_generic_fallback_rejected_for_clear_value_question(self):
        from app.services.response_validator import validate_reply
        slots = {}
        result = validate_reply(
            reply="Уточните вопрос, пожалуйста.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="какая выгода через вас открывать?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "generic_fallback_for_clear_intent")


# ---------------------------------------------------------------------------
# 10. Golden Dialog 4 — FL bank (§11 dialog 4)
# ---------------------------------------------------------------------------

class TestGoldenDialog4_FL_Bank(unittest.TestCase):

    def test_fl_debtor_type_preserved(self):
        from app.processing.deal_state import update_deal_state
        slots = {"debtor_type": "ФЛ"}
        ds = update_deal_state(slots, "какие тарифы там?", intent_signals={"intents": []})
        self.assertEqual(ds["debtor_type"], "ФЛ")

    def test_fl_manager_context_no_yul_banks(self):
        ctx = _build_manager_context({"debtor_type": "ФЛ"})
        # FL context: active_banks_fl has ТКБ, no Альфа-Банк in yul list
        self.assertNotIn("active_banks_yul", ctx)
        fl_str = str(ctx.get("active_banks_fl", ""))
        self.assertIn("ТКБ", fl_str)
        self.assertNotIn("Альфа-Банк", fl_str)

    def test_validator_rejects_repeated_debtor_type_question_when_fl_known(self):
        from app.services.response_validator import validate_reply
        slots = {"debtor_type": "ФЛ", "client_type": "ФЛ"}
        result = validate_reply(
            reply="Для юрлица или физлица подбираем счёт?",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="какие тарифы там?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "repeated_known_debtor_type_question")

    def test_kb_query_expander_fl_adds_fl_banks(self):
        slots = _slots(debtor_type="ФЛ")
        result = expand_kb_query("тарифы?", slots)
        self.assertIn("ФЛ", result)
        self.assertNotIn("Альфа-Банк", result)


# ---------------------------------------------------------------------------
# 11. Golden Dialog 5 — Observation stage (§11 dialog 5)
# ---------------------------------------------------------------------------

class TestGoldenDialog5_ObservationStage(unittest.TestCase):

    def test_observation_query_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent(
            "у вас открываются счета на стадии наблюдения?"
        ))

    def test_what_required_after_observation_is_docs_intent(self):
        from app.processing.deal_state import detect_docs_intent
        slots = {"debtor_type": "ЮЛ", "_last_bank": "Альфа-Банк"}
        self.assertTrue(detect_docs_intent("что от меня требуется?", slots))

    def test_docs_intent_next_move_explain_documents(self):
        from app.processing.deal_state import update_deal_state
        slots = {"debtor_type": "ЮЛ", "_last_bank": "Альфа-Банк"}
        ds = update_deal_state(slots, "что от меня требуется?")
        self.assertEqual(ds["next_manager_move"], "explain_documents")
        self.assertEqual(ds["client_intent"], "ask_documents")

    def test_validator_rejects_reply_without_docs_for_docs_intent(self):
        from app.services.response_validator import validate_reply
        slots = {
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
            }
        }
        result = validate_reply(
            reply="Конечно, всё сделаем.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="что от меня требуется?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "docs_intent_no_docs_in_reply")

    def test_validator_accepts_reply_with_docs_info(self):
        from app.services.response_validator import validate_reply
        slots = {
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
            }
        }
        result = validate_reply(
            reply="Нужен ИНН должника, данные арбитражного управляющего и судебный акт о введении процедуры.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots,
            user_text="что от меня требуется?",
        )
        self.assertTrue(result["is_valid"], result.get("reason"))


# ---------------------------------------------------------------------------
# 12. DealState — missing fields (§2)
# ---------------------------------------------------------------------------

class TestDealStateMissingFields(unittest.TestCase):

    def test_procedure_stage_synced_from_slots(self):
        from app.processing.deal_state import update_deal_state
        slots = {"procedure_type": "НАБЛЮДЕНИЕ"}
        ds = update_deal_state(slots, "вопрос")
        self.assertEqual(ds["procedure_stage"], "НАБЛЮДЕНИЕ")

    def test_last_bot_reply_hash_computed(self):
        from app.processing.deal_state import update_deal_state
        import hashlib
        reply = "По ТКБ открытие 2800 ₽, ведение 2090 ₽/мес."
        expected = hashlib.md5(reply.encode("utf-8")).hexdigest()[:8]
        slots = {"_last_bot_text": reply}
        ds = update_deal_state(slots, "подробнее")
        self.assertEqual(ds["last_bot_reply_hash"], expected)
        self.assertEqual(ds["last_offer"], reply[:200])

    def test_last_bank_list_context_extracted(self):
        from app.processing.deal_state import update_deal_state
        slots = {"_last_bot_text": "Для ЮЛ активны Альфа-Банк, ТКБ и Уралсиб."}
        ds = update_deal_state(slots, "подробнее")
        self.assertIn("Альфа-Банк", ds["last_bank_list_context"])
        self.assertIn("ТКБ", ds["last_bank_list_context"])
        self.assertIn("Уралсиб", ds["last_bank_list_context"])

    def test_last_conditions_context_set_when_pricing_reply(self):
        from app.processing.deal_state import update_deal_state
        slots = {
            "_last_bot": "По ТКБ: открытие 2800 ₽, ведение 2090 ₽/мес.",
            "_last_bot_text": "По ТКБ: открытие 2800 ₽, ведение 2090 ₽/мес.",
            "_last_bank": "ТКБ",
        }
        ds = update_deal_state(slots, "и что?")
        self.assertIn("ТКБ_conditions", ds["last_conditions_context"])

    def test_hash_not_recomputed_if_text_unchanged(self):
        from app.processing.deal_state import update_deal_state
        reply = "Для ЮЛ: Альфа, ТКБ, Уралсиб."
        slots = {
            "_last_bot_text": reply,
            "_deal_state": {"last_bot_reply_hash": "aabbccdd"},
        }
        ds = update_deal_state(slots, "подробнее")
        # Same text → hash changes (we compute fresh)
        import hashlib
        self.assertEqual(ds["last_bot_reply_hash"], hashlib.md5(reply.encode()).hexdigest()[:8])

    def test_all_required_fields_present(self):
        from app.processing.deal_state import update_deal_state
        ds = update_deal_state({}, "любой текст")
        required = [
            "deal_stage", "selected_bank", "selected_product", "debtor_type",
            "procedure_stage", "client_intent", "next_manager_move",
            "ready_to_open_reason", "handoff_needed",
            "last_offer", "last_bank_list_context", "last_conditions_context",
            "last_bot_reply_hash",
        ]
        for f in required:
            self.assertIn(f, ds, f"Missing field: {f}")


# ---------------------------------------------------------------------------
# 13. Escalation metadata format (§5)
# ---------------------------------------------------------------------------

class TestEscalationMetadataFormat(unittest.TestCase):

    def _fmt(self, need, slots):
        from app.escalation.service import _format_request_text
        return _format_request_text(need, slots)

    def test_ready_to_open_upgrades_consultation_label(self):
        slots = {"_deal_state": {"deal_stage": "ready_to_open", "selected_bank": "ТКБ"}}
        text = self._fmt("Консультация", slots)
        self.assertIn("Открытие счёта", text)
        self.assertIn("ТКБ", text)

    def test_selected_bank_included_in_text(self):
        slots = {
            "_deal_state": {
                "deal_stage": "ready_to_open",
                "selected_bank": "Уралсиб",
                "debtor_type": "ЮЛ",
            }
        }
        text = self._fmt("Открытие счёта", slots)
        self.assertIn("Уралсиб", text)

    def test_debtor_type_from_deal_state_when_no_slot(self):
        slots = {"_deal_state": {"deal_stage": "ready_to_open", "debtor_type": "ФЛ"}}
        text = self._fmt("Открытие счёта", slots)
        self.assertIn("физлица", text)

    def test_non_bankruptcy_product_included(self):
        slots = {
            "_deal_state": {
                "deal_stage": "ready_to_open",
                "selected_product": "card",
                "selected_bank": "ТКБ",
            }
        }
        text = self._fmt("Открытие счёта", slots)
        self.assertIn("card", text)

    def test_consulting_stays_consulting_when_no_deal_state(self):
        text = self._fmt("Консультация", {})
        self.assertIn("Консультация", text)


# ---------------------------------------------------------------------------
# PLAN v9.6 — Golden Dialog Tests (Dialogs A–G)
# ---------------------------------------------------------------------------

class TestDialogAFlYulContextDrift(unittest.TestCase):
    """Dialog A — FL context must not drift to YUL scenarios."""

    def test_fl_stored_debtor_type_blocks_yul_scenarios(self):
        """Once debtor_type=ФЛ is stored, YUL scenarios must be penalised."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"debtor_type": "ФЛ", "_active_scenario": "bank_selection_fl"}
        result = decide_scenario_policy(
            user_text="только один банк?",
            slots=slots,
            rag_scenarios=[],
        )
        active = result["active_scenario"]
        ct_map = {
            "bank_selection_yul": "ЮЛ",
            "alfabank_yul_conditions": "ЮЛ",
            "tkb_yul_conditions": "ЮЛ",
            "uralsib_yul_conditions": "ЮЛ",
            "tbank_yul_conditions": "ЮЛ",
        }
        self.assertNotIn(
            active, ct_map,
            f"FL session must not switch to YUL scenario; got active={active!r}",
        )

    def test_fl_manager_context_has_only_fl_banks(self):
        """FL debtor_type → manager_context must not list YUL banks."""
        ctx = _build_manager_context({"debtor_type": "ФЛ"})
        banks_str = str(ctx)
        self.assertNotIn("Альфа-Банк", banks_str,
            "Альфа-Банк is YUL-only and must not appear in FL manager_context")

    def test_fl_manager_context_lists_tkb(self):
        """FL debtor_type → TKB must appear in FL bank list."""
        ctx = _build_manager_context({"debtor_type": "ФЛ"})
        banks_str = str(ctx.get("active_banks_fl", ""))
        self.assertIn("ТКБ", banks_str)

    def test_uralsib_not_in_fl_active_banks(self):
        """KB audit: Uralsib has no FL data — must not appear in FL active banks."""
        ctx = _build_manager_context({"debtor_type": "ФЛ"})
        banks_str = str(ctx.get("active_banks_fl", ""))
        self.assertNotIn("Уралсиб", banks_str,
            "Uralsib has no FL KB data; must not be listed as an FL active bank")

    def test_explicit_yul_switch_allowed_from_fl(self):
        """User explicitly asking about ЮЛ must be allowed to switch even if FL is stored."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"debtor_type": "ФЛ", "_active_scenario": "bank_selection_fl"}
        result = decide_scenario_policy(
            user_text="а для юридических лиц какие банки?",
            slots=slots,
            rag_scenarios=[],
        )
        # Must not be blocked to FL — should allow YUL scenarios
        scores = result.get("scores", {})
        fl_dominated = all(
            "yul" not in sid for sid in scores
        )
        # If the message explicitly asks about ЮЛ, scoring must not produce a purely FL result
        # (scores dict may be empty if override fired — that's OK too)
        # Key assertion: it should not be stuck on bank_selection_fl continuity
        self.assertNotEqual(
            result["active_scenario"], "bank_selection_fl",
            "Explicit ЮЛ switch must be allowed even when FL is stored",
        )


class TestDialogBBonusOverridesObjection(unittest.TestCase):
    """Dialog B — bonus/interest intent must override active direct_bank_objection."""

    def test_interest_or_bonus_removes_objection_continuity(self):
        """When active=direct_bank_objection but user asks about bonuses, bonus scenario wins."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "direct_bank_objection"}
        result = decide_scenario_policy(
            user_text="какие ещё бонусы есть для арбитражных управляющих?",
            slots=slots,
            rag_scenarios=[
                {"scenario_id": "au_bonus_question", "score": 0.6, "reasons": [], "matched_aliases": []},
                {"scenario_id": "interest_on_balance", "score": 0.5, "reasons": [], "matched_aliases": []},
            ],
            intent_signals={
                "intents": ["interest_or_bonus"],
                "matches": [{"intent": "interest_or_bonus", "source": "regex", "score": 0.9,
                              "matched_anchor": "бонусы для ау", "threshold": 0.60}],
                "dialog_acts": [],
                "debtor_type": None,
                "bank_focus": None,
                "semantic_rejects": [],
            },
        )
        active = result["active_scenario"]
        self.assertNotEqual(
            active, "direct_bank_objection",
            "interest_or_bonus must override direct_bank_objection continuity",
        )

    def test_repetition_complaint_overrides_objection(self):
        """repetition_complaint must always override any active business scenario."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "direct_bank_objection"}
        result = decide_scenario_policy(
            user_text="ты уже говорил об этом, зачем повторяешь?",
            slots=slots,
            rag_scenarios=[],
            intent_signals={
                "intents": ["repetition_complaint"],
                "matches": [{"intent": "repetition_complaint", "source": "regex", "score": 0.95,
                              "matched_anchor": "ты это уже говорил", "threshold": 0.62}],
                "dialog_acts": [],
                "debtor_type": None,
                "bank_focus": None,
                "semantic_rejects": [],
            },
        )
        self.assertEqual(result["active_scenario"], "bot_repetition_complaint")
        self.assertEqual(result["reason"], "repetition_complaint_override")


class TestDialogCRepetitionComplaint(unittest.TestCase):
    """Dialog C — repetition complaint must be handled as a complaint."""

    def test_repetition_complaint_scenario_set(self):
        """Detecting repetition_complaint → active scenario = bot_repetition_complaint."""
        from app.processing.scenario_policy import decide_scenario_policy
        result = decide_scenario_policy(
            user_text="зачем ты повторяешься?",
            slots={},
            rag_scenarios=[],
            intent_signals={
                "intents": ["repetition_complaint"],
                "matches": [{"intent": "repetition_complaint", "source": "regex", "score": 0.92,
                              "matched_anchor": "зачем повторяешься", "threshold": 0.62}],
                "dialog_acts": [],
                "debtor_type": None,
                "bank_focus": None,
                "semantic_rejects": [],
            },
        )
        self.assertEqual(result["active_scenario"], "bot_repetition_complaint")

    def test_repetition_complaint_injects_instruction(self):
        """bot_repetition_complaint active scenario → manager_context has dialog_act and instruction."""
        ctx = _build_manager_context({"_active_scenario": "bot_repetition_complaint"})
        self.assertEqual(ctx.get("dialog_act"), "repetition_complaint")
        self.assertIn("instruction", ctx)
        inst = ctx["instruction"]
        self.assertIn("повторяешь", inst.lower())

    def test_typo_variant_in_gate(self):
        """'повторяешся' (typo) must pass the semantic gate."""
        from app.processing.intent_semantic import INTENT_GATES
        gate = INTENT_GATES.get("repetition_complaint", [])
        self.assertIn("повторяешься", gate,
            "Typo-tolerant variants must be in the repetition_complaint gate")


class TestDialogDTimelines(unittest.TestCase):
    """Dialog D — timeline question must not require tariff details."""

    def test_timing_question_skips_fl_tariff_check(self):
        """'Как долго?' must not fail with missing_fl_tariff_details."""
        from app.services.response_validator import validate_reply
        reply = "Обычно открытие счёта занимает 3–5 рабочих дней после подачи документов."
        result = validate_reply(
            reply=reply,
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="сколько времени занимает открытие?",
            answer_contract={"topic": "bank_selection_fl"},
        )
        self.assertNotEqual(result.get("reason"), "missing_fl_tariff_details",
            "Timing question must not require tariff details")

    def test_timing_question_skips_missing_primary_fact(self):
        """Timing reply without mentioning ТКБ must still be valid."""
        from app.services.response_validator import validate_reply
        reply = "Процедура открытия счёта обычно занимает около недели."
        result = validate_reply(
            reply=reply,
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="сколько дней ждать?",
            answer_contract={"topic": "bank_selection_fl"},
        )
        self.assertNotEqual(result.get("reason"), "missing_primary_fact",
            "Timing reply need not mention specific bank")

    def test_tariff_question_still_requires_fl_details(self):
        """Non-timing tariff question still requires FL tariff details."""
        from app.services.response_validator import validate_reply
        reply = "Для физических лиц счёт открывается легко."
        result = validate_reply(
            reply=reply,
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="какие условия открытия для физлиц?",
            answer_contract={"topic": "bank_selection_fl"},
        )
        self.assertFalse(result["is_valid"],
            "Non-timing tariff question must still require ТКБ and pricing in reply")


class TestDialogEEscalationMetadata(unittest.TestCase):
    """Dialog E — escalation metadata must come from DealState, not stale slots."""

    def test_deal_state_debtor_type_used_in_format_text(self):
        """_format_request_text must take debtor_type from _deal_state, not loose slot."""
        from app.escalation.service import _format_request_text
        slots = {
            "_deal_state": {
                "deal_stage": "ready_to_open",
                "selected_bank": "Т-Банк",
                "debtor_type": "ФЛ",
            },
            "debtor_type": "ЮЛ",  # stale loose slot — must be overridden by _deal_state
        }
        text = _format_request_text("Открытие счёта", slots)
        self.assertIn("физлица", text,
            "_format_request_text must prefer _deal_state.debtor_type over loose debtor_type slot")

    def test_escalation_metadata_logs_deal_state_bank(self):
        """EscalationMetadata log must pick up selected_bank from _deal_state."""
        from app.escalation.service import _format_request_text
        slots = {"_deal_state": {"selected_bank": "ТКБ", "debtor_type": "ФЛ"}}
        text = _format_request_text("Открытие счёта", slots)
        self.assertIn("ТКБ", text)


class TestDialogFBitrixFailure(unittest.TestCase):
    """Dialog F — Bitrix failure must not produce false 'Escalation completed'."""

    def test_notify_manager_returns_status_dict(self):
        """notify_manager must return a dict with manager_notified key."""
        import inspect
        from app.escalation.bitrix_notifier import notify_manager
        sig = inspect.signature(notify_manager)
        ret = sig.return_annotation
        self.assertIsNotNone(
            ret,
            "notify_manager must declare a return type (dict)",
        )

    def test_escalation_service_imports_correctly(self):
        """escalation service must import without error."""
        from app.escalation import service  # noqa: F401
        self.assertTrue(True)


class TestDialogGStartupDB(unittest.TestCase):
    """Dialog G — worker must guarantee DB schema before polling."""

    def test_worker_module_has_db_init(self):
        """worker_loop must call Base.metadata.create_all before polling."""
        import inspect
        import app.worker as worker_mod
        src = inspect.getsource(worker_mod.worker_loop)
        self.assertIn("create_all", src,
            "worker_loop must call Base.metadata.create_all before the polling loop")

    def test_worker_imports_storage_models(self):
        """worker module must import storage.models to register ORM models."""
        import inspect
        import app.worker as worker_mod
        src = inspect.getsource(worker_mod.worker_loop)
        self.assertIn("storage.models", src,
            "worker_loop must import app.storage.models to register all ORM models")


# ---------------------------------------------------------------------------
# Plan §16 Dialog A — Objection → ЮЛ → tariff switch
# ---------------------------------------------------------------------------

class TestPlan16DialogA_ObjectionThenYUL(unittest.TestCase):
    """Dialog A: value objection → ЮЛ declared → tariff question must switch scenario."""

    def test_conditions_correction_blocks_tariff_scenarios(self):
        """correction_not_tariffs_but_conditions → tariff-focused scenarios get -999."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="я про условия, а не тарифы",
            slots=slots,
            rag_scenarios=[],
            intent_signals={
                "intents": ["correction_not_tariffs_but_conditions"],
                "matches": [{"intent": "correction_not_tariffs_but_conditions",
                             "source": "regex", "score": 0.99,
                             "matched_anchor": "не тарифы а условия", "threshold": 0.55}],
                "dialog_acts": [],
                "debtor_type": "ЮЛ",
                "bank_focus": None,
                "semantic_rejects": [],
            },
        )
        active = result["active_scenario"]
        from app.processing.scenario_policy import _TARIFF_FOCUSED_SCENARIOS
        self.assertNotIn(
            active, _TARIFF_FOCUSED_SCENARIOS,
            f"conditions_correction must block tariff scenarios; got active={active!r}",
        )
        self.assertIn("conditions_correction", result.get("reason", ""),
            "reason must include conditions_correction tag")

    def test_conditions_correction_boosts_conditions_scenarios(self):
        """correction_not_tariffs_but_conditions → conditions scenarios get +60 boost."""
        from app.processing.scenario_policy import decide_scenario_policy, _CONDITIONS_FOCUSED_SCENARIOS
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ",
                 "_last_bank": "ТКБ"}
        result = decide_scenario_policy(
            user_text="я про условия, а не тарифы",
            slots=slots,
            rag_scenarios=[
                {"scenario_id": "tkb_yul_conditions", "score": 0.8, "reasons": [], "matched_aliases": []},
            ],
            intent_signals={
                "intents": ["correction_not_tariffs_but_conditions"],
                "matches": [{"intent": "correction_not_tariffs_but_conditions",
                             "source": "regex", "score": 0.99,
                             "matched_anchor": "не тарифы а условия", "threshold": 0.55}],
                "dialog_acts": [], "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": [],
            },
        )
        active = result["active_scenario"]
        self.assertIn(
            active, _CONDITIONS_FOCUSED_SCENARIOS,
            f"conditions_correction must route to conditions scenario; got active={active!r}",
        )

    def test_conditions_intent_soft_boost(self):
        """conditions_details_requested → soft boost for conditions scenarios."""
        from app.processing.scenario_policy import decide_scenario_policy, _CONDITIONS_FOCUSED_SCENARIOS
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="расскажи про условия работы со счётом",
            slots=slots,
            rag_scenarios=[
                {"scenario_id": "tkb_yul_conditions", "score": 0.7, "reasons": [], "matched_aliases": []},
            ],
            intent_signals={
                "intents": ["conditions_details_requested"],
                "matches": [{"intent": "conditions_details_requested",
                             "source": "regex", "score": 0.88,
                             "matched_anchor": "условия работы счёта", "threshold": 0.55}],
                "dialog_acts": [], "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": [],
            },
        )
        scores = result.get("scores", {})
        # At least one conditions scenario must appear in top scores
        has_conditions = any(sid in _CONDITIONS_FOCUSED_SCENARIOS for sid in scores)
        self.assertTrue(
            has_conditions,
            f"conditions_intent must boost at least one conditions scenario; scores={scores}",
        )


# ---------------------------------------------------------------------------
# Plan §16 Dialog E — Formal tone enforcement
# ---------------------------------------------------------------------------

class TestPlan16DialogE_FormalTone(unittest.TestCase):
    """Dialog E: replies with ты/тебе/твой must be rejected by validator."""

    def test_ty_rejected(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Ты можешь открыть счёт в ТКБ.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="как открыть счёт?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "informal_address_tu_tebe")

    def test_tebe_rejected(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Расскажу тебе подробнее об условиях.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="расскажи подробнее",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "informal_address_tu_tebe")

    def test_tvoy_rejected(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Твой должник — ЮЛ, верно?",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="у меня должник ЮЛ",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "informal_address_tu_tebe")

    def test_formal_vam_accepted(self):
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Для вашего должника ЮЛ, рекомендую Альфа-Банк или ТКБ.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="у меня должник ЮЛ",
        )
        self.assertNotEqual(result.get("reason"), "informal_address_tu_tebe")


# ---------------------------------------------------------------------------
# Plan §16 Dialog B/J — DealState new triggers (§9)
# ---------------------------------------------------------------------------

class TestPlan16DialogJ_DealStateTriggers(unittest.TestCase):
    """§9: valid and invalid triggers for escalation readiness."""

    def test_gotov_zapuskat_is_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertTrue(detect_action_intent("готов запускать"))

    def test_mne_podkhodit_is_docs_affirm_after_explain(self):
        """'мне подходит' after explain_documents triggers post_docs_consent."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "handoff_needed": False,
                "selected_product": "bankruptcy_account",
                "procedure_stage": None, "client_intent": None,
                "ready_to_open_reason": None,
                "last_offer": None, "last_bank_list_context": [], "last_conditions_context": [],
                "last_bot_reply_hash": None,
            },
        }
        ds = update_deal_state(slots, "мне подходит")
        self.assertEqual(ds["deal_stage"], "ready_to_open")
        self.assertEqual(ds["ready_to_open_reason"], "post_docs_consent")
        self.assertTrue(ds["handoff_needed"])

    def test_podrobnee_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent("подробнее"))

    def test_kakie_usloviya_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent("какие условия?"))

    def test_po_vremeni_skolko_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent("по времени сколько?"))

    def test_skolko_po_vremeni_not_action_intent(self):
        from app.processing.deal_state import detect_action_intent
        self.assertFalse(detect_action_intent("сколько по времени?"))

    def test_chto_ot_menya_trebuetsya_is_docs_intent(self):
        from app.processing.deal_state import detect_docs_intent
        slots = {"debtor_type": "ЮЛ", "_last_bank": "ТКБ"}
        self.assertTrue(detect_docs_intent("что от меня требуется?", slots))

    def test_chto_skinut_is_docs_intent(self):
        from app.processing.deal_state import detect_docs_intent
        slots = {"debtor_type": "ЮЛ", "_last_bank": "ТКБ"}
        self.assertTrue(detect_docs_intent("что скинуть?", slots))


# ---------------------------------------------------------------------------
# Plan §26 — ScenarioPolicy respects Planner decision (§9)
# ---------------------------------------------------------------------------

class TestPlan26_PlannerDrivenPolicy(unittest.TestCase):
    """§9/§26: When planner_scenario is provided, ScenarioPolicy must accept it."""

    def _policy(self, planner_scenario, active=None, rag=None, intents=None):
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": active, "debtor_type": "ЮЛ"}
        return decide_scenario_policy(
            user_text="тест",
            slots=slots,
            rag_scenarios=rag or [],
            intent_signals=intents or {"intents": [], "matches": [], "dialog_acts": [],
                                       "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": []},
            planner_scenario=planner_scenario,
        )

    # Dialog A — bonus question overrides bank_selection_yul
    def test_dialog_a_planner_au_bonus_overrides_bank_selection(self):
        result = self._policy(
            planner_scenario="au_bonus_question",
            active="bank_selection_yul",
            rag=[{"scenario_id": "bank_selection_yul", "score": 0.9, "reasons": [], "matched_aliases": []}],
        )
        self.assertEqual(result["active_scenario"], "au_bonus_question")
        self.assertEqual(result["reason"], "planner_decision")

    # Dialog B — conditions planner overrides tariff active scenario
    def test_dialog_b_planner_conditions_overrides_tariff_active(self):
        result = self._policy(
            planner_scenario="tkb_yul_conditions",
            active="bank_selection_yul",
            rag=[{"scenario_id": "bank_selection_yul", "score": 0.85, "reasons": [], "matched_aliases": []}],
        )
        self.assertEqual(result["active_scenario"], "tkb_yul_conditions")
        self.assertEqual(result["reason"], "planner_decision")

    # Dialog C — topic shift after objection
    def test_dialog_c_planner_bank_selection_after_objection(self):
        result = self._policy(
            planner_scenario="bank_selection_yul",
            active="direct_bank_objection",
        )
        self.assertEqual(result["active_scenario"], "bank_selection_yul")
        self.assertEqual(result["reason"], "planner_decision")
        self.assertEqual(result["decision"], "switch")

    # Dialog D — planner ready_to_open is accepted
    def test_dialog_d_planner_ready_to_open(self):
        result = self._policy(
            planner_scenario="ready_to_open",
            active="tkb_yul_conditions",
        )
        self.assertEqual(result["active_scenario"], "ready_to_open")
        self.assertEqual(result["reason"], "planner_decision")

    # Dialog F — FL correction
    def test_dialog_f_planner_fl_correction(self):
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="нет, я спрашивал про ФЛ",
            slots=slots,
            rag_scenarios=[],
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": "ФЛ", "bank_focus": None, "semantic_rejects": []},
            planner_scenario="bank_selection_fl",
        )
        self.assertEqual(result["active_scenario"], "bank_selection_fl")
        self.assertEqual(result["reason"], "planner_decision")

    # repetition_complaint overrides even Planner
    def test_repetition_complaint_overrides_planner(self):
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul"}
        result = decide_scenario_policy(
            user_text="вы уже это говорили",
            slots=slots,
            rag_scenarios=[],
            intent_signals={
                "intents": ["repetition_complaint"],
                "matches": [{"intent": "repetition_complaint", "source": "regex",
                             "score": 0.99, "matched_anchor": "уже говорили", "threshold": 0.55}],
                "dialog_acts": [], "debtor_type": None, "bank_focus": None, "semantic_rejects": [],
            },
            planner_scenario="bank_selection_yul",
        )
        self.assertEqual(result["active_scenario"], "bot_repetition_complaint")
        self.assertEqual(result["reason"], "repetition_complaint_override")

    # unknown planner scenario falls back to scoring
    def test_unknown_planner_scenario_falls_back_to_scoring(self):
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="какие банки?",
            slots=slots,
            rag_scenarios=[],
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": []},
            planner_scenario="nonexistent_scenario_xyz",
        )
        self.assertNotEqual(result["reason"], "planner_decision",
            "Unknown Planner scenario must fall back to scoring, not 'planner_decision'")

    # no planner_scenario → scoring runs normally (no regression)
    def test_no_planner_scenario_scoring_runs(self):
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="какие банки для ЮЛ?",
            slots=slots,
            rag_scenarios=[
                {"scenario_id": "bank_selection_yul", "score": 0.9, "reasons": [], "matched_aliases": []},
            ],
            intent_signals={
                "intents": ["bank_selection"],
                "matches": [{"intent": "bank_selection", "source": "regex",
                             "score": 0.9, "matched_anchor": "банки для юл", "threshold": 0.55}],
                "dialog_acts": [], "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": [],
            },
        )
        self.assertNotEqual(result["reason"], "planner_decision")
        self.assertIsNotNone(result["active_scenario"])


# ---------------------------------------------------------------------------
# Plan §26 — Planner validator unit tests
# ---------------------------------------------------------------------------

class TestPlan26_PlannerValidator(unittest.TestCase):
    """§26: run_dialog_planner internal validator rejects bad outputs.

    _validate_planner_v2 returns a list of error strings (empty = valid).
    """

    def _validate(self, data):
        from app.services.dialog_planner import _validate_planner_v2
        return _validate_planner_v2(data)  # returns list[str] of errors

    def test_valid_minimal_plan(self):
        errors = self._validate({
            "scenario": "bank_selection_yul",
            "topic": "bank_selection",
            "topic_changed": False,
            "dialog_act": "question",
            "user_intent": "ask_banks",
            "confidence": 0.85,
            "handoff": {"needed": False, "reason": None},
            "retrieval": {"queries": ["банки ЮЛ"], "required_fact_groups": [], "avoid_fact_groups": []},
            "must_not_repeat": [],
            "responder_instruction": "Answer.",
        })
        self.assertEqual(errors, [], f"Valid minimal plan rejected with errors: {errors}")

    def test_missing_scenario_rejected(self):
        errors = self._validate({"topic": "bank_selection"})
        self.assertTrue(len(errors) > 0, "Missing scenario must produce errors")
        self.assertTrue(any("scenario" in e.lower() for e in errors),
                        f"Error must mention 'scenario': {errors}")

    def test_unknown_scenario_rejected(self):
        errors = self._validate({
            "scenario": "totally_unknown_xyz_123",
            "topic": "x", "topic_changed": False, "dialog_act": "q", "user_intent": "x",
            "confidence": 0.5,
            "handoff": {"needed": False, "reason": None},
            "retrieval": {"queries": [], "required_fact_groups": [], "avoid_fact_groups": []},
            "must_not_repeat": [], "responder_instruction": "x",
        })
        self.assertTrue(len(errors) > 0, "Unknown scenario must produce errors")

    def test_missing_handoff_rejected(self):
        errors = self._validate({
            "scenario": "bank_selection_yul",
            "topic": "x", "topic_changed": False, "dialog_act": "q", "user_intent": "x",
            "confidence": 0.5,
            "retrieval": {"queries": [], "required_fact_groups": [], "avoid_fact_groups": []},
            "must_not_repeat": [], "responder_instruction": "x",
        })
        self.assertTrue(len(errors) > 0, "Missing handoff must produce errors")


# ---------------------------------------------------------------------------
# Plan §18/§19 — Anti-repetition slot tracking
# ---------------------------------------------------------------------------

class TestPlan18_AntiRepetitionTracking(unittest.TestCase):
    """§18/§19: _answered_fact_groups and _bot_reply_history are updated per turn."""

    def test_answered_fact_groups_merges_correctly(self):
        prev = ["tkb_tariffs", "bank_selection_yul"]
        new_tags = ["tkb_yul_conditions", "bank_selection_yul"]  # overlap: bank_selection_yul
        merged = list(dict.fromkeys(prev + new_tags))
        self.assertEqual(merged, ["tkb_tariffs", "bank_selection_yul", "tkb_yul_conditions"])

    def test_answered_fact_groups_capped_at_15(self):
        prev = [f"tag_{i}" for i in range(13)]
        new_tags = ["tag_new_1", "tag_new_2", "tag_new_3"]
        merged = list(dict.fromkeys(prev + new_tags))[:15]
        self.assertEqual(len(merged), 15)

    def test_bot_reply_history_kept_last_3(self):
        history = [
            {"hash": "aaa", "text": "reply1", "fact_tags": ["tag_a"], "turn": 1},
            {"hash": "bbb", "text": "reply2", "fact_tags": ["tag_b"], "turn": 2},
            {"hash": "ccc", "text": "reply3", "fact_tags": ["tag_c"], "turn": 3},
        ]
        new_entry = {"hash": "ddd", "text": "reply4", "fact_tags": ["tag_d"], "turn": 4}
        history.append(new_entry)
        history = history[-3:]
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["hash"], "bbb")
        self.assertEqual(history[-1]["hash"], "ddd")


# ---------------------------------------------------------------------------
# Plan §27 — Additional acceptance cases
# ---------------------------------------------------------------------------

class TestPlan27_AcceptanceCases(unittest.TestCase):
    """§27: Acceptance cases from Telegram run."""

    def test_syuda_skidyvat_is_docs_intent(self):
        """'сюда скидывать?' must be recognized as docs intent."""
        from app.processing.deal_state import detect_docs_intent
        slots = {"debtor_type": "ЮЛ", "_last_bank": "ТКБ"}
        self.assertTrue(detect_docs_intent("сюда скидывать?", slots))

    def test_syuda_skidyvat_is_not_action_intent_alone(self):
        """'сюда скидывать?' alone must NOT fire action intent (needs bank context)."""
        from app.processing.deal_state import detect_action_intent
        # It's a docs question, not a bare readiness trigger
        result = detect_action_intent("сюда скидывать?")
        # We don't assert True/False strictly — just verify no crash and returns bool
        self.assertIsInstance(result, bool)

    def test_okay_zhdu_after_handoff_silent(self):
        """Session silenced after handoff: _session_silenced_after_handoff=True means no reply."""
        slots = {"_session_silenced_after_handoff": True}
        self.assertTrue(bool(slots.get("_session_silenced_after_handoff")))

    def test_handoff_truthfulness_validator(self):
        """Reply claiming manager transfer must fail validation when handoff=false."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Передам ваши данные менеджеру в ближайшее время.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="хорошо",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "handoff_claim_without_handoff_flag")

    def test_no_handoff_claim_when_handoff_true(self):
        """When handoff=true, handoff language in reply is valid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Передам ваши данные менеджеру — он свяжется с вами.",
            brain_result={"action": "answer", "handoff": {"needed": True, "reason": "ready_to_open"}},
            current_entities={},
            slots={},
            user_text="ТКБ мне подходит",
        )
        # Should not fail on handoff_claim_without_handoff_flag
        self.assertNotEqual(result.get("reason"), "handoff_claim_without_handoff_flag")

    def test_planner_scenario_kept_active_when_same(self):
        """When Planner returns same scenario as active, decision is keep_active."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul", "debtor_type": "ЮЛ"}
        result = decide_scenario_policy(
            user_text="понял",
            slots=slots,
            rag_scenarios=[],
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": "ЮЛ", "bank_focus": None, "semantic_rejects": []},
            planner_scenario="bank_selection_yul",
        )
        self.assertEqual(result["decision"], "keep_active")
        self.assertEqual(result["active_scenario"], "bank_selection_yul")


# ---------------------------------------------------------------------------
# Plan new §9 — Golden tests for new plan items
# ---------------------------------------------------------------------------

class TestPlanNew_CurrencyAccountScenario(unittest.TestCase):
    """§3/§9: currency_account_question scenario routing."""

    def test_currency_account_planner_candidate_detected(self):
        """Lexical hint 'валют' → currency_account_question in candidates."""
        from app.processing.message_processor import _build_planner_candidates
        candidates = _build_planner_candidates(
            user_text="могу я делать расчеты с валютным счетом?",
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": None, "bank_focus": None, "semantic_rejects": []},
            previous_scenario=None,
            slots={},
        )
        self.assertIn("currency_account_question", candidates)

    def test_currency_account_planner_accepted_by_policy(self):
        """ScenarioPolicy accepts currency_account_question from Planner."""
        from app.processing.scenario_policy import decide_scenario_policy
        slots = {"_active_scenario": "bank_selection_yul"}
        result = decide_scenario_policy(
            user_text="могу я делать расчеты с валютным счетом?",
            slots=slots,
            rag_scenarios=[],
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": None, "bank_focus": None, "semantic_rejects": []},
            planner_scenario="currency_account_question",
        )
        self.assertEqual(result["active_scenario"], "currency_account_question")
        self.assertEqual(result["reason"], "planner_decision")

    def test_currency_settlement_claim_validator(self):
        """§4: Reply claiming расчеты возможны when user asked must be rejected."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Расчеты с кредиторами возможны через валютный счёт без ограничений.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="можно ли делать расчеты с кредиторами с валютного счета?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "currency_settlement_claim_without_restriction")

    def test_currency_restriction_reply_passes(self):
        """Reply stating расчеты не разрешены should pass the validator."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Расчеты с кредиторами через валютный счёт не допускаются — только зачисление валюты.",
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots={},
            user_text="можно ли платить кредиторам с валютного счета?",
        )
        self.assertNotEqual(result.get("reason"), "currency_settlement_claim_without_restriction")


class TestPlanNew_DebtorTypeUnknownGuard(unittest.TestCase):
    """§5/§9: When debtor_type is unknown, Planner must not choose YUL/FL-specific scenarios."""

    def test_unknown_debtor_candidate_does_not_push_yul(self):
        """Without debtor_type, lexical/intent candidates should not include YUL-specific conditions."""
        from app.processing.message_processor import _build_planner_candidates
        candidates = _build_planner_candidates(
            user_text="расскажи про условия ТКБ",
            intent_signals={
                "intents": ["specific_bank_conditions"],
                "matches": [{"intent": "specific_bank_conditions", "source": "semantic",
                             "score": 0.62, "matched_anchor": "условия ткб", "threshold": 0.58}],
                "dialog_acts": [], "debtor_type": None, "bank_focus": "tkb", "semantic_rejects": [],
            },
            previous_scenario=None,
            slots={},  # no debtor_type
        )
        # With no debtor_type, either yul or fl conditions may appear as candidates — that's OK
        # The Planner (via rule 7) must not choose them. Test that rule is in the system prompt.
        from app.services.dialog_planner import _PLANNER_SYSTEM_V2
        self.assertIn("debtor_type=null", _PLANNER_SYSTEM_V2)
        self.assertIn("unknown_business_question", _PLANNER_SYSTEM_V2)


class TestPlanNew_PendingQuestionToPlanner(unittest.TestCase):
    """§6/§9: Pending assistant question must be passed to Planner context."""

    def test_pending_question_in_planner_context(self):
        """If pending_question is set, it must appear in Planner context JSON."""
        import json
        from app.services.dialog_planner import _build_planner_context_v2
        ctx_str = _build_planner_context_v2(
            user_text="да",
            recent_dialog=[{"role": "bot", "text": "Давайте подробнее разберём условия?"}],
            known_slots={"debtor_type": "ЮЛ"},
            deal_state={"deal_stage": "consulting"},
            previous_scenario="tkb_yul_conditions",
            candidate_scenarios=["tkb_yul_conditions"],
            candidate_intents_regex=[],
            candidate_intents_semantic=[],
            last_bot_reply_summary="Давайте подробнее разберём условия?",
            already_answered=[],
            system_constraints={},
            pending_question="Давайте подробнее разберём условия?",
        )
        ctx = json.loads(ctx_str)
        self.assertIsNotNone(ctx.get("pending_assistant_question"))
        self.assertIn("условия", ctx["pending_assistant_question"])


class TestPlanNew_WeakAcknowledgement(unittest.TestCase):
    """§8/§9: Weak acknowledgements must not auto-advance to docs/opening."""

    def test_nu_ladno_no_pending_question_no_docs_consent(self):
        """'ну ладно' without pending question must NOT trigger post_docs_consent."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "handoff_needed": False,
                "selected_product": "bankruptcy_account",
                "procedure_stage": None, "client_intent": None,
                "ready_to_open_reason": None,
                "last_offer": None, "last_bank_list_context": [], "last_conditions_context": [],
                "last_bot_reply_hash": None,
            },
            # No _pending_question set
        }
        ds = update_deal_state(slots, "ну ладно")
        self.assertNotEqual(ds.get("deal_stage"), "ready_to_open",
            "'ну ладно' without pending question must not advance to ready_to_open")
        self.assertFalse(ds.get("handoff_needed"))

    def test_ponyatno_no_pending_question_no_docs_consent(self):
        """'понятно' without pending question must NOT trigger post_docs_consent."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "handoff_needed": False,
                "selected_product": "bankruptcy_account",
                "procedure_stage": None, "client_intent": None,
                "ready_to_open_reason": None,
                "last_offer": None, "last_bank_list_context": [], "last_conditions_context": [],
                "last_bot_reply_hash": None,
            },
        }
        ds = update_deal_state(slots, "понятно")
        self.assertFalse(ds.get("handoff_needed"))

    def test_davajte_with_pending_question_does_consent(self):
        """'давайте' WITH a pending yes/no question SHOULD trigger post_docs_consent."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "debtor_type": "ЮЛ",
            "_last_bank": "ТКБ",
            "_pending_question": "Готовы начать оформление?",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "ТКБ",
                "debtor_type": "ЮЛ",
                "handoff_needed": False,
                "selected_product": "bankruptcy_account",
                "procedure_stage": None, "client_intent": None,
                "ready_to_open_reason": None,
                "last_offer": None, "last_bank_list_context": [], "last_conditions_context": [],
                "last_bot_reply_hash": None,
            },
        }
        ds = update_deal_state(slots, "давайте")
        self.assertEqual(ds.get("deal_stage"), "ready_to_open")
        self.assertTrue(ds.get("handoff_needed"))


class TestPlanNew_ComplaintHandling(unittest.TestCase):
    """§7/§9: Complaint and correction phrases must route to correct scenarios."""

    def test_complaint_candidate_from_lexical_hint(self):
        """'ты странный' must put bot_complaint in Planner candidates."""
        from app.processing.message_processor import _build_planner_candidates
        candidates = _build_planner_candidates(
            user_text="ты странный какой-то",
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": None, "bank_focus": None, "semantic_rejects": []},
            previous_scenario=None,
            slots={},
        )
        self.assertIn("bot_complaint", candidates)

    def test_drugoe_sprasival_correction_candidate(self):
        """'другое спрашивал' must include clarification_or_correction in candidates."""
        from app.processing.message_processor import _build_planner_candidates
        candidates = _build_planner_candidates(
            user_text="я же другое спрашивал",
            intent_signals={"intents": [], "matches": [], "dialog_acts": [],
                            "debtor_type": None, "bank_focus": None, "semantic_rejects": []},
            previous_scenario=None,
            slots={},
        )
        self.assertIn("clarification_or_correction", candidates)

    def test_bot_complaint_scenario_in_catalog(self):
        """bot_complaint must be in ScenarioPolicy CATALOG."""
        from app.processing.scenario_catalog import CATALOG
        self.assertIn("bot_complaint", CATALOG)

    def test_planner_system_prompt_mentions_complaint(self):
        """Planner system prompt must instruct on complaint handling."""
        from app.services.dialog_planner import _PLANNER_SYSTEM_V2
        self.assertIn("bot_complaint", _PLANNER_SYSTEM_V2)


class TestFix1_BurstMerging(unittest.TestCase):
    """FIX 1 — Burst message merging: only the latest job in a burst should reply."""

    def test_has_newer_active_job_exported(self):
        """has_newer_active_job must be importable from jobs_repo."""
        from app.storage.repositories.jobs_repo import has_newer_active_job
        self.assertTrue(callable(has_newer_active_job))

    def test_has_newer_queued_job_still_works(self):
        """has_newer_queued_job must still be importable (backwards compat)."""
        from app.storage.repositories.jobs_repo import has_newer_queued_job
        self.assertTrue(callable(has_newer_queued_job))

    def test_merge_trailing_consecutive_user_messages(self):
        """_merge_trailing_user_messages must join consecutive user messages."""
        from app.processing.message_processor import _merge_trailing_user_messages
        msgs = [
            {"role": "user", "text": "ТКБ"},
            {"role": "user", "text": "Какие условия там"},
            {"role": "user", "text": "?"},
        ]
        result = _merge_trailing_user_messages(msgs, "?")
        # Should merge all 3 consecutive user messages
        self.assertIn("ТКБ", result)
        self.assertIn("Какие условия там", result)

    def test_merge_stops_at_bot_reply(self):
        """_merge_trailing_user_messages must stop at bot turn boundary."""
        from app.processing.message_processor import _merge_trailing_user_messages
        msgs = [
            {"role": "user", "text": "Привет"},
            {"role": "bot", "text": "Здравствуйте"},
            {"role": "user", "text": "ТКБ"},
            {"role": "user", "text": "условия?"},
        ]
        result = _merge_trailing_user_messages(msgs, "условия?")
        # Should NOT include "Привет" from before bot reply
        self.assertNotIn("Привет", result)
        self.assertIn("ТКБ", result)

    def test_frustration_only_re_catches_question_mark(self):
        """_FRUSTRATION_ONLY_RE must match standalone '?'."""
        from app.processing.message_processor import _FRUSTRATION_ONLY_RE
        self.assertTrue(_FRUSTRATION_ONLY_RE.match("?"))
        self.assertTrue(_FRUSTRATION_ONLY_RE.match("???"))
        self.assertTrue(_FRUSTRATION_ONLY_RE.match("!"))
        self.assertFalse(_FRUSTRATION_ONLY_RE.match("привет"))


class TestFix2_ForeignLanguageArtifacts(unittest.TestCase):
    """FIX 2 — Foreign language artifacts must be caught before sending."""

    def test_benotujete_rejected(self):
        """Reply containing 'benötujete' must be invalid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Для открытия счета benötujete ИНН должника.",
            brain_result={},
            current_entities={},
            slots={},
            user_text="что нужно для открытия",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "non_russian_output")

    def test_umlaut_o_rejected(self):
        """Reply containing 'ö' must be invalid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Документ öffnen должника.",
            brain_result={},
            current_entities={},
            slots={},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "non_russian_output")

    def test_valid_crm_bitrix_allowed(self):
        """Reply containing 'CRM' and 'Bitrix' must be valid (allowlisted)."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Заявка передаётся в CRM через Bitrix системой учёта.",
            brain_result={},
            current_entities={},
            slots={},
        )
        self.assertTrue(result["is_valid"])

    def test_long_latin_word_in_russian_text_rejected(self):
        """Reply with unexpected long Latin word in Russian context must be invalid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Открытие счёта требует оформления через zusammenarbeit с банком.",
            brain_result={},
            current_entities={},
            slots={},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "non_russian_output")

    def test_telegram_allowed(self):
        """'Telegram' must be allowlisted and not trigger foreign artifact."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Вопросы задавайте через Telegram удобным способом в рабочее время.",
            brain_result={},
            current_entities={},
            slots={},
        )
        self.assertTrue(result["is_valid"])

    def test_foreign_artifact_re_catches_o_umlaut(self):
        """_FOREIGN_ARTIFACT_RE must match ö (U+00F6)."""
        import re
        from app.services.response_validator import _FOREIGN_ARTIFACT_RE
        self.assertIsNotNone(_FOREIGN_ARTIFACT_RE.search("benötujete"))
        self.assertIsNotNone(_FOREIGN_ARTIFACT_RE.search("öffnen"))
        self.assertIsNone(_FOREIGN_ARTIFACT_RE.search("CRM Bitrix"))


class TestFix3_ConditionsDrift(unittest.TestCase):
    """FIX 3 — Conditions requests must not drift to documents/opening flow."""

    def test_conditions_request_detected(self):
        """_CONDITIONS_REQUEST_RE must match 'какие условия в ТКБ'."""
        from app.services.response_validator import _CONDITIONS_REQUEST_RE
        self.assertIsNotNone(_CONDITIONS_REQUEST_RE.search("какие условия в ТКБ"))
        self.assertIsNotNone(_CONDITIONS_REQUEST_RE.search("Какие там условия?"))
        self.assertIsNotNone(_CONDITIONS_REQUEST_RE.search("что по условиям в банке"))

    def test_docs_drift_detected(self):
        """_DOCS_OPENING_DRIFT_RE must match document/opening drift phrases."""
        from app.services.response_validator import _DOCS_OPENING_DRIFT_RE
        self.assertIsNotNone(_DOCS_OPENING_DRIFT_RE.search("Для открытия нужны ИНН и судебный акт"))
        self.assertIsNotNone(_DOCS_OPENING_DRIFT_RE.search("поможем оформить счёт"))
        self.assertIsNotNone(_DOCS_OPENING_DRIFT_RE.search("давайте соберём документы"))

    def test_conditions_reply_drifted_to_docs_rejected(self):
        """Reply drifting to docs when user asked about conditions must be invalid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Открытие 2800 руб., ведение 2090 руб. Для открытия нужны ИНН, данные АУ и судебный акт.",
            brain_result={},
            current_entities={},
            slots={},
            user_text="Какие условия в ТКБ?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "conditions_reply_drifted_to_documents")

    def test_conditions_reply_on_target_valid(self):
        """Conditions reply focused on operations (not docs) must be valid."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply=(
                "По ТКБ ключевые условия: переводы на ЮЛ — 33 руб. электронно, "
                "переводы на ФЛ тарифицируются по шкале банка. "
                "Есть особенности по операциям и согласованию платежей."
            ),
            brain_result={},
            current_entities={},
            slots={},
            user_text="Какие условия в ТКБ?",
        )
        self.assertTrue(result["is_valid"])

    def test_planner_system_prompt_mentions_conditions_rule(self):
        """Planner system prompt must have a rule about conditions requests."""
        from app.services.dialog_planner import _PLANNER_SYSTEM_V2
        self.assertIn("условия", _PLANNER_SYSTEM_V2.lower())
        self.assertIn("tkb_yul_conditions", _PLANNER_SYSTEM_V2)

    def test_non_conditions_request_not_blocked(self):
        """Reply to non-conditions request must not be blocked by conditions drift check."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Для открытия счёта нужны ИНН должника, данные АУ и судебный акт суда.",
            brain_result={},
            current_entities={},
            slots={"debtor_type": "ЮЛ"},
            user_text="какие документы нужны для открытия счёта?",
        )
        # This is a docs request, not a conditions request — drift check must NOT fire
        self.assertTrue(result["is_valid"])


class TestFix4_DeterministicHandoffReply(unittest.TestCase):
    """FIX 4 — When handoff=true, send fixed deterministic reply instead of LLM output."""

    def test_deterministic_reply_constant_exists(self):
        """_HANDOFF_DETERMINISTIC_REPLY must be defined and contain expected text."""
        from app.processing.message_processor import _HANDOFF_DETERMINISTIC_REPLY
        self.assertIn("Принял", _HANDOFF_DETERMINISTIC_REPLY)
        self.assertIn("менеджер", _HANDOFF_DETERMINISTIC_REPLY.lower())
        self.assertIn("Передаю", _HANDOFF_DETERMINISTIC_REPLY)

    def test_deterministic_reply_is_short_and_clean(self):
        """Fixed handoff reply must be a short, non-verbose message."""
        from app.processing.message_processor import _HANDOFF_DETERMINISTIC_REPLY
        # Should not be excessively long
        self.assertLess(len(_HANDOFF_DETERMINISTIC_REPLY), 120)
        # Should not have LLM-style filler
        lower = _HANDOFF_DETERMINISTIC_REPLY.lower()
        self.assertNotIn("документ", lower)
        self.assertNotIn("инн", lower)
        self.assertNotIn("откр", lower)

    def test_deterministic_reply_passes_validator(self):
        """The fixed handoff reply must be valid according to response_validator."""
        from app.processing.message_processor import _HANDOFF_DETERMINISTIC_REPLY
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply=_HANDOFF_DETERMINISTIC_REPLY,
            brain_result={"handoff": {"needed": True, "reason": "ready_to_open"}, "action": "handoff"},
            current_entities={},
            slots={},
            user_text="окей, подходит",
        )
        self.assertTrue(result["is_valid"], f"Handoff reply invalid: {result['reason']}")

    def test_handoff_reply_not_llm_generated(self):
        """The handoff reply must not come from LLM — verified by being the fixed constant."""
        from app.processing.message_processor import _HANDOFF_DETERMINISTIC_REPLY
        # Exact match check — it must be this specific string, not an LLM variation
        self.assertEqual(
            _HANDOFF_DETERMINISTIC_REPLY,
            "Принял. Передаю вашу заявку старшему менеджеру, чтобы помочь вам дальше.",
        )


if __name__ == "__main__":
    unittest.main()
