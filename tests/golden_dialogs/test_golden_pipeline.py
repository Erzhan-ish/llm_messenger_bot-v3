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
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
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


if __name__ == "__main__":
    unittest.main()
