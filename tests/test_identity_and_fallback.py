"""Tests for identity guard, conversational signals, fallback, and validator rules.

Run: python -m pytest tests/test_identity_and_fallback.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# 1. _build_user_message has real XML tags
# ---------------------------------------------------------------------------
class TestBuildUserMessage(unittest.TestCase):
    def test_build_user_message_has_real_tags(self):
        from app.services.conversation_brain import _build_user_message
        msg = _build_user_message({"user_message": "test", "fact_pack": {}})
        self.assertIn("<CONTEXT_DO_NOT_COPY>", msg)
        self.assertIn("</CONTEXT_DO_NOT_COPY>", msg)
        self.assertIn("<OUTPUT_SCHEMA>", msg)
        self.assertIn("</OUTPUT_SCHEMA>", msg)
        self.assertIn("Не копируй CONTEXT_DO_NOT_COPY", msg)


# ---------------------------------------------------------------------------
# 2. Identity guard
# ---------------------------------------------------------------------------
class TestIdentityGuard(unittest.TestCase):
    def _guard(self, text, introduced=False):
        from app.services.identity_guard import check_identity_guard
        return check_identity_guard(text, {"_introduced": introduced})

    def test_greeting_intro_not_introduced(self):
        r = self._guard("добрый день", introduced=False)
        self.assertIsNotNone(r)
        reply = r["reply"]
        self.assertIn("Алексей", reply)
        self.assertIn("В плюсе", reply)
        self.assertIn("счета", reply)
        self.assertTrue(r.get("set_introduced"))

    def test_greeting_passes_through_if_introduced(self):
        # "добрый день" alone after introduced should NOT be intercepted
        # (it's a short greeting — the guard returns None so LLM handles context)
        r = self._guard("добрый день", introduced=True)
        self.assertIsNone(r)

    def test_identity_guard_who_are_you(self):
        r = self._guard("вы кто? почему не представляетесь", introduced=True)
        self.assertIsNotNone(r)
        reply = r["reply"]
        self.assertIn("В плюсе", reply)
        self.assertIn("счет", reply.lower())
        self.assertNotIn("нужны данные для открытия счета", reply)

    def test_confusion_guard(self):
        r = self._guard("о чем ты? ты кто", introduced=True)
        self.assertIsNotNone(r)
        reply = r["reply"]
        self.assertIn("Извините", reply)
        self.assertIn("В плюсе", reply)

    def test_bot_question_guard(self):
        r = self._guard("ты бот?", introduced=True)
        self.assertIsNotNone(r)
        reply = r["reply"]
        self.assertNotIn("я бот", reply.lower())
        self.assertNotIn("я живой человек", reply.lower())
        self.assertIn("В плюсе", reply)

    def test_non_identity_passes_through(self):
        r = self._guard("какие условия у Уралсиба?", introduced=True)
        self.assertIsNone(r)

    def test_card_question_passes_through(self):
        r = self._guard("карта для должника — можно?", introduced=True)
        self.assertIsNone(r)


# ---------------------------------------------------------------------------
# 3. Conversational signals
# ---------------------------------------------------------------------------
class TestConversationalSignals(unittest.TestCase):
    def _sig(self, text):
        from app.services.context_builder import build_conversational_signals
        return build_conversational_signals(text)

    def test_greeting_only(self):
        s = self._sig("добрый день")
        self.assertTrue(s["is_greeting_only"])
        self.assertFalse(s["asks_identity"])

    def test_asks_identity(self):
        s = self._sig("вы кто? почему не представляетесь")
        self.assertTrue(s["asks_identity"])

    def test_complains_confusion(self):
        s = self._sig("о чем ты несёшь")
        self.assertTrue(s["complains_confusion"])

    def test_asks_account_type_difference(self):
        s = self._sig("задатковый и залоговый — в чем разница?")
        self.assertTrue(s["asks_account_type_difference"])

    def test_normal_question_no_signals(self):
        s = self._sig("какие документы нужны для Уралсиба?")
        self.assertFalse(s["is_greeting_only"])
        self.assertFalse(s["asks_identity"])
        self.assertFalse(s["complains_confusion"])
        self.assertFalse(s["asks_account_type_difference"])


# ---------------------------------------------------------------------------
# 4. Validator — new rules
# ---------------------------------------------------------------------------
class TestValidatorNewRules(unittest.TestCase):
    def _val(self, reply, user_text="", answer_contract=None, scenario_facts=None, slots=None):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            brain_result={"action": "answer", "handoff": {"needed": False}},
            current_entities={},
            slots=slots or {},
            user_text=user_text,
            answer_contract=answer_contract,
            scenario_facts=scenario_facts,
        )

    def test_role_confusion_rejected(self):
        v = self._val("о чем я?", user_text="ты кто?")
        self.assertFalse(v["is_valid"])
        self.assertEqual(v["reason"], "role_confusion")

    def test_too_short_for_specific_question(self):
        v = self._val("Хорошо.", user_text="какие условия у банков для открытия счёта?")
        self.assertFalse(v["is_valid"])
        self.assertEqual(v["reason"], "too_short_for_specific_question")

    def test_identity_wrong_answer(self):
        # Reply has no mention of «В плюсе», «счет», «должник», or «банкрот»
        v = self._val(
            "Привет! Рад вам помочь. Уточните ваш вопрос.",
            user_text="вы кто?",
            answer_contract={"topic": "identity"},
        )
        self.assertFalse(v["is_valid"])
        self.assertEqual(v["reason"], "identity_question_wrong_answer")

    def test_identity_correct_answer(self):
        v = self._val(
            "Я Алексей, консультант компании «В плюсе». Помогаю по счетам должников.",
            user_text="вы кто?",
            answer_contract={"topic": "identity"},
        )
        self.assertTrue(v["is_valid"])

    def test_generic_greeting_to_specific_question_rejected(self):
        v = self._val(
            "Здравствуйте! Как я могу вам помочь?",
            user_text="есть разница в открытии счетов?",
        )
        self.assertFalse(v["is_valid"])
        self.assertEqual(v["reason"], "generic_greeting_reply_to_specific_question")

    def test_repair_revalidated_falls_back(self):
        # If repaired reply is "о чем я?" — should be rejected
        v = self._val("о чем я?", user_text="ты кто")
        self.assertFalse(v["is_valid"])

    def test_card_answer_valid(self):
        v = self._val(
            "Карта при реализации имущества оформляется финансовым управляющим.",
            user_text="карта для должника",
        )
        self.assertTrue(v["is_valid"])


# ---------------------------------------------------------------------------
# 5. Context fallback — scenario-aware cases
# ---------------------------------------------------------------------------
class TestContextFallback(unittest.TestCase):
    def _fallback(self, fact_pack=None, current_entities=None, slots=None, scenario_facts=None, signals=None):
        from app.processing.message_processor import _build_context_fallback
        return _build_context_fallback(
            fact_pack or {},
            current_entities or {},
            slots or {},
            scenario_facts=scenario_facts,
            signals=signals,
        )

    def test_card_no_reply_context_fallback(self):
        reply = self._fallback(
            scenario_facts={"debtor_card_realization": {"constraints": ["финансовый управляющий"]}},
        )
        self.assertIn("реализация", reply.lower())
        self.assertIn("финансовый управляющий", reply.lower())
        self.assertNotIn("какой именно вопрос", reply.lower())

    def test_partner_banks_fallback(self):
        reply = self._fallback(
            scenario_facts={"partner_banks": {}},
        )
        self.assertIn("Альфа-Банк", reply)
        self.assertIn("ТКБ", reply)
        self.assertIn("Уралсиб", reply)

    def test_account_type_difference_no_generic(self):
        reply = self._fallback(
            signals={"asks_account_type_difference": True},
        )
        self.assertIn("задатков", reply.lower())
        self.assertNotIn("Как могу помочь", reply)

    def test_uralsib_fallback(self):
        reply = self._fallback(
            current_entities={"mentioned_bank": "Уралсиб"},
            slots={"client_type": "ЮЛ"},
        )
        self.assertIn("3500", reply)
        self.assertIn("1600", reply)


if __name__ == "__main__":
    unittest.main()
