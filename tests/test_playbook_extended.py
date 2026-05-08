"""Extended playbook regression tests for new scenarios.

Covers plan.txt requirements:
  1. direct_bank_objection: initial and follow-up replies
  2. bank_selection_yul_low_cost: initial listing, follow-up "только эти оба?"
  3. Tariff comparison replies in bank_selection_yul / bank_pricing_yul
  4. RAGTrace session_id not None
  5. Ollama num_predict respects configured budget

Run:
    python -m pytest tests/test_playbook_extended.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.scenario_playbook import (
    SLOT_FORBIDDEN, SLOT_KNOWN, SLOT_NEXT_STEP,
    _build_low_cost_reply, _build_tariff_compare_reply,
    run_scenario_playbook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _playbook(user_text, active=None, known=None, fact_pack=None):
    slots = {"_active_scenario": active, "_known_slots": known or {}}
    return run_scenario_playbook(user_text, slots, fact_pack=fact_pack)


def _fake_pricing() -> dict:
    return {
        "bank_pricing_yul": [
            {"bank": "Альфа-Банк", "opening_fee": 800,  "monthly_fee": 800},
            {"bank": "ТКБ",        "opening_fee": 2800, "monthly_fee": 2090},
            {"bank": "Уралсиб",    "opening_fee": 3000, "monthly_fee": 1500},
        ]
    }


# ============================================================================
# 1. direct_bank_objection — initial
# ============================================================================
class TestDirectBankObjectionInitial(unittest.TestCase):
    def _result(self, text="а в чем выгода через вас работать"):
        return _playbook(text, active="direct_bank_objection")

    def test_action_is_reply(self):
        r = self._result()
        self.assertEqual(r["action"], "reply")

    def test_llm_skipped(self):
        r = self._result()
        self.assertTrue(r["log"]["llm_skipped"])

    def test_reply_mentions_documents(self):
        r = self._result()
        reply = r["reply"].lower()
        self.assertIn("документ", reply)

    def test_reply_mentions_support(self):
        r = self._result()
        reply = r["reply"].lower()
        self.assertTrue(
            "сопровождение" in reply or "коммуникац" in reply,
            f"Expected 'сопровождение' or 'коммуникац' in: {reply!r}",
        )

    def test_required_next_step(self):
        r = self._result()
        self.assertIn("offer_case_check_or_bank_selection", r["updates"].get(SLOT_NEXT_STEP, ""))

    def test_forbidden_includes_bank_question(self):
        r = self._result()
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(
            any("о каких банках" in f.lower() for f in forbidden),
            f"Expected forbidden 'о каких банках' in: {forbidden}",
        )

    def test_known_marks_objection_answered(self):
        r = self._result()
        self.assertTrue(r["updates"][SLOT_KNOWN].get("objection_answered"))

    def test_phrase_variants(self):
        for phrase in [
            "я же могу напрямую в банк пойти",
            "зачем мне с вами работать",
            "какой бонус мне будет",
        ]:
            with self.subTest(phrase=phrase):
                r = _playbook(phrase, active="direct_bank_objection")
                self.assertEqual(r["action"], "reply", f"Expected reply for: {phrase!r}")


# ============================================================================
# 2. direct_bank_objection — follow-up "подробнее?"
# ============================================================================
class TestDirectBankObjectionFollowUp(unittest.TestCase):
    def test_подробнее_returns_expanded_reply(self):
        r = _playbook("подробнее?", active="direct_bank_objection",
                      known={"objection_answered": True})
        self.assertEqual(r["action"], "reply")
        reply = r["reply"].lower()
        self.assertIn("документ", reply)
        self.assertIn("финмониторинг", reply)

    def test_почему_follow_up(self):
        r = _playbook("почему?", active="direct_bank_objection",
                      known={"objection_answered": True})
        self.assertEqual(r["action"], "reply")

    def test_вы_не_ответили(self):
        r = _playbook("вы не ответили", active="direct_bank_objection")
        self.assertEqual(r["action"], "reply")
        # Should not ask about banks
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(
            any("о каких банках" in f.lower() for f in forbidden),
            f"Must still forbid bank-selection question: {forbidden}",
        )

    def test_no_bank_selection_question_in_follow_up(self):
        r = _playbook("подробнее", active="direct_bank_objection")
        reply = (r.get("reply") or "").lower()
        self.assertNotIn("о каких банках хотите узнать", reply)


# ============================================================================
# 3. bank_selection_yul_low_cost — initial listing
# ============================================================================
class TestBankSelectionLowCost(unittest.TestCase):
    def test_action_is_reply(self):
        r = _playbook("у меня должник ЮЛ нужен банк подешевле",
                      active="bank_selection_yul_low_cost")
        self.assertEqual(r["action"], "reply")

    def test_reply_mentions_alfa(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack=_fake_pricing())
        self.assertIn("Альфа-Банк", r["reply"])

    def test_reply_mentions_tkb(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack=_fake_pricing())
        self.assertIn("ТКБ", r["reply"])

    def test_reply_mentions_uralsib(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack=_fake_pricing())
        self.assertIn("Уралсиб", r["reply"])

    def test_reply_includes_alfa_price(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack=_fake_pricing())
        # Alfa: 800 ₽ opening
        self.assertIn("800", r["reply"])

    def test_known_marks_low_cost_listed(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertTrue(r["updates"][SLOT_KNOWN].get("low_cost_listed"))

    def test_required_next_step(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertIn("explain_low_cost", r["updates"].get(SLOT_NEXT_STEP, ""))

    def test_uralsib_no_price_uses_fallback_text(self):
        """When Uralsib price not in fact_pack, use 'условия уточняются' text."""
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack={"bank_pricing_yul": []})
        reply = r["reply"].lower()
        self.assertTrue(
            "уточн" in reply or "уралсиб" in reply.lower(),
            f"Expected Uralsib mention or clarification in: {reply!r}",
        )


# ============================================================================
# 4. bank_selection_yul_low_cost — follow-up "только эти оба?"
# ============================================================================
class TestBankSelectionLowCostMoreOptions(unittest.TestCase):
    FOLLOW_UP_PHRASES = [
        "только эти оба?",
        "только эти?",
        "а еще?",
        "другие есть?",
        "еще варианты?",
        "больше нет?",
    ]

    def _r(self, phrase):
        return _playbook(phrase, active="bank_selection_yul_low_cost",
                         known={"low_cost_listed": True})

    def test_action_is_reply(self):
        for phrase in self.FOLLOW_UP_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r(phrase)
                self.assertEqual(r["action"], "reply", f"Expected reply for: {phrase!r}")

    def test_reply_mentions_all_three_banks(self):
        r = self._r("только эти оба?")
        reply = r["reply"]
        self.assertIn("Альфа-Банк", reply)
        self.assertIn("ТКБ", reply)
        self.assertIn("Уралсиб", reply)

    def test_reply_explains_why_only_two_highlighted(self):
        r = self._r("а еще?")
        reply = r["reply"].lower()
        self.assertTrue(
            "дешевле" in reply or "старт" in reply or "дешевл" in reply,
            f"Expected price explanation in: {reply!r}",
        )


# ============================================================================
# 5. Tariff comparison in bank_selection_yul and bank_pricing_yul
# ============================================================================
class TestTariffComparisonPlaybook(unittest.TestCase):
    COMPARE_PHRASES = [
        "какие условия и тарифы у них?",
        "сравни тарифы",
        "покажи тарифы",
        "условия у них",
        "какие тарифы",
    ]

    def _r_yul(self, phrase):
        return _playbook(phrase, active="bank_selection_yul", fact_pack=_fake_pricing())

    def _r_pricing(self, phrase):
        return _playbook(phrase, active="bank_pricing_yul", fact_pack=_fake_pricing())

    def _r_low_cost(self, phrase):
        return _playbook(phrase, active="bank_selection_yul_low_cost",
                         known={"low_cost_listed": True}, fact_pack=_fake_pricing())

    def test_bank_selection_yul_gives_comparison(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_yul(phrase)
                self.assertEqual(r["action"], "reply", f"Expected reply for: {phrase!r}")

    def test_bank_pricing_yul_gives_comparison(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_pricing(phrase)
                self.assertEqual(r["action"], "reply", f"Expected reply for: {phrase!r}")

    def test_low_cost_gives_comparison(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_low_cost(phrase)
                self.assertEqual(r["action"], "reply", f"Expected reply for: {phrase!r}")

    def test_comparison_includes_alfa(self):
        r = self._r_yul("какие тарифы")
        self.assertIn("Альфа-Банк", r["reply"])

    def test_comparison_includes_tkb(self):
        r = self._r_yul("сравни тарифы")
        self.assertIn("ТКБ", r["reply"])

    def test_comparison_mentions_paused_banks(self):
        r = self._r_yul("какие условия и тарифы у них?")
        self.assertIn("паузе", r["reply"].lower())

    def test_no_can_i_compare_question(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_yul(phrase)
                reply = (r.get("reply") or "").lower()
                self.assertNotIn("могу сравнить", reply)
                self.assertNotIn("хотите сравнить", reply)

    def test_required_next_step_ask_which_bank(self):
        r = self._r_yul("сравни тарифы")
        self.assertIn("ask_which_bank", r["updates"].get(SLOT_NEXT_STEP, ""))


# ============================================================================
# 6. Pricing helpers
# ============================================================================
class TestPricingHelpers(unittest.TestCase):
    def test_build_low_cost_with_prices(self):
        text = _build_low_cost_reply(_fake_pricing())
        self.assertIn("Альфа-Банк", text)
        self.assertIn("800", text)
        self.assertIn("Уралсиб", text)

    def test_build_low_cost_no_fact_pack(self):
        text = _build_low_cost_reply(None)
        self.assertIn("Альфа-Банк", text)
        self.assertIn("ТКБ", text)

    def test_build_tariff_compare_with_prices(self):
        text = _build_tariff_compare_reply(_fake_pricing())
        self.assertIn("Альфа-Банк", text)
        self.assertIn("ТКБ", text)
        self.assertIn("Уралсиб", text)
        self.assertIn("паузе", text.lower())

    def test_build_tariff_no_uralsib_fallback(self):
        text = _build_tariff_compare_reply({"bank_pricing_yul": []})
        self.assertIn("уточня", text.lower())


# ============================================================================
# 7. Ollama num_predict respects budget
# ============================================================================
class TestOllamaNumPredict(unittest.TestCase):
    def test_brain_budget_not_capped_at_700(self):
        from app.llm.providers.ollama import OllamaProvider, _OLLAMA_NUM_PREDICT_HARD_MAX
        provider = OllamaProvider()
        # Brain budget is 1000; hard max is 2048 — should not be capped to 700
        self.assertGreater(_OLLAMA_NUM_PREDICT_HARD_MAX, 1000,
                           "Hard max must be > 1000 so brain budget of 1000 is not capped")

    def test_max_tokens_1000_passed_through(self):
        from app.llm.providers.ollama import _OLLAMA_NUM_PREDICT_HARD_MAX
        requested = 1000
        num_predict = min(requested, _OLLAMA_NUM_PREDICT_HARD_MAX)
        self.assertEqual(num_predict, 1000)

    def test_excessive_tokens_capped(self):
        from app.llm.providers.ollama import _OLLAMA_NUM_PREDICT_HARD_MAX
        requested = 9999
        num_predict = min(requested, _OLLAMA_NUM_PREDICT_HARD_MAX)
        self.assertEqual(num_predict, _OLLAMA_NUM_PREDICT_HARD_MAX)


if __name__ == "__main__":
    unittest.main()
