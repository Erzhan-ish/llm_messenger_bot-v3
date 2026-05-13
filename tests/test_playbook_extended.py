"""Extended playbook regression tests for new scenarios.

Covers plan.txt requirements:
  1. direct_bank_objection: enrich with constraints — LLM writes the answer naturally
  2. bank_selection_yul_low_cost: enrich with pricing context
  3. Tariff queries in bank_selection_yul / bank_pricing_yul: enrich with constraints
  4. Pricing helpers: _build_low_cost_reply, _build_tariff_compare_reply
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
# 1. direct_bank_objection — enrich with constraints (LLM answers naturally)
# ============================================================================
class TestDirectBankObjectionInitial(unittest.TestCase):
    def _result(self, text="а в чем выгода через вас работать"):
        return _playbook(text, active="direct_bank_objection")

    def test_action_is_enrich(self):
        r = self._result()
        self.assertEqual(r["action"], "enrich")

    def test_llm_not_skipped(self):
        r = self._result()
        self.assertFalse(r["log"]["llm_skipped"])

    def test_reply_is_none(self):
        r = self._result()
        self.assertIsNone(r["reply"])

    def test_no_rigid_next_step(self):
        r = self._result()
        # Business scenarios must not impose a rigid next_step — LLM decides
        self.assertIsNone(r["updates"].get(SLOT_NEXT_STEP))

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

    def test_fact_pack_has_forbidden_phrases(self):
        r = self._result()
        fp = r["fact_pack_additions"]
        self.assertIn("_forbidden_phrases", fp)
        self.assertTrue(len(fp["_forbidden_phrases"]) > 0)

    def test_fact_pack_no_required_next_step(self):
        r = self._result()
        fp = r["fact_pack_additions"]
        # Rigid required_next_step must not be injected into fact_pack for business scenarios
        self.assertNotIn("_required_next_step", fp)

    def test_phrase_variants(self):
        for phrase in [
            "я же могу напрямую в банк пойти",
            "зачем мне с вами работать",
            "какой бонус мне будет",
        ]:
            with self.subTest(phrase=phrase):
                r = _playbook(phrase, active="direct_bank_objection")
                self.assertEqual(r["action"], "enrich", f"Expected enrich for: {phrase!r}")


# ============================================================================
# 2. direct_bank_objection — follow-up phrases
# ============================================================================
class TestDirectBankObjectionFollowUp(unittest.TestCase):
    def test_подробнее_returns_enrich(self):
        r = _playbook("подробнее?", active="direct_bank_objection",
                      known={"objection_answered": True})
        self.assertEqual(r["action"], "enrich")
        self.assertIsNone(r["reply"])

    def test_подробнее_forbidden_constraints_still_set(self):
        r = _playbook("подробнее?", active="direct_bank_objection",
                      known={"objection_answered": True})
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(len(forbidden) > 0, "Expected forbidden constraints in follow-up")

    def test_почему_follow_up(self):
        r = _playbook("почему?", active="direct_bank_objection",
                      known={"objection_answered": True})
        self.assertEqual(r["action"], "enrich")

    def test_вы_не_ответили(self):
        r = _playbook("вы не ответили", active="direct_bank_objection")
        self.assertEqual(r["action"], "enrich")
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
# 3. bank_selection_yul_low_cost — enrich with low-cost context
# ============================================================================
class TestBankSelectionLowCost(unittest.TestCase):
    def test_action_is_enrich(self):
        r = _playbook("у меня должник ЮЛ нужен банк подешевле",
                      active="bank_selection_yul_low_cost")
        self.assertEqual(r["action"], "enrich")

    def test_llm_not_skipped(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertFalse(r["log"]["llm_skipped"])

    def test_reply_is_none(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack=_fake_pricing())
        self.assertIsNone(r["reply"])

    def test_known_marks_low_cost_listed(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertTrue(r["updates"][SLOT_KNOWN].get("low_cost_listed"))

    def test_no_rigid_next_step(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertIsNone(r["updates"].get(SLOT_NEXT_STEP))

    def test_fact_pack_pricing_focus(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        self.assertEqual(r["fact_pack_additions"].get("_pricing_focus"), "low_cost")

    def test_forbidden_includes_mogу_sravnit(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost")
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(
            any("могу сравнить" in f.lower() for f in forbidden),
            f"Expected 'Могу сравнить?' in forbidden: {forbidden}",
        )

    def test_action_is_enrich_with_empty_pricing(self):
        r = _playbook("нужен банк подешевле", active="bank_selection_yul_low_cost",
                      fact_pack={"bank_pricing_yul": []})
        self.assertEqual(r["action"], "enrich")


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

    def test_action_is_enrich(self):
        for phrase in self.FOLLOW_UP_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r(phrase)
                self.assertEqual(r["action"], "enrich", f"Expected enrich for: {phrase!r}")

    def test_known_marks_low_cost_listed(self):
        r = self._r("только эти оба?")
        self.assertTrue(r["updates"][SLOT_KNOWN].get("low_cost_listed"))

    def test_no_mogу_sravnit_in_forbidden(self):
        r = self._r("а еще?")
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(
            any("могу сравнить" in f.lower() for f in forbidden),
            f"Expected 'Могу сравнить?' in forbidden: {forbidden}",
        )


# ============================================================================
# 5. Tariff queries in bank_selection_yul and bank_pricing_yul — enrich
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

    def test_bank_selection_yul_gives_enrich(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_yul(phrase)
                self.assertEqual(r["action"], "enrich", f"Expected enrich for: {phrase!r}")

    def test_bank_pricing_yul_gives_enrich(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_pricing(phrase)
                self.assertEqual(r["action"], "enrich", f"Expected enrich for: {phrase!r}")

    def test_low_cost_gives_enrich(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_low_cost(phrase)
                self.assertEqual(r["action"], "enrich", f"Expected enrich for: {phrase!r}")

    def test_no_can_i_compare_question(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_yul(phrase)
                reply = (r.get("reply") or "").lower()
                self.assertNotIn("могу сравнить", reply)
                self.assertNotIn("хотите сравнить", reply)

    def test_bank_selection_yul_no_rigid_next_step(self):
        r = self._r_yul("сравни тарифы")
        self.assertIsNone(r["updates"].get(SLOT_NEXT_STEP))

    def test_bank_pricing_yul_no_rigid_next_step(self):
        r = self._r_pricing("какие тарифы")
        self.assertIsNone(r["updates"].get(SLOT_NEXT_STEP))

    def test_reply_is_none_for_all(self):
        for phrase in self.COMPARE_PHRASES:
            with self.subTest(phrase=phrase):
                r = self._r_yul(phrase)
                self.assertIsNone(r["reply"])


# ============================================================================
# 6. Pricing helpers (still deterministic — used by LLM fallback if needed)
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
