"""Regression tests for bank/pricing scenario switching and debtor_type handling.

Covers plan.txt requirements:
  1. allowed_stages → explicit bank query → switches to bank_selection_yul
  2. debtor_type slot extraction
  3. Known debtor_type prevents "юрлицо или физлицо?" question
  4. "какие там тарифы" → switches to bank_selection_yul
  5. "Уралсиб что?" with/without bank_focus returns Uralsib-specific scenario
  6. bank_focus must not use generic partner_banks fallback
  7. _is_followup does NOT suppress bank/pricing intent even with known_slots
  8. Validator rejects repeated debtor-type question and wrong-bank reply

Run:
    python -m pytest tests/test_bank_pricing_policy.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.scenario_policy import (
    _bank_pricing_switch_target,
    _is_bank_pricing_intent,
    _is_followup,
    decide_scenario_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy(user_text, active=None, client_type=None, bank_name=None,
            known_slots=None, rag_scenarios=None):
    slots = {"_active_scenario": active}
    if client_type:
        slots["client_type"] = client_type
    if bank_name:
        slots["bank_name"] = bank_name
    if known_slots:
        slots["_known_slots"] = known_slots
    return decide_scenario_policy(
        user_text=user_text,
        slots=slots,
        rag_scenarios=rag_scenarios or [],
    )


def _switch_target(user_text, active=None, client_type=None,
                   bank_name=None, last_bank=None):
    slots = {}
    if client_type:
        slots["client_type"] = client_type
    if bank_name:
        slots["bank_name"] = bank_name
    if last_bank:
        slots["_last_bank"] = last_bank
    return _bank_pricing_switch_target(user_text, active or "", slots)


# ============================================================================
# 1. _is_bank_pricing_intent — detection layer
# ============================================================================
class TestIsBankPricingIntent(unittest.TestCase):
    INTENT = [
        "с какими банками сотрудничаете",
        "какие банки",
        "тарифы",
        "тарифы у уралсиба",
        "условия у них",
        "стоимость",
        "сколько стоит",
        "уралсиб",
        "альфа-банк",
        "ткб",
        "т-банк",
        "мкб",
        "росбанк",
    ]
    NOT_INTENT = [
        "да",
        "хорошо",
        "что делать",
        "документы нужны",
        "у нас банкрот",
    ]

    def test_intent_detected(self):
        for t in self.INTENT:
            with self.subTest(t=t):
                self.assertTrue(_is_bank_pricing_intent(t), f"Should be bank intent: {t!r}")

    def test_not_intent(self):
        for t in self.NOT_INTENT:
            with self.subTest(t=t):
                self.assertFalse(_is_bank_pricing_intent(t), f"Should NOT be bank intent: {t!r}")


# ============================================================================
# 2. _bank_pricing_switch_target — target resolution
# ============================================================================
class TestBankPricingSwitchTarget(unittest.TestCase):

    # 2a. Explicit bank name in message
    def test_уралсиб_yul_returns_uralsib_yul_conditions(self):
        target = _switch_target("уралсиб что?", client_type="ЮЛ")
        self.assertEqual(target, "uralsib_yul_conditions")

    def test_уралсиб_fl_returns_uralsib_fl_conditions(self):
        target = _switch_target("уралсиб что?", client_type="ФЛ")
        self.assertEqual(target, "uralsib_fl_conditions")

    def test_альфа_yul_returns_alfabank_yul_conditions(self):
        target = _switch_target("альфа тарифы", client_type="ЮЛ")
        self.assertEqual(target, "alfabank_yul_conditions")

    def test_ткб_yul_returns_tkb_yul_conditions(self):
        target = _switch_target("условия у ткб", client_type="ЮЛ")
        self.assertEqual(target, "tkb_yul_conditions")

    def test_мкб_returns_mkb_yul(self):
        target = _switch_target("мкб тарифы", client_type="ЮЛ")
        self.assertEqual(target, "mkb_yul_conditions")

    def test_росбанк_returns_rosbank_yul(self):
        target = _switch_target("росбанк условия", client_type="ЮЛ")
        self.assertEqual(target, "rosbank_yul_conditions")

    # 2b. Already in that scenario — no switch (returns None)
    def test_already_in_uralsib_scenario_returns_none(self):
        target = _switch_target("уралсиб что?", active="uralsib_yul_conditions", client_type="ЮЛ")
        self.assertIsNone(target)

    # 2c. Partner banks query → bank_selection_yul / partner_banks
    def test_partner_banks_yul(self):
        target = _switch_target("с какими банками сотрудничаете", client_type="ЮЛ")
        self.assertEqual(target, "bank_selection_yul")

    def test_partner_banks_fl(self):
        target = _switch_target("с какими банками сотрудничаете", client_type="ФЛ")
        self.assertEqual(target, "partner_banks")

    def test_какие_банки_yul(self):
        target = _switch_target("какие банки", client_type="ЮЛ")
        self.assertEqual(target, "bank_selection_yul")

    # 2d. Generic tariffs → bank_selection_yul / bank_selection_fl
    def test_тарифы_yul(self):
        target = _switch_target("тарифы", client_type="ЮЛ")
        self.assertEqual(target, "bank_selection_yul")

    def test_тарифы_fl(self):
        target = _switch_target("тарифы", client_type="ФЛ")
        self.assertEqual(target, "bank_selection_fl")

    def test_стоимость_yul(self):
        target = _switch_target("стоимость", client_type="ЮЛ")
        self.assertEqual(target, "bank_selection_yul")

    # 2e. Bank in slots + conditions/tariffs intent
    def test_conditions_intent_with_bank_in_slots(self):
        target = _switch_target("условия у них", client_type="ЮЛ", bank_name="Уралсиб")
        self.assertEqual(target, "uralsib_yul_conditions")

    def test_tariff_intent_with_bank_in_last_bank(self):
        target = _switch_target("тарифы у них", client_type="ЮЛ", last_bank="ТКБ")
        self.assertEqual(target, "tkb_yul_conditions")

    # 2f. No client_type — defaults to non-YUL (no "ЮЛ" in ct)
    def test_no_client_type_partner_banks(self):
        target = _switch_target("с какими банками сотрудничаете")
        self.assertEqual(target, "partner_banks")

    def test_no_client_type_tariffs(self):
        target = _switch_target("тарифы")
        self.assertEqual(target, "bank_selection_fl")


# ============================================================================
# 3. decide_scenario_policy — step 3.5 integration
# ============================================================================
class TestDecidePolicyBankSwitch(unittest.TestCase):

    # 3a. allowed_stages → "с какими банками сотрудничаете" → switches
    def test_allowed_stages_to_bank_selection_yul(self):
        result = _policy(
            "у меня юр лицо должник. с какими банками сотрудничаете?",
            active="allowed_stages",
            client_type="ЮЛ",
        )
        self.assertEqual(result["decision"], "switch")
        self.assertEqual(result["active_scenario"], "bank_selection_yul")
        self.assertTrue(result["scenario_switch_allowed"])
        self.assertIn("bank_pricing_intent", result["reason"])

    def test_allowed_stages_to_uralsib_when_named(self):
        result = _policy(
            "уралсиб что? и условия у них",
            active="allowed_stages",
            client_type="ЮЛ",
        )
        self.assertEqual(result["decision"], "switch")
        self.assertEqual(result["active_scenario"], "uralsib_yul_conditions")

    # 3b. allowed_stages + known_slots → "тарифы" still switches (bank intent beats follow-up)
    def test_tariffs_beats_known_slots_follow_up(self):
        result = _policy(
            "тарифы",
            active="allowed_stages",
            client_type="ЮЛ",
            known_slots={"account_opening_allowed": True},
        )
        self.assertEqual(result["decision"], "switch")
        self.assertIn(result["active_scenario"], ("bank_selection_yul", "bank_selection_fl"))

    def test_bank_name_in_message_beats_known_slots(self):
        result = _policy(
            "уралсиб",
            active="allowed_stages",
            client_type="ЮЛ",
            known_slots={"account_opening_allowed": True},
        )
        self.assertEqual(result["decision"], "switch")
        self.assertEqual(result["active_scenario"], "uralsib_yul_conditions")

    # 3c. Partner banks query
    def test_какие_банки_from_allowed_stages(self):
        result = _policy(
            "какие банки",
            active="allowed_stages",
            client_type="ЮЛ",
        )
        self.assertEqual(result["decision"], "switch")
        self.assertEqual(result["active_scenario"], "bank_selection_yul")

    # 3d. Specific bank + условия у них (bank in slots)
    def test_условия_у_них_with_uralsib_in_slots(self):
        result = _policy(
            "условия у них",
            active="allowed_stages",
            client_type="ЮЛ",
            bank_name="Уралсиб",
        )
        self.assertEqual(result["decision"], "switch")
        self.assertEqual(result["active_scenario"], "uralsib_yul_conditions")

    # 3e. No switch when already in the right scenario
    def test_no_switch_when_already_in_bank_selection_yul(self):
        result = _policy(
            "тарифы",
            active="bank_selection_yul",
            client_type="ЮЛ",
        )
        # Already in bank_selection_yul — should NOT switch (target == active)
        self.assertNotEqual(result["active_scenario"], "bank_selection_yul_again")
        # The policy stays at bank_selection_yul (keep or switch to same = no change)
        self.assertEqual(result["active_scenario"], "bank_selection_yul")


# ============================================================================
# 4. _is_followup — bank intent bypasses known_slots heuristic
# ============================================================================
class TestIsFollowupBankBypass(unittest.TestCase):

    def _followup(self, text, slots=None):
        return _is_followup(text, slots or {})

    def test_тарифы_with_known_slots_not_followup(self):
        slots = {"_known_slots": {"account_opening_allowed": True}}
        self.assertFalse(self._followup("тарифы", slots))

    def test_уралсиб_with_known_slots_not_followup(self):
        slots = {"_known_slots": {"account_opening_allowed": True}}
        self.assertFalse(self._followup("уралсиб", slots))

    def test_с_какими_банками_with_known_slots_not_followup(self):
        slots = {"_known_slots": {"account_opening_allowed": True}}
        self.assertFalse(self._followup("с какими банками сотрудничаете", slots))

    def test_non_bank_short_message_with_known_slots_is_followup(self):
        slots = {"_known_slots": {"account_opening_allowed": True}}
        self.assertTrue(self._followup("что дальше", slots))

    def test_non_bank_short_message_2words_with_known_slots_is_followup(self):
        slots = {"_known_slots": {"realization_started": True}}
        self.assertTrue(self._followup("всё ясно", slots))

    def test_bank_message_without_known_slots_not_bank_bypass(self):
        # No known_slots — bank intent doesn't matter for this check
        # (followup is still False because the text isn't a confirm/followup phrase)
        self.assertFalse(self._followup("уралсиб", {}))


# ============================================================================
# 5. debtor_type slot extraction
# ============================================================================
class TestDebtorTypeExtraction(unittest.TestCase):

    def _extract(self, text):
        from app.processing.slots import extract_runtime_slots
        slots = {}
        extract_runtime_slots(text, slots)
        return slots

    def test_юр_лицо_должник(self):
        s = self._extract("у меня юр лицо должник")
        self.assertEqual(s.get("debtor_type"), "ЮЛ")

    def test_должник_юр_лицо(self):
        s = self._extract("должник юр лицо")
        self.assertEqual(s.get("debtor_type"), "ЮЛ")

    def test_для_юрлица(self):
        s = self._extract("для юрлица открываем счет")
        self.assertEqual(s.get("debtor_type"), "ЮЛ")

    def test_ооо_должник(self):
        s = self._extract("ооо должник")
        self.assertEqual(s.get("debtor_type"), "ЮЛ")

    def test_должник_физ_лицо(self):
        s = self._extract("должник физ лицо")
        self.assertEqual(s.get("debtor_type"), "ФЛ")

    def test_для_физлица(self):
        s = self._extract("должник для физлица")
        self.assertEqual(s.get("debtor_type"), "ФЛ")

    def test_neutral_text_no_debtor_type(self):
        s = self._extract("добрый день, расскажите про счет")
        self.assertIsNone(s.get("debtor_type"))

    def test_not_overwritten_if_already_set(self):
        from app.processing.slots import extract_runtime_slots
        slots = {"debtor_type": "ФЛ"}
        extract_runtime_slots("должник юр лицо", slots)
        self.assertEqual(slots.get("debtor_type"), "ФЛ")  # not overwritten


# ============================================================================
# 6. Validator — repeated debtor-type question
# ============================================================================
class TestValidatorDebtorTypeQuestion(unittest.TestCase):

    def _validate(self, reply, slots):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            {"action": "answer", "handoff": {"needed": False}},
            {},
            slots,
            user_text="какие тарифы",
        )

    def test_rejects_юрлицо_или_физлицо_when_type_known_via_client_type(self):
        slots = {"_introduced": True, "client_type": "ЮЛ"}
        r = self._validate(
            "Могу помочь. Счёт подбираем для юрлица или физлица?", slots
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "repeated_known_debtor_type_question")

    def test_rejects_юрлицо_или_физлицо_when_type_known_via_debtor_type(self):
        slots = {"_introduced": True, "debtor_type": "ФЛ"}
        r = self._validate(
            "Для юрлица или физлица подбираем счёт?", slots
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "repeated_known_debtor_type_question")

    def test_accepts_reply_without_debtor_question_when_type_known(self):
        slots = {"_introduced": True, "client_type": "ЮЛ"}
        r = self._validate(
            "Для юрлиц актуальные варианты: Альфа-Банк, ТКБ, Уралсиб.", slots
        )
        self.assertTrue(r["is_valid"])

    def test_accepts_question_when_type_unknown(self):
        slots = {"_introduced": True}
        r = self._validate(
            "Счёт подбираем для юрлица или физлица?", slots
        )
        # No client_type known — question is valid
        self.assertNotEqual(r.get("reason"), "repeated_known_debtor_type_question")


# ============================================================================
# 7. Validator — bank focus not in reply
# ============================================================================
class TestValidatorBankFocusInReply(unittest.TestCase):

    def _validate(self, reply, active_scenario, user_text="условия"):
        from app.services.response_validator import validate_reply
        slots = {"_introduced": True, "_active_scenario": active_scenario}
        return validate_reply(
            reply,
            {"action": "answer", "handoff": {"needed": False}},
            {},
            slots,
            user_text=user_text,
        )

    def test_uralsib_scenario_reply_missing_uralsib_rejected(self):
        r = self._validate(
            "Для юрлиц доступны Альфа-Банк и ТКБ с разными тарифами и условиями.",
            "uralsib_yul_conditions",
        )
        self.assertFalse(r["is_valid"])
        self.assertIn("уралсиб", r["reason"])

    def test_uralsib_scenario_reply_with_uralsib_accepted(self):
        r = self._validate(
            "По Уралсибу для юрлиц: открытие 3500 руб., ведение 1600 руб. в месяц.",
            "uralsib_yul_conditions",
        )
        self.assertTrue(r["is_valid"])

    def test_tkb_scenario_reply_missing_tkb_rejected(self):
        r = self._validate(
            "Для юрлиц доступны Альфа-Банк и Уралсиб с выгодными условиями открытия.",
            "tkb_yul_conditions",
        )
        self.assertFalse(r["is_valid"])
        self.assertIn("ткб", r["reason"])

    def test_short_reply_not_checked(self):
        # Reply 40–59 chars: too_short check passes (≥40), bank focus skipped (<60)
        r = self._validate(
            "Уточняю детали по условиям, сейчас смотрю.",  # 42 chars
            "uralsib_yul_conditions",
        )
        self.assertNotEqual(r.get("reason"), "bank_focus_not_in_reply:уралсиб")

    def test_non_bank_scenario_not_checked(self):
        r = self._validate(
            "Для юрлиц доступны Альфа-Банк и ТКБ.",
            "allowed_stages",
        )
        # allowed_stages is not in _BANK_CONDITIONS_SCENARIO_KW → no bank check
        self.assertNotIn("bank_focus_not_in_reply", r.get("reason") or "")


# ============================================================================
# 8. Full scenario flow simulation
# ============================================================================
class TestFullBankFlow(unittest.TestCase):
    """Simulate the exact dialog from plan.txt step by step."""

    def test_allowed_stages_then_bank_query_switches(self):
        # Turn 1: sets active=allowed_stages
        r1 = _policy("у вас открываются счета на стадии наблюдения?", active=None)
        # With no active, RAG would set it. Simulate: active becomes allowed_stages.

        # Turn 2: user declares debtor type + asks banks
        r2 = _policy(
            "у меня юр лицо должник. с какими банками сотрудничаете?",
            active="allowed_stages",
            client_type="ЮЛ",
            known_slots={"account_opening_allowed": True},
        )
        self.assertEqual(r2["decision"], "switch")
        self.assertEqual(r2["active_scenario"], "bank_selection_yul")

    def test_bank_selection_then_tariff_query_switches_to_same(self):
        # Turn 3: user asks tariffs after bank_selection_yul is active
        r3 = _policy(
            "я же сказал для юр лица. какие там тарифы?",
            active="bank_selection_yul",
            client_type="ЮЛ",
        )
        # Already in bank_selection_yul, tariffs doesn't move to different scenario
        self.assertEqual(r3["active_scenario"], "bank_selection_yul")

    def test_bank_selection_then_uralsib_query(self):
        # Turn 4: user specifically asks about Уралсиб
        r4 = _policy(
            "уралсиб что? и условия у них",
            active="bank_selection_yul",
            client_type="ЮЛ",
        )
        self.assertEqual(r4["decision"], "switch")
        self.assertEqual(r4["active_scenario"], "uralsib_yul_conditions")

    def test_uralsib_scenario_active_then_conditions_kept(self):
        # Turn 5: follow-up "расскажите подробнее" in uralsib scenario
        r5 = _policy(
            "расскажите подробнее",
            active="uralsib_yul_conditions",
            client_type="ЮЛ",
            known_slots={"bank_focus": "Уралсиб"},
        )
        # "расскажите подробнее" has known_slots + 2 words → follow-up → keep_active
        self.assertEqual(r5["decision"], "keep_active")
        self.assertEqual(r5["active_scenario"], "uralsib_yul_conditions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
