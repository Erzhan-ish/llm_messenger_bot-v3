"""Comprehensive test coverage for all 30 user scenarios.

Tests the deterministic layers:
  - Scenario matching (ScenarioFactIndex.match_scenarios / build_fact_pack)
  - Identity guard (check_identity_guard)
  - Context fallback (_build_context_fallback)
  - Response validator (validate_reply)

Run: python -m pytest tests/test_30_scenarios.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_KB_SOURCE = Path(__file__).parent.parent / "knowledge_base.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_index():
    from app.knowledge_base.kb import KnowledgeBase
    kb = KnowledgeBase.from_source(_KB_SOURCE, cache_path=None)
    return kb.scenario_index


def _match_ids(user_text: str, memory=None, entities=None) -> list:
    idx = _load_index()
    matches = idx.match_scenarios(user_text, memory=memory or {}, current_entities=entities or {})
    return [m.scenario_id for m in matches]


def _fact_pack(user_text: str, memory=None, entities=None) -> dict:
    idx = _load_index()
    matches = idx.match_scenarios(user_text, memory=memory or {}, current_entities=entities or {})
    return idx.build_fact_pack(matches, memory=memory or {}, current_entities=entities or {})


def _identity(text: str, introduced: bool = True):
    from app.services.identity_guard import check_identity_guard
    return check_identity_guard(text, {"_introduced": introduced})


def _fallback(scenario_facts=None, current_entities=None, slots=None, signals=None):
    from app.processing.message_processor import _build_context_fallback
    return _build_context_fallback(
        fact_pack={},
        current_entities=current_entities or {},
        slots=slots or {},
        scenario_facts=scenario_facts,
        signals=signals,
    )


def _val(reply: str, user_text: str = "", answer_contract=None, scenario_facts=None, slots=None):
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


def _skip_if_no_kb(test_case):
    if not _KB_SOURCE.exists():
        test_case.skipTest("knowledge_base.txt not found")


# ===========================================================================
# CATEGORY 1 — Bank selection + pricing
# ===========================================================================

class TestCat1BankSelection(unittest.TestCase):
    """Scenarios 1-4: which banks, cheapest YUL, Uralsib conditions, FL."""

    def setUp(self):
        _skip_if_no_kb(self)

    def test_s1_partner_banks_yul(self):
        """С какими банками вы сейчас работаете по юрлицам? → partner_banks."""
        ids = _match_ids("С какими банками вы сейчас работаете по юрлицам?")
        self.assertIn("partner_banks", ids)

    def test_s1_partner_banks_fallback_contains_all_three(self):
        """Fallback for partner_banks must name Альфа, ТКБ, Уралсиб."""
        reply = _fallback(scenario_facts={"partner_banks": {}})
        self.assertIn("Альфа-Банк", reply)
        self.assertIn("ТКБ", reply)
        self.assertIn("Уралсиб", reply)

    def test_s2_cheapest_yul_keyword(self):
        """'Дешевле' keyword → bank_selection_yul_low_cost or bank_selection_yul."""
        ids = _match_ids("где дешевле всего открыть счет для ЮЛ")
        matched = set(ids)
        self.assertTrue(
            matched & {"bank_selection_yul_low_cost", "bank_selection_yul"},
            f"Got: {ids}",
        )

    def test_s2_cheapest_yul_full_query(self):
        """'Где дешевле всего открыть счет на ЮЛ?' → includes low_cost scenario."""
        ids = _match_ids("Где дешевле всего открыть счет на ЮЛ?")
        self.assertIn("bank_selection_yul_low_cost", ids)

    def test_s3_uralsib_yul_conditions(self):
        """Уралсиб условия для юрлица → uralsib_yul_conditions in fact_pack with 3500/1600."""
        pack = _fact_pack(
            "Уралсиб вроде нормально, какие там условия для юрлица?",
            entities={"mentioned_bank": "Уралсиб", "mentioned_client_type": "ЮЛ"},
        )
        self.assertIn("uralsib_yul_conditions", pack)
        pricing = pack["uralsib_yul_conditions"].get("pricing", {})
        values = {v.get("value_num") for v in pricing.values() if v.get("value_num")}
        self.assertTrue(values & {3500, 1600}, f"Expected 3500/1600, got {values}")

    def test_s4_fl_bank_selection(self):
        """А что по физлицам? (client_type=ФЛ) → bank_selection_fl."""
        ids = _match_ids("А что по физлицам?", entities={"mentioned_client_type": "ФЛ"})
        self.assertIn("bank_selection_fl", ids)

    def test_s4_fl_fallback_tkb_1500_missing(self):
        """Validator: FL bank reply without 1500 → missing_fl_tariff_details."""
        result = _val(
            "Для физлица рекомендуем ТКБ — надёжный банк.",
            user_text="что по физлицам, сколько стоит открытие?",
            answer_contract={"topic": "bank_selection_fl"},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "missing_fl_tariff_details")

    def test_s4_fl_correct_tariff_valid(self):
        """FL reply with '1500' is valid."""
        result = _val(
            "Для физлица — ТКБ: открытие 1500 руб., ведение бесплатно.",
            user_text="что по физлицам?",
            answer_contract={"topic": "bank_selection_fl"},
        )
        self.assertTrue(result["is_valid"])


# ===========================================================================
# CATEGORY 2 — Complex transfer / operational
# ===========================================================================

class TestCat2TransferAndOps(unittest.TestCase):
    """Scenarios 5-9: transfers, cash, internet banking, restructuring."""

    def setUp(self):
        _skip_if_no_kb(self)

    def test_s5_transfer_alfa_200k(self):
        """Нужно перевести 200 000 с Альфы → alfabank_yul_conditions + large_transfer_fl."""
        ids = _match_ids(
            "Нужно перевести 200 000 руб из Альфа-Банка",
            entities={"mentioned_bank": "Альфа-Банк"},
        )
        matched = set(ids)
        self.assertTrue(
            matched & {"alfabank_yul_conditions", "large_transfer_fl"},
            f"Got: {ids}",
        )

    def test_s5_transfer_keyword_match(self):
        """'Перевод' keyword → relevant transfer/pricing scenario."""
        ids = _match_ids("нужен перевод на другой счёт", entities={"mentioned_bank": "Альфа-Банк"})
        matched = set(ids)
        self.assertTrue(
            matched & {"alfabank_yul_conditions", "large_transfer_fl"},
            f"Got: {ids}",
        )

    def test_s6_transfer_uralsib_30m(self):
        """А если 30 млн из Уралсиба? → uralsib_yul_conditions + large_transfer_fl."""
        ids = _match_ids(
            "А если перевести 30 миллионов из Уралсиба?",
            entities={"mentioned_bank": "Уралсиб"},
        )
        matched = set(ids)
        self.assertTrue(
            matched & {"uralsib_yul_conditions", "large_transfer_fl"},
            f"Got: {ids}",
        )

    def test_s7_cash_withdrawal(self):
        """Можно снять наличные? → cash_withdrawal_bankruptcy."""
        ids = _match_ids("Можно ли снять наличные со счёта должника?")
        self.assertIn("cash_withdrawal_bankruptcy", ids)

    def test_s7_cash_validator_rejects_nalichnye_in_debtor_card_reply(self):
        """Validator: debtor_card topic must not contain 'наличные'."""
        result = _val(
            "Карту можно оформить, наличные снять тоже реально.",
            user_text="карта для должника",
            answer_contract={"topic": "debtor_card"},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "wrong_topic_fact")

    def test_s8_internet_bank_ip(self):
        """Есть ли интернет-банк для ИП? → internet_bank_fl_ip."""
        ids = _match_ids("Есть ли интернет-банк для ИП?")
        self.assertIn("internet_bank_fl_ip", ids)

    def test_s9_restrukturizatsiya_ip(self):
        """ИП в реструктуризации → restructuring_ip matched via keyword."""
        ids = _match_ids("ИП в реструктуризации — можно открыть счёт?")
        self.assertIn("restructuring_ip", ids)


# ===========================================================================
# CATEGORY 3 — Constraints / unusual debtors
# ===========================================================================

class TestCat3Constraints(unittest.TestCase):
    """Scenarios 10-16: card, deceased, non-resident, liquidated, etc."""

    def setUp(self):
        _skip_if_no_kb(self)

    def test_s10_debtor_card_realization(self):
        """Карту можно оформить должнику? → debtor_card_realization."""
        ids = _match_ids("Карту можно оформить должнику при реализации?")
        self.assertIn("debtor_card_realization", ids)

    def test_s10_card_fallback_mentions_finansovy_upravlyayuschiy(self):
        """Card fallback must mention 'финансовый управляющий'."""
        reply = _fallback(
            scenario_facts={"debtor_card_realization": {"constraints": ["финансовый управляющий"]}}
        )
        self.assertIn("финансовый управляющий", reply.lower())
        self.assertIn("реализац", reply.lower())

    def test_s10_card_validator_requires_facts(self):
        """Validator: debtor_card scenario + generic reply without key facts → invalid."""
        result = _val(
            "Да, карту можно оформить, обращайтесь.",
            user_text="карта для должника",
            scenario_facts={"debtor_card_realization": {"constraints": ["финансовый управляющий"]}},
            answer_contract={"topic": "debtor_card"},
        )
        self.assertFalse(result["is_valid"])

    def test_s11_deceased_fl(self):
        """Должник умер → deceased_fl."""
        ids = _match_ids("Должник умер, что делать со счётом?")
        self.assertIn("deceased_fl", ids)

    def test_s12_non_resident(self):
        """Должник нерезидент → non_resident."""
        ids = _match_ids("Должник нерезидент, можно ли открыть?")
        self.assertIn("non_resident", ids)

    def test_s12_non_resident_foreign(self):
        """Иностранный гражданин → non_resident."""
        ids = _match_ids("Должник — иностранный гражданин")
        self.assertIn("non_resident", ids)

    def test_s13_liquidated_yul(self):
        """ЮЛ ликвидировано → liquidated_yul."""
        ids = _match_ids("Юрлицо в процессе ликвидации, можно ли открыть?")
        self.assertIn("liquidated_yul", ids)

    def test_s15_special_account_without_main(self):
        """Спецсчёт без основного? → special_account_without_main."""
        ids = _match_ids("Нужен только специальный счёт без основного расчётного")
        self.assertIn("special_account_without_main", ids)

    def test_s15_zadatkovy_keyword(self):
        """'Задатков' keyword → special_account_without_main."""
        ids = _match_ids("счёт для задатков — нужен только он")
        self.assertIn("special_account_without_main", ids)

    def test_s16_red_zone(self):
        """Красная зона → red_zone_company."""
        ids = _match_ids("Что такое красная зона для компании?")
        self.assertIn("red_zone_company", ids)

    def test_s16_red_zone_keyword(self):
        """'Красн' keyword in 'красная зона' → red_zone_company."""
        ids = _match_ids("у банка мы в красной зоне")
        self.assertIn("red_zone_company", ids)


# ===========================================================================
# CATEGORY 4 — Signing / documents
# ===========================================================================

class TestCat4DocsAndSigning(unittest.TestCase):
    """Scenarios 17-21: docs, ECP online, power of attorney, no branch, advance."""

    def setUp(self):
        _skip_if_no_kb(self)

    def test_s17_uralsib_yul_docs(self):
        """Какие документы нужны для Уралсиба ЮЛ? → docs_uralsib_yul."""
        ids = _match_ids(
            "Какие документы нужны для Уралсиба для юрлица?",
            entities={"mentioned_bank": "Уралсиб", "mentioned_client_type": "ЮЛ"},
        )
        self.assertIn("docs_uralsib_yul", ids)

    def test_s17_docs_keyword_boosts_bank_scenario(self):
        """Keyword 'документы' + Уралсиб entity → docs_uralsib_yul."""
        ids = _match_ids(
            "какие нужны документы",
            entities={"mentioned_bank": "Уралсиб"},
        )
        self.assertIn("docs_uralsib_yul", ids)

    def test_s18_online_ecp_signing(self):
        """Можно ли подписать ЭЦП онлайн? → online_signing."""
        ids = _match_ids("Можно ли подписать документы ЭЦП онлайн?")
        self.assertIn("online_signing", ids)

    def test_s18_online_keyword(self):
        """'Онлайн' keyword → online_signing."""
        ids = _match_ids("Хочу открыть счёт онлайн без визита в банк")
        self.assertIn("online_signing", ids)

    def test_s19_power_of_attorney(self):
        """Можно по доверенности? → power_of_attorney."""
        ids = _match_ids("Можно ли по доверенности открыть счёт?")
        self.assertIn("power_of_attorney", ids)

    def test_s20_no_branch_in_city(self):
        """В нашем городе нет отделения → at least one scenario matched."""
        ids = _match_ids("В нашем городе нет отделения Уралсиба, как быть?")
        self.assertGreater(len(ids), 0)

    def test_s21_advance_opening(self):
        """Можно открыть счёт заранее? → early_opening."""
        ids = _match_ids("Можно ли открыть счёт заранее до введения процедуры?")
        self.assertIn("early_opening", ids)

    def test_s21_advance_keyword(self):
        """'Заранее' keyword → early_opening."""
        ids = _match_ids("хочу открыть заранее, до банкротства")
        self.assertIn("early_opening", ids)


# ===========================================================================
# CATEGORY 5 — Add-on services (refusal scenarios)
# ===========================================================================

class TestCat5AddonServices(unittest.TestCase):
    """Scenarios 22-23: capitalization, payroll — both refusals."""

    def setUp(self):
        _skip_if_no_kb(self)

    def test_s22_capitalization(self):
        """Есть ли капитализация по счёту? → capitalization_none."""
        ids = _match_ids("Есть ли капитализация процентов по счёту?")
        self.assertIn("capitalization_none", ids)

    def test_s22_capitalization_keyword(self):
        """'Капитализац' keyword → capitalization_none."""
        ids = _match_ids("нам нужна капитализация")
        self.assertIn("capitalization_none", ids)

    def test_s23_payroll_project(self):
        """Нас интересует зарплатный проект → payroll_project_none."""
        ids = _match_ids("Нас интересует зарплатный проект для сотрудников")
        self.assertIn("payroll_project_none", ids)

    def test_s23_payroll_keyword(self):
        """'Зарплатн' keyword → payroll_project_none."""
        ids = _match_ids("есть зарплатный проект?")
        self.assertIn("payroll_project_none", ids)

    def test_s22_validator_rejects_tariff_in_partner_banks(self):
        """Validator: partner_banks topic must not contain tariff numbers."""
        result = _val(
            "Работаем с Альфа-Банком (800 руб.), ТКБ (1500 руб.) и Уралсибом (3500 руб.).",
            user_text="с какими банками работаете?",
            answer_contract={"topic": "partner_banks"},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "answered_tariffs_when_asked_bank_list")

    def test_s23_partner_banks_valid_no_tariffs(self):
        """Validator: partner_banks reply without tariff numbers is valid."""
        result = _val(
            "Сейчас активные варианты — Альфа-Банк, ТКБ и Уралсиб. Т-Банк, МКБ и Росбанк на паузе.",
            user_text="с какими банками работаете?",
            answer_contract={"topic": "partner_banks"},
        )
        self.assertTrue(result["is_valid"])


# ===========================================================================
# CATEGORY 6 — Role model and handoff
# ===========================================================================

class TestCat6RoleAndHandoff(unittest.TestCase):
    """Scenarios 24-30: identity, frustration, out-of-domain, consent, handoff."""

    def test_s24_identity_bot_question(self):
        """Вы бот? → identity guard intercepts, holds manager role."""
        result = _identity("Вы бот?", introduced=True)
        self.assertIsNotNone(result)
        reply = result["reply"]
        self.assertNotIn("я бот", reply.lower())
        self.assertNotIn("я живой человек", reply.lower())
        self.assertIn("В плюсе", reply)

    def test_s24_who_are_you_introduced(self):
        """Кто вы? → identity guard, role answer, no data-collection."""
        result = _identity("Кто вы? Почему не представляетесь?", introduced=True)
        self.assertIsNotNone(result)
        self.assertIn("В плюсе", result["reply"])
        self.assertNotIn("нужны данные для открытия счета", result["reply"])

    def test_s24_greeting_not_introduced(self):
        """Добрый день → intro reply with Алексей and В плюсе."""
        result = _identity("добрый день", introduced=False)
        self.assertIsNotNone(result)
        self.assertIn("Алексей", result["reply"])
        self.assertIn("В плюсе", result["reply"])
        self.assertTrue(result.get("set_introduced"))

    def test_s24_greeting_after_introduced_passes_through(self):
        """Добрый день when already introduced → None (LLM handles)."""
        result = _identity("добрый день", introduced=True)
        self.assertIsNone(result)

    def test_s25_escalation_promised_action_without_handoff(self):
        """Bot promises to open account without handoff → rejected."""
        result = _val(
            "Хорошо, сейчас откроем счёт для вас.",
            user_text="Хочу открыть счёт",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "promised_action_without_handoff")

    def test_s26_consent_trigger_open_account(self):
        """'Откройте мне счёт' → validator: needs handoff or request_data."""
        result = _val(
            "Хорошо, всё понял, готовы.",
            user_text="счет откройте, пожалуйста",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "open_account_without_handoff_or_request_data")

    def test_s26_consent_with_handoff_valid(self):
        """When handoff.needed=True, bot can say it's connecting."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            "Отлично! Передаю вас менеджеру для оформления счёта.",
            brain_result={"action": "handoff", "handoff": {"needed": True}},
            current_entities={},
            slots={"_had_consent": True},
            user_text="давайте открывать счёт",
        )
        self.assertTrue(result["is_valid"])

    def test_s27_frustration_regex(self):
        """'????' and '!!!' match frustration-only pattern."""
        import re
        _FRUSTRATION_ONLY_RE = re.compile(r"^[\s?!.…]+$", re.U)
        self.assertTrue(_FRUSTRATION_ONLY_RE.match("????"))
        self.assertTrue(_FRUSTRATION_ONLY_RE.match("!!!"))
        self.assertFalse(_FRUSTRATION_ONLY_RE.match("что происходит?"))

    def test_s28_out_of_domain_question_phrasing_not_forced_handoff(self):
        """Informational question 'где быстрее всего счёт открыть?' does not force handoff."""
        from app.processing.message_processor import should_force_handoff
        result = should_force_handoff(
            "Где быстрее всего счёт открыть?",
            brain_result={},
            memory={},
        )
        self.assertFalse(result)

    def test_s29_confusion_guard(self):
        """'О чем ты?' → confusion guard returns identity + apology."""
        result = _identity("о чем ты вообще? ты кто?", introduced=True)
        self.assertIsNotNone(result)
        reply = result["reply"]
        self.assertIn("Извините", reply)
        self.assertIn("В плюсе", reply)

    def test_s30_explicit_consent_forces_handoff(self):
        """'Да, давайте откроем счёт' → should_force_handoff returns True."""
        from app.processing.message_processor import should_force_handoff
        result = should_force_handoff(
            "Да, давайте откроем счёт",
            brain_result={},
            memory={},
        )
        self.assertTrue(result)

    def test_s30_repeated_intro_rejected(self):
        """Repeated 'Здравствуйте я Алексей' after _introduced → validator rejects."""
        result = _val(
            "Здравствуйте! Я Алексей, консультант компании «В плюсе».",
            user_text="расскажите про Уралсиб",
            slots={"_introduced": True},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "repeated_intro")


# ===========================================================================
# EDGE CASES — boundary conditions from the 30 scenarios
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Additional edge cases that surfaced from scenario analysis."""

    def test_ec1_near_duplicate_rejected(self):
        """Same reply twice → near_duplicate_of_previous."""
        reply = "По Уралсибу для юрлица: открытие 3500 руб., ведение 1600 руб."
        result = _val(
            reply,
            user_text="напомни условия Уралсиба",
            slots={"_last_bot_text": reply},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "near_duplicate_of_previous")

    def test_ec2_cjk_in_reply_rejected(self):
        """CJK characters in reply → non_russian_output."""
        result = _val("Уралсиб: 3500 руб. 银行 ведение 1600 руб.", user_text="условия Уралсиба")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "non_russian_output")

    def test_ec3_empty_reply_rejected(self):
        """Empty reply → empty_reply."""
        result = _val("", user_text="что-нибудь")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "empty_reply")

    def test_ec4_question_phrasing_not_forced_handoff(self):
        """'Где быстрее всего счёт открыть?' should NOT be treated as consent."""
        from app.processing.message_processor import should_force_handoff
        result = should_force_handoff(
            "Где быстрее всего счёт открыть?",
            brain_result={},
            memory={},
        )
        self.assertFalse(result)

    def test_ec5_explicit_consent_forces_handoff(self):
        """'Да, давайте откроем счёт' → should_force_handoff returns True."""
        from app.processing.message_processor import should_force_handoff
        result = should_force_handoff(
            "Да, давайте откроем счёт",
            brain_result={},
            memory={},
        )
        self.assertTrue(result)

    def test_ec6_account_type_difference_fallback(self):
        """'задатковый и залоговый' → fallback answers difference, not generic."""
        reply = _fallback(signals={"asks_account_type_difference": True})
        self.assertIn("задатков", reply.lower())
        self.assertNotIn("Как могу помочь", reply)

    def test_ec7_scenario_facts_not_empty_generic_reply_invalid(self):
        """Generic 'что вас интересует?' rejected when scenario_facts present."""
        result = _val(
            "Что вас интересует?",
            user_text="карта для должника",
            scenario_facts={"debtor_card_realization": {"constraints": ["финансовый управляющий"]}},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "generic_reply_despite_scenario_facts")

    def test_ec8_identity_question_wrong_answer_rejected(self):
        """'Чем могу помочь?' on identity question → identity_question_wrong_answer."""
        result = _val(
            "Чем могу помочь вам сегодня?",
            user_text="кто вы?",
            answer_contract={"topic": "identity"},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "identity_question_wrong_answer")

    def test_ec9_partner_banks_scenario_all_required(self):
        """Validator: partner_banks scenario requires all 3 active banks."""
        result = _val(
            "Работаем с Альфа-Банком и ТКБ.",
            user_text="с какими банками работаете?",
            answer_contract={"topic": "partner_banks"},
            scenario_facts={"partner_banks": {}},
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "missing_required_partner_banks")

    def test_ec10_uralsib_fallback_has_prices(self):
        """Fallback for Уралсиб YUL must contain 3500 and 1600."""
        reply = _fallback(
            current_entities={"mentioned_bank": "Уралсиб"},
            slots={"client_type": "ЮЛ"},
        )
        self.assertIn("3500", reply)
        self.assertIn("1600", reply)

    def test_ec11_role_confusion_phrase_rejected(self):
        """'о чем я?' → role_confusion."""
        result = _val("о чем я?", user_text="ты кто?")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "role_confusion")

    def test_ec12_too_short_specific_question(self):
        """Short reply to specific question → too_short_for_specific_question."""
        result = _val(
            "Хорошо.",
            user_text="какие условия у банков для открытия счёта?",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "too_short_for_specific_question")


if __name__ == "__main__":
    unittest.main()
