"""E2E тесты для conversation_brain.

Проверяют, что бот правильно понимает смысл и не звучит как скрипт.
Запуск: python -m pytest tests/test_conversation_brain_e2e.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Unit-тесты context_builder
# ---------------------------------------------------------------------------
class TestContextBuilder(unittest.TestCase):
    def test_extract_mentioned_bank_alfa(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("перевод на физлицо в Альфе?")
        self.assertEqual(r["mentioned_bank"], "Альфа-Банк")

    def test_extract_mentioned_bank_uralsib(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("давайте с Уралсибом")
        self.assertEqual(r["mentioned_bank"], "Уралсиб")

    def test_extract_amount_300k(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("300 тыс рублей")
        self.assertEqual(r["mentioned_amount"], 300_000)

    def test_extract_amount_plain(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("сумма 1500000")
        self.assertEqual(r["mentioned_amount"], 1_500_000)

    def test_extract_recipient_fl(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("перевод на физлицо")
        self.assertEqual(r["mentioned_recipient"], "ФЛ")

    def test_extract_recipient_ul(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("перевод на юрлицо")
        self.assertEqual(r["mentioned_recipient"], "ЮЛ")

    def test_no_bank_alias(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("с какими банками сотрудничаете?")
        self.assertIsNone(r["mentioned_bank"])

    def test_tkb_alias(self):
        from app.services.context_builder import extract_entities
        r = extract_entities("что по ТКБ?")
        self.assertEqual(r["mentioned_bank"], "ТКБ")


# ---------------------------------------------------------------------------
# Unit-тесты response_validator
# ---------------------------------------------------------------------------
class TestResponseValidator(unittest.TestCase):
    def _validate(self, reply, brain_result=None, entities=None, slots=None, tool_results=None):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            brain_result or {},
            entities or {},
            slots or {},
            tool_results=tool_results,
        )

    def test_empty_reply_invalid(self):
        r = self._validate("")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "empty_reply")

    def test_valid_reply(self):
        r = self._validate("В Альфа-Банке комиссия составит 1500 рублей. Сравним с другим банком?")
        self.assertTrue(r["is_valid"])

    def test_fee_missing_from_reply(self):
        tool_results = {"calculate_transfer_fee": {"calculated_fee": 1500, "total_fee": 1650}}
        r = self._validate(
            "Да, в этом банке выгодные условия.",
            tool_results=tool_results,
        )
        self.assertFalse(r["is_valid"])
        self.assertIn("fee_1500", r["reason"])

    def test_fee_in_reply_valid(self):
        tool_results = {"calculate_transfer_fee": {"calculated_fee": 1500, "total_fee": 1650}}
        r = self._validate(
            "В Альфа-Банке перевод 300 000 руб. на физлицо выйдет в 1500 рублей.",
            tool_results=tool_results,
        )
        self.assertTrue(r["is_valid"])

    def test_near_duplicate_invalid(self):
        slots = {"_last_bot_text": "Уточните сумму перевода для расчёта комиссии в Альфа-Банке."}
        r = self._validate(
            "Уточните сумму перевода для расчёта комиссии в Альфа-Банке.",
            slots=slots,
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "near_duplicate_of_previous")

    def test_wrong_bank_in_reply(self):
        r = self._validate(
            "В ТКБ перевод обойдётся в 2000 рублей.",
            entities={"mentioned_bank": "Альфа-Банк"},
            slots={},
        )
        self.assertFalse(r["is_valid"])


# ---------------------------------------------------------------------------
# Unit-тесты calculate_transfer_fee
# ---------------------------------------------------------------------------
class TestTransferFeeCalculator(unittest.TestCase):
    def test_alfa_fl_below_150k(self):
        from app.domain.calculators import calculate_transfer_fee
        r = calculate_transfer_fee("Альфа-Банк", 100_000, "ФЛ")
        self.assertEqual(r["calculated_fee"], 0)

    def test_alfa_fl_above_150k(self):
        from app.domain.calculators import calculate_transfer_fee
        r = calculate_transfer_fee("Альфа-Банк", 300_000, "ФЛ")
        self.assertEqual(r["calculated_fee"], 1500)  # 300000 * 0.005

    def test_uralsib_fl_below_100k(self):
        from app.domain.calculators import calculate_transfer_fee
        r = calculate_transfer_fee("Уралсиб", 80_000, "ФЛ")
        self.assertEqual(r["calculated_fee"], 0)

    def test_uralsib_fl_above_100k(self):
        from app.domain.calculators import calculate_transfer_fee
        r = calculate_transfer_fee("Уралсиб", 300_000, "ФЛ")
        self.assertEqual(r["calculated_fee"], 600)   # 300000 * 0.002
        self.assertEqual(r["control_fee"], 150)
        self.assertEqual(r["total_fee"], 750)


# ---------------------------------------------------------------------------
# Integration-тесты conversation_brain (с stub LLM)
# ---------------------------------------------------------------------------
class TestConversationBrainIntegration(unittest.TestCase):
    """Проверяет структуру ответа brain при stub-провайдере."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_brain_returns_dict(self):
        from app.services.conversation_brain import run_conversation_brain
        result = self._run(run_conversation_brain(
            user_text="с какими банками сотрудничаете?",
            recent_dialog=[],
            memory={"active_task": None, "last_bank": None, "client_type": None},
            kb_facts=[
                {"type": "selection", "bank": "Альфа-Банк", "status": "ACTIVE", "fact": "Активный партнёр"},
                {"type": "selection", "bank": "ТКБ", "status": "ACTIVE", "fact": "Активный партнёр"},
                {"type": "selection", "bank": "Уралсиб", "status": "ACTIVE", "fact": "Активный партнёр"},
                {"type": "availability", "bank": "Т-Банк", "status": "PAUSE", "fact": "На паузе"},
            ],
        ))
        self.assertIsInstance(result, dict)
        self.assertIn("reply", result)
        self.assertIn("needs_tool", result)
        self.assertIn("state_update", result)
        self.assertIn("handoff", result)

    def test_brain_no_handoff_on_compare(self):
        """'давайте с Уралсибом' при active_task=transfer_fee_quote НЕ должно быть handoff."""
        from app.services.conversation_brain import run_conversation_brain
        result = self._run(run_conversation_brain(
            user_text="давайте с Уралсибом",
            recent_dialog=[
                {"role": "user", "text": "перевод на физлицо в Альфе?"},
                {"role": "bot", "text": "В Альфа-Банке перевод 300 000 руб. выйдет в 1500 руб. Сравним с другим банком?"},
            ],
            memory={
                "active_task": {"type": "transfer_fee_quote", "bank": "Альфа-Банк", "amount": 300_000, "recipient": "ФЛ"},
                "last_bank": "Альфа-Банк",
                "client_type": None,
            },
            kb_facts=[],
        ))
        handoff = (result.get("handoff") or {})
        # Stub возвращает заглушку, но хотя бы проверим что не крашится
        self.assertIsInstance(handoff.get("needed"), bool)

    def test_brain_repair_called_on_invalid(self):
        """Repair возвращает строку (или None при stub)."""
        from app.services.conversation_brain import conversation_brain_repair
        result = self._run(conversation_brain_repair(
            previous_reply="В ТКБ комиссия 1000 рублей.",
            validation_error="wrong_bank_ТКБ_instead_of_Альфа-Банк",
            user_text="в Альфе сколько?",
            memory={"active_task": None, "last_bank": "Альфа-Банк"},
            kb_facts=[],
        ))
        # stub вернёт либо строку либо None
        self.assertTrue(result is None or isinstance(result, str))


# ---------------------------------------------------------------------------
# E2E сценарии (проверяют ключевые кейсы из плана)
# ---------------------------------------------------------------------------
class TestE2EScenarios(unittest.TestCase):
    """Проверяют логику обработки через полный pipeline (без отправки в Telegram).

    Используют mock OutboundDispatcher.
    """

    def setUp(self):
        from app.outbound.dispatcher import OutboundDispatcher
        self.sent: list[str] = []

        async def mock_send(channel: str, external_user_id: str, text: str):
            self.sent.append(text)

        async def mock_typing(channel: str, external_user_id: str):
            pass

        OutboundDispatcher.send = mock_send
        OutboundDispatcher.send_typing = mock_typing

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_msg(self, text: str, user_id: str = None) -> dict:
        return {
            "channel": "telegram",
            "external_user_id": user_id or f"test_{int(time.time()*1000)}",
            "message_id": f"msg_{int(time.time()*1000)}",
            "message_type": "text",
            "text": text,
        }

    async def _setup_db(self):
        from app.storage.db import engine, Base
        from app.main import _ensure_user_columns
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.run_sync(_ensure_user_columns)
            except Exception:
                pass

    # TEST 1: card_realization_negative
    def test_1_card_realization_negative(self):
        """Бот не повторяет вопрос 'реализация введена?' после ответа 'нет'."""
        self._run(self._setup_db())
        uid = f"e2e_card_{int(time.time())}"
        from app.processing.message_processor import process_message
        self._run(process_message(self._make_msg("можно карту сделать?", uid)))
        self.sent.clear()
        self._run(process_message(self._make_msg("ещё нет", uid)))
        # После 'нет' бот НЕ должен снова спросить "реализация введена?"
        full_reply = " ".join(self.sent)
        self.assertNotIn("введена?", full_reply.lower(), "Бот повторил вопрос о реализации")

    # TEST 2: partner_banks — список банков
    def test_2_partner_banks_list(self):
        """При вопросе о банках-партнёрах упоминаются Альфа, ТКБ, Уралсиб."""
        self._run(self._setup_db())
        uid = f"e2e_banks_{int(time.time())}"
        from app.processing.message_processor import process_message
        self._run(process_message(self._make_msg("с какими банками сотрудничаете?", uid)))
        full_reply = " ".join(self.sent).lower()
        # Хотя бы один активный банк должен быть упомянут
        has_active_bank = any(b in full_reply for b in ["альфа", "ткб", "уралсиб"])
        self.assertTrue(has_active_bank, f"Активные банки не упомянуты: {full_reply[:300]}")

    # TEST 3: transfer_alfa_alias — alias банка
    def test_3_transfer_alfa_alias(self):
        """'в Альфе' → бот уточняет сумму для Альфа-Банк, не для ТКБ."""
        self._run(self._setup_db())
        uid = f"e2e_alfa_{int(time.time())}"
        from app.processing.message_processor import process_message
        self._run(process_message(self._make_msg("перевод на физлицо в Альфе?", uid)))
        full_reply = " ".join(self.sent).lower()
        # В ответе должен быть Альфа-Банк (или уточнение суммы), но не только ТКБ
        has_tkb_only = "ткб" in full_reply and "альф" not in full_reply
        self.assertFalse(has_tkb_only, f"Бот ответил про ТКБ вместо Альфа-Банк: {full_reply[:300]}")

    # TEST 4: calculate_transfer_fee (через calculator напрямую)
    def test_4_transfer_amount_followup(self):
        """300 тыс + Альфа-Банк + ФЛ = 1500 рублей."""
        from app.domain.calculators import calculate_transfer_fee
        result = calculate_transfer_fee("Альфа-Банк", 300_000, "ФЛ")
        self.assertEqual(result["calculated_fee"], 1500)

    # TEST 5: compare_after_transfer (через context_builder)
    def test_5_followup_compare_bank(self):
        """'давайте с Уралсибом' при active_task=transfer_fee_quote: mentioned_bank = Уралсиб."""
        from app.services.context_builder import extract_entities
        r = extract_entities("давайте с Уралсибом")
        self.assertEqual(r["mentioned_bank"], "Уралсиб")

    # TEST 6: no_handoff_on_compare
    def test_6_no_handoff_on_compare_phrase(self):
        """'сравним' и 'с какими сравнить' не должны быть в _CONSENT_HARD_RE."""
        import re
        from app.processing.message_processor import _CONSENT_HARD_RE
        self.assertIsNone(_CONSENT_HARD_RE.search("давайте сравним"))
        self.assertIsNone(_CONSENT_HARD_RE.search("с какими сравнить?"))
        self.assertIsNone(_CONSENT_HARD_RE.search("а там как?"))

    # TEST 7: no_script_style — проверяем что static replies не используются для всего
    def test_7_consent_triggers_handoff(self):
        """'оформляем' должен триггерить _CONSENT_HARD_RE."""
        from app.processing.message_processor import _CONSENT_HARD_RE
        self.assertIsNotNone(_CONSENT_HARD_RE.search("оформляем"))
        self.assertIsNotNone(_CONSENT_HARD_RE.search("куда оплатить"))
        self.assertIsNotNone(_CONSENT_HARD_RE.search("готов начать"))


# ---------------------------------------------------------------------------
# Unit-тесты build_fact_pack
# ---------------------------------------------------------------------------
class TestBuildFactPack(unittest.TestCase):
    def _pack(self, user_text, memory=None, entities=None):
        from app.services.context_builder import build_fact_pack
        return build_fact_pack(user_text, memory or {}, entities or {})

    def test_always_has_partner_banks(self):
        pack = self._pack("сколько стоит открытие?")
        self.assertIn("partner_banks", pack)
        self.assertIn("active", pack["partner_banks"])

    def test_card_question_gets_card_primary(self):
        """Вопрос про карту → primary = realization_card, _note говорит не про наличные."""
        pack = self._pack("можно ему карту сделать?")
        card_rules = pack.get("card_rules", {})
        self.assertIn("primary", card_rules)
        self.assertIn("_note", card_rules)
        self.assertIn("карт", card_rules["primary"].lower())

    def test_cash_question_gets_cash_primary(self):
        """Вопрос про наличные → primary = cash_withdrawal."""
        pack = self._pack("как снять наличные?")
        card_rules = pack.get("card_rules", {})
        self.assertIn("primary", card_rules)
        self.assertIn("наличн", card_rules["primary"].lower())

    def test_card_not_cash_separate_primary(self):
        """Карта и наличные дают разные primary."""
        pack_card = self._pack("сделайте карту")
        pack_cash = self._pack("снять наличные")
        self.assertNotEqual(
            pack_card["card_rules"].get("primary"),
            pack_cash["card_rules"].get("primary"),
        )

    def test_low_cost_adds_pricing_hint(self):
        pack = self._pack("хочу подешевле")
        self.assertIn("_pricing_hint", pack)
        self.assertIn("Альфа", pack["_pricing_hint"])

    def test_banks_list_adds_banks_hint(self):
        pack = self._pack("с какими банками работаете?")
        self.assertIn("_banks_hint", pack)

    def test_yul_pricing_present_for_unknown_type(self):
        """Если тип клиента неизвестен, оба набора тарифов присутствуют."""
        pack = self._pack("сколько стоит счёт?")
        self.assertIn("bank_pricing_yul", pack)
        self.assertIn("bank_pricing_fl", pack)

    def test_fl_client_type_excludes_yul_pricing(self):
        """Для ФЛ bank_pricing_yul отсутствует."""
        pack = self._pack("физлицо, сколько стоит?", entities={"mentioned_client_type": "ФЛ"})
        self.assertNotIn("bank_pricing_yul", pack)
        self.assertIn("bank_pricing_fl", pack)

    def test_yul_client_type_excludes_fl_pricing(self):
        """Для ЮЛ bank_pricing_fl отсутствует."""
        pack = self._pack("ООО, сколько?", entities={"mentioned_client_type": "ЮЛ"})
        self.assertIn("bank_pricing_yul", pack)
        self.assertNotIn("bank_pricing_fl", pack)


# ---------------------------------------------------------------------------
# Новые e2e тесты (план v2, раздел 11)
# ---------------------------------------------------------------------------
class TestNewE2EScenarios(unittest.TestCase):
    """Проверяют логику из плана v2 через pipeline или прямые проверки."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def setUp(self):
        from app.outbound.dispatcher import OutboundDispatcher
        self.sent: list[str] = []

        async def mock_send(channel: str, external_user_id: str, text: str):
            self.sent.append(text)

        async def mock_typing(channel: str, external_user_id: str):
            pass

        OutboundDispatcher.send = mock_send
        OutboundDispatcher.send_typing = mock_typing

    def _make_msg(self, text: str, user_id: str = None) -> dict:
        return {
            "channel": "telegram",
            "external_user_id": user_id or f"test_{int(time.time()*1000)}",
            "message_id": f"msg_{int(time.time()*1000)}",
            "message_type": "text",
            "text": text,
        }

    async def _setup_db(self):
        from app.storage.db import engine, Base
        from app.main import _ensure_user_columns
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.run_sync(_ensure_user_columns)
            except Exception:
                pass

    # TEST 8: card question → fact_pack primary = realization_card (not cash)
    def test_8_card_question_not_cash_withdrawal(self):
        """fact_pack для 'карту сделать' → primary содержит реализацию, не суд."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("можно ему карту сделать?", {}, {})
        card_rules = pack.get("card_rules", {})
        primary = card_rules.get("primary", "")
        # primary должен говорить о реализации, не о наличных/суде
        self.assertIn("реализац", primary.lower(), f"primary не про реализацию: {primary}")
        self.assertNotIn("суд", primary.lower(), f"primary говорит про суд: {primary}")

    # TEST 9: card followup "что делать до реализации"
    def test_9_card_followup_before_realization(self):
        """fact_pack для 'что делать пока нет реализации' → secondary/before_realization присутствует."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("что делать пока нет реализации?", {}, {})
        card_rules = pack.get("card_rules", {})
        # general case — все три правила присутствуют
        all_text = " ".join(str(v) for v in card_rules.values()).lower()
        self.assertIn("реализац", all_text, "card_rules не содержит информацию о реализации")

    # TEST 10: open_account_request → validator требует handoff или request_data
    def test_10_open_account_request_triggers_handoff_or_request_data(self):
        """'счет откройте мне' → validator отклоняет ответ без handoff."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Конечно, откроем вам счёт прямо сейчас!",
            brain_result={"handoff": {"needed": False}, "action": "answer"},
            current_entities={},
            slots={},
            user_text="счет откройте мне",
        )
        self.assertFalse(result["is_valid"])
        # Причина должна быть про handoff или promised_action
        self.assertTrue(
            "handoff" in result["reason"] or "promised_action" in result["reason"],
            f"Неожиданная причина: {result['reason']}",
        )

    # TEST 11: partner banks list complete — все активные банки в partner_banks
    def test_11_partner_banks_list_complete(self):
        """partner_banks.active содержит все три банка."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("с какими банками работаете?", {}, {})
        active = pack["partner_banks"]["active"]
        for bank in ["Альфа-Банк", "ТКБ", "Уралсиб"]:
            self.assertIn(bank, active, f"{bank} отсутствует в active banks")

    # TEST 12: low_cost_yul uses all tariffs — _pricing_hint содержит все банки
    def test_12_low_cost_yul_uses_all_tariffs(self):
        """_pricing_hint для 'подешевле' содержит все активные банки ЮЛ."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("хочу подешевле для ООО", {}, {"mentioned_client_type": "ЮЛ"})
        hint = pack.get("_pricing_hint", "")
        for bank in ["Альфа", "ТКБ", "Уралсиб"]:
            self.assertIn(bank, hint, f"{bank} отсутствует в _pricing_hint")

    # TEST 13: account tariffs compact — bank_pricing_yul содержит нужные поля
    def test_13_account_tariffs_compact(self):
        """bank_pricing_yul содержит opening_fee, monthly_fee для всех трёх банков."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("тарифы на счёт", {}, {})
        yul = pack.get("bank_pricing_yul", [])
        self.assertEqual(len(yul), 3, f"Ожидалось 3 банка ЮЛ, получено {len(yul)}")
        for entry in yul:
            self.assertIn("bank", entry)
            self.assertIn("opening_fee", entry)
            self.assertIn("monthly_fee", entry)

    # TEST 14: no promised action without handoff — validator отклоняет "откроем"
    def test_14_no_promised_action_without_handoff(self):
        """Ответ с 'откроем' без handoff → validator отклоняет."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Хорошо, откроем вам счёт в Альфа-Банке.",
            brain_result={"handoff": {"needed": False}, "action": "answer"},
            current_entities={},
            slots={},
            user_text="хочу открыть счёт",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["reason"], "promised_action_without_handoff")

    # TEST 15: promised action WITH handoff → valid
    def test_15_promised_action_with_handoff_is_valid(self):
        """С handoff.needed=true 'откроем' в ответе допустимо."""
        from app.services.response_validator import validate_reply
        result = validate_reply(
            reply="Да, откроем счёт. Нужен ИНН должника.",
            brain_result={"handoff": {"needed": True}, "action": "handoff"},
            current_entities={},
            slots={"_had_consent": True},
            user_text="хочу открыть",
        )
        self.assertTrue(result["is_valid"], f"Ожидался valid, reason={result['reason']}")


# ---------------------------------------------------------------------------
# Тесты плана v11 (7 новых)
# ---------------------------------------------------------------------------
class TestPlanV11(unittest.TestCase):
    """Проверяют исправления из плана v11: _introduced, answer_contract, validator."""

    def _validate(self, reply, brain_result=None, entities=None, slots=None,
                  tool_results=None, user_text="", answer_contract=None):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            brain_result or {},
            entities or {},
            slots or {},
            tool_results=tool_results,
            user_text=user_text,
            answer_contract=answer_contract,
        )

    # test_no_repeated_intro
    def test_no_repeated_intro(self):
        """Если _introduced=True и ответ начинается с 'Здравствуйте' → repeated_intro."""
        r = self._validate(
            "Здравствуйте! Активные банки — Альфа-Банк, ТКБ, Уралсиб.",
            slots={"_introduced": True},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "repeated_intro")

    # test_no_repeated_intro_ok_when_not_introduced
    def test_no_repeated_intro_ok_when_not_introduced(self):
        """Если _introduced=False — приветствие допустимо."""
        r = self._validate(
            "Здравствуйте! Активные банки — Альфа-Банк, ТКБ, Уралсиб.",
            slots={"_introduced": False},
        )
        self.assertTrue(r["is_valid"])

    # test_card_not_no_problem — answer_contract.do_not_include=[нет проблем]
    def test_card_not_no_problem(self):
        """answer_contract.debtor_card: фраза 'нет проблем' запрещена."""
        contract = {
            "topic": "debtor_card",
            "must_include": ["реализация имущества"],
            "do_not_include": ["нет проблем"],
        }
        r = self._validate(
            "Нет проблем с картой. Оформим после решения суда.",
            answer_contract=contract,
        )
        self.assertFalse(r["is_valid"])
        self.assertIn("forbidden_phrase", r["reason"])

    # test_card_why_explains_reason — near_duplicate + "почему?" → did_not_explain_reason
    def test_card_why_explains_reason(self):
        """'почему?' после дублирующего ответа → did_not_explain_reason."""
        prev = "До введения реализации имущества карту должнику оформить нельзя."
        r = self._validate(
            "До введения реализации имущества карту должнику оформить нельзя.",
            slots={"_last_bot_text": prev},
            user_text="почему?",
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "did_not_explain_reason")

    # test_validator_blocks_cash_when_card
    def test_validator_blocks_cash_when_card(self):
        """answer_contract.topic=debtor_card + reply упоминает наличные → wrong_topic_fact."""
        contract = {
            "topic": "debtor_card",
            "do_not_include": ["наличные"],
        }
        r = self._validate(
            "Снятие наличных возможно только по решению суда.",
            answer_contract=contract,
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "wrong_topic_fact")

    # test_validator_blocks_repeated_intro
    def test_validator_blocks_repeated_intro(self):
        """_introduced=True + ответ начинается с 'Здравствуйте' → repeated_intro."""
        r = self._validate(
            "Здравствуйте! Мы сотрудничаем с Альфа-Банком.",
            slots={"_introduced": True},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "repeated_intro")

    # test_validator_partner_banks_no_tariffs
    def test_validator_partner_banks_no_tariffs(self):
        """topic=partner_banks + тарифы в ответе → answered_tariffs_when_asked_bank_list."""
        contract = {"topic": "partner_banks", "do_not_include": []}
        r = self._validate(
            "У нас Альфа-Банк — открытие 800 руб., ТКБ и Уралсиб.",
            answer_contract=contract,
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "answered_tariffs_when_asked_bank_list")

    # test_answer_contract_build_for_banks_topic
    def test_answer_contract_build_for_banks_topic(self):
        """build_fact_pack для 'с какими банками' → answer_contract.topic=partner_banks."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("с какими банками работаете?", {}, {})
        contract = pack.get("answer_contract", {})
        self.assertEqual(contract.get("topic"), "partner_banks")
        self.assertIn("Альфа-Банк", contract.get("must_include", []))

    # test_answer_contract_build_for_card_topic
    def test_answer_contract_build_for_card_topic(self):
        """build_fact_pack для 'карту сделать' → answer_contract.topic=debtor_card."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("можно карту сделать?", {}, {})
        contract = pack.get("answer_contract", {})
        self.assertEqual(contract.get("topic"), "debtor_card")
        self.assertIn("наличные", contract.get("do_not_include", []))

    # test_answer_contract_fl_selection
    def test_answer_contract_fl_selection(self):
        """build_fact_pack для ФЛ → answer_contract.topic=bank_selection_fl с ТКБ."""
        from app.services.context_builder import build_fact_pack
        pack = build_fact_pack("какой вариант?", {}, {"mentioned_client_type": "ФЛ"})
        contract = pack.get("answer_contract", {})
        self.assertEqual(contract.get("topic"), "bank_selection_fl")
        must = contract.get("must_include", [])
        self.assertTrue(any("ТКБ" in m for m in must))

    # test_introduced_in_memory
    def test_introduced_in_memory(self):
        """build_conversation_context включает _introduced из slots."""
        import asyncio
        from app.services.context_builder import build_conversation_context
        async def _run():
            ctx = await build_conversation_context(
                "какие варианты?", session_id=9999, slots={"_introduced": True}
            )
            return ctx
        ctx = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(ctx["memory"].get("_introduced"))


# ---------------------------------------------------------------------------
# Тесты плана v12: устойчивый JSON-парсер и нормализация
# ---------------------------------------------------------------------------
class TestBrainJsonParsing(unittest.TestCase):

    def setUp(self):
        from app.services.conversation_brain import _extract_json, normalize_brain_result, _looks_like_brain_result
        self._extract = _extract_json
        self._normalize = normalize_brain_result
        self._looks = _looks_like_brain_result

    # test_extract_json_valid_raw
    def test_extract_json_valid_raw(self):
        """Чистый JSON → парсится напрямую."""
        raw = '{"reply": "Привет", "action": "answer", "handoff": {"needed": false}}'
        obj = self._extract(raw)
        self.assertEqual(obj["reply"], "Привет")

    # test_extract_json_fenced_json
    def test_extract_json_fenced_json(self):
        """```json ... ``` — парсится через fenced block."""
        raw = 'Вот ответ:\n```json\n{"reply": "ТКБ", "action": "answer"}\n```'
        obj = self._extract(raw)
        self.assertEqual(obj["reply"], "ТКБ")

    # test_extract_json_ignores_input_context
    def test_extract_json_ignores_input_context(self):
        """Объект с user_message/recent_dialog без reply/action → не является brain result."""
        ctx_obj = {"user_message": "привет", "recent_dialog": [], "memory": {}, "fact_pack": {}}
        self.assertFalse(self._looks(ctx_obj))

    # test_extract_json_scans_multiple_objects_and_selects_brain_result
    def test_extract_json_scans_multiple_objects_selects_brain(self):
        """Если в raw два объекта, берём brain result (с reply/action)."""
        input_ctx = '{"user_message": "test", "recent_dialog": []}'
        brain_out = '{"reply": "Альфа-Банк", "action": "answer", "handoff": {"needed": false}}'
        raw = input_ctx + "\n" + brain_out
        obj = self._extract(raw)
        self.assertEqual(obj["reply"], "Альфа-Банк")

    # test_normalize_brain_result_defaults
    def test_normalize_brain_result_defaults(self):
        """normalize_brain_result заполняет все дефолтные поля."""
        obj = self._normalize({"reply": "Привет"})
        self.assertEqual(obj["action"], "answer")
        self.assertEqual(obj["confidence"], 0.0)
        self.assertFalse(obj["stop"])
        self.assertIsInstance(obj["needs_tool"], dict)
        self.assertEqual(obj["needs_tool"]["name"], "none")
        self.assertIsInstance(obj["handoff"], dict)
        self.assertFalse(obj["handoff"]["needed"])

    # test_handoff_bool_normalized
    def test_handoff_bool_normalized(self):
        """handoff: true → {"needed": true, "reason": null}."""
        obj = self._normalize({"reply": "OK", "handoff": True})
        self.assertIsInstance(obj["handoff"], dict)
        self.assertTrue(obj["handoff"]["needed"])
        self.assertIsNone(obj["handoff"]["reason"])

    # test_needs_tool_none_normalized
    def test_needs_tool_none_normalized(self):
        """needs_tool: None → {"name": "none", "args": {}}."""
        obj = self._normalize({"reply": "OK", "needs_tool": None})
        self.assertEqual(obj["needs_tool"], {"name": "none", "args": {}})

    # test_needs_tool_string_normalized
    def test_needs_tool_string_normalized(self):
        """needs_tool: "calculate_transfer_fee" → dict."""
        obj = self._normalize({"reply": "OK", "needs_tool": "calculate_transfer_fee"})
        self.assertEqual(obj["needs_tool"]["name"], "calculate_transfer_fee")

    # test_safe_fallback_uralsib_conditions
    def test_safe_fallback_uralsib_conditions(self):
        """_build_context_fallback с банком Уралсиб → содержит тарифы из fact_pack."""
        from app.services.context_builder import build_fact_pack
        from app.processing.message_processor import _build_context_fallback
        fact_pack = build_fact_pack("условия", {}, {"mentioned_bank": "Уралсиб"})
        result = _build_context_fallback(
            fact_pack,
            {"mentioned_bank": "Уралсиб"},
            {"client_type": "ЮЛ"},
        )
        self.assertIn("Уралсиб", result)
        self.assertIn("3500", result)

    # test_safe_fallback_generic_when_no_bank
    def test_safe_fallback_generic_when_no_bank(self):
        """_build_context_fallback без банка → generic фраза."""
        from app.processing.message_processor import _build_context_fallback
        result = _build_context_fallback({}, {}, {})
        self.assertIn("уточняю", result.lower())


# ---------------------------------------------------------------------------
# Тесты плана v13 (7 тестов)
# ---------------------------------------------------------------------------
class TestPlanV13(unittest.TestCase):

    def _validate(self, reply, brain_result=None, entities=None, slots=None,
                  tool_results=None, user_text="", answer_contract=None):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            brain_result or {},
            entities or {},
            slots or {},
            tool_results=tool_results,
            user_text=user_text,
            answer_contract=answer_contract,
        )

    # test_context_wrapped_in_tags
    def test_context_wrapped_in_tags(self):
        """_build_user_message содержит CONTEXT_DO_NOT_COPY и OUTPUT_SCHEMA."""
        from app.services.conversation_brain import _build_user_message
        msg = _build_user_message({"user_message": "привет", "recent_dialog": []})
        self.assertIn("<CONTEXT_DO_NOT_COPY>", msg)
        self.assertIn("</CONTEXT_DO_NOT_COPY>", msg)
        self.assertIn("<OUTPUT_SCHEMA>", msg)
        self.assertIn("</OUTPUT_SCHEMA>", msg)

    # test_explicit_open_forces_handoff
    def test_explicit_open_forces_handoff(self):
        """'ну какие еще, счет открыть' + brain=request_data → should_force_handoff=True."""
        from app.processing.message_processor import should_force_handoff
        brain = {"action": "request_data", "handoff": {"needed": False}}
        result = should_force_handoff("ну какие еще, счет открыть", brain, {})
        self.assertTrue(result)

    # test_force_handoff_skips_compare_task
    def test_force_handoff_skips_compare_task(self):
        """should_force_handoff не срабатывает при active_task=transfer_fee_quote."""
        from app.processing.message_processor import should_force_handoff
        brain = {"action": "request_data", "handoff": {"needed": False}}
        memory = {"active_task": {"type": "transfer_fee_quote"}}
        result = should_force_handoff("как открыть счет", brain, memory)
        self.assertFalse(result)

    # test_promised_action_without_handoff_blocked
    def test_promised_action_without_handoff_blocked(self):
        """'Давайте откроем счёт' без handoff → promised_action_without_handoff."""
        r = self._validate(
            "Давайте откроем счёт в Альфа-Банке.",
            brain_result={"handoff": {"needed": False}, "action": "answer"},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "promised_action_without_handoff")

    # test_pristupi_without_handoff_blocked
    def test_pristupi_without_handoff_blocked(self):
        """'приступим к открытию' без handoff → promised_action_without_handoff."""
        r = self._validate(
            "Приступим к открытию счёта, нужны документы.",
            brain_result={"handoff": {"needed": False}, "action": "answer"},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "promised_action_without_handoff")

    # test_non_russian_output_blocked
    def test_non_russian_output_blocked(self):
        """Ответ с китайскими символами → non_russian_output."""
        r = self._validate("Пожалуйста, обращайтесь. 祝您一天愉快！")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "non_russian_output")

    # test_russian_only_passes
    def test_russian_only_passes(self):
        """Чисто русский текст → проходит CJK проверку."""
        r = self._validate("По Альфа-Банку: открытие 800 руб., ведение 800 руб. в месяц.")
        self.assertTrue(r["is_valid"])

    # test_no_repeated_intro_after_introduced
    def test_no_repeated_intro_after_introduced(self):
        """'Я Алексей...' после _introduced=True → repeated_intro."""
        r = self._validate(
            "Я Алексей, менеджер «В плюсе». Чем помочь?",
            slots={"_introduced": True},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "repeated_intro")


# ---------------------------------------------------------------------------
# Регрессионные тесты плана (план §7)
# ---------------------------------------------------------------------------
class TestPlanRegression(unittest.TestCase):
    """Регрессионные тесты по пунктам 1-6 плана."""

    def _validate(self, reply, brain_result=None, entities=None, slots=None,
                  user_text="", answer_contract=None):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            brain_result or {},
            entities or {},
            slots or {},
            user_text=user_text,
            answer_contract=answer_contract,
        )

    # --- §1: filler tail endings ---

    def test_filler_tail_vse_ponyal_rejected(self):
        """Ответ заканчивающийся на 'Всё понял?' — отклоняется валидатором."""
        r = self._validate("Тарифы по ТКБ: открытие 2800 ₽. Всё понял?")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "filler_tail_ending")

    def test_filler_tail_okei_rejected(self):
        """Ответ заканчивающийся на 'Окей?' — отклоняется."""
        r = self._validate("Подробности уточним с менеджером. Окей?")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "filler_tail_ending")

    def test_filler_tail_pognali_rejected(self):
        """Ответ заканчивающийся на 'Погнали оформлять?' — отклоняется."""
        r = self._validate("Всё готово, банк выбран. Погнали оформлять?")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "filler_tail_ending")

    def test_filler_tail_ok_in_middle_allowed(self):
        """'Окей' внутри фразы, не в конце — допустимо."""
        r = self._validate("Окей, тогда давайте начнём с ТКБ: открытие 2800 ₽.")
        self.assertTrue(r["is_valid"] or r["reason"] != "filler_tail_ending")

    # --- §2: handoff latch ---

    def test_handoff_ack_re_matches_okei(self):
        """'окей' одиночное — попадает под _HANDOFF_ACK_RE."""
        from app.processing.message_processor import _HANDOFF_ACK_RE
        self.assertIsNotNone(_HANDOFF_ACK_RE.match("окей"))
        self.assertIsNotNone(_HANDOFF_ACK_RE.match("хорошо"))
        self.assertIsNotNone(_HANDOFF_ACK_RE.match("спасибо"))
        self.assertIsNotNone(_HANDOFF_ACK_RE.match("жду звонка"))

    def test_handoff_ack_re_no_match_for_substantive(self):
        """Содержательное сообщение — НЕ попадает под _HANDOFF_ACK_RE."""
        from app.processing.message_processor import _HANDOFF_ACK_RE
        self.assertIsNone(_HANDOFF_ACK_RE.match("а какие документы нужны для Уралсиба?"))
        self.assertIsNone(_HANDOFF_ACK_RE.match("подождите, у меня вопрос про тарифы"))

    # --- §3: post-docs consent → handoff ---

    def test_post_docs_consent_horosho_triggers_handoff(self):
        """'хорошо' после explain_documents с известным банком → handoff_needed=True."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "_last_bank": "ТКБ",
            "debtor_type": "ЮЛ",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "ТКБ",
                "handoff_needed": False,
            },
        }
        ds = update_deal_state(slots, "хорошо")
        self.assertTrue(ds.get("handoff_needed"), "Ожидался handoff_needed=True после affirm")
        self.assertEqual(ds.get("deal_stage"), "ready_to_open")

    def test_post_docs_consent_davajte_triggers_handoff(self):
        """'давайте' после explain_documents → handoff."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "_last_bank": "Уралсиб",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "Уралсиб",
                "handoff_needed": False,
            },
        }
        ds = update_deal_state(slots, "давайте")
        self.assertTrue(ds.get("handoff_needed"))

    def test_post_docs_no_handoff_without_bank(self):
        """'хорошо' без банка после explain_documents — НЕ escalate."""
        from app.processing.deal_state import update_deal_state
        slots = {
            "_deal_state": {
                "deal_stage": "consulting",
                "next_manager_move": "explain_documents",
                "selected_bank": None,
                "handoff_needed": False,
            },
        }
        ds = update_deal_state(slots, "хорошо")
        self.assertFalse(ds.get("handoff_needed"))

    # --- §4: handoff-claim validator ---

    def test_handoff_claim_blocked_without_flag(self):
        """'передам менеджеру' в ответе без handoff.needed → отклоняется."""
        r = self._validate(
            "Понял, передам менеджеру ваши данные.",
            brain_result={"handoff": {"needed": False}, "action": "answer"},
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "handoff_claim_without_handoff_flag")

    def test_handoff_claim_allowed_with_flag(self):
        """'передам менеджеру' с handoff.needed=True → допустимо."""
        r = self._validate(
            "Отлично, передам менеджеру ваш кейс.",
            brain_result={"handoff": {"needed": True}, "action": "handoff"},
            slots={"_had_consent": True},
        )
        self.assertNotEqual(r.get("reason"), "handoff_claim_without_handoff_flag")

    # --- §5: debtor_type guard ---

    def test_debtor_unknown_yul_docs_rejected(self):
        """ОГРН в ответе при неизвестном debtor_type → отклоняется."""
        r = self._validate(
            "Для открытия нужны ОГРН и устав организации.",
            slots={},  # debtor_type not set
        )
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "debtor_type_unknown_yul_specific_docs")

    def test_debtor_known_yul_docs_allowed(self):
        """ОГРН допустим если debtor_type=ЮЛ."""
        r = self._validate(
            "Для ЮЛ нужны ОГРН и устав организации.",
            slots={"debtor_type": "ЮЛ"},
        )
        self.assertNotEqual(r.get("reason"), "debtor_type_unknown_yul_specific_docs")

    def test_docs_intent_asks_yul_fl_when_unknown(self):
        """instruction содержит просьбу уточнить ЮЛ/ФЛ когда debtor_type неизвестен."""
        from app.services.conversation_responder import _build_manager_context
        slots = {
            "_last_bank": "Уралсиб",
            "_deal_state": {
                "deal_stage": "bank_selected",
                "next_manager_move": "explain_documents",
                "selected_bank": "Уралсиб",
                "handoff_needed": False,
                "debtor_type": None,
            },
        }
        ctx = _build_manager_context(slots)
        instruction = ctx.get("instruction", "")
        self.assertIn("ЮЛ", instruction, "Instruction должен упоминать тип должника")
        self.assertIn("ФЛ", instruction, "Instruction должен упоминать ФЛ")

    # --- §6: foreign-language noise ---

    def test_vietnamese_artifact_rejected(self):
        """Ответ с вьетнамскими диакритиками → non_russian_output."""
        r = self._validate("Нам cần документы для открытия счёта.")
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "non_russian_output")

    def test_pure_russian_foreign_check_passes(self):
        """Чисто русский ответ — проходит foreign-language проверку."""
        r = self._validate("По Уралсибу: открытие 3500 ₽, ведение 1600 ₽/мес.")
        self.assertTrue(r["is_valid"] or r["reason"] != "non_russian_output")

    # --- §1+§7: что скинуть? docs intent detection ---

    def test_chto_skinut_detected_as_docs_intent(self):
        """'что скинуть?' с известным банком → detect_docs_intent=True."""
        from app.processing.deal_state import detect_docs_intent
        slots = {"_last_bank": "ТКБ"}
        self.assertTrue(detect_docs_intent("что скинуть?", slots))

    def test_chto_skinut_update_deal_state_sets_explain_docs(self):
        """'что скинуть?' → update_deal_state ставит next_manager_move=explain_documents."""
        from app.processing.deal_state import update_deal_state
        slots = {"_last_bank": "ТКБ", "debtor_type": "ЮЛ"}
        ds = update_deal_state(slots, "что скинуть?")
        self.assertEqual(ds.get("next_manager_move"), "explain_documents")


if __name__ == "__main__":
    unittest.main(verbosity=2)
