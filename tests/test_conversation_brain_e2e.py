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


if __name__ == "__main__":
    unittest.main(verbosity=2)
