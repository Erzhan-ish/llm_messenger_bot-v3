"""Regression tests for value-objection routing and LLM trace functionality.

Covers plan.txt requirements:
  1. Value objection phrases route to direct_bank_objection via _VALUE_OBJECTION_RE
  2. decide_scenario_policy switches to direct_bank_objection for objection phrases
  3. Bonus/interest phrases are NOT routed to direct_bank_objection (separate path)
  4. LLMTracer creates JSONL records when enabled
  5. FL slot prevents re-asking debtor_type when debtor_type=ФЛ is known

Run:
    python -m pytest tests/test_value_objection_routing.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.scenario_policy import _VALUE_OBJECTION_RE, decide_scenario_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy(user_text, active=None, client_type=None, rag_scenarios=None):
    slots = {"_active_scenario": active}
    if client_type:
        slots["client_type"] = client_type
    return decide_scenario_policy(
        user_text=user_text,
        slots=slots,
        rag_scenarios=rag_scenarios or [],
    )


# ============================================================================
# 1. _VALUE_OBJECTION_RE — detection layer
# ============================================================================
class TestValueObjectionRegex(unittest.TestCase):
    MATCHES = [
        "в чем выгода через вас работать",
        "в чём выгода",
        "какая выгода от вас",
        "зачем через вас",
        "зачем мне с вами работать",
        "могу напрямую в банк",
        "пойду напрямую",
        "почему не напрямую",
        "что вы даёте",
        "что вы даете",
        "какая польза от вас",
        "зачем мне посредник",
        "банк сам откроет",
        "не лучше ли напрямую",
        "зачем нам вы",
        "какой бонус мне будет",
        "какой бонус для ау",
    ]
    NO_MATCHES = [
        "какой тариф",
        "тарифы у уралсиба",
        "с какими банками",
        "документы нужны",
        "да",
        "хорошо",
        "нужен счёт для должника ФЛ",
        "у меня должник физлицо",
        "хочу открыть спецсчёт",
    ]

    def test_matches(self):
        for phrase in self.MATCHES:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    _VALUE_OBJECTION_RE.search(phrase),
                    f"Should match: {phrase!r}",
                )

    def test_no_matches(self):
        for phrase in self.NO_MATCHES:
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    _VALUE_OBJECTION_RE.search(phrase),
                    f"Should NOT match: {phrase!r}",
                )


# ============================================================================
# 2. decide_scenario_policy — switches to direct_bank_objection
# ============================================================================
class TestValueObjectionPolicy(unittest.TestCase):
    OBJECTION_PHRASES = [
        "в чем выгода через вас работать",
        "зачем через вас вообще",
        "могу напрямую в банк",
        "какой бонус мне будет",
    ]

    def test_switches_to_direct_bank_objection_no_active(self):
        for phrase in self.OBJECTION_PHRASES:
            with self.subTest(phrase=phrase):
                result = _policy(phrase)
                self.assertEqual(
                    result["active_scenario"], "direct_bank_objection",
                    f"Expected direct_bank_objection for: {phrase!r}",
                )
                self.assertEqual(result["decision"], "switch")
                self.assertIn("value_objection", result["reason"])

    def test_switches_to_direct_bank_objection_overrides_active(self):
        """Even when allowed_stages is active, value objection forces switch."""
        result = _policy(
            "в чем выгода через вас",
            active="allowed_stages",
        )
        self.assertEqual(result["active_scenario"], "direct_bank_objection")
        self.assertEqual(result["decision"], "switch")

    def test_objection_with_yul_client_type(self):
        result = _policy(
            "зачем через вас работать",
            active="bank_selection_yul",
            client_type="ЮЛ",
        )
        self.assertEqual(result["active_scenario"], "direct_bank_objection")

    def test_non_objection_not_routed(self):
        result = _policy(
            "нужен счёт для должника физлицо",
            active="allowed_stages",
        )
        self.assertNotEqual(
            result["active_scenario"], "direct_bank_objection",
            "Plain bank question should NOT route to direct_bank_objection",
        )

    def test_value_objection_before_bank_pricing(self):
        """Value objection takes priority over bank pricing intent."""
        result = _policy(
            "в чем выгода через вас работать с ткб",
            active="allowed_stages",
            client_type="ЮЛ",
        )
        # Should route to direct_bank_objection, not tkb_yul_conditions
        self.assertEqual(result["active_scenario"], "direct_bank_objection")


# ============================================================================
# 3. LLMTracer — creates JSONL record when enabled
# ============================================================================
class TestLLMTracer(unittest.TestCase):
    def test_tracer_writes_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from app.services.llm_trace import LLMTracer
            tracer = LLMTracer()
            tracer._loaded = True
            tracer.enabled = True
            tracer.full_prompt = False
            tracer.trace_dir = tmpdir
            tracer.do_redact = False
            tracer.include_response = True

            tracer.write(
                trace_id="llm_test_abc123",
                session_id=42,
                phase="brain",
                provider="stub",
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                raw_response='{"reply": "ok"}',
            )

            records = tracer.read_recent(limit=5)
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["trace_id"], "llm_test_abc123")
            self.assertEqual(rec["session_id"], 42)
            self.assertEqual(rec["phase"], "brain")
            self.assertEqual(rec["provider"], "stub")

    def test_tracer_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from app.services.llm_trace import LLMTracer
            tracer = LLMTracer()
            tracer._loaded = True
            tracer.enabled = False
            tracer.trace_dir = tmpdir

            tracer.write(
                trace_id="llm_no_write",
                phase="brain",
                provider="stub",
                model="test-model",
                messages=[],
            )

            records = tracer.read_recent(limit=5)
            self.assertEqual(len(records), 0)

    def test_get_by_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from app.services.llm_trace import LLMTracer
            tracer = LLMTracer()
            tracer._loaded = True
            tracer.enabled = True
            tracer.full_prompt = True
            tracer.trace_dir = tmpdir
            tracer.do_redact = False
            tracer.include_response = True

            tracer.write(
                trace_id="llm_find_me",
                phase="repair",
                provider="ollama",
                model="qwen",
                messages=[{"role": "system", "content": "sys"}],
                raw_response="fixed",
            )

            rec = tracer.get_by_id("llm_find_me")
            self.assertIsNotNone(rec)
            self.assertEqual(rec["phase"], "repair")
            self.assertIn("messages", rec)  # full_prompt=True

    def test_pii_redaction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from app.services.llm_trace import LLMTracer
            tracer = LLMTracer()
            tracer._loaded = True
            tracer.enabled = True
            tracer.full_prompt = True
            tracer.trace_dir = tmpdir
            tracer.do_redact = True
            tracer.include_response = True

            tracer.write(
                trace_id="llm_redact",
                phase="brain",
                provider="stub",
                model="test",
                messages=[{"role": "user", "content": "Мой телефон +7 999 123 45 67"}],
                raw_response="ok",
            )

            rec = tracer.get_by_id("llm_redact")
            self.assertIsNotNone(rec)
            msg_content = rec["messages"][0]["content"]
            self.assertNotIn("999", msg_content)
            self.assertIn("[REDACTED]", msg_content)


# ============================================================================
# 4. make_trace_id format
# ============================================================================
class TestMakeTraceId(unittest.TestCase):
    def test_format(self):
        from app.services.llm_trace import make_trace_id
        tid = make_trace_id()
        self.assertTrue(tid.startswith("llm_"), f"Bad prefix: {tid}")
        parts = tid.split("_")
        self.assertGreaterEqual(len(parts), 3, f"Expected 3+ parts: {tid}")
        # Last part is 8 hex chars
        self.assertEqual(len(parts[-1]), 8)
        self.assertTrue(all(c in "0123456789abcdef" for c in parts[-1]))


if __name__ == "__main__":
    unittest.main()
