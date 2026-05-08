"""Regression tests for app/processing/scenario_playbook.py.

Covers all 5 verification cases from the task:
  1. debtor_card_realization — yes answer
  2. debtor_card_realization — follow-up after confirmed realization
  3. allowed_stages — "что требуется"
  4. Gratitude close
  5. Real refusal / disinterest  (should NOT be intercepted by playbook)

Also verifies:
  - Log fields present for every action
  - Validator rejects contradictory / forbidden replies
  - scenario_policy._is_followup respects _known_slots and _pending_slot
  - NOT_INTERESTED guard stays IN_PROGRESS when _active_scenario is set

Run:
    python -m pytest tests/test_scenario_playbook.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.scenario_playbook import (
    SLOT_FORBIDDEN, SLOT_KNOWN, SLOT_NEXT_STEP, SLOT_PENDING,
    detect_gratitude, detect_what_next, detect_what_required, detect_yes_no,
    run_scenario_playbook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card_slots(**extra):
    return {"_active_scenario": "debtor_card_realization", **extra}


def _stages_slots(**extra):
    return {"_active_scenario": "allowed_stages", **extra}


def _run(text, slots):
    return run_scenario_playbook(text, slots)


def _log_keys(result):
    return set(result["log"].keys())


# ============================================================================
# 1. detect_yes_no
# ============================================================================
class TestDetectYesNo(unittest.TestCase):
    YES_CASES = ["да", "ага", "угу", "ну да", "конечно", "введена", "введено", "да, введена"]
    NO_CASES  = ["нет", "не введена", "ещё нет", "еще нет", "пока нет", "нее", "неа"]

    def test_yes_cases(self):
        for t in self.YES_CASES:
            with self.subTest(t=t):
                self.assertEqual(detect_yes_no(t), "yes", f"Expected 'yes' for {t!r}")

    def test_no_cases(self):
        for t in self.NO_CASES:
            with self.subTest(t=t):
                self.assertEqual(detect_yes_no(t), "no", f"Expected 'no' for {t!r}")

    def test_neutral_returns_none(self):
        for t in ["что мне делать", "хорошо", "счет", "расскажите подробнее"]:
            with self.subTest(t=t):
                self.assertIsNone(detect_yes_no(t))


# ============================================================================
# 2. detect_gratitude
# ============================================================================
class TestDetectGratitude(unittest.TestCase):
    GRATITUDE_CASES = [
        "хорошо спасибо",
        "хорошо, спасибо",
        "спасибо хорошо",
        "понял спасибо",
        "поняла спасибо",
        "понятно спасибо",
        "принял спасибо",
        "ладно спасибо",
        "ок спасибо",
        "благодарю",
        "хорошо\nспасибо",      # merged multi-line
        "Хорошо, Спасибо!",     # mixed case + punct
    ]
    NOT_GRATITUDE = [
        "хорошо",               # alone – not gratitude
        "нет спасибо",
        "что от меня требуется",
        "спасибо но у меня есть ещё вопрос",
    ]

    def test_gratitude_detected(self):
        for t in self.GRATITUDE_CASES:
            with self.subTest(t=t):
                self.assertTrue(detect_gratitude(t), f"Should be gratitude: {t!r}")

    def test_not_gratitude(self):
        for t in self.NOT_GRATITUDE:
            with self.subTest(t=t):
                self.assertFalse(detect_gratitude(t), f"Should NOT be gratitude: {t!r}")


# ============================================================================
# 3. detect_what_required
# ============================================================================
class TestDetectWhatRequired(unittest.TestCase):
    MATCH = [
        "что от меня требуется",
        "хорошо, что от меня требуется",
        "что нужно от меня",
        "от меня что нужно",
        "ну что от меня",
        "что надо от меня сделать",
        "что нужно для открытия",
        "что нужно для оформления",
        "какие нужны документы",
    ]
    NO_MATCH = [
        "где находится банк",
        "сколько стоит открытие",
        "добрый день",
    ]

    def test_match(self):
        for t in self.MATCH:
            with self.subTest(t=t):
                self.assertTrue(detect_what_required(t))

    def test_no_match(self):
        for t in self.NO_MATCH:
            with self.subTest(t=t):
                self.assertFalse(detect_what_required(t))


# ============================================================================
# 4. detect_what_next
# ============================================================================
class TestDetectWhatNext(unittest.TestCase):
    MATCH = [
        "что мне делать",
        "что мне делать получается",
        "что мне дальше",
        "дальше что",
        "следующий шаг",
        "что теперь",
    ]

    def test_match(self):
        for t in self.MATCH:
            with self.subTest(t=t):
                self.assertTrue(detect_what_next(t))


# ============================================================================
# 5. debtor_card_realization — full state machine
# ============================================================================
class TestDebtorCardPlaybook(unittest.TestCase):

    # 5a. Turn 1: initial question — playbook does nothing (LLM/fallback asks the question)
    def test_turn1_initial_question_is_skipped(self):
        r = _run("можно ему карту сделать?", _card_slots())
        self.assertEqual(r["action"], "skip")
        self.assertFalse(r["log"].get("llm_skipped"))

    def test_turn1_scenario_not_set_is_skipped(self):
        r = _run("да", {})          # no active_scenario
        self.assertEqual(r["action"], "skip")

    # 5b. Turn 2: user says "да" → store realization_started=True, deterministic reply
    def test_turn2_yes_stores_realization_true(self):
        r = _run("да", _card_slots(_known_slots={}))
        self.assertEqual(r["action"], "reply")
        updates = r["updates"]
        self.assertTrue(updates[SLOT_KNOWN]["realization_started"])
        self.assertIsNone(updates[SLOT_PENDING])

    def test_turn2_yes_reply_mentions_documents(self):
        r = _run("да", _card_slots(_known_slots={}))
        reply = r["reply"].lower()
        self.assertIn("финансовый управляющий", reply)
        self.assertIn("документ", reply)

    def test_turn2_yes_reply_does_not_say_wait(self):
        r = _run("да", _card_slots(_known_slots={}))
        reply = r["reply"].lower()
        self.assertNotIn("дождитесь введения реализации", reply)
        self.assertNotIn("реализация уже введена", reply)

    def test_turn2_yes_sets_forbidden_actions(self):
        r = _run("да", _card_slots(_known_slots={}))
        forbidden = r["updates"].get(SLOT_FORBIDDEN) or []
        self.assertTrue(any("дождитесь" in f for f in forbidden))

    def test_turn2_yes_sets_required_next_step(self):
        r = _run("да", _card_slots(_known_slots={}))
        self.assertEqual(r["updates"][SLOT_NEXT_STEP], "explain_card_opening_process")

    def test_turn2_log_shows_yn_yes(self):
        r = _run("да", _card_slots(_known_slots={}))
        self.assertEqual(r["log"]["yes_no_answer_detected"], "yes")
        self.assertTrue(r["log"]["llm_skipped"])

    # 5c. Turn 2: user says "нет" → realization_started=False, short reply
    def test_turn2_no_stores_realization_false(self):
        r = _run("нет", _card_slots(_known_slots={}))
        self.assertEqual(r["action"], "reply")
        self.assertFalse(r["updates"][SLOT_KNOWN]["realization_started"])

    def test_turn2_no_reply_says_cannot_open(self):
        r = _run("нет", _card_slots(_known_slots={}))
        reply = r["reply"].lower()
        self.assertIn("нельзя", reply)

    def test_turn2_no_reply_does_not_ask_question_again(self):
        r = _run("нет", _card_slots(_known_slots={}))
        self.assertNotIn("реализация уже введена?", r["reply"].lower())

    # 5d. Turn 3: follow-up after confirmed realization
    def test_turn3_what_to_do_returns_next_steps(self):
        r = _run("что мне делать получается",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertEqual(r["action"], "reply")
        reply = r["reply"].lower()
        self.assertIn("реализация введена", reply)

    def test_turn3_reply_has_numbered_steps(self):
        r = _run("что мне делать получается",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertIn("1.", r["reply"])

    def test_turn3_reply_does_not_say_wait(self):
        r = _run("что мне делать получается",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertNotIn("дождитесь введения реализации", r["reply"].lower())

    def test_turn3_что_дальше_variant(self):
        r = _run("что дальше",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertEqual(r["action"], "reply")

    def test_turn3_следующий_шаг_variant(self):
        r = _run("следующий шаг",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertEqual(r["action"], "reply")

    # 5e. Turn 3+: LLM still called for other messages (enrich, not reply)
    def test_unrelated_message_after_yes_goes_to_enrich(self):
        r = _run("а стоимость открытия какая",
                 _card_slots(_known_slots={"realization_started": True}))
        self.assertEqual(r["action"], "enrich")
        # Forbidden actions injected into fact_pack
        fp = r["fact_pack_additions"]
        self.assertIn("_forbidden_phrases", fp)

    # 5f. Log completeness
    def test_log_has_required_keys(self):
        r = _run("да", _card_slots(_known_slots={}))
        required = {
            "pending_slot_before", "known_slots_before", "required_next_step_before",
            "llm_skipped", "pending_slot_after", "known_slots_after",
            "required_next_step_after", "scenario_playbook_applied",
        }
        self.assertTrue(required.issubset(_log_keys(r)), f"Missing: {required - _log_keys(r)}")


# ============================================================================
# 6. allowed_stages — full state machine
# ============================================================================
class TestAllowedStagesPlaybook(unittest.TestCase):

    # 6a. Turn 1: confirms account opening allowed → enrich, not reply
    def test_turn1_confirmation_is_enrich(self):
        r = _run("у вас открываются счета на стадии наблюдения?", _stages_slots())
        self.assertEqual(r["action"], "enrich")
        self.assertFalse(r["log"].get("llm_skipped"))

    def test_turn1_sets_account_opening_allowed_in_known(self):
        r = _run("у вас открываются счета на стадии наблюдения?", _stages_slots())
        self.assertTrue(r["updates"][SLOT_KNOWN].get("account_opening_allowed"))

    # 6b. "что от меня требуется" → deterministic requirements reply
    def test_what_required_returns_reply(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("что от меня требуется", slots)
        self.assertEqual(r["action"], "reply")

    def test_what_required_reply_mentions_documents(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("что от меня требуется", slots)
        reply = r["reply"].lower()
        self.assertIn("инн", reply)
        self.assertIn("суд", reply)

    def test_what_required_reply_asks_debtor_type(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("что от меня требуется", slots)
        reply = r["reply"].lower()
        self.assertIn("юрлицо", reply.replace(" ", "") + reply)

    def test_what_required_reply_no_generic_fallback(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("что от меня требуется", slots)
        self.assertNotIn("секунду", r["reply"].lower())
        self.assertNotIn("уточните", r["reply"].lower())

    def test_what_required_phrase_variants(self):
        for phrase in ["что нужно от меня", "ну что от меня", "что нужно для открытия"]:
            with self.subTest(phrase=phrase):
                slots = _stages_slots(_known_slots={"account_opening_allowed": True})
                r = _run(phrase, slots)
                self.assertEqual(r["action"], "reply")

    def test_хорошо_что_требуется_variant(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("хорошо, что от меня требуется", slots)
        self.assertEqual(r["action"], "reply")

    def test_ну_что_нужно_от_меня_variant(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("ну что нужно от меня", slots)
        self.assertEqual(r["action"], "reply")

    def test_required_next_step_set_in_log(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("что от меня требуется", slots)
        self.assertEqual(r["log"]["required_next_step_after"], "explain_requirements_for_opening")
        self.assertTrue(r["log"]["llm_skipped"])

    # 6c. Generic repeat of original question → enrich (LLM handles it)
    def test_repeat_original_question_is_enrich(self):
        slots = _stages_slots(_known_slots={"account_opening_allowed": True})
        r = _run("счет на стадии наблюдения", slots)
        self.assertEqual(r["action"], "enrich")


# ============================================================================
# 7. Gratitude close
# ============================================================================
class TestGratitudeClose(unittest.TestCase):

    def test_хорошо_спасибо_returns_reply(self):
        slots = _card_slots(_known_slots={"realization_started": True})
        r = _run("хорошо\nспасибо", slots)
        self.assertEqual(r["action"], "reply")
        self.assertTrue(r["log"]["gratitude_close_detected"])

    def test_reply_is_polite_closing(self):
        r = _run("хорошо спасибо", _card_slots())
        self.assertIn("пожалуйста", r["reply"].lower())

    def test_reply_does_not_repeat_scenario_question(self):
        r = _run("хорошо спасибо", _card_slots())
        self.assertNotIn("реализация уже введена", r["reply"].lower())

    def test_reply_does_not_use_context_fallback_text(self):
        r = _run("хорошо спасибо", _card_slots())
        self.assertNotIn("секунду", r["reply"].lower())
        self.assertNotIn("уточняю", r["reply"].lower())

    def test_gratitude_takes_priority_over_active_scenario(self):
        # Even inside debtor_card scenario, gratitude short-circuits
        slots = _card_slots(_known_slots={"realization_started": True})
        r = _run("понял спасибо", slots)
        self.assertEqual(r["action"], "reply")
        self.assertTrue(r["log"]["gratitude_close_detected"])

    def test_gratitude_clears_pending_slot(self):
        r = _run("хорошо спасибо",
                 _card_slots(_pending_slot="realization_started"))
        self.assertIsNone(r["updates"][SLOT_PENDING])

    def test_благодарю_variant(self):
        r = _run("благодарю", {})
        self.assertEqual(r["action"], "reply")
        self.assertTrue(r["log"]["gratitude_close_detected"])

    def test_llm_skipped_true_for_gratitude(self):
        r = _run("хорошо спасибо", {})
        self.assertTrue(r["log"]["llm_skipped"])


# ============================================================================
# 8. Real refusal / disinterest — playbook must NOT intercept
# ============================================================================
class TestRealRefusal(unittest.TestCase):

    def test_не_интересно_is_skipped(self):
        r = _run("нет, мне не интересно", _card_slots())
        # Playbook returns skip — yes/no detection: "нет, мне не интересно" has extra text
        self.assertEqual(r["action"], "skip")

    def test_не_нужно_is_skipped(self):
        r = _run("не нужно, спасибо", {})
        # gratitude check: "не нужно, спасибо" does not match _GRATITUDE_RE (no keyword before/after)
        self.assertEqual(r["action"], "skip")

    def test_short_нет_in_card_scenario_is_detected(self):
        # Bare "нет" IS a valid no-answer to "Реализация введена?"
        r = _run("нет", _card_slots(_known_slots={}))
        self.assertEqual(r["action"], "reply")
        self.assertEqual(detect_yes_no("нет"), "no")

    def test_нет_мне_не_интересно_not_treated_as_no_answer(self):
        # Multi-word refusal is not caught by strict _NO_RE
        self.assertIsNone(detect_yes_no("нет, мне не интересно"))

    def test_no_active_scenario_always_skip(self):
        for text in ["да", "нет", "хорошо", "что делать"]:
            with self.subTest(text=text):
                r = _run(text, {})   # no _active_scenario, no gratitude
                # Only gratitude can produce reply without active_scenario
                if not detect_gratitude(text):
                    self.assertEqual(r["action"], "skip")


# ============================================================================
# 9. Response validator — forbidden_actions and realization_started checks
# ============================================================================
class TestValidatorForbiddenRules(unittest.TestCase):

    def _validate(self, reply, slots=None, user_text="что делать"):
        from app.services.response_validator import validate_reply
        return validate_reply(
            reply,
            {"action": "answer", "handoff": {"needed": False}},
            {},
            slots or {"_introduced": True},
            user_text=user_text,
        )

    def test_forbidden_action_rejected(self):
        slots = {
            "_introduced": True,
            "_forbidden_actions": ["дождитесь введения реализации"],
        }
        r = self._validate("Дождитесь введения реализации имущества.", slots)
        self.assertFalse(r["is_valid"])
        self.assertIn("forbidden_action", r["reason"])

    def test_realization_started_true_rejects_wait_phrase(self):
        slots = {
            "_introduced": True,
            "_known_slots": {"realization_started": True},
        }
        r = self._validate("Дождитесь судебного решения о введении реализации.", slots)
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["reason"], "contradicts_realization_started")

    def test_realization_started_true_accepts_correct_reply(self):
        slots = {
            "_introduced": True,
            "_known_slots": {"realization_started": True},
        }
        r = self._validate(
            "Да, карту можно оформить. Документы подписывает финансовый управляющий.",
            slots,
        )
        self.assertTrue(r["is_valid"])

    def test_realization_started_false_no_rejection(self):
        # realization=False — the "дождитесь" phrase is actually correct
        slots = {
            "_introduced": True,
            "_known_slots": {"realization_started": False},
        }
        r = self._validate(
            "Дождитесь судебного решения о введении реализации имущества.", slots
        )
        # Should NOT fail on contradicts_realization_started (realization is False, not True)
        self.assertNotEqual(r.get("reason"), "contradicts_realization_started")

    def test_forbidden_actions_none_does_not_crash(self):
        r = self._validate("Карту можно оформить.", {"_introduced": True, "_forbidden_actions": None})
        self.assertTrue(r["is_valid"])

    def test_forbidden_actions_empty_list_does_not_crash(self):
        r = self._validate("Карту можно оформить.", {"_introduced": True, "_forbidden_actions": []})
        self.assertTrue(r["is_valid"])


# ============================================================================
# 10. scenario_policy._is_followup respects _known_slots and _pending_slot
# ============================================================================
class TestScenarioPolicyFollowup(unittest.TestCase):

    def _followup(self, text, slots):
        from app.processing.scenario_policy import _is_followup
        return _is_followup(text, slots)

    def test_known_slots_nonempty_short_message_is_followup(self):
        slots = {"_known_slots": {"realization_started": True}}
        self.assertTrue(self._followup("что дальше", slots))

    def test_known_slots_nonempty_6word_is_followup(self):
        slots = {"_known_slots": {"realization_started": True}}
        self.assertTrue(self._followup("что мне делать получается теперь", slots))

    def test_known_slots_nonempty_7word_is_not_forced_followup(self):
        # 7 words exceeds the ≤6 threshold — not auto-followup (unless matched by regex)
        slots = {"_known_slots": {"realization_started": True}}
        result = self._followup("давайте посмотрим другой банк для открытия счета", slots)
        # should NOT be forced follow-up by _known_slots check (7 words > 6)
        self.assertFalse(result)

    def test_pending_slot_short_message_is_followup(self):
        slots = {"_pending_slot": "realization_started"}
        self.assertTrue(self._followup("да введена", slots))

    def test_empty_known_slots_short_message_not_auto_followup(self):
        # "ещё вопрос" is short but matches no FOLLOWUP/CONFIRM/CONFUSION pattern.
        # Without known_slots or pending_slot it should NOT be forced into follow-up.
        self.assertFalse(self._followup("ещё вопрос", {}))

    def test_existing_confirm_exact_still_works(self):
        self.assertTrue(self._followup("да", {}))
        self.assertTrue(self._followup("ок", {}))

    def test_confusion_signal_still_works(self):
        self.assertTrue(self._followup("я же говорил что не про это", {}))


# ============================================================================
# 11. NOT_INTERESTED guard in state machine (unit-level check)
# ============================================================================
class TestNotInterestedGuard(unittest.TestCase):
    """Verify that _active_scenario keeps NOT_INTERESTED from firing."""

    def _state(self, text):
        from app.processing.state_detector import detect_state
        return detect_state(text)

    def test_bare_нет_without_entities_is_not_interested(self):
        from app.processing.state_detector import DialogState
        self.assertEqual(self._state("нет"), DialogState.NOT_INTERESTED)

    def test_guard_logic_with_active_scenario(self):
        from app.processing.state_detector import DialogState
        state = self._state("нет")
        slots = {"_active_scenario": "debtor_card_realization"}
        # Simulate the guard in message_processor
        if state in (DialogState.NOT_INTERESTED, DialogState.LATER) and (
            slots.get("_active_scenario")
        ):
            state = DialogState.IN_PROGRESS
        self.assertEqual(state, DialogState.IN_PROGRESS)


# ============================================================================
# 12. Boundary / regression edge cases
# ============================================================================
class TestEdgeCases(unittest.TestCase):

    def test_empty_text_does_not_crash(self):
        r = _run("", _card_slots(_known_slots={}))
        self.assertIn(r["action"], ("skip", "enrich", "reply"))

    def test_none_active_scenario_skips(self):
        r = _run("да", {"_active_scenario": None})
        self.assertEqual(r["action"], "skip")

    def test_unknown_active_scenario_skips(self):
        r = _run("да", {"_active_scenario": "some_unknown_scenario"})
        self.assertEqual(r["action"], "skip")

    def test_result_always_has_all_keys(self):
        for text, slots in [
            ("да", _card_slots(_known_slots={})),
            ("нет", _card_slots(_known_slots={})),
            ("что требуется", _stages_slots()),
            ("хорошо спасибо", {}),
            ("нет, мне не интересно", {}),
        ]:
            with self.subTest(text=text):
                r = _run(text, slots)
                self.assertIn("action", r)
                self.assertIn("reply", r)
                self.assertIn("updates", r)
                self.assertIn("fact_pack_additions", r)
                self.assertIn("log", r)

    def test_updates_is_always_dict(self):
        r = _run("какой-то случайный текст", _card_slots())
        self.assertIsInstance(r["updates"], dict)

    def test_fact_pack_additions_always_dict(self):
        r = _run("да", _card_slots(_known_slots={}))
        self.assertIsInstance(r["fact_pack_additions"], dict)

    def test_log_always_has_base_keys(self):
        base = {"pending_slot_before", "known_slots_before", "required_next_step_before"}
        for text in ["да", "нет", "хорошо спасибо", "что требуется", "нет интересно"]:
            with self.subTest(text=text):
                r = _run(text, _card_slots(_known_slots={}))
                self.assertTrue(base.issubset(_log_keys(r)), f"Missing keys for {text!r}")

    def test_gratitude_before_any_scenario(self):
        # gratitude without active_scenario still returns reply
        r = _run("хорошо спасибо", {})
        self.assertEqual(r["action"], "reply")

    def test_card_scenario_realization_none_not_yes_not_no_is_skip(self):
        # Generic question in card scenario before realization answered → skip
        r = _run("расскажите про карту подробнее", _card_slots(_known_slots={}))
        self.assertEqual(r["action"], "skip")

    def test_card_scenario_realization_true_unknown_question_is_enrich(self):
        # After realization confirmed, random question → enrich (LLM with constraints)
        r = _run("а какой банк лучше", _card_slots(_known_slots={"realization_started": True}))
        self.assertEqual(r["action"], "enrich")
        self.assertIn("_forbidden_phrases", r["fact_pack_additions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
