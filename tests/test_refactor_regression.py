"""Regression tests for the refactored pipeline.

Covers 5 test blocks (deterministic paths only — no DB, no LLM):
  Block 1: Service/short replies
  Block 2: Clarify slot flow
  Block 3: Bank selection planning
  Block 4: Specific pricing planning
  Block 5: Handoff routing

All tests use rule-based paths; LLM-dependent paths are mocked where needed.
"""
from __future__ import annotations

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers — minimal stubs so imports don't blow up without full app env
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# BLOCK 1 — Service / short replies
# dialog_analyzer must route these via rule-based paths (no LLM)
# ---------------------------------------------------------------------------

class TestBlock1Service:
    """Rule-based routing for service/short messages."""

    def _classify(self, text: str) -> dict:
        from app.services.dialog_analyzer import _get_rule_based_decision
        return _get_rule_based_decision(text)

    def test_greeting(self):
        d = self._classify("привет")
        assert d is not None
        assert d["stage"] == "GREETING"
        assert d["query_mode"] == "service"
        assert d["needs_kb"] is False

    def test_greeting_formal(self):
        d = self._classify("Добрый день")
        assert d is not None
        assert d["stage"] == "GREETING"

    def test_thanks(self):
        d = self._classify("спасибо")
        assert d is not None
        assert d["stage"] == "THANKS"
        assert d["query_mode"] == "service"

    def test_ack_ok(self):
        d = self._classify("ок")
        assert d is not None
        assert d["stage"] == "ACK"
        assert d["query_mode"] == "service"

    def test_ack_understood(self):
        d = self._classify("понял")
        assert d is not None
        assert d["stage"] == "ACK"

    def test_intro_who_are_you(self):
        d = self._classify("вы кто?")
        assert d is not None
        assert d["stage"] == "INTRO"
        assert d["query_mode"] == "intro"

    def test_intro_what_company(self):
        d = self._classify("что за компания?")
        assert d is not None
        assert d["query_mode"] == "intro"


# ---------------------------------------------------------------------------
# BLOCK 1b — _plan_service mapping
# ---------------------------------------------------------------------------

class TestBlock1PlanService:
    """Plan builder returns correct service intent."""

    def _plan(self, stage: str) -> dict:
        from app.processing.message_processor import _make_base, _plan_service
        base = _make_base()
        return _plan_service(base, stage)

    def test_greeting_plan(self):
        p = self._plan("GREETING")
        assert p["action"] == "service"
        assert p["intent"] == "greeting"

    def test_ack_plan(self):
        p = self._plan("ACK")
        assert p["action"] == "service"
        assert p["intent"] == "ack"

    def test_thanks_plan(self):
        p = self._plan("THANKS")
        assert p["action"] == "service"
        assert p["intent"] == "thanks"

    def test_service_texts_have_greeting(self):
        from app.processing.message_processor import _SERVICE_TEXTS
        text = _SERVICE_TEXTS.get("greeting", "")
        assert "Алексей" in text or "алексей" in text.lower()
        assert len(text) > 0

    def test_render_service_no_llm(self):
        from app.processing.message_processor import render_manager_text
        plan = {"action": "service", "intent": "greeting"}
        text = _run(render_manager_text(plan))
        assert len(text) > 5
        assert "Алексей" in text


# ---------------------------------------------------------------------------
# BLOCK 2 — Clarify slot flow
# ---------------------------------------------------------------------------

class TestBlock2ClarifySlots:
    """Clarify routing and static variant selection."""

    def test_clarify_plan_has_question(self):
        from app.processing.message_processor import _make_base, _plan_clarify
        base = _make_base()
        slots = {}
        p = _plan_clarify(base, "bank_selection", "client_type", slots)
        assert p["action"] == "clarify"
        assert p["question_to_ask"] == "client_type"
        assert slots.get("_pending_question_type") == "client_type"

    def test_clarify_text_client_type(self):
        from app.processing.message_processor import _clarify_text
        plan = {"question_to_ask": "client_type"}
        text = _clarify_text(plan, seed="хочу открыть счет")
        assert "ИП" in text or "ООО" in text or "физ" in text.lower()

    def test_clarify_text_priority(self):
        from app.processing.message_processor import _clarify_text
        plan = {"question_to_ask": "priority"}
        text = _clarify_text(plan, seed="подберите")
        assert len(text) > 10

    def test_clarify_text_deterministic_same_seed(self):
        from app.processing.message_processor import _clarify_text
        plan = {"question_to_ask": "client_type"}
        t1 = _clarify_text(plan, seed="открыть счет")
        t2 = _clarify_text(plan, seed="открыть счет")
        assert t1 == t2

    def test_clarify_text_different_seeds_may_differ(self):
        from app.processing.message_processor import _clarify_text, _CLARIFY_VARIANTS
        plan = {"question_to_ask": "client_type"}
        variants = _CLARIFY_VARIANTS["client_type"]
        # Collect outputs for different seeds — at least 2 unique variants over 20 seeds
        results = {_clarify_text(plan, seed=str(i)) for i in range(20)}
        assert len(results) >= 2

    def test_render_clarify_no_llm(self):
        from app.processing.message_processor import render_manager_text
        plan = {"action": "clarify", "question_to_ask": "client_type", "_seed": "хочу счет"}
        text = _run(render_manager_text(plan))
        assert len(text) > 10

    def test_validate_plan_clarify_ok(self):
        from app.services.fact_validator import validate_plan
        p = {"action": "clarify", "question_to_ask": "client_type"}
        assert validate_plan(p)["is_valid"] is True

    def test_validate_plan_clarify_no_question_fails(self):
        from app.services.fact_validator import validate_plan
        p = {"action": "clarify", "question_to_ask": None}
        assert validate_plan(p)["is_valid"] is False

    def test_pending_slot_resolution_client_type_юл(self):
        from app.processing.slots import extract_runtime_slots
        slots = {"_pending_question_type": "client_type"}
        extract_runtime_slots("ООО", slots)
        assert slots.get("client_type") == "ЮЛ"
        assert "_pending_question_type" not in slots

    def test_pending_slot_resolution_priority_price(self):
        from app.processing.slots import extract_runtime_slots
        slots = {"_pending_question_type": "priority"}
        extract_runtime_slots("подешевле", slots)
        assert slots.get("priority_criteria") == "price"
        assert "_pending_question_type" not in slots

    def test_pending_slot_resolution_priority_speed(self):
        from app.processing.slots import extract_runtime_slots
        slots = {"_pending_question_type": "priority"}
        extract_runtime_slots("срочно", slots)
        assert slots.get("priority_criteria") == "speed"


# ---------------------------------------------------------------------------
# BLOCK 3 — Bank selection planning (mock facts, no DB)
# ---------------------------------------------------------------------------

def _make_active_candidate(bank: str, of: float, mf: float, speed: bool = False) -> dict:
    feat = "Онлайн открытие за 1 день" if speed else "Универсальный пакет"
    return {
        "bank": bank,
        "client_type": "ЮЛ",
        "status": "ACTIVE",
        "opening_fee": of,
        "monthly_fee": mf,
        "positioning": feat,
        "main_feature": feat,
        "features": [],
        "rank_score": 0.5,
    }


class TestBlock3BankSelection:
    """Plan builder for bank_selection mode."""

    def _mock_facts(self, candidates: list) -> dict:
        return {"all_found_banks": candidates}

    def _mock_facts_result(self, candidates: list, confidence: float = 0.8) -> dict:
        return {
            "facts": self._mock_facts(candidates),
            "confidence": confidence,
            "retrieval_reason": "top_matches",
            "matched_fields": [],
            "missing_fields": [],
            "source_chunks": [],
        }

    def test_single_candidate_gives_answer(self):
        from app.processing.message_processor import _make_base, _plan_bank_selection
        cand = _make_active_candidate("Альфа-Банк", 800, 1000)
        cand["rank_score"] = 0.5
        facts = self._mock_facts([cand])
        slots = {"client_type": "ЮЛ", "priority_criteria": None}
        base = _make_base("ЮЛ")
        p = _plan_bank_selection(base, facts, slots, "ЮЛ", None)
        assert p["action"] == "answer"
        assert p["bank"] == "Альфа-Банк"

    def test_multiple_candidates_gives_compare(self):
        from app.processing.message_processor import _make_base, _plan_bank_selection
        candidates = [
            _make_active_candidate("Альфа-Банк", 800, 1000),
            _make_active_candidate("ТКБ", 0, 500),
            _make_active_candidate("Уралсиб", 1000, 0),
        ]
        for c in candidates:
            c["rank_score"] = 0.5
        facts = self._mock_facts(candidates)
        slots = {"client_type": "ЮЛ", "priority_criteria": "price"}
        base = _make_base("ЮЛ")
        p = _plan_bank_selection(base, facts, slots, "ЮЛ", "price")
        assert p["action"] == "compare"
        assert len(p["candidates"]) <= 3

    def test_no_client_type_asks_clarify(self):
        from app.processing.message_processor import _make_base, _plan_bank_selection
        candidates = [_make_active_candidate("Альфа-Банк", 800, 1000)]
        candidates[0]["rank_score"] = 0.5
        facts = self._mock_facts(candidates)
        slots = {}
        base = _make_base()
        p = _plan_bank_selection(base, facts, slots, None, None)
        assert p["action"] == "clarify"
        assert p["question_to_ask"] == "client_type"

    def test_pause_banks_excluded(self):
        from app.processing.message_processor import _make_base, _plan_bank_selection
        paused = _make_active_candidate("Росбанк", 500, 500)
        paused["status"] = "PAUSE"
        active = _make_active_candidate("ТКБ", 0, 500)
        active["rank_score"] = 0.5
        facts = self._mock_facts([paused, active])
        slots = {"client_type": "ЮЛ"}
        base = _make_base("ЮЛ")
        p = _plan_bank_selection(base, facts, slots, "ЮЛ", None)
        bank_names = [c["bank"] for c in p.get("candidates", [])]
        if p.get("bank"):
            bank_names.append(p["bank"])
        assert "Росбанк" not in bank_names

    def test_unknown_status_excluded(self):
        from app.processing.message_processor import _make_base, _plan_bank_selection
        unknown = _make_active_candidate("МКБ", 0, 0)
        unknown["status"] = None  # unknown — must be excluded
        unknown["rank_score"] = 0.5
        facts = self._mock_facts([unknown])
        slots = {"client_type": "ЮЛ"}
        base = _make_base("ЮЛ")
        p = _plan_bank_selection(base, facts, slots, "ЮЛ", None)
        # No ACTIVE candidates → clarify or no_candidates
        assert p["action"] in ("clarify", "service")

    def test_validate_compare_plan(self):
        from app.services.fact_validator import validate_plan
        p = {
            "action": "compare",
            "candidates": [
                {"bank": "Альфа-Банк", "opening_fee": 800},
                {"bank": "ТКБ", "opening_fee": 0},
            ],
        }
        assert validate_plan(p)["is_valid"] is True

    def test_validate_compare_no_candidates_fails(self):
        from app.services.fact_validator import validate_plan
        p = {"action": "compare", "candidates": []}
        assert validate_plan(p)["is_valid"] is False


# ---------------------------------------------------------------------------
# BLOCK 4 — Specific pricing / factual planning
# ---------------------------------------------------------------------------

class TestBlock4SpecificPricing:
    """_plan_factual with mock bank_profile data."""

    def _mock_facts_result(self, bank: str, of=None, mf=None, confidence=0.85,
                            client_type="ФЛ") -> dict:
        profile = {
            "bank": bank,
            "client_type": client_type,
            "opening_fee": of,
            "monthly_fee": mf,
            "status": "ACTIVE",
            "docs": [],
            "constraints": [],
            "source_conflicts": [],
        }
        return {
            "facts": {**profile, "bank_profile": profile},
            "confidence": confidence,
            "retrieval_reason": "top_matches",
            "matched_fields": ["bank"],
            "missing_fields": [],
            "source_chunks": [],
        }

    def test_answer_plan_has_bank_and_items(self):
        from app.processing.message_processor import _make_base, _plan_factual
        fr = self._mock_facts_result("ТКБ", of=1500, mf=0, client_type="ФЛ")
        base = _make_base("ФЛ")
        slots = {}
        decision = {"action": "ANSWER"}
        p = _plan_factual(base, "specific_bank", fr, fr["facts"], slots, decision, "ФЛ", 0.85)
        assert p["action"] == "answer"
        assert p["bank"] == "ТКБ"
        assert any(i["label"] == "Открытие счёта" for i in p["items"])
        assert any(i["label"] == "Ведение счёта" for i in p["items"])

    def test_no_bank_low_confidence_asks_clarify(self):
        from app.processing.message_processor import _make_base, _plan_factual
        fr = {
            "facts": {"bank_profile": {"bank": None, "opening_fee": None, "monthly_fee": None}},
            "confidence": 0.15,
            "retrieval_reason": "low_score",
            "matched_fields": [],
            "missing_fields": [],
            "source_chunks": [],
        }
        base = _make_base()
        slots = {}
        decision = {"action": "ANSWER"}
        p = _plan_factual(base, "specific_bank", fr, fr["facts"], slots, decision, None, 0.15)
        assert p["action"] == "clarify"

    def test_validate_answer_with_bank_and_items(self):
        from app.services.fact_validator import validate_plan
        p = {
            "action": "answer",
            "bank": "Альфа-Банк",
            "items": [{"label": "Открытие счёта", "value": "800 руб."}],
            "candidates": [],
        }
        assert validate_plan(p)["is_valid"] is True

    def test_validate_answer_no_data_fails(self):
        from app.services.fact_validator import validate_plan
        p = {"action": "answer", "bank": None, "items": [], "candidates": []}
        assert validate_plan(p)["is_valid"] is False

    def test_render_prompt_contains_bank(self):
        from app.llm.prompts.manager.loader import build_render_prompt
        plan = {
            "action": "answer",
            "intent": "specific_bank",
            "bank": "Альфа-Банк",
            "client_type": "ФЛ",
            "items": [{"label": "Открытие счёта", "value": "800 руб."}],
            "candidates": [],
            "docs": [],
            "constraints": [],
            "question_to_ask": None,
            "status": "ACTIVE",
        }
        prompt = build_render_prompt(plan)
        assert "Альфа-Банк" in prompt
        assert "800" in prompt

    def test_render_prompt_number_whitelist_raw_digits(self):
        from app.llm.prompts.manager.loader import build_render_prompt
        plan = {
            "action": "answer",
            "intent": "pricing",
            "bank": "ТКБ",
            "client_type": "ФЛ",
            "items": [
                {"label": "Открытие счёта", "value": "1500 руб."},
                {"label": "Ведение счёта",  "value": "0 руб./мес."},
            ],
            "candidates": [],
            "docs": [],
            "constraints": [],
            "question_to_ask": None,
            "status": "ACTIVE",
        }
        prompt = build_render_prompt(plan)
        # Prompt should list raw numbers without suffixes
        assert "1500" in prompt
        # "0" may appear in various contexts — just check the whitelist line has digits
        assert "ТОЛЬКО эти числа" in prompt or "Не называй никаких цифр" in prompt

    def test_render_prompt_no_data_forbids_banks(self):
        from app.llm.prompts.manager.loader import build_render_prompt
        plan = {
            "action": "answer",
            "intent": "pricing",
            "bank": None,
            "client_type": None,
            "items": [],
            "candidates": [],
            "docs": [],
            "constraints": [],
            "question_to_ask": None,
            "status": None,
        }
        prompt = build_render_prompt(plan)
        assert "Не называй ни одного банка" in prompt
        assert "Не называй никаких цифр" in prompt

    def test_compare_prompt_has_n_plus_one_limit(self):
        from app.llm.prompts.manager.loader import build_render_prompt
        plan = {
            "action": "compare",
            "intent": "bank_selection",
            "bank": None,
            "client_type": "ЮЛ",
            "items": [],
            "candidates": [
                {"bank": "Альфа-Банк", "opening_fee": 800, "monthly_fee": 1000},
                {"bank": "ТКБ",        "opening_fee": 0,   "monthly_fee": 500},
            ],
            "docs": [],
            "constraints": [],
            "question_to_ask": "priority",
            "status": None,
        }
        prompt = build_render_prompt(plan)
        assert "Максимум 3 предложений" in prompt  # n_banks=2 → max 3
        assert "самый" not in prompt or "оценочных" in prompt  # no superlatives instruction

    def test_compare_prompt_forbids_superlatives(self):
        from app.llm.prompts.manager.loader import build_render_prompt
        plan = {
            "action": "compare",
            "intent": "bank_selection",
            "bank": None,
            "client_type": "ЮЛ",
            "items": [],
            "candidates": [
                {"bank": "Альфа-Банк", "opening_fee": 800, "monthly_fee": 1000},
                {"bank": "ТКБ",        "opening_fee": 0,   "monthly_fee": 500},
                {"bank": "Уралсиб",    "opening_fee": 1000, "monthly_fee": 0},
            ],
            "docs": [],
            "constraints": [],
            "question_to_ask": "priority",
            "status": None,
        }
        prompt = build_render_prompt(plan)
        assert "лучший" in prompt  # should appear in forbidden list
        assert "рекомендую" in prompt  # in forbidden list


# ---------------------------------------------------------------------------
# BLOCK 5 — Handoff routing
# ---------------------------------------------------------------------------

class TestBlock5Handoff:
    """Rule-based handoff detection."""

    def _classify(self, text: str) -> dict:
        from app.services.dialog_analyzer import _get_rule_based_decision
        return _get_rule_based_decision(text)

    def test_explicit_human_request(self):
        d = self._classify("соедините с человеком")
        assert d is not None
        assert d["needs_handoff"] is True
        assert d["action"] == "HANDOFF"
        assert d["handoff_reason"] == "human_request"

    def test_operator_request(self):
        d = self._classify("мне нужен оператор")
        assert d is not None
        assert d["needs_handoff"] is True

    def test_manager_request(self):
        d = self._classify("позвоните мне")
        assert d is not None
        assert d["needs_handoff"] is True

    def test_consent_ready_to_open(self):
        d = self._classify("оформляем")
        assert d is not None
        assert d["needs_handoff"] is True
        assert d["handoff_reason"] == "ready_to_open"

    def test_consent_oformit(self):
        d = self._classify("хочу открыть")
        # "открыть" is in CONSENT_RE
        assert d is not None
        assert d["needs_handoff"] is True

    def test_consent_poekhali(self):
        d = self._classify("поехали")
        assert d is not None
        assert d["needs_handoff"] is True
        assert d["handoff_reason"] == "ready_to_open"

    def test_handoff_plan_action(self):
        from app.processing.message_processor import _make_base, _plan_handoff
        base = _make_base()
        p = _plan_handoff(base, "service", "human_request")
        assert p["action"] == "handoff"
        assert p["handoff_reason"] == "human_request"

    def test_render_handoff_no_llm(self):
        from app.processing.message_processor import render_manager_text
        plan = {"action": "handoff", "handoff_reason": "human_request"}
        text = _run(render_manager_text(plan))
        assert len(text) > 10

    def test_validate_handoff_always_valid(self):
        from app.services.fact_validator import validate_plan
        p = {"action": "handoff"}
        assert validate_plan(p)["is_valid"] is True


# ---------------------------------------------------------------------------
# BLOCK 5b — rank_score / speed signal (unit level)
# ---------------------------------------------------------------------------

class TestRankScore:
    """Deterministic rank scoring."""

    def _profile(self, status="ACTIVE", of=None, mf=None, positioning="", features=None) -> dict:
        return {
            "status": status,
            "opening_fee": of,
            "monthly_fee": mf,
            "positioning": positioning,
            "features": features or [],
        }

    def test_pause_scores_zero(self):
        from app.services.fact_retriever import _calc_rank_score
        p = self._profile(status="PAUSE", of=0, mf=0)
        assert _calc_rank_score(p, None) == 0.0

    def test_unknown_status_scores_zero(self):
        from app.services.fact_retriever import _calc_rank_score
        p = self._profile(status=None, of=0, mf=0)
        assert _calc_rank_score(p, None) == 0.0

    def test_active_base_score(self):
        from app.services.fact_retriever import _calc_rank_score
        p = self._profile(of=1000, mf=500)
        score = _calc_rank_score(p, None)
        assert score >= 0.4

    def test_price_priority_cheap_higher(self):
        from app.services.fact_retriever import _calc_rank_score
        cheap  = self._profile(of=0,    mf=0)
        expensive = self._profile(of=5000, mf=5000)
        assert _calc_rank_score(cheap, "price") > _calc_rank_score(expensive, "price")

    def test_speed_priority_keyword_wins(self):
        from app.services.fact_retriever import _calc_rank_score, _has_speed_signal
        fast = self._profile(positioning="Открытие счёта онлайн за 1 день")
        slow = self._profile(positioning="Стандартное обслуживание")
        assert _has_speed_signal(fast) is True
        assert _has_speed_signal(slow) is False
        assert _calc_rank_score(fast, "speed") > _calc_rank_score(slow, "speed")

    def test_has_speed_signal_keywords(self):
        from app.services.fact_retriever import _has_speed_signal
        assert _has_speed_signal({"positioning": "Быстро открываем", "features": []}) is True
        assert _has_speed_signal({"positioning": "Дистанционно", "features": []}) is True
        assert _has_speed_signal({"positioning": "Без посещения офиса", "features": []}) is True
        assert _has_speed_signal({"positioning": "Стандартный тариф", "features": []}) is False


# ---------------------------------------------------------------------------
# BLOCK 6 — Slots extraction (unit)
# ---------------------------------------------------------------------------

class TestSlotExtraction:
    """extract_runtime_slots unit tests."""

    def test_detect_yul(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("открытие счета для ООО", s)
        assert s.get("client_type") == "ЮЛ"

    def test_detect_ip(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("я ИП, нужен счёт", s)
        assert s.get("client_type") == "ИП"

    def test_detect_fl(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("для физического лица", s)
        assert s.get("client_type") == "ФЛ"

    def test_detect_price_priority(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("хочу подешевле", s)
        assert s.get("priority_criteria") == "price"

    def test_detect_speed_priority(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("нужно срочно", s)
        assert s.get("priority_criteria") == "speed"

    def test_detect_alfa_bank(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("тарифы Альфа банка", s)
        assert s.get("bank_name") == "Альфа-Банк"

    def test_detect_tkb(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("условия в ТКБ", s)
        assert s.get("bank_name") == "ТКБ"

    def test_detect_tinkoff_alias(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("тарифы тинькофф", s)
        assert s.get("bank_name") == "Т-Банк"

    def test_inn_labeled(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("мой ИНН 7707083893", s)
        assert s.get("inn") == "7707083893"

    def test_inn_unlabeled_10_digits(self):
        from app.processing.slots import extract_runtime_slots
        s = {}
        extract_runtime_slots("7707083893", s)
        assert s.get("inn") == "7707083893"
