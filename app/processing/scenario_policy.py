"""Dialog Scenario Policy — catalog-driven scoring replaces linear if-elif chain.

Architecture:
    RAG proposes  (scenario_matches from ScenarioFactIndex)
    Policy scores  ALL catalog scenarios, picks winner
    LLM formulates (filtered fact_pack)
    Validator enforces (validate_reply)

Hard rules (value objection, follow-up, bank switch) are implemented as
extreme score overrides so the scoring stays the single decision point.
"""
from __future__ import annotations

import re
from typing import Optional

from app.logging import logger

# ---------------------------------------------------------------------------
# Keep-active triggers
# ---------------------------------------------------------------------------

_CONFIRM_EXACT = frozenset({
    "да", "нет", "ок", "окей", "ладно", "хорошо", "понял", "поняла",
    "понятно", "принял", "принято", "угу", "ясно", "конечно", "согласен",
    "согласна", "ага", "нее", "неа", "ну да", "ну нет",
})

_FOLLOWUP_RE = re.compile(
    r"^\s*("
    r"что\s+дальше|как\s+дальше|дальше\s+что|и\s+что\s+теперь"
    r"|от\s+меня\s+что\s+(требуется|нужно|надо)"
    r"|что\s+(от\s+меня|мне\s+нужно\s+сделать|надо\s+сделать)"
    r"|следующий\s+шаг"
    r"|хорошо\s+что\s+дальше|ладно\s+что\s+дальше|и\s+что"
    r"|ваши\s+требования|что\s+требуется|что\s+нужно"
    r")\s*[?!.]?\s*$",
    re.I | re.U,
)

_CONFUSION_RE = re.compile(
    r"("
    r"причём\s+тут|причем\s+тут"
    r"|я\s+же\s+говорил|я\s+же\s+сказал"
    r"|я\s+ответил|я\s+не\s+про\s+это"
    r"|это\s+не\s+(тот\s+)?вопрос|это\s+не\s+то"
    r"|о\s+другом|не\s+об\s+этом"
    r"|я\s+имел\s+в\s+виду\s+другое"
    r"|вы\s+не\s+поняли|ты\s+не\s+понял"
    r"|не\s+о\s+том|не\s+то\s+имею"
    r")",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Bank / pricing explicit-switch triggers
# ---------------------------------------------------------------------------

# bank keyword (lowercase) → (yul_scenario, fl_scenario)
_BANK_SCENARIO_MAP: tuple[tuple[str, tuple[str, str]], ...] = (
    ("альфа",    ("alfabank_yul_conditions",  "bank_selection_fl")),
    ("ткб",      ("tkb_yul_conditions",       "tkb_fl_conditions")),
    ("уралсиб",  ("uralsib_yul_conditions",   "bank_selection_fl")),  # no Uralsib FL KB data
    ("т-банк",   ("tbank_yul_conditions",     "bank_selection_fl")),
    ("тинькофф", ("tbank_yul_conditions",     "bank_selection_fl")),
    ("мкб",      ("mkb_yul_conditions",       "bank_selection_fl")),
    ("росбанк",  ("rosbank_yul_conditions",   "bank_selection_fl")),
)

_PARTNER_BANKS_INTENT_RE = re.compile(
    r"\b("
    r"с\s+какими\s+банками|какие\s+банки|список\s+банков"
    r"|банки[-\s]партнёры|банки[-\s]партнеры"
    r"|сотрудничаете\b|с\s+кем\s+работаете"
    r")\b",
    re.I | re.U,
)

_PRICING_INTENT_RE = re.compile(
    r"\b("
    r"тариф\w*|тарификаци\w*"
    r"|стоимость\b|ценник\b|почём\b"
    r"|сколько\s+стоит"
    r"|условия\s+у\s+(них|него|банка\w*)"
    r"|условия\s+банк\w*"
    r")\b",
    re.I | re.U,
)

_VALUE_OBJECTION_RE = re.compile(
    r"\b("
    r"в\s+чём\s+выгода|в\s+чем\s+выгода"
    r"|какая\s+выгода|зачем\s+через\s+вас"
    r"|зачем\s+(мне\s+)?с\s+вами\s+работать"
    r"|могу\s+напрямую\s+в\s+банк|пойду\s+напрямую"
    r"|почему\s+не\s+напрямую|что\s+вы\s+(даёте|дает|даете)"
    r"|какая\s+польза\s+от\s+вас|зачем\s+(мне\s+)?посредник"
    r"|банк\s+сам\s+откроет|не\s+лучше\s+ли\s+напрямую"
    r"|зачем\s+нам\s+вы|какой\s+бонус\s+(мне|нам|будет|для\s+ау|для\s+меня)"
    r")\b",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Switch triggers
# ---------------------------------------------------------------------------

_NEW_TOPIC_RE = re.compile(
    r"\b("
    r"другой\s+вопрос|другой\s+случай|другой\s+должник|другой\s+человек"
    r"|ещё\s+(один\s+)?вопрос|еще\s+(один\s+)?вопрос"
    r"|новый\s+вопрос|по[-\s]другому|другое\s+дело"
    r"|а\s+вот\s+(ещё|еще)|теперь\s+про|теперь\s+о"
    r"|по\s+(?:физлицам|физ\s*лицам|юрлицам|юр\s*лицам|ип|нерезидентам|умершим)"
    r")\b",
    re.I | re.U,
)

# keyword → the scenario it strongly implies (for explicit domain-shift detection)
_DOMAIN_SHIFT_MAP: tuple[tuple[str, str], ...] = (
    ("нерезидент",      "non_resident"),
    ("иностранн",       "non_resident"),
    ("иностранец",      "non_resident"),
    ("умершего",        "deceased_fl"),
    ("умершей",         "deceased_fl"),
    ("умерший",         "deceased_fl"),
    ("покойн",          "deceased_fl"),
    ("посмертн",        "deceased_fl"),
    ("ликвидирован",    "liquidated_yul"),
    ("красная зона",    "red_zone_company"),
    ("красной зоны",    "red_zone_company"),
    ("красной зоне",    "red_zone_company"),
)

_FL_EXPLICIT_RE = re.compile(
    r"\b(физ\s*лиц[ауое]?\b|физлиц\w*|фл\b|физическое\s+лицо)\b",
    re.I | re.U,
)
_YUL_EXPLICIT_RE = re.compile(
    r"\b(юр\s*лиц[ауое]?\b|юрлиц\w*|юл\b|ооо\b|зао\b|организаци\w*|компани\w*)\b",
    re.I | re.U,
)

# Scenarios typed by client_type — used for incompatibility detection
_SCENARIO_CLIENT_TYPE: dict[str, str] = {
    "bank_selection_yul":           "ЮЛ",
    "bank_selection_yul_low_cost":  "ЮЛ",
    "alfabank_yul_conditions":      "ЮЛ",
    "tkb_yul_conditions":           "ЮЛ",
    "uralsib_yul_conditions":       "ЮЛ",
    "tbank_yul_conditions":         "ЮЛ",
    "mkb_yul_conditions":           "ЮЛ",
    "rosbank_yul_conditions":       "ЮЛ",
    "red_zone_company":             "ЮЛ",
    "liquidated_yul":               "ЮЛ",
    "bank_selection_fl":            "ФЛ",
    "tkb_fl_conditions":            "ФЛ",
    "uralsib_fl_conditions":        "ФЛ",
    "deceased_fl":                  "ФЛ",
    "internet_bank_fl_ip":          "ФЛ",
    "restructuring_ip":             "ИП",
}

# ---------------------------------------------------------------------------
# Decision helpers (pure functions)
# ---------------------------------------------------------------------------

def _is_bank_pricing_intent(text: str) -> bool:
    """True when the message explicitly asks about banks, tariffs, or a specific bank."""
    t_low = (text or "").lower()
    if _PARTNER_BANKS_INTENT_RE.search(text) or _PRICING_INTENT_RE.search(text):
        return True
    return any(kw in t_low for kw, _ in _BANK_SCENARIO_MAP)


def _is_followup(user_text: str, slots: dict) -> bool:
    t = (user_text or "").strip()
    cleaned = re.sub(r"[?!.,\s]+$", "", t.lower()).strip()
    if cleaned in _CONFIRM_EXACT:
        return True
    if _FOLLOWUP_RE.match(t):
        return True
    if _CONFUSION_RE.search(t):
        return True
    # Reply to a pending slot question — if very short it's almost certainly an answer
    pending = (slots.get("_pending_question") or slots.get("_pending_question_type")
               or slots.get("_pending_slot"))
    if pending and len(t.split()) <= 4:
        return True
    # Short message while scenario slots are being filled → follow-up.
    # Exception: explicit bank/pricing intent always gets its own switch, not treated as follow-up.
    if not _is_bank_pricing_intent(t) and slots.get("_known_slots") and len(t.split()) <= 6:
        return True
    return False


def _bank_pricing_switch_target(
    user_text: str, active_scenario: str, slots: dict
) -> Optional[str]:
    """Return target scenario when user explicitly asks about a bank or pricing.

    Priority:
      1. Explicit bank name in message → that bank's conditions scenario
      2. Bank in slots + pricing/conditions intent → that bank's conditions scenario
      3. Partner-banks list query → bank_selection_yul / partner_banks
      4. Generic tariffs query → bank_selection_yul / bank_selection_fl
    """
    t_low = (user_text or "").lower()
    ct = slots.get("client_type") or ""
    is_yul = ct in ("ЮЛ", "ИП")

    # 1. Explicit bank name in the message
    for kw, (yul_s, fl_s) in _BANK_SCENARIO_MAP:
        if kw in t_low:
            target = yul_s if is_yul else fl_s
            if target != active_scenario:
                return target

    # 2. Bank in slots + pricing or "условия" intent
    bank_in_focus = (
        slots.get("_current_bank_mention")
        or slots.get("bank_name")
        or slots.get("_last_bank")
    ) or ""
    if bank_in_focus and (
        _PRICING_INTENT_RE.search(user_text)
        or re.search(r"\bусловия\b", user_text, re.I | re.U)
    ):
        for kw, (yul_s, fl_s) in _BANK_SCENARIO_MAP:
            if kw in bank_in_focus.lower():
                target = yul_s if is_yul else fl_s
                if target != active_scenario:
                    return target

    # 3. Partner banks list
    if _PARTNER_BANKS_INTENT_RE.search(user_text):
        target = "bank_selection_yul" if is_yul else "partner_banks"
        if target != active_scenario:
            return target

    # 4. Generic tariffs/pricing
    if _PRICING_INTENT_RE.search(user_text):
        target = "bank_selection_yul" if is_yul else "bank_selection_fl"
        if target != active_scenario:
            return target

    return None


def _domain_shift_target(user_text: str, active_scenario: str) -> Optional[str]:
    """Return target scenario_id if a strong domain-shift keyword is present
    and the target differs from the active scenario. None if no shift."""
    t_low = (user_text or "").lower()
    for kw, target in _DOMAIN_SHIFT_MAP:
        if kw in t_low and target != active_scenario:
            return target
    return None


def _client_type_incompatible(user_text: str, active_scenario: str) -> bool:
    active_ct = _SCENARIO_CLIENT_TYPE.get(active_scenario)
    if active_ct is None:
        return False
    if active_ct == "ЮЛ" and _FL_EXPLICIT_RE.search(user_text):
        return True
    if active_ct == "ФЛ" and _YUL_EXPLICIT_RE.search(user_text):
        return True
    return False


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_S_INTENT_MATCH          = 2.0    # per trigger_intent matched by REGEX
_S_INTENT_MATCH_SEMANTIC = 0.8    # TASK 4 — semantic-only match scores less
_S_RAG_BONUS     = 1.5    # RAG found this scenario
_S_CONTINUITY    = 3.0    # currently active scenario
_S_FOLLOWUP      = 10.0   # follow-up → strong continuity keep
_S_VALUE_OBJ     = 100.0  # value objection via REGEX → hard route
_S_VALUE_OBJ_SEM = 10.0   # TASK 4 — semantic-only objection → modest boost only
_S_BANK_SWITCH   = 50.0   # explicit bank/pricing switch target
_S_DOMAIN_SHIFT  = 50.0   # domain-shift keyword detected
_S_CT_MISMATCH   = -999.0 # client type mismatch → hard block
_S_BLOCK_INTENT  = -999.0 # blocked_if_intents present
_S_NEW_TOPIC     = -3.0   # explicit "new topic" signal (reduces continuity)

# TASK 5 — Explicit account/bank/debtor_type terms override direct_bank_objection
_ACCOUNT_REQUEST_RE = re.compile(
    r"\b("
    r"нужен\s+счет|нужен\s+банк|банк\s+подобрать|подобрать\s+банк"
    r"|счет\s+открыть|открыть\s+счет|подешевле"
    r"|\bюл\b|\bю\.л\b|\bюрлиц\w*|\bооо\b|\bзао\b"
    r"|\bфл\b|\bф\.л\b|\bфизлиц\w*"
    r")\b",
    re.I | re.U,
)

# Maps bank_focus codes → which scenario IDs they signal (for slot boost check)
_BANK_FOCUS_SCENARIOS: dict[str, tuple[str, str]] = {
    "alfabank": ("alfabank_yul_conditions", "bank_selection_fl"),
    "tkb":      ("tkb_yul_conditions",      "tkb_fl_conditions"),
    "uralsib":  ("uralsib_yul_conditions",  "bank_selection_fl"),  # no Uralsib FL KB data
    "tbank":    ("tbank_yul_conditions",    "bank_selection_fl"),
    "mkb":      ("mkb_yul_conditions",      "bank_selection_fl"),
    "rosbank":  ("rosbank_yul_conditions",  "bank_selection_fl"),
}

# §8 — Conditions vs tariffs differentiation
_TARIFF_FOCUSED_SCENARIOS: frozenset[str] = frozenset({
    "bank_selection_yul", "bank_selection_fl", "bank_pricing_yul", "bank_pricing_fl",
    "bank_tariff_comparison", "bank_selection_yul_low_cost",
})
_CONDITIONS_FOCUSED_SCENARIOS: frozenset[str] = frozenset({
    "alfabank_yul_conditions", "tkb_yul_conditions", "uralsib_yul_conditions",
    "tbank_yul_conditions", "mkb_yul_conditions", "rosbank_yul_conditions",
    "tkb_fl_conditions", "uralsib_fl_conditions",
})
_S_CONDITIONS_CORRECTION = 60.0  # hard-route to conditions when user corrects "not tariffs"

# Secondary-intent → required_next_step when this scenario is primary
_SECONDARY_NEXT_STEP: dict[tuple[str, str], str] = {
    ("bank_selection_yul_low_cost", "tariff_comparison_requested"): "give_tariff_comparison",
    ("bank_selection_yul",          "tariff_comparison_requested"): "give_tariff_comparison",
    ("bank_selection_fl",           "tariff_comparison_requested"): "give_tariff_comparison",
    ("direct_bank_objection",       "bank_selection"):              "list_banks",
    ("direct_bank_objection",       "tariff_comparison_requested"): "give_tariff_comparison",
    ("bank_pricing_yul",            "bank_selection"):              "list_banks",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_scenario_score(
    spec,
    detected_intents: list[str],
    regex_intents: set,
    semantic_intents: set,
    slots: dict,
    active: Optional[str],
    rag_ids: list[str],
    is_fup: bool,
    is_value_obj_regex: bool,
    is_value_obj_semantic: bool,
    is_account_request: bool,
    bank_switch_tgt: Optional[str],
    domain_shift_tgt: Optional[str],
    is_new_topic: bool,
    bank_focus: Optional[str],
    user_text: str,
    stored_debtor_type: Optional[str] = None,
    is_conditions_correction: bool = False,
    is_conditions_intent: bool = False,
) -> float:
    score = 0.0
    sid = spec.scenario_id

    # TASK 4 — Base: matching trigger intents, score differs by source
    for intent in spec.trigger_intents:
        if intent in regex_intents:
            score += _S_INTENT_MATCH           # 2.0 for regex / exact
        elif intent in semantic_intents:
            score += _S_INTENT_MATCH_SEMANTIC  # 0.8 for semantic-only

    # Slot boosts (truthy check only)
    for slot_name, boost in spec.slot_boosts.items():
        if slots.get(slot_name):
            score += boost

    # RAG confirmation
    if sid in rag_ids:
        score += _S_RAG_BONUS

    # Active scenario continuity
    if sid == active:
        if is_fup:
            score += _S_FOLLOWUP
        elif is_new_topic:
            pass  # no continuity bonus — user explicitly changed topic
        else:
            # TASK 2 — Topic switch overrides: clear continuity when higher-priority intent present
            _drop_continuity = False
            # interest_or_bonus/bonus_interest overrides direct_bank_objection
            if sid in ("direct_bank_objection", "value_proposition", "value_followup"):
                if not is_value_obj_regex and not is_value_obj_semantic:
                    if "interest_or_bonus" in regex_intents or "interest_or_bonus" in semantic_intents:
                        _drop_continuity = True
            # timelines intent overrides tariff/bank-listing scenarios
            if sid in ("bank_selection_yul", "bank_selection_fl", "bank_pricing_yul", "bank_pricing_fl",
                       "bank_tariff_comparison", "bank_selection_yul_low_cost"):
                if "timelines" in regex_intents or "timelines" in semantic_intents:
                    _drop_continuity = True
            if not _drop_continuity:
                score += _S_CONTINUITY

    # TASK 4+5 — Value objection override: full bonus only for regex detection.
    # If account/bank terms present alongside, block the override entirely.
    if sid == "direct_bank_objection" and not is_account_request:
        if is_value_obj_regex:
            score += _S_VALUE_OBJ       # 100.0 — hard route
        elif is_value_obj_semantic:
            score += _S_VALUE_OBJ_SEM   # 10.0 — soft boost only

    # TASK 5 — If explicit account/bank/debtor_type terms present, strongly boost
    # bank_selection over direct_bank_objection (even if objection was active)
    if is_account_request and sid in (
        "bank_selection_yul", "bank_selection_fl", "bank_selection_yul_low_cost", "partner_banks",
    ):
        score += _S_BANK_SWITCH * 0.6  # +30 — outranks continuity and semantic objection

    # TASK 3 — yes_to_offer_bank_selection intent routes to bank_selection by debtor_type
    if "yes_to_offer_bank_selection" in regex_intents or "yes_to_offer_bank_selection" in semantic_intents:
        debtor_type = (slots.get("debtor_type") or slots.get("client_type") or "").upper()
        if debtor_type in ("ЮЛ", "ИП", "LEGAL_ENTITY") and sid == "bank_selection_yul":
            score += _S_BANK_SWITCH
        elif debtor_type in ("ФЛ", "INDIVIDUAL") and sid == "bank_selection_fl":
            score += _S_BANK_SWITCH

    if bank_switch_tgt and sid == bank_switch_tgt:
        score += _S_BANK_SWITCH

    if domain_shift_tgt and sid == domain_shift_tgt:
        score += _S_DOMAIN_SHIFT

    # Bank focus: boost the matching bank scenario
    if bank_focus and bank_focus in _BANK_FOCUS_SCENARIOS:
        is_yul = (slots.get("client_type") or "") in ("ЮЛ", "ИП")
        yul_s, fl_s = _BANK_FOCUS_SCENARIOS[bank_focus]
        if (is_yul and sid == yul_s) or (not is_yul and sid == fl_s):
            score += _S_BANK_SWITCH

    # Client type mismatch penalty (TASK 1: also applies from stored debtor_type)
    ct = _SCENARIO_CLIENT_TYPE.get(sid)
    if ct == "ЮЛ" and _FL_EXPLICIT_RE.search(user_text):
        score += _S_CT_MISMATCH
    elif ct == "ФЛ" and _YUL_EXPLICIT_RE.search(user_text):
        score += _S_CT_MISMATCH
    elif stored_debtor_type:
        # Persistent session debtor_type: penalize mismatched scenarios unless user explicitly switches
        if ct == "ЮЛ" and stored_debtor_type == "ФЛ" and not _YUL_EXPLICIT_RE.search(user_text):
            score += _S_CT_MISMATCH
        elif ct == "ФЛ" and stored_debtor_type in ("ЮЛ", "ИП") and not _FL_EXPLICIT_RE.search(user_text):
            score += _S_CT_MISMATCH

    # Blocked intents penalty
    for intent in spec.blocked_if_intents:
        if intent in detected_intents:
            score += _S_BLOCK_INTENT

    # §8 — Conditions correction: hard-route to conditions, block tariff scenarios
    if is_conditions_correction:
        if sid in _TARIFF_FOCUSED_SCENARIOS:
            score = -999.0   # hard block — overrides all accumulated score
        elif sid in _CONDITIONS_FOCUSED_SCENARIOS:
            score += _S_CONDITIONS_CORRECTION  # +60 ensures win over continuity
    elif is_conditions_intent:
        # Soft boost: conditions_details_requested scores above tariff continuity
        if sid in _CONDITIONS_FOCUSED_SCENARIOS:
            score += _S_BANK_SWITCH * 0.7   # +35 beats continuity (3) + intent match (2)
        elif sid in _TARIFF_FOCUSED_SCENARIOS:
            score -= _S_CONTINUITY           # cancel continuity if active tariff scenario

    # Priority tiebreaker (tiny offset)
    score += spec.priority * 0.01

    return score


def _get_required_next_step(
    scenario: Optional[str],
    detected_intents: list[str],
) -> Optional[str]:
    """Return required_next_step when a secondary intent should be served immediately."""
    if not scenario:
        return None
    for intent in detected_intents:
        step = _SECONDARY_NEXT_STEP.get((scenario, intent))
        if step:
            return step
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_scenario_policy(
    user_text: str,
    slots: dict,
    rag_scenarios: list[dict],
    dialog_state: Optional[str] = None,
    intent_signals: Optional[dict] = None,
    planner_scenario: Optional[str] = None,
) -> dict:
    """Score all catalog scenarios and pick the winner.

    Hard rules (value objection, follow-up, bank switch) are extreme score bonuses,
    not separate if-elif branches, so the scoring is the single decision point.

    Args:
        user_text:      Current user message.
        slots:          Session slots (_active_scenario, _pending_question, …).
        rag_scenarios:  Match dicts from ScenarioFactIndex.match_scenarios().
        dialog_state:   Optional DialogState string.
        intent_signals: Output of extract_intent_signals() — provides detected intents.

    Returns:
        {
            "scenario_switch_allowed": bool,
            "decision": "keep_active" | "switch" | "compare",
            "active_scenario": str | None,
            "candidate_scenarios": list[str],
            "scores": dict[str, float],   # top-5 scored scenarios
            "reason": str,
            "required_next_step": str | None,  # multi-intent: secondary intent to serve
        }
    """
    from app.processing.scenario_catalog import CATALOG

    active = slots.get("_active_scenario")
    sigs = intent_signals or {}

    # TASK 4 — Split intents by source for differentiated scoring
    regex_intents: set[str] = set()
    semantic_intents: set[str] = set()
    for m in sigs.get("matches", []):
        if m.get("source") == "regex":
            regex_intents.add(m["intent"])
        elif m.get("source") == "semantic":
            semantic_intents.add(m["intent"])
    # Anything in intents list not explicitly semantic → treat as regex
    for i in sigs.get("intents", []):
        if i not in semantic_intents:
            regex_intents.add(i)
    detected_intents: list[str] = list(regex_intents | semantic_intents)

    bank_focus: Optional[str] = sigs.get("bank_focus")
    rag_ids = [m["scenario_id"] for m in rag_scenarios]

    # TASK 1 — Stored debtor_type: persists FL/YUL context across turns
    _ds = slots.get("_deal_state") or {}
    stored_debtor_type: Optional[str] = (
        slots.get("debtor_type")
        or slots.get("client_type")
        or _ds.get("debtor_type")
    ) or None

    # Pre-compute signals
    is_fup = _is_followup(user_text, slots)
    # TASK 4 — only REGEX detection triggers the hard value-objection override
    is_value_obj_regex = bool(
        _VALUE_OBJECTION_RE.search(user_text)
        or "direct_bank_objection" in regex_intents
    )
    is_value_obj_semantic = (
        not is_value_obj_regex and "direct_bank_objection" in semantic_intents
    )
    # TASK 5 — account/bank/debtor_type terms present → overrides active objection
    is_account_request = bool(_ACCOUNT_REQUEST_RE.search(user_text))
    bank_switch_tgt = _bank_pricing_switch_target(user_text, active, slots)
    domain_shift_tgt = _domain_shift_target(user_text, active)
    is_new_topic = bool(_NEW_TOPIC_RE.search(user_text))
    # §8 — Conditions vs tariffs
    is_conditions_correction = "correction_not_tariffs_but_conditions" in regex_intents
    is_conditions_intent = (
        not is_conditions_correction
        and "conditions_details_requested" in (regex_intents | semantic_intents)
    )

    # TASK 3 — Repetition complaint: hard override, skips all business scenarios
    if "repetition_complaint" in regex_intents or "repetition_complaint" in semantic_intents:
        return {
            "scenario_switch_allowed": True,
            "decision": "switch",
            "active_scenario": "bot_repetition_complaint",
            "candidate_scenarios": [],
            "scores": {"bot_repetition_complaint": 999.0},
            "reason": "repetition_complaint_override",
            "required_next_step": None,
        }

    # §9 — Planner-driven routing: validate Planner scenario and accept it directly
    if planner_scenario:
        if planner_scenario in CATALOG:
            required_next_step = _get_required_next_step(planner_scenario, detected_intents)
            decision = "switch" if planner_scenario != active else "keep_active"
            rag_ids = [m["scenario_id"] for m in rag_scenarios]
            return {
                "scenario_switch_allowed": decision == "switch",
                "decision": decision,
                "active_scenario": planner_scenario,
                "candidate_scenarios": [sid for sid in rag_ids if sid != planner_scenario][:3],
                "scores": {planner_scenario: 999.0},
                "reason": "planner_decision",
                "required_next_step": required_next_step,
            }
        else:
            logger.warning(
                "ScenarioPolicy | Planner scenario '{}' not in catalog — falling back to scoring",
                planner_scenario,
            )

    # Score every catalog scenario
    scored: list[tuple[float, str]] = []
    for sid, spec in CATALOG.items():
        s = _compute_scenario_score(
            spec=spec,
            detected_intents=detected_intents,
            regex_intents=regex_intents,
            semantic_intents=semantic_intents,
            slots=slots,
            active=active,
            rag_ids=rag_ids,
            is_fup=is_fup,
            is_value_obj_regex=is_value_obj_regex,
            is_value_obj_semantic=is_value_obj_semantic,
            is_account_request=is_account_request,
            bank_switch_tgt=bank_switch_tgt,
            domain_shift_tgt=domain_shift_tgt,
            is_new_topic=is_new_topic,
            bank_focus=bank_focus,
            user_text=user_text,
            stored_debtor_type=stored_debtor_type,
            is_conditions_correction=is_conditions_correction,
            is_conditions_intent=is_conditions_intent,
        )
        scored.append((s, sid))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Expose top-5 scores for logging/debugging
    top_scores = {sid: round(s, 3) for s, sid in scored[:5] if s > 0}

    # Winner is the highest-scoring scenario (with positive score)
    positive = [(s, sid) for s, sid in scored if s > 0]
    if positive:
        winner_score, winner = positive[0]
        candidates = [sid for _, sid in positive[1:4]]
    elif rag_ids:
        winner, winner_score = rag_ids[0], 0.0
        candidates = rag_ids[1:]
    else:
        winner, winner_score = active, 0.0
        candidates = []

    # Determine decision
    if winner == active or is_fup:
        decision = "keep_active"
        if is_fup:
            winner = active  # never abandon active on follow-up
    elif winner != active:
        # Switch unless scores are very close and no explicit signal
        second_score = positive[1][0] if len(positive) > 1 else 0.0
        gap = winner_score - second_score
        if winner_score < _S_CONTINUITY and gap < 0.5 and active and not is_new_topic:
            # Ambiguous: too close to call, expose both
            decision = "compare"
        else:
            decision = "switch"
    else:
        decision = "keep_active"

    # Multi-intent: required_next_step (TASK 6)
    required_next_step = _get_required_next_step(winner, detected_intents)

    # Build reason tags
    reasons: list[str] = []
    if is_value_obj_regex:
        reasons.append("value_objection_regex")
    if is_value_obj_semantic:
        reasons.append("value_objection_semantic")
    if is_account_request:
        reasons.append("account_request")
    if is_fup:
        reasons.append("followup")
    if bank_switch_tgt:
        reasons.append(f"bank_switch:{bank_switch_tgt}")
    if domain_shift_tgt:
        reasons.append(f"domain_shift:{domain_shift_tgt}")
    if is_new_topic:
        reasons.append("new_topic")
    if is_conditions_correction:
        reasons.append("conditions_correction")
    elif is_conditions_intent:
        reasons.append("conditions_intent")
    if not reasons:
        reasons.append("scoring")

    return {
        "scenario_switch_allowed": decision == "switch",
        "decision": decision,
        "active_scenario": winner,
        "candidate_scenarios": list(candidates),
        "scores": top_scores,
        "reason": "|".join(reasons),
        "required_next_step": required_next_step,
    }


def apply_scenario_policy_to_fact_pack(
    fact_pack: dict,
    policy: dict,
    full_scenario_facts: dict,
) -> dict:
    """Filter or restructure fact_pack['scenario_facts'] per the policy decision.

    keep_active → only active scenario's facts remain primary; candidates exposed as IDs only.
    switch      → use full RAG scenario_facts unchanged.
    compare     → active scenario facts first, then candidates; both visible to LLM.
    """
    fp = dict(fact_pack)
    decision = policy["decision"]
    active = policy["active_scenario"]
    candidates = policy["candidate_scenarios"]

    if decision == "keep_active":
        if active and full_scenario_facts:
            primary = {k: v for k, v in full_scenario_facts.items() if k == active}
            # Fall back to full facts only if the active scenario has no RAG data at all
            # (supplemental fetch via get_all_facts_for_scenario happens in the caller)
            fp["scenario_facts"] = primary if primary else full_scenario_facts
        # Expose candidate IDs as lightweight metadata (not full facts — avoids context bloat)
        if candidates:
            fp["_candidate_scenario_ids"] = candidates[:3]
        # Hint for the LLM: which scenario context is locked
        fp["_active_scenario"] = active

    elif decision == "switch":
        fp["scenario_facts"] = full_scenario_facts
        fp.pop("_candidate_scenario_ids", None)
        fp["_active_scenario"] = active

    elif decision == "compare":
        # Active scenario first in the ordered dict so brain naturally reads it first
        if active and active in full_scenario_facts:
            ordered: dict = {active: full_scenario_facts[active]}
            ordered.update({k: v for k, v in full_scenario_facts.items() if k != active})
            fp["scenario_facts"] = ordered
        else:
            fp["scenario_facts"] = full_scenario_facts
        if candidates:
            fp["_candidate_scenario_ids"] = candidates[:3]
        fp["_active_scenario"] = active

    return fp
