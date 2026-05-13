"""Scenario state machine — deterministic slot filling and required_next_step.

Runs between KB retrieval and the LLM call.
  action=reply   → deterministic text ready, skip LLM entirely.
  action=enrich  → add constraints to fact_pack, let LLM continue.
  action=skip    → no change, normal pipeline.

Slot keys written to session slots:
  _known_slots          dict    accumulated factual answers for the active scenario
  _pending_slot         str     which slot we are waiting the user to fill
  _expected_answer_type str     "yes_no" | "text" | None
  _last_bot_question    str     the last clarifying question the bot posed
  _required_next_step   str     intent the LLM reply must fulfil
  _forbidden_actions    list    phrases the LLM reply must not contain
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Slot key constants (single source of truth)
# ---------------------------------------------------------------------------
SLOT_KNOWN     = "_known_slots"
SLOT_PENDING   = "_pending_slot"
SLOT_EXPECTED  = "_expected_answer_type"
SLOT_LAST_Q    = "_last_bot_question"
SLOT_NEXT_STEP = "_required_next_step"
SLOT_FORBIDDEN = "_forbidden_actions"

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

_YES_RE = re.compile(
    r"^\s*(да|ага|угу|ну\s+да|конечно|само\s+собой|точно|верно|есть"
    r"|введена|введено|введён|да[\s,]+введена|всё\s+верно|всё\s+так)\s*[.!]?\s*$",
    re.I | re.U,
)

_NO_RE = re.compile(
    r"^\s*(нет|не\s+введена|не\s+введено|ещё\s+нет|еще\s+нет|пока\s+нет"
    r"|нее|неа|не|нет[\s,]+пока)\s*[.!]?\s*$",
    re.I | re.U,
)

_GRATITUDE_RE = re.compile(
    r"^\s*("
    r"хорошо[\s,]+спасибо|спасибо[\s,]+хорошо"
    r"|понял[\s,]+спасибо|спасибо[\s,]+понял"
    r"|поняла[\s,]+спасибо|спасибо[\s,]+поняла"
    r"|понятно[\s,]+спасибо|спасибо[\s,]+понятно"
    r"|ладно[\s,]+спасибо|спасибо[\s,]+ладно"
    r"|ок[\s,]+спасибо|спасибо[\s,]+ок"
    r"|принял[\s,]+спасибо|спасибо[\s,]+принял"
    r"|принята[\s,]+спасибо"
    r"|благодарю"
    r")\s*[.!]?\s*$",
    re.I | re.U,
)

_WHAT_REQUIRED_RE = re.compile(
    r"("
    r"что\s+(от\s+меня\s+)?требуется"
    r"|от\s+меня\s+что\s+(нужно|требуется|надо)"
    r"|что\s+нужно\s+(от\s+меня|мне|сделать|подготовить)"
    r"|что\s+надо\s+(от\s+меня|мне|сделать|принести)"
    r"|какие\s+(нужны|требуются)\s+(документы|бумаги|данные)"
    r"|что\s+(нужно|надо|требуется)\s+(для\s+)?(открытия|оформления)"
    r"|ну\s+(что\s+)?от\s+меня"
    r"|что\s+нужно$"
    r")",
    re.I | re.U,
)

_WHAT_NEXT_RE = re.compile(
    r"("
    r"что\s+(мне\s+)?(делать|дальше|теперь)"
    r"|как\s+(мне\s+)?(действовать|поступить|быть)"
    r"|дальше\s+что|следующий\s+шаг|что\s+дальше"
    r"|куда\s+(мне\s+)?(идти|обращаться|звонить|обратиться)"
    r")",
    re.I | re.U,
)

_OBJECTION_ELABORATE_RE = re.compile(
    r"^\s*("
    r"подробнее|подробней|поподробнее"
    r"|почему(\s+через\s+вас)?"
    r"|зачем(\s+через\s+вас|\s+мне\s+с\s+вами(\s+работать)?)?"
    r"|не\s+понял|в\s+чем\s+именно|в\s+чём\s+именно"
    r"|какая\s+выгода"
    r"|вы\s+не\s+ответили"
    r"|ну\s+зачем\s+мне\s+с\s+вами(\s+работать)?"
    r"|расскажи\s+(подробнее|ещё|еще)"
    r"|расскажите\s+(подробнее|ещё|еще)"
    r"|что\s+именно"
    r")\s*[?!.]?\s*$",
    re.I | re.U,
)

_ONLY_MORE_RE = re.compile(
    r"("
    r"только\s+(эти\s+)?(оба|два|оба\s+варианта)"
    r"|только\s+эти\b"
    r"|а\s+(ещё|еще)\s*[?!]?"
    r"|другие\s+(банки\s+)?есть"
    r"|ещё\s+(варианты|банки|есть)\b"
    r"|еще\s+(варианты|банки|есть)\b"
    r"|больше\s+(нет|ничего|вариантов)"
    r"|и\s+всё\s*\?"
    r")",
    re.I | re.U,
)

_TARIFF_COMPARE_RE = re.compile(
    r"("
    r"какие\s+(условия(\s+и\s+тарифы)?|тарифы)"
    r"|сравни\s+(тарифы|условия|их)"
    r"|условия\s+у\s+(них|него|банков)"
    r"|тарифы\s+у\s+(них|него|банков)"
    r"|дай\s+(сравнение|сравнить|тарифы)"
    r"|покажи\s+(тарифы|условия|сравнение)"
    r"|какой\s+тариф"
    r"|условия\s+и\s+тарифы"
    r")",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Bank scenario constants
# ---------------------------------------------------------------------------

_BANK_COND_SCENARIOS = frozenset({
    "uralsib_yul_conditions", "uralsib_fl_conditions",
    "alfabank_yul_conditions",
    "tkb_yul_conditions",     "tkb_fl_conditions",
    "tbank_yul_conditions",   "mkb_yul_conditions",
    "rosbank_yul_conditions",
})

_TARIFF_COMPARE_SCENARIOS = frozenset({
    "bank_selection_yul", "bank_selection_yul_low_cost", "bank_pricing_yul",
})

# ---------------------------------------------------------------------------
# Pricing helpers (read from fact_pack to keep playbook deterministic)
# ---------------------------------------------------------------------------

def _pricing_line(name: str, p: dict, fallback_open: int | None, fallback_mon: int | None) -> str:
    of = p.get("opening_fee", fallback_open)
    mf = p.get("monthly_fee", fallback_mon)
    if of is not None and mf is not None:
        return f"{name}: открытие {int(of)} ₽, ведение {int(mf)} ₽/мес."
    return f"{name}: условия уточняются отдельно."


def _build_low_cost_reply(fact_pack: dict | None) -> str:
    pricing = (fact_pack or {}).get("bank_pricing_yul") or []
    banks = {p.get("bank"): p for p in pricing if p.get("bank")}

    alfa = _pricing_line("Альфа-Банк", banks.get("Альфа-Банк", {}), 800, 800)
    tkb  = _pricing_line("ТКБ",        banks.get("ТКБ",  {}),        2800, 2090)

    ural_p = banks.get("Уралсиб", {})
    ural_of = ural_p.get("opening_fee")
    ural_mf = ural_p.get("monthly_fee")
    if ural_of is not None and ural_mf is not None:
        ural = f"Уралсиб: открытие {int(ural_of)} ₽, ведение {int(ural_mf)} ₽/мес. — тоже активный вариант."
    else:
        ural = (
            "Уралсиб тоже активный вариант, "
            "но по стартовым затратам обычно выходит дороже — условия уточняются отдельно."
        )

    return (
        f"Для юрлица самый дешёвый старт — {alfa} "
        f"Также можно рассмотреть {tkb} "
        f"{ural}"
    )


def _build_tariff_compare_reply(fact_pack: dict | None) -> str:
    pricing = (fact_pack or {}).get("bank_pricing_yul") or []
    banks = {p.get("bank"): p for p in pricing if p.get("bank")}

    lines = []
    for name, fallback_of, fallback_mf in [
        ("Альфа-Банк", 800,  800),
        ("ТКБ",        2800, 2090),
    ]:
        lines.append("— " + _pricing_line(name, banks.get(name, {}), fallback_of, fallback_mf))

    ural_p = banks.get("Уралсиб", {})
    ural_of = ural_p.get("opening_fee")
    ural_mf = ural_p.get("monthly_fee")
    if ural_of is not None and ural_mf is not None:
        lines.append(f"— Уралсиб: открытие {int(ural_of)} ₽, ведение {int(ural_mf)} ₽/мес.")
    else:
        lines.append("— Уралсиб: условия уточняются отдельно.")

    body = "\n".join(lines)
    return (
        f"По активным вариантам для юрлиц:\n{body}\n"
        "Т-Банк, МКБ и Росбанк сейчас на паузе."
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_yes_no(text: str) -> Optional[str]:
    """Return 'yes', 'no', or None."""
    t = (text or "").strip()
    if _YES_RE.match(t):
        return "yes"
    if _NO_RE.match(t):
        return "no"
    return None


def detect_gratitude(text: str) -> bool:
    flat = re.sub(r"\s+", " ", (text or "").strip())
    return bool(_GRATITUDE_RE.match(flat))


def detect_what_required(text: str) -> bool:
    return bool(_WHAT_REQUIRED_RE.search(text or ""))


def detect_what_next(text: str) -> bool:
    return bool(_WHAT_NEXT_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Playbook engine
# ---------------------------------------------------------------------------

def run_scenario_playbook(
    user_text: str,
    slots: dict,
    fact_pack: Optional[dict] = None,
    kb_facts: Optional[list] = None,
) -> dict:
    """Evaluate the deterministic scenario state machine.

    Returns:
      {"action": "reply",  "reply": str,  "updates": dict, "fact_pack_additions": dict, "log": dict}
      {"action": "enrich", "reply": None, "updates": dict, "fact_pack_additions": dict, "log": dict}
      {"action": "skip",   "reply": None, "updates": {},   "fact_pack_additions": {},   "log": dict}
    """
    active  = slots.get("_active_scenario")
    known   = dict(slots.get(SLOT_KNOWN) or {})
    pending = slots.get(SLOT_PENDING)

    log: dict = {
        "pending_slot_before":        pending,
        "known_slots_before":         dict(known),
        "required_next_step_before":  slots.get(SLOT_NEXT_STEP),
    }

    def _skip() -> dict:
        log["llm_skipped"] = False
        return {"action": "skip", "reply": None, "updates": {}, "fact_pack_additions": {}, "log": log}

    def _reply(text: str, updates: dict) -> dict:
        log.update({
            "llm_skipped":            True,
            "pending_slot_after":     updates.get(SLOT_PENDING, pending),
            "known_slots_after":      updates.get(SLOT_KNOWN, known),
            "required_next_step_after": updates.get(SLOT_NEXT_STEP, slots.get(SLOT_NEXT_STEP)),
        })
        return {"action": "reply", "reply": text, "updates": updates,
                "fact_pack_additions": {}, "log": log}

    def _enrich(updates: dict, fp: dict) -> dict:
        log.update({
            "llm_skipped":            False,
            "pending_slot_after":     updates.get(SLOT_PENDING, pending),
            "known_slots_after":      updates.get(SLOT_KNOWN, known),
            "required_next_step_after": updates.get(SLOT_NEXT_STEP, slots.get(SLOT_NEXT_STEP)),
        })
        return {"action": "enrich", "reply": None, "updates": updates,
                "fact_pack_additions": fp, "log": log}

    # -----------------------------------------------------------------------
    # Gratitude / closing — always first
    # -----------------------------------------------------------------------
    if detect_gratitude(user_text):
        log["gratitude_close_detected"] = True
        return _reply(
            "Пожалуйста! Если появятся вопросы по счетам или документам — я на связи.",
            {SLOT_PENDING: None, SLOT_NEXT_STEP: None},
        )

    # -----------------------------------------------------------------------
    # direct_bank_objection — enrich with constraints, let LLM write naturally
    # -----------------------------------------------------------------------
    if active == "direct_bank_objection":
        log["scenario_playbook_applied"] = "direct_bank_objection"
        return _enrich(
            {
                SLOT_KNOWN:     {**known, "objection_answered": True},
                SLOT_FORBIDDEN: ["о каких банках хотите узнать", "уточните вопрос"],
            },
            {
                "_known_slots":       {**known, "objection_answered": True},
                "_forbidden_phrases": ["о каких банках хотите узнать", "уточните вопрос"],
            },
        )

    # -----------------------------------------------------------------------
    # debtor_card_realization
    # -----------------------------------------------------------------------
    if active == "debtor_card_realization":
        log["scenario_playbook_applied"] = "debtor_card_realization"
        realization_known = known.get("realization_started")

        # First entry: no realization info yet, user hasn't answered yet
        # → skip so ActionExecutor (explain_card_rules) handles the initial question
        if realization_known is None and pending is None and detect_yes_no(user_text) is None:
            return _skip()

        # Waiting for yes/no on "Реализация уже введена?"
        if realization_known is None and (pending == "realization_started" or pending is None):
            yn = detect_yes_no(user_text)
            log["yes_no_answer_detected"] = yn
            if yn == "yes":
                new_known = {**known, "realization_started": True}
                return _reply(
                    "Да, тогда карту можно оформить. Документы подписывает финансовый управляющий — "
                    "он же контролирует операции по карте. Следующий шаг — выбрать банк и подготовить "
                    "документы по делу: определение суда о введении реализации, данные и ИНН должника.",
                    {
                        SLOT_KNOWN:    new_known,
                        SLOT_PENDING:  None,
                        SLOT_EXPECTED: None,
                        SLOT_NEXT_STEP: "explain_card_opening_process",
                        SLOT_FORBIDDEN: [
                            "дождитесь введения реализации",
                            "дождитесь судебного решения о введении",
                        ],
                    },
                )
            if yn == "no":
                new_known = {**known, "realization_started": False}
                return _reply(
                    "Тогда карту пока оформить нельзя. "
                    "Обычно её оформляют после введения реализации имущества.",
                    {
                        SLOT_KNOWN:    new_known,
                        SLOT_PENDING:  None,
                        SLOT_EXPECTED: None,
                        SLOT_NEXT_STEP: None,
                    },
                )

        # Реализация подтверждена — handle follow-up questions
        if realization_known is True:
            if detect_what_next(user_text) or detect_what_required(user_text):
                return _reply(
                    "Раз реализация введена, следующие шаги:\n"
                    "1. Финансовый управляющий подписывает документы на открытие карты.\n"
                    "2. Выбираете банк — мы поможем с условиями и заявкой.\n"
                    "3. Банк открывает карту на имя должника под контролем ФУ.\n"
                    "Какой банк рассматриваете или нужна помощь с выбором?",
                    {
                        SLOT_NEXT_STEP: "explain_next_steps_after_realization",
                        SLOT_FORBIDDEN: [
                            "дождитесь введения реализации",
                            "дождитесь судебного решения о введении",
                        ],
                    },
                )
            # Let LLM answer but forbid contradictory phrasing
            return _enrich(
                {
                    SLOT_FORBIDDEN: [
                        "дождитесь введения реализации",
                        "дождитесь судебного решения о введении",
                    ],
                },
                {
                    "_known_slots": known,
                    "_forbidden_phrases": [
                        "дождитесь введения реализации",
                        "дождитесь судебного решения о введении",
                    ],
                    "_required_next_step": "answer_within_realization_context",
                },
            )

    # -----------------------------------------------------------------------
    # allowed_stages
    # -----------------------------------------------------------------------
    if active == "allowed_stages":
        log["scenario_playbook_applied"] = "allowed_stages"

        if known.get("account_opening_allowed") is None:
            known = {**known, "account_opening_allowed": True, "procedure_stage": "observation"}

        if detect_what_required(user_text) or detect_what_next(user_text):
            new_known = {**known, "account_opening_allowed": True}
            log["required_next_step_after"] = "explain_requirements_for_opening"
            return _reply(
                "Для открытия счёта на стадии наблюдения нужны:\n"
                "— данные должника (название организации или ФИО и ИНН),\n"
                "— определение суда о введении наблюдения,\n"
                "— паспорт арбитражного управляющего и его ИНН.\n"
                "После этого подбираем подходящий банк и готовим заявку. "
                "Должник — юрлицо или физлицо?",
                {SLOT_KNOWN: new_known, SLOT_NEXT_STEP: "explain_requirements_for_opening"},
            )

        return _enrich(
            {SLOT_KNOWN: {**known, "account_opening_allowed": True}},
            {"_known_slots": {**known, "account_opening_allowed": True}},
        )

    # -----------------------------------------------------------------------
    # bank_selection_yul_low_cost — cheapest options for ЮЛ
    # -----------------------------------------------------------------------
    if active == "bank_selection_yul_low_cost":
        log["scenario_playbook_applied"] = "bank_selection_yul_low_cost"
        low_cost_listed = known.get("low_cost_listed")

        # Enrich with constraints — LLM answers naturally with low-cost focus
        return _enrich(
            {
                SLOT_KNOWN:     {**known, "low_cost_listed": True, "debtor_type": "legal_entity"},
                SLOT_FORBIDDEN: ["Счёт подбираем для юрлица или физлица", "Могу сравнить?"],
            },
            {
                "_known_slots":       {**known, "low_cost_listed": True},
                "_forbidden_phrases": ["Счёт подбираем для юрлица или физлица", "Могу сравнить?"],
                "_pricing_focus":     "low_cost",
            },
        )

    # -----------------------------------------------------------------------
    # bank_selection_fl — deterministic bank list for физлицо
    # -----------------------------------------------------------------------
    if active == "bank_selection_fl":
        log["scenario_playbook_applied"] = "bank_selection_fl"
        banks_listed = known.get("banks_listed")

        # Enrich with FL context — LLM answers naturally
        return _enrich(
            {
                SLOT_KNOWN:     {**known, "banks_listed": True},
                SLOT_FORBIDDEN: ["Счёт подбираем для юрлица или физлица"],
            },
            {
                "_known_slots":       {**known, "banks_listed": True},
                "_forbidden_phrases": ["Счёт подбираем для юрлица или физлица"],
            },
        )

    # -----------------------------------------------------------------------
    # bank_selection_yul — first-turn deterministic bank list for ЮЛ
    # -----------------------------------------------------------------------
    if active == "bank_selection_yul":
        log["scenario_playbook_applied"] = "bank_selection_yul"
        is_yul = (
            slots.get("debtor_type") in ("ЮЛ", "ИП")
            or slots.get("client_type") in ("ЮЛ", "ИП")
        )
        banks_listed = known.get("banks_listed")

        # Enrich with YUL context — LLM answers naturally
        forbidden_list = (
            ["Счёт подбираем для юрлица или физлица"] if is_yul else []
        )
        return _enrich(
            {
                SLOT_KNOWN: {**known, "banks_listed": True},
                **(
                    {SLOT_FORBIDDEN: forbidden_list} if forbidden_list else {}
                ),
            },
            {
                "_known_slots": {**known, "banks_listed": True},
                **(
                    {"_forbidden_phrases": forbidden_list} if forbidden_list else {}
                ),
            },
        )

    # -----------------------------------------------------------------------
    # bank_pricing_yul — LLM enrichment with YUL tariff constraints
    # -----------------------------------------------------------------------
    if active == "bank_pricing_yul":
        log["scenario_playbook_applied"] = "bank_pricing_yul"
        is_yul = (
            slots.get("debtor_type") in ("ЮЛ", "ИП")
            or slots.get("client_type") in ("ЮЛ", "ИП")
        )

        forbidden_list = (
            ["Счёт подбираем для юрлица или физлица", "Могу сравнить?"] if is_yul else []
        )
        return _enrich(
            {
                SLOT_KNOWN: known,
                **(
                    {SLOT_FORBIDDEN: forbidden_list} if forbidden_list else {}
                ),
            },
            {
                "_known_slots":           known,
                "_uralsib_fallback_hint": (
                    "По Уралсибу условия лучше уточнить отдельно, "
                    "но банк сейчас активен для юрлиц."
                ),
                **(
                    {"_forbidden_phrases": forbidden_list} if forbidden_list else {}
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Bank-specific conditions (uralsib_yul_conditions, tkb_yul_conditions, …)
    # -----------------------------------------------------------------------
    if active in _BANK_COND_SCENARIOS:
        log["scenario_playbook_applied"] = active

        # If client is ready to open — switch from tariff enrichment to handoff/docs mode
        _ds = slots.get("_deal_state") or {}
        if _ds.get("deal_stage") == "ready_to_open" or _ds.get("handoff_needed"):
            log["deal_stage_override"] = "ready_to_open"
            _forbidden_tariff = [
                "Счёт подбираем для юрлица или физлица",
                "Могу сравнить?",
            ]
            return _enrich(
                {
                    SLOT_KNOWN:     {**known, "ready_to_open": True},
                    SLOT_NEXT_STEP: "handoff_to_manager",
                    SLOT_FORBIDDEN: _forbidden_tariff,
                },
                {
                    "_known_slots":       {**known, "ready_to_open": True},
                    "_required_next_step": "handoff_to_manager",
                    "_forbidden_phrases": _forbidden_tariff,
                    "_deal_stage":        "ready_to_open",
                },
            )

        # Docs/requirements question in deal context — enrich with explain_documents hint
        if detect_what_required(user_text) or detect_what_next(user_text):
            if _ds.get("deal_stage") in ("bank_selected", "ready_to_open") or _ds.get("selected_bank"):
                log["docs_in_deal_context"] = True
                return _enrich(
                    {SLOT_KNOWN: known, SLOT_NEXT_STEP: "explain_documents"},
                    {"_known_slots": known, "_required_next_step": "explain_documents"},
                )

        is_yul = (
            slots.get("debtor_type") in ("ЮЛ", "ИП")
            or slots.get("client_type") in ("ЮЛ", "ИП")
        )
        forbidden_list = (
            ["Счёт подбираем для юрлица или физлица"] if is_yul else []
        )
        return _enrich(
            {
                SLOT_KNOWN: known,
                **(
                    {SLOT_FORBIDDEN: forbidden_list} if forbidden_list else {}
                ),
            },
            {
                "_known_slots": known,
                **(
                    {"_forbidden_phrases": forbidden_list} if forbidden_list else {}
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Global: detect action/docs intent at deal stage regardless of active scenario
    # -----------------------------------------------------------------------
    _ds_global = slots.get("_deal_state") or {}
    if _ds_global.get("handoff_needed") or _ds_global.get("deal_stage") == "ready_to_open":
        log["scenario_playbook_applied"] = "deal_state_ready_to_open"
        _forbidden_selling = ["Счёт подбираем для юрлица или физлица", "Могу сравнить?"]
        if detect_what_required(user_text) or detect_what_next(user_text):
            return _enrich(
                {
                    SLOT_KNOWN: {**known, "ready_to_open": True},
                    SLOT_NEXT_STEP: "explain_documents",
                    SLOT_FORBIDDEN: _forbidden_selling,
                },
                {
                    "_known_slots":       {**known, "ready_to_open": True},
                    "_required_next_step": "explain_documents",
                    "_forbidden_phrases": _forbidden_selling,
                    "_deal_stage":        "ready_to_open",
                },
            )
        return _enrich(
            {
                SLOT_KNOWN:     {**known, "ready_to_open": True},
                SLOT_NEXT_STEP: "handoff_to_manager",
                SLOT_FORBIDDEN: _forbidden_selling,
            },
            {
                "_known_slots":       {**known, "ready_to_open": True},
                "_required_next_step": "handoff_to_manager",
                "_forbidden_phrases": _forbidden_selling,
                "_deal_stage":        "ready_to_open",
            },
        )

    return _skip()
