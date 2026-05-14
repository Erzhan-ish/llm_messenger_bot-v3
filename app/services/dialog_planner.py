"""LLM dialog planner — главный semantic router.

The planner returns a strict JSON plan. It replaces broad semantic regex routing,
while hard guards and CRM actions remain deterministic in application code.
Receives: user_text, slots, active_task, recent_dialog, kb_context, scenario_hints.

v2: run_dialog_planner() — new scenario-based planner per new arch (plan §4-§14).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from app.config import llm_token_budget as _budget
from app.llm.providers import ask_llm
from app.logging import logger

ALLOWED_INTENTS = {
    "greeting", "company_info", "bank_selection", "pricing", "conditions", "docs",
    "timing", "timing_docs", "constraint", "objection", "handoff",
    "redirect_to_domain", "clarify", "smalltalk", "specific_bank", "process", "service",
    "transfer_fee_quote", "extra_fees", "bonus", "signing", "access_cards", "operations",
}

PLANNER_PROMPT = """
Ты — диалоговый планировщик для Telegram-консультанта компании «В плюсе».
Ты НЕ пишешь ответ клиенту. Верни только JSON.
Ты — главный semantic router. Код не делает routing вместо тебя.

Роль компании: помощь арбитражным/финансовым управляющим в открытии и сопровождении счетов должников в процедурах банкротства.

ТВОЯ ЗАДАЧА — определить СМЫСЛ сообщения клиента:

ДОСТУПНЫЕ INTENT-Ы:
- greeting / company_info / smalltalk — приветствие, представление
- bank_selection — подбор банка
- pricing — тарифы открытие/ведение
- transfer_fee_quote — расчёт стоимости конкретного перевода (по сумме)
- extra_fees — дополнительные платежи/комиссии кроме основных
- bonus — бонус на остаток для АУ
- docs — документы для открытия
- timing — сроки открытия
- timing_docs — сроки + документы
- conditions — широкий вопрос «какие условия»
- signing — подписание, ЭЦП
- access_cards — доступ в интернет-банк, карты
- operations — операции (пенсия, крупный перевод)
- constraint — можно/нельзя/регламентное ограничение
- objection — дорого / подумаю / у других дешевле
- handoff — клиент готов оформлять
- redirect_to_domain — вопрос вне домена
- clarify — нужно уточнить вопрос

КЛЮЧЕВЫЕ ПРАВИЛА:

1. ACTIVE_TASK: если клиент пишет follow-up к предыдущей теме, используй active_task.
   - Если active_task.intent=transfer_fee_quote и клиент написал сумму → intent=transfer_fee_quote, amount=сумма из текста.
   - Если active_task.intent=transfer_fee_quote и клиент написал другой банк → intent=transfer_fee_quote, bank_name=новый банк, amount из active_task.
   - Если active_task.pending_question=realization_status и клиент написал "да введена"/"введена" → intent=constraint, next_action=explain_card_process.
   - Если active_task.pending_question=client_type и клиент ответил типом клиента → intent=bank_selection, client_type=из ответа.
   - Короткие follow-up: "у них сколько?", "а там?", "130 тыс", "для ЮЛ", "да введена" — всегда интерпретируй через active_task.

2. FEE vs CONSTRAINT:
   - "сколько будет стоить перевод/комиссия/платёж" → transfer_fee_quote или extra_fees, НЕ constraint.
   - "дополнительные платежи/скрытые/кроме основной" → extra_fees.
   - "можно ли / нельзя ли / есть ограничение" → constraint.
   - constraint_topic можно заполнить как доп. поле, но intent остаётся fee/pricing, если клиент спросил стоимость.

3. CLIENT_TYPE:
   - НЕ выводи client_type=ФЛ из слов "должник", "банкрот" — это описание должника, а не типа счёта.
   - client_type устанавливай только если клиент явно назвал (ЮЛ/ИП/ФЛ).

4. BANK_NAME:
   - Банк из текущего сообщения имеет приоритет над историей.
   - Если клиент назвал банк в тексте — используй его, не заменяй.

5. HANDOFF:
   - Если клиент говорит "что дальше", "оформляем", "подходит", "давайте", "куда оплатить", "готов начать", "позовите менеджера" → should_handoff=true.

6. OUT_OF_SCOPE:
   - Если вопрос явно не про счета/банкротство/процедуры/тарифы → domain=out_of_scope.

7. SCENARIO_HINTS — это подсказки, но финальное решение принимаешь ты.

8. CONFIDENCE:
   - Если ты уверен → 0.8–1.0.
   - Если follow-up без контекста → 0.6–0.8.
   - Если вообще непонятно → 0.4, intent=clarify.

Верни строго JSON без markdown:
{
  "domain": "in_scope|out_of_scope|unclear",
  "stage": "GREETING|QUALIFY|PRESENTATION|OBJECTION|CONSTRAINT|DOCS|HANDOFF|OUT_OF_SCOPE|SERVICE|OTHER",
  "intent": "greeting|company_info|bank_selection|pricing|transfer_fee_quote|extra_fees|bonus|conditions|docs|timing|timing_docs|signing|access_cards|operations|constraint|objection|handoff|redirect_to_domain|clarify|smalltalk|specific_bank|process|service",
  "query_mode": "pricing|transfer_fee_quote|extra_fees|bonus|bank_selection|conditions|docs|timing|timing_docs|signing|access_cards|operations|constraint|specific_bank|process|out_of_scope|service|smalltalk",
  "scenario_topic": "string or null",
  "constraint_topic": "string or null",
  "next_action": "string or null",
  "client_type": "ФЛ|ЮЛ|ИП|null",
  "bank_name": "ТКБ|Уралсиб|Альфа-Банк|Т-Банк|МКБ|Росбанк|null",
  "amount": null,
  "transfer_target": "ФЛ|ЮЛ|null",
  "answer_focus": "short description",
  "must_use_facts": [],
  "missing_info": [],
  "should_handoff": false,
  "should_stop": false,
  "next_question": "one short question or null",
  "confidence": 0.0
}
""".strip()


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty planner response")
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"no json object in planner response: {raw[:200]!r}")
    return json.loads(m.group(0))


def _planner_to_decision(plan: Dict[str, Any]) -> Dict[str, Any]:
    intent = plan.get("intent") or "smalltalk"
    if intent not in ALLOWED_INTENTS:
        intent = "smalltalk"

    if plan.get("domain") == "out_of_scope":
        return {
            "stage": "OUT_OF_SCOPE", "action": "ANSWER", "query_mode": "out_of_scope",
            "needs_kb": False, "needs_handoff": False,
            "confidence": float(plan.get("confidence") or 0.8), "handoff_reason": None,
            "planner": plan,
        }

    if plan.get("should_handoff") or intent == "handoff":
        return {
            "stage": "DOC_TRANSFER", "action": "HANDOFF", "query_mode": "service",
            "needs_kb": False, "needs_handoff": True,
            "confidence": float(plan.get("confidence") or 0.9),
            "handoff_reason": "ready_to_open", "planner": plan,
        }

    qmode_map = {
        "company_info":         "intro",
        "redirect_to_domain":   "out_of_scope",
        "constraint":           "constraint",
        "conditions":           "conditions",
        "docs":                 "docs",
        "timing":               "timing",
        "timing_docs":          "timing_docs",
        "pricing":              "pricing",
        "bank_selection":       "bank_selection",
        "specific_bank":        "specific_bank",
        "process":              "process",
        "greeting":             "service",
        "smalltalk":            "smalltalk",
        "clarify":              "service",
        "transfer_fee_quote":   "transfer_fee_quote",
        "extra_fees":           "extra_fees",
        "bonus":                "bonus",
        "signing":              "constraint",
        "access_cards":         "access_cards",
        "operations":           "operations",
    }
    # Prefer explicit query_mode from planner if valid
    explicit_qmode = plan.get("query_mode")
    valid_qmodes = {
        "transfer_fee_quote", "extra_fees", "bonus", "access_cards", "operations",
        "constraint", "pricing", "bank_selection", "docs", "timing", "timing_docs",
        "conditions", "specific_bank", "process", "out_of_scope", "service", "smalltalk", "intro",
    }
    if explicit_qmode and explicit_qmode in valid_qmodes:
        qmode = explicit_qmode
    else:
        qmode = qmode_map.get(intent, intent)

    # Hard normalisations
    if intent == "transfer_fee_quote":
        qmode = "transfer_fee_quote"
    elif intent == "extra_fees":
        qmode = "extra_fees"

    return {
        "stage":         plan.get("stage") or "PRESENTATION",
        "action":        "ANSWER" if intent != "clarify" else "CLARIFY",
        "query_mode":    qmode,
        "needs_kb":      qmode not in ("service", "smalltalk", "intro", "out_of_scope"),
        "needs_handoff": False,
        "confidence":    float(plan.get("confidence") or 0.65),
        "handoff_reason": None,
        "planner":       plan,
        # Forward new fields for plan_builder / renderer
        "scenario_topic":   plan.get("scenario_topic"),
        "constraint_topic": plan.get("constraint_topic"),
        "next_action":      plan.get("next_action"),
        "amount":           plan.get("amount"),
        "transfer_target":  plan.get("transfer_target"),
        "bank_name":        plan.get("bank_name"),
    }


async def plan_dialog(
    user_text: str,
    *,
    slots: Optional[dict] = None,
    active_task: Optional[dict] = None,
    recent_dialog: str = "",
    kb_context: list[str] | None = None,
    rule_hints: Optional[dict] = None,
) -> Dict[str, Any]:
    slots = slots or {}
    active_task = active_task or {}

    # Build scenario hints from catalog
    try:
        from app.domain.scenario_catalog import build_scenario_hints
        scenario_hints = build_scenario_hints(user_text)
    except Exception:
        scenario_hints = []

    payload = {
        "user_message": user_text,
        "slots": {
            "client_type":      slots.get("client_type"),
            "bank_name":        slots.get("_current_bank_mention") or slots.get("bank_name") or slots.get("_last_bank"),
            "last_mode":        slots.get("_last_mode"),
            "sales_stage":      slots.get("sales_stage"),
            "transfer_amount":  slots.get("_transfer_amount"),
            "transfer_target":  slots.get("_transfer_target"),
        },
        "active_task": active_task,
        "recent_dialog": recent_dialog[-1800:],
        "kb_context":   (kb_context or [])[:8],
        "rule_hints":   (rule_hints or {}) | {"scenario_hints": scenario_hints},
    }
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("ANALYZER"))
        plan = _extract_json(raw)
        decision = _planner_to_decision(plan)
        logger.info(
            "Dialog planner | intent={} | domain={} | qmode={} | scenario={} | constraint={}",
            plan.get("intent"), plan.get("domain"), decision.get("query_mode"),
            plan.get("scenario_topic"), plan.get("constraint_topic"),
        )
        return decision
    except Exception:
        logger.exception("dialog_planner failed; using safe smalltalk fallback")
        return {
            "stage": "OTHER", "action": "ANSWER", "query_mode": "smalltalk",
            "needs_kb": False, "needs_handoff": False, "confidence": 0.0,
            "handoff_reason": None,
        }


# ===========================================================================
# v2 — Scenario-based Dialog Planner (plan §4–§14)
# Replaces ScenarioPolicy as the main business-scenario router.
# ===========================================================================

_SCENARIO_DESCRIPTIONS_V2: dict[str, str] = {
    "bank_selection_yul":          "Выбор банка для ЮЛ/ИП — список, сравнение",
    "bank_selection_yul_low_cost": "Выбор банка для ЮЛ — критерий 'дешевле всего'",
    "bank_selection_fl":           "Выбор банка для ФЛ",
    "alfabank_yul_conditions":     "Условия Альфа-Банка для ЮЛ (не тарифы — переводы, ограничения)",
    "tkb_yul_conditions":          "Условия ТКБ для ЮЛ (не тарифы — переводы, ограничения)",
    "tkb_fl_conditions":           "Условия ТКБ для ФЛ",
    "uralsib_yul_conditions":      "Условия Уралсиба для ЮЛ",
    "uralsib_fl_conditions":       "Условия Уралсиба для ФЛ",
    "tbank_yul_conditions":        "Условия Т-Банка для ЮЛ (на паузе)",
    "mkb_yul_conditions":          "Условия МКБ для ЮЛ (на паузе)",
    "rosbank_yul_conditions":      "Условия Росбанка для ЮЛ (на паузе)",
    "bank_pricing_yul":            "Тарифы банков для ЮЛ: открытие/ведение в рублях",
    "bank_pricing_fl":             "Тарифы банков для ФЛ",
    "bank_tariff_comparison":      "Сравнение тарифов нескольких банков",
    "au_bonus_question":           "Бонус, вознаграждение, процент на остаток для АУ",
    "interest_on_balance":         "Процент на остаток (годовых)",
    "direct_bank_objection":       "Возражение: 'зачем через вас, пойду напрямую в банк'",
    "documents_required":          "Документы нужные для открытия счёта",
    "ready_to_open":               "Клиент готов открыть счёт — требуется handoff",
    "red_zone_company":            "ЮЛ в красной зоне (банки отказывают напрямую)",
    "non_resident":                "Должник — нерезидент",
    "deceased_fl":                 "Умерший должник (ФЛ)",
    "liquidated_yul":              "Ликвидируемое ЮЛ",
    "debtor_card_realization":     "Карта для должника при реализации имущества",
    "allowed_stages":              "Стадии процедуры на которых открывается счёт",
    "timeline_question":           "Сроки открытия счёта",
    "no_branch_remote_opening":    "Открытие без визита в офис / дистанционно",
    "currency_account_question":   "Вопросы об операциях с валютным счётом: расчёты, платежи кредиторам, что разрешено",
    "bot_repetition_complaint":    "Пользователь жалуется что бот повторяется",
    "bot_complaint":               "Пользователь недоволен ботом: 'ты странный', 'что-то не то говоришь', непонимание",
    "clarification_or_correction": "Пользователь поправляет бота или уточняет вопрос: 'я же другое спрашивал'",
    "small_talk":                  "Светская беседа не по теме",
    "greeting":                    "Приветствие",
    "gratitude_close":             "Благодарность / завершение диалога",
    "unknown_business_question":   "Неизвестный бизнес-вопрос — широкий KB-поиск",
}

_ALLOWED_SCENARIOS_V2: frozenset[str] = frozenset(_SCENARIO_DESCRIPTIONS_V2)

_CATALOG_TEXT_V2 = "\n".join(
    f"- {sid}: {desc}" for sid, desc in _SCENARIO_DESCRIPTIONS_V2.items()
)

_PLANNER_SYSTEM_V2 = f"""Ты — Dialog Planner для русскоязычного чат-бота «В плюсе».
Компания помогает арбитражным управляющим (АУ) открывать банковские счета для должников в банкротных процедурах.

АКТИВНЫЕ БАНКИ для ЮЛ/ИП: Альфа-Банк (800/800 руб.), ТКБ (2800/2090 руб.), Уралсиб (3500/1600 руб.)
АКТИВНЫЕ БАНКИ для ФЛ: ТКБ (1500/0), Уралсиб (0/0)
НА ПАУЗЕ: Т-Банк, МКБ, Росбанк

ТВОЯ ЗАДАЧА: проанализировать диалог и вернуть СТРОГО JSON без какого-либо другого текста.

КАТАЛОГ СЦЕНАРИЕВ:
{_CATALOG_TEXT_V2}

СХЕМА ОТВЕТА:
{{
  "scenario": "<scenario_id из каталога>",
  "topic": "<краткое название темы>",
  "topic_changed": <true|false>,
  "dialog_act": "<question|correction|confirmation|objection|small_talk|handoff>",
  "user_intent": "<краткое намерение>",
  "confidence": <0.0–1.0>,
  "slots_update": {{
    "debtor_type": "<ЮЛ|ФЛ|ИП|null>",
    "bank_focus": "<alfabank|tkb|uralsib|tbank|mkb|rosbank|null>",
    "selected_bank": "<Альфа-Банк|ТКБ|Уралсиб|Т-Банк|МКБ|Росбанк|null>"
  }},
  "deal_state_update": {{
    "deal_stage": "<consulting|comparing_banks|bank_selected|ready_to_open|null>",
    "client_intent": "<ask_info|compare|open_account|ask_documents|null>",
    "next_manager_move": "<answer_question|explain_documents|handoff_to_manager|collect_missing_slot|null>"
  }},
  "handoff": {{
    "needed": <true|false>,
    "reason": "<ready_to_open|docs_affirm|user_explicit|null>"
  }},
  "retrieval": {{
    "queries": ["<запрос 1 на русском>", "<запрос 2>"],
    "required_fact_groups": ["<scenario_id>"],
    "avoid_fact_groups": ["<scenario_id>"]
  }},
  "must_not_repeat": ["<fact_group>"],
  "responder_instruction": "<одна чёткая фраза что ответить и чего избегать>"
}}

КРИТИЧЕСКИЕ ПРАВИЛА:
1. scenario — ТОЛЬКО из каталога выше. Нельзя выдумывать.
2. topic_changed=true при явной смене темы (коррекция, новая тема, новый банк).
3. handoff.needed=true ТОЛЬКО если клиент явно готов открывать + банк известен + тип должника известен.
4. retrieval.queries: 1-3 конкретных запроса на русском для поиска в KB.
5. avoid_fact_groups: если клиент сказал "не про тарифы" / "уже знаю" → добавь то что не нужно.
6. НИКАКОГО текста кроме JSON.
7. Если debtor_type=null и пользователь не указал ЮЛ/ФЛ/ИП, НЕ используй сценарии bank_selection_yul, bank_selection_fl, *_yul_conditions, *_fl_conditions. Вместо этого используй unknown_business_question и в responder_instruction попроси уточнить тип должника.
8. Если есть pending_assistant_question (бот задал вопрос) и пользователь отвечает "да"/"ок"/"хорошо" — интерпретируй как ответ на этот вопрос, не переходи к документам/открытию без явного согласия.
9. Слабые подтверждения (ну ладно, понятно, хорошо, ок) без pending_assistant_question НЕ означают готовность открывать счёт.
10. Фразы "ты странный", "что-то не то говоришь", "непонятно" → scenario=bot_complaint, dialog_act=correction.
11. Фразы "я же другое спрашивал", "я про другое" → scenario=clarification_or_correction, topic_changed=true.
12. Вопросы о валютном счёте (расчёты, платежи кредиторам) → scenario=currency_account_question, retrieval.queries должны искать "валютный счёт операции".
13. Если пользователь спрашивает "какие условия в ТКБ/Альфе/Уралсибе" → scenario=tkb_yul_conditions/alfabank_yul_conditions/uralsib_yul_conditions, responder_instruction ОБЯЗАТЕЛЬНО должен содержать "отвечай только про операционные условия банка, не переходи к документам для открытия и не предлагай начать оформление". handoff.needed=false.
""".strip()


def _extract_json_v2(text: str) -> str:
    """Extract JSON object from raw output (handles markdown blocks)."""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end + 1]


def _validate_planner_v2(data: dict) -> list[str]:
    errors: list[str] = []
    for f in ("scenario", "topic_changed", "handoff", "retrieval"):
        if f not in data:
            errors.append(f"missing:{f}")
    scenario = data.get("scenario", "")
    if scenario and scenario not in _ALLOWED_SCENARIOS_V2:
        errors.append(f"unknown_scenario:{scenario!r}")
    if not isinstance(data.get("handoff") or {}, dict):
        errors.append("handoff:not_dict")
    if not isinstance(data.get("retrieval") or {}, dict):
        errors.append("retrieval:not_dict")
    return errors


def _build_planner_context_v2(
    user_text: str,
    recent_dialog: list[dict],
    known_slots: dict,
    deal_state: dict,
    previous_scenario: Optional[str],
    candidate_scenarios: list[str],
    candidate_intents_regex: list[str],
    candidate_intents_semantic: list[str],
    last_bot_reply_summary: Optional[str],
    already_answered: list[str],
    system_constraints: dict,
    pending_question: Optional[str] = None,
) -> str:
    dialog_compact = []
    for m in (recent_dialog or [])[-6:]:
        role = m.get("role", "")
        text = (m.get("text") or "")[:250]
        if role and text:
            dialog_compact.append({"role": role, "content": text})

    ctx = {
        "current_user_message": user_text[:400],
        "recent_dialog": dialog_compact,
        "known_slots": {
            "debtor_type": known_slots.get("debtor_type") or known_slots.get("client_type"),
            "bank_focus": known_slots.get("_last_bank") or known_slots.get("bank_name"),
            "selected_bank": (deal_state or {}).get("selected_bank"),
            "procedure_stage": known_slots.get("procedure_type") or (deal_state or {}).get("procedure_stage"),
        },
        "deal_state": {
            "deal_stage": (deal_state or {}).get("deal_stage"),
            "client_intent": (deal_state or {}).get("client_intent"),
            "next_manager_move": (deal_state or {}).get("next_manager_move"),
        },
        "previous_scenario": previous_scenario,
        "candidate_scenarios": candidate_scenarios[:4],
        "candidate_intents_regex": candidate_intents_regex[:4],
        "candidate_intents_semantic": candidate_intents_semantic[:4],
        "last_bot_reply_summary": (last_bot_reply_summary or "")[:150],
        "already_answered": already_answered[:4],
        "system_constraints": system_constraints,
        "pending_assistant_question": (pending_question or "")[:200] or None,
    }
    return json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))


async def run_dialog_planner(
    user_text: str,
    recent_dialog: list[dict],
    known_slots: dict,
    deal_state: dict,
    candidate_scenarios: list[str],
    candidate_intents_regex: list[str],
    candidate_intents_semantic: list[str],
    previous_scenario: Optional[str] = None,
    last_bot_reply_summary: Optional[str] = None,
    already_answered: Optional[list[str]] = None,
    pending_question: Optional[str] = None,
    system_constraints: Optional[dict] = None,
    trace_ctx: Optional[dict] = None,
) -> Optional[dict]:
    """Call LLM Dialog Planner v2 and return validated result, or None on failure.

    On None the caller must fall back to ScenarioPolicy.
    """
    trace_ctx = trace_ctx or {}
    trace_id = trace_ctx.get("trace_id", "")
    session_id = trace_ctx.get("session_id", "")

    planner_input = _build_planner_context_v2(
        user_text=user_text,
        recent_dialog=recent_dialog,
        known_slots=known_slots,
        deal_state=deal_state,
        previous_scenario=previous_scenario,
        candidate_scenarios=candidate_scenarios,
        candidate_intents_regex=candidate_intents_regex,
        candidate_intents_semantic=candidate_intents_semantic,
        last_bot_reply_summary=last_bot_reply_summary,
        already_answered=already_answered or [],
        system_constraints=system_constraints or {},
        pending_question=pending_question,
    )

    messages_v2 = [
        {"role": "system", "content": _PLANNER_SYSTEM_V2},
        {"role": "user",   "content": planner_input},
    ]

    logger.info(
        "PlannerCall | trace_id={} | session={} | previous_scenario={}"
        " | candidate_scenarios={} | message={!r}",
        trace_id, session_id, previous_scenario,
        candidate_scenarios[:3], user_text[:80],
    )

    t0 = time.monotonic()
    for attempt in range(2):
        try:
            raw = await ask_llm(
                messages_v2,
                max_tokens=512,
                trace_ctx={**trace_ctx, "phase": f"planner_v2_attempt_{attempt}"},
            )
        except Exception:
            logger.exception("PlannerCall LLM error | trace_id={} | attempt={}", trace_id, attempt)
            if attempt == 0:
                continue
            return None

        json_str = _extract_json_v2(raw)
        if not json_str:
            logger.warning("PlannerResult | no JSON | attempt={} | raw={!r}", attempt, raw[:200])
            if attempt == 0:
                messages_v2.append({"role": "assistant", "content": raw})
                messages_v2.append({"role": "user", "content": "Верни ТОЛЬКО JSON по схеме без текста."})
                continue
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("PlannerResult | JSON parse error: {} | attempt={}", e, attempt)
            if attempt == 0:
                messages_v2.append({"role": "assistant", "content": raw})
                messages_v2.append({"role": "user", "content": "JSON невалидный. Исправь и верни только JSON."})
                continue
            return None

        errors = _validate_planner_v2(data)
        if errors:
            logger.warning("PlannerValidation | errors={} | attempt={}", errors, attempt)
            if attempt == 0 and any("unknown_scenario" in e for e in errors):
                messages_v2.append({"role": "assistant", "content": raw})
                messages_v2.append({
                    "role": "user",
                    "content": f"Ошибка: {errors}. Используй только scenario из каталога.",
                })
                continue
            if any("missing" in e for e in errors):
                return None
            # Minor errors — patch unknown scenario to fallback
            if any("unknown_scenario" in e for e in errors):
                data["scenario"] = "unknown_business_question"

        elapsed = round(time.monotonic() - t0, 3)
        scenario = data.get("scenario", "")
        topic_changed = data.get("topic_changed", False)
        handoff_needed = (data.get("handoff") or {}).get("needed", False)
        retrieval_queries = (data.get("retrieval") or {}).get("queries", [])
        user_intent = data.get("user_intent", "")

        logger.info(
            "PlannerResult | trace_id={} | session={} | scenario={} | topic_changed={}"
            " | user_intent={} | handoff={} | retrieval_queries={} | elapsed={}s",
            trace_id, session_id, scenario, topic_changed,
            user_intent, handoff_needed, retrieval_queries[:2], elapsed,
        )

        if topic_changed and previous_scenario and previous_scenario != scenario:
            logger.info(
                "PlannerTopicSwitch | trace_id={} | from={} | to={} | reason={}",
                trace_id, previous_scenario, scenario, user_intent or "planner_decision",
            )

        if handoff_needed:
            logger.info(
                "PlannerHandoff | trace_id={} | needed=true | reason={}",
                trace_id, (data.get("handoff") or {}).get("reason"),
            )

        logger.info(
            "PlannerValidation | trace_id={} | valid=true | errors=[]",
            trace_id,
        )
        return data

    return None
