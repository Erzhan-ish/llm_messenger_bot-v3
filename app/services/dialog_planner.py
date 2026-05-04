"""LLM dialog planner for semantic routing.

The planner returns a strict JSON plan. It replaces broad semantic regex routing,
while hard guards and CRM actions remain deterministic in application code.
"""
from __future__ import annotations

import json
import re
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

Роль компании: помощь арбитражным/финансовым управляющим в открытии и сопровождении счетов должников в процедурах банкротства.
Бот консультирует по банкам, тарифам, документам, срокам, ограничениям, картам/счетам должников и передает теплый диалог менеджеру.
Бот не продает товары и не отвечает на посторонние темы.

ПРИОРИТЕТ ОПРЕДЕЛЕНИЯ INTENT — читай ВНИМАТЕЛЬНО:
Не выбирай intent='constraint' только потому, что в KB есть ограничение.
Сначала определи, что ХОЧЕТ клиент:

- выбрать банк → intent=bank_selection
- узнать открытие/ведение → intent=pricing
- рассчитать стоимость перевода по конкретной сумме → intent=transfer_fee_quote
- узнать дополнительные комиссии/платежи кроме основных → intent=extra_fees
- узнать бонус на остаток → intent=bonus
- узнать документы → intent=docs
- узнать сроки → intent=timing
- подписание/ЭЦП → intent=signing
- карты, доступ в ЛК → intent=access_cards
- операции (пенсия, крупный перевод) → intent=operations
- спросить можно/нельзя/регламентное ограничение → intent=constraint
- широкий вопрос «какие условия» → intent=conditions

ВАЖНО: Если пользователь спрашивает про комиссии, платежи, стоимость перевода, «сколько будет стоить», «дополнительные платежи» — это pricing / transfer_fee_quote / extra_fees, НЕ constraint.
constraint_topic можно заполнить дополнительно, если в ответе нужно упомянуть регламентное ограничение.

Примеры:
- "Сколько перевод 200 000 на физлицо?" → intent=transfer_fee_quote
- "Есть ли доп платежи?" → intent=extra_fees
- "Можно открыть спецсчёт без основного?" → intent=constraint, constraint_topic=special_account_without_main
- "Можно карту должнику?" → intent=constraint, constraint_topic=fl_realization_card_signed_by_financial_manager
- "Можно онлайн по ЭЦП?" → intent=signing
- "Какие тарифы?" → intent=pricing

Дополнительные правила:
- Вне домена: domain="out_of_scope", intent="redirect_to_domain".
- Если клиент готов оформлять, спрашивает «что дальше», «куда оплатить», «давайте», «оформляем», «мне подходит» — should_handoff=true.
- Если в KB_CONTEXT есть ограничение/constraint, оно важнее тарифов и продажной логики — но НЕ меняй intent с fee/pricing на constraint.
- Если вопрос про документы — intent="docs".
- Если вопрос про сроки/как долго/когда откроется — intent="timing".
- Если в одном сообщении сроки + документы — intent="timing_docs".
- Если клиент отвечает на уточнение после constraint, продолжи воронку: client_type → bank_selection, а не повторяй constraint.
- Не придумывай факты. must_use_facts заполняй только фактами из KB_CONTEXT.
- Каждый план должен двигать диалог дальше через next_question, кроме STOP/HANDOFF.
- client_type возвращай только если клиент явно указал (ЮЛ/ИП/ФЛ). НЕ выводи ФЛ по умолчанию из слов "должник", "банкрот".
- Если в текущем сообщении упомянут банк (bank_name) — используй его. Не заменяй на банк из истории.
- scenario_hints содержат подсказки из каталога сценариев — используй их как сильный сигнал.

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

    # Protect fee intents from becoming constraint even if constraint_topic is set
    fee_intents = {"transfer_fee_quote", "extra_fees", "pricing", "bonus"}
    if intent in fee_intents and plan.get("constraint_topic"):
        # constraint_topic stays as additional info, but intent and qmode stay fee-focused
        pass

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
    # Prefer explicit query_mode from planner if it's a known mode
    explicit_qmode = plan.get("query_mode")
    if explicit_qmode and explicit_qmode in {
        "transfer_fee_quote", "extra_fees", "bonus", "access_cards", "operations",
        "constraint", "pricing", "bank_selection", "docs", "timing", "timing_docs",
        "conditions", "specific_bank", "process", "out_of_scope", "service", "smalltalk",
    }:
        qmode = explicit_qmode
    else:
        qmode = qmode_map.get(intent, intent)

    # Normalise: transfer_fee_quote / extra_fees must always map to themselves
    if intent == "transfer_fee_quote":
        qmode = "transfer_fee_quote"
    elif intent == "extra_fees":
        qmode = "extra_fees"

    return {
        "stage": plan.get("stage") or "PRESENTATION",
        "action": "ANSWER" if intent != "clarify" else "CLARIFY",
        "query_mode": qmode,
        "needs_kb": qmode not in ("service", "smalltalk", "intro", "out_of_scope"),
        "needs_handoff": False,
        "confidence": float(plan.get("confidence") or 0.65),
        "handoff_reason": None,
        "planner": plan,
        # Pass through new fields so plan_builder and renderer can use them
        "scenario_topic":    plan.get("scenario_topic"),
        "constraint_topic":  plan.get("constraint_topic"),
        "next_action":       plan.get("next_action"),
        "amount":            plan.get("amount"),
        "transfer_target":   plan.get("transfer_target"),
        "bank_name":         plan.get("bank_name"),
    }


async def plan_dialog(
    user_text: str,
    *,
    slots: Optional[dict] = None,
    recent_dialog: str = "",
    kb_context: list[str] | None = None,
    rule_hints: Optional[dict] = None,
) -> Dict[str, Any]:
    slots = slots or {}

    # Build scenario hints from catalog
    try:
        from app.domain.scenario_catalog import build_scenario_hints
        scenario_hints = build_scenario_hints(user_text)
    except Exception:
        scenario_hints = []

    payload = {
        "user_message": user_text,
        "slots": {
            "client_type": slots.get("client_type"),
            "bank_name": slots.get("_current_bank_mention") or slots.get("bank_name") or slots.get("_last_bank"),
            "last_mode": slots.get("_last_mode"),
            "sales_stage": slots.get("sales_stage"),
            "transfer_amount": slots.get("_transfer_amount"),
            "transfer_target": slots.get("_transfer_target"),
        },
        "recent_dialog": recent_dialog[-1800:],
        "kb_context": (kb_context or [])[:8],
        "rule_hints": (rule_hints or {}) | {"scenario_hints": scenario_hints},
    }
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("ANALYZER"))
        plan = _extract_json(raw)
        decision = _planner_to_decision(plan)
        logger.info(
            "Dialog planner decision | intent={} | domain={} | qmode={} | scenario_topic={} | constraint_topic={}",
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
