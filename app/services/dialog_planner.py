"""LLM dialog planner — главный semantic router.

The planner returns a strict JSON plan. It replaces broad semantic regex routing,
while hard guards and CRM actions remain deterministic in application code.
Receives: user_text, slots, active_task, recent_dialog, kb_context, scenario_hints.
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
