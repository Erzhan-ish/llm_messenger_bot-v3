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
}

PLANNER_PROMPT = """
Ты — диалоговый планировщик для Telegram-консультанта компании «В плюсе».
Ты НЕ пишешь ответ клиенту. Верни только JSON.

Роль компании: помощь арбитражным/финансовым управляющим в открытии и сопровождении счетов должников в процедурах банкротства.
Бот консультирует по банкам, тарифам, документам, срокам, ограничениям, картам/счетам должников и передает теплый диалог менеджеру.
Бот не продает товары и не отвечает на посторонние темы.

Правила:
- Вне домена: domain="out_of_scope", intent="redirect_to_domain".
- Если клиент готов оформлять, спрашивает «что дальше», «куда оплатить», «давайте», «оформляем», «мне подходит» — should_handoff=true.
- Если вопрос про ограничение/можно ли/нельзя ли, intent="constraint" или "process".
- Если вопрос про документы — intent="docs".
- Если вопрос про сроки/как долго/когда откроется — intent="timing".
- Если в одном сообщении сроки + документы — intent="timing_docs".
- Если вопрос «какие условия/подробнее/что входит» по банку — intent="conditions".
- Не придумывай факты. must_use_facts заполняй только фактами из KB_CONTEXT.

Верни строго JSON без markdown:
{
  "domain": "in_scope|out_of_scope|unclear",
  "stage": "GREETING|QUALIFY|PRESENTATION|OBJECTION|CONSTRAINT|DOCS|HANDOFF|OUT_OF_SCOPE|SERVICE|OTHER",
  "intent": "greeting|company_info|bank_selection|pricing|conditions|docs|timing|timing_docs|constraint|objection|handoff|redirect_to_domain|clarify|smalltalk|specific_bank|process|service",
  "client_type": "ФЛ|ЮЛ|ИП|null",
  "bank_name": "ТКБ|Уралсиб|Альфа-Банк|Т-Банк|МКБ|Росбанк|null",
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
        "company_info": "intro",
        "redirect_to_domain": "out_of_scope",
        "constraint": "constraint",
        "conditions": "conditions",
        "docs": "docs",
        "timing": "timing",
        "timing_docs": "timing_docs",
        "pricing": "pricing",
        "bank_selection": "bank_selection",
        "specific_bank": "specific_bank",
        "process": "process",
        "greeting": "service",
        "smalltalk": "smalltalk",
        "clarify": "service",
    }
    qmode = qmode_map.get(intent, intent)
    return {
        "stage": plan.get("stage") or "PRESENTATION",
        "action": "ANSWER" if intent != "clarify" else "CLARIFY",
        "query_mode": qmode,
        "needs_kb": qmode not in ("service", "smalltalk", "intro", "out_of_scope"),
        "needs_handoff": False,
        "confidence": float(plan.get("confidence") or 0.65),
        "handoff_reason": None,
        "planner": plan,
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
    payload = {
        "user_message": user_text,
        "slots": {
            "client_type": slots.get("client_type"),
            "bank_name": slots.get("bank_name") or slots.get("_last_bank"),
            "last_mode": slots.get("_last_mode"),
            "sales_stage": slots.get("sales_stage"),
        },
        "recent_dialog": recent_dialog[-1800:],
        "kb_context": (kb_context or [])[:8],
        "rule_hints": rule_hints or {},
    }
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("ANALYZER"))
        plan = _extract_json(raw)
        decision = _planner_to_decision(plan)
        logger.info("Dialog planner decision | intent={} | domain={} | qmode={}", plan.get("intent"), plan.get("domain"), decision.get("query_mode"))
        return decision
    except Exception:
        logger.exception("dialog_planner failed; using safe smalltalk fallback")
        return {
            "stage": "OTHER", "action": "ANSWER", "query_mode": "smalltalk",
            "needs_kb": False, "needs_handoff": False, "confidence": 0.0,
            "handoff_reason": None,
        }
