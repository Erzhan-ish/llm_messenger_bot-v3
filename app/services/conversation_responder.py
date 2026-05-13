"""Conversation Responder — natural LLM-based answer generator.

Primary response path. LLM writes the answer directly in manager style.
Returns a slim JSON schema: reply, intent, next_step, needs_clarification, handoff.

The responder wraps its output in a brain_result-compatible dict so the existing
validator/repair/handoff logic in message_processor.py requires no changes.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import llm_token_budget as _budget
from app.config import settings
from app.llm.providers import ask_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — natural manager style (based on 358f68c + current constraints)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Ты — Алексей, менеджер-консультант компании «В плюсе».
Помогаешь арбитражным управляющим открывать банковские счета для должников в банкротных процедурах.

СТИЛЬ ДИАЛОГА:
Веди диалог как живой менеджер. Intents и scenarios — это подсказки, а не жёсткий скрипт.
- Краткость: 1–3 предложения. Telegram-стиль, живой деловой язык.
- Слова: «давайте разберём», «всё понял», «секунду», «погнали», «окей».
- Вопрос в конце — только если реально нужны данные для движения вперёд.
- НЕ используй «Рад помочь», «Чем могу быть полезен», «Ваш запрос понятен».
- НЕ извиняйся.

РАБОТА С КОНТЕКСТОМ:
- «там», «у них», «у него» — резолви из previous_reply или recent_dialog.
- «подробнее», «расскажи ещё», «и что» — расширяй предыдущий ответ, не повторяй его дословно.
- Если debtor_type/client_type уже известен — НЕ спрашивай ЮЛ/ФЛ снова.
- Если last_bank или selected_bank известен — отвечай про этот банк.
- «да давайте» после предложения → продолжи предыдущую тему, не спрашивай снова что нужно.

ФАКТЫ:
- Используй только данные из kb_facts и relevant_facts для цен, тарифов, условий. Не выдумывай цифры.
- Активные банки для ЮЛ/ИП: Альфа-Банк (800/800), ТКБ (2800/2090), Уралсиб (3500/1600).
- Активные банки для ФЛ: ТКБ (1500/0), Уралсиб (0/0, переводы до 100тыс бесплатно).
- На паузе: Т-Банк, МКБ, Росбанк.
- Выгода через нас: льготные условия, сопровождение документации и финмониторинг.
- Если пользователь спрашивает тарифы/условия — давай сразу цифры, без «Могу сравнить?».
- Не давай generic-ответов «уточните вопрос» при ясном намерении.
- Не пиши маркированные списки если можно без них.

ПЕРЕХОД К ДЕЙСТВИЮ:
Если клиент выбрал банк и хочет открыть счёт — переходи к следующему бизнес-шагу:
подтверди что берём в работу, объясни минимально нужные данные/документы, передай менеджеру.
Смотри поле instruction в manager_context — если оно есть, следуй ему точно.

HANDOFF:
Если пользователь явно хочет открыть счёт → вернуть handoff.needed=true.
Явные сигналы: «оформляем», «хочу открыть», «давайте откроем», «подключите менеджера», «куда оплатить», «пришлю документы».

Верни только JSON (никакого текста вне JSON, никакого markdown):
{
  "reply": "текст ответа клиенту",
  "intent": "bank_selection|tariff_comparison|value_proposition|docs|clarification|card|handoff|small_talk",
  "next_step": "следующий шаг или null",
  "needs_clarification": false,
  "handoff": {"needed": false, "reason": null}
}""".strip()

# ---------------------------------------------------------------------------
# Output schema keys
# ---------------------------------------------------------------------------

_RESULT_KEYS = frozenset({"reply", "intent", "needs_clarification", "handoff"})


# ---------------------------------------------------------------------------
# JSON extraction (same pattern as conversation_brain)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and _RESULT_KEYS & obj.keys():
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and _RESULT_KEYS & obj.keys():
                return obj
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        pos = raw.find("{", idx)
        if pos == -1:
            break
        try:
            obj, _ = decoder.raw_decode(raw, pos)
            if isinstance(obj, dict) and _RESULT_KEYS & obj.keys():
                return obj
        except json.JSONDecodeError:
            pass
        idx = pos + 1

    raise ValueError(f"no valid responder JSON: {raw[:200]!r}")


def _normalize(obj: dict) -> dict:
    obj.setdefault("reply", None)
    obj.setdefault("intent", "answer")
    obj.setdefault("next_step", None)
    obj.setdefault("needs_clarification", False)
    h = obj.get("handoff")
    if not isinstance(h, dict):
        obj["handoff"] = {"needed": bool(h), "reason": None}
    else:
        h.setdefault("needed", False)
        h.setdefault("reason", None)
    return obj


def _default_result() -> dict:
    return _normalize({
        "reply": None,
        "intent": "answer",
        "next_step": None,
        "needs_clarification": False,
        "handoff": {"needed": False, "reason": None},
    })


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA = """{
  "reply": "текст ответа клиенту",
  "intent": "bank_selection|tariff_comparison|value_proposition|docs|clarification|card|handoff|small_talk",
  "next_step": "следующий шаг или null",
  "needs_clarification": false,
  "handoff": {"needed": false, "reason": null}
}"""


def _build_user_message(payload: dict) -> str:
    ctx = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"<CONTEXT>\n{ctx}\n</CONTEXT>\n\n"
        f"<OUTPUT_SCHEMA>\n{_OUTPUT_SCHEMA}\n</OUTPUT_SCHEMA>\n\n"
        "Верни только JSON по OUTPUT_SCHEMA. Не копируй CONTEXT. Никакого markdown."
    )


def _kb_facts_to_text(kb_facts: list[dict], top_n: int = 6) -> list[str]:
    lines: list[str] = []
    for f in (kb_facts or [])[:top_n]:
        if isinstance(f, dict):
            t = f.get("text") or f.get("content") or ""
            if t:
                lines.append(t[:300])
    return lines


_ACTIVE_BANKS_YUL = ["Альфа-Банк", "ТКБ", "Уралсиб"]
_ACTIVE_BANKS_FL  = ["ТКБ", "Уралсиб"]
_PAUSED_BANKS     = ["Т-Банк", "МКБ", "Росбанк"]

_TARIFF_YUL = {
    "Альфа-Банк": "открытие 800 ₽, ведение 800 ₽/мес",
    "ТКБ":        "открытие 2800 ₽, ведение 2090 ₽/мес",
    "Уралсиб":    "открытие 3500 ₽, ведение 1600 ₽/мес",
}
_TARIFF_FL = {
    "ТКБ":     "открытие 1500 ₽, ведение бесплатно",
    "Уралсиб": "ведение бесплатно, переводы до 100 тыс. бесплатно",
}


def _build_manager_context(known_slots: dict, kb_facts: list[dict] | None = None) -> dict:
    """Build compact structured manager_context per plan §8."""
    ctx: dict[str, Any] = {}
    debtor_type = known_slots.get("debtor_type") or known_slots.get("client_type") or ""
    last_bank = known_slots.get("_last_bank") or known_slots.get("bank_focus") or ""
    last_topic = known_slots.get("_last_topic") or ""
    deal_state = known_slots.get("_deal_state") or {}
    selected_bank = deal_state.get("selected_bank") or last_bank

    # Debtor type
    if debtor_type:
        ctx["debtor_type"] = debtor_type

    # DealState fields
    if deal_state.get("deal_stage"):
        ctx["deal_stage"] = deal_state["deal_stage"]
    if deal_state.get("client_intent"):
        ctx["client_intent"] = deal_state["client_intent"]
    if deal_state.get("next_manager_move"):
        ctx["next_manager_move"] = deal_state["next_manager_move"]
    if selected_bank:
        ctx["selected_bank"] = selected_bank
    if last_bank:
        ctx["last_bank_focus"] = last_bank

    # active_topic — derived from deal stage / last topic
    _deal_stage = deal_state.get("deal_stage") or ""
    if _deal_stage == "ready_to_open" and selected_bank:
        _dt_label = debtor_type or (deal_state.get("debtor_type") or "")
        ctx["active_topic"] = f"opening {selected_bank} account{' for ' + _dt_label if _dt_label else ''}"
    elif last_topic:
        ctx["active_topic"] = last_topic

    # Active and paused banks lists per debtor type (§8)
    if debtor_type in ("ЮЛ", "ИП"):
        ctx["active_banks_yul"] = [
            f"{b} ({_TARIFF_YUL[b]})" for b in _ACTIVE_BANKS_YUL
        ]
        ctx["paused_banks"] = _PAUSED_BANKS
    elif debtor_type == "ФЛ":
        ctx["active_banks_fl"] = [
            f"{b} ({_TARIFF_FL[b]})" for b in _ACTIVE_BANKS_FL
        ]
        ctx["paused_banks"] = _PAUSED_BANKS
    else:
        # Unknown debtor type — show both sets so LLM can pick
        ctx["active_banks_yul"] = [f"{b} ({_TARIFF_YUL[b]})" for b in _ACTIVE_BANKS_YUL]
        ctx["active_banks_fl"]  = [f"{b} ({_TARIFF_FL[b]})"  for b in _ACTIVE_BANKS_FL]
        ctx["paused_banks"] = _PAUSED_BANKS

    # previous_bot_reply
    prev_reply = known_slots.get("_last_bot_text") or ""
    if prev_reply:
        ctx["previous_bot_reply"] = prev_reply[:300]

    # relevant_facts — top KB facts summarised as bullet strings (§8)
    if kb_facts:
        _facts: list[str] = []
        for f in kb_facts[:5]:
            if not isinstance(f, dict):
                continue
            text = (f.get("text") or f.get("fact") or "")[:180].strip()
            bank = f.get("bank") or ""
            if text:
                _facts.append(f"{bank + ': ' if bank else ''}{text}")
        if _facts:
            ctx["relevant_facts"] = _facts

    # instruction — escalation directive when client is ready to open (§8)
    if deal_state.get("handoff_needed") or _deal_stage == "ready_to_open":
        _bank = selected_bank or "выбранный банк"
        _dt   = deal_state.get("debtor_type") or debtor_type or ""
        ctx["instruction"] = (
            f"Клиент готов к открытию ({_bank}{', ' + _dt if _dt else ''}). "
            "НЕ повторяй тарифы. Не спрашивай снова банк/тип должника. "
            "Подтверди что берём в работу. "
            "Назови минимально нужные данные: ИНН должника, данные АУ, судебный акт. "
            "Скажи что передаю кейс менеджеру. Верни handoff.needed=true."
        )
    elif deal_state.get("next_manager_move") == "explain_documents":
        ctx["instruction"] = (
            "Клиент спросил что требуется. Ответь конкретно: "
            "ИНН должника, данные арбитражного управляющего, судебный акт о введении процедуры. "
            "Если уже передаём менеджеру — напомни об этом."
        )

    return ctx


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

async def run_conversation_responder(
    user_text: str,
    recent_dialog: list[dict],
    known_slots: dict,
    candidate_intents: list[str],
    candidate_scenarios: list[str],
    kb_facts: list[dict],
    dialog_state: str | None = None,
    fact_pack: dict | None = None,
    trace_ctx: dict | None = None,
) -> dict:
    """Call LLM in natural manager mode and return responder result dict."""
    # Expose only user-relevant slots (no internal _ prefixes)
    public_slots = {
        k: v for k, v in (known_slots or {}).items()
        if v and not k.startswith("_")
        and k not in ("ready_to_open", "sales_stage")
    }
    # Include key internal slots that help the LLM
    for k in ("_last_bank", "_last_topic", "_active_scenario"):
        if known_slots.get(k):
            public_slots[k.lstrip("_")] = known_slots[k]

    manager_ctx = _build_manager_context(known_slots or {}, kb_facts=kb_facts)

    payload: dict[str, Any] = {
        "user_message": user_text,
        "recent_dialog": [
            {"role": m.get("role", ""), "text": (m.get("text") or "")[:300]}
            for m in (recent_dialog or [])[-8:]
        ],
        "manager_context": manager_ctx,
        "known_slots": public_slots,
        "candidate_intents": candidate_intents or [],
        "candidate_scenarios": candidate_scenarios or [],
        "kb_facts": _kb_facts_to_text(kb_facts),
    }
    if dialog_state:
        payload["dialog_state"] = dialog_state

    # Inject primary fact_pack hints compactly
    fp = dict(fact_pack or {})
    if fp.get("answer_contract"):
        payload["answer_contract"] = fp["answer_contract"]
    if fp.get("_pricing_hint"):
        payload["pricing_hint"] = fp["_pricing_hint"]
    if fp.get("_banks_hint"):
        payload["banks_hint"] = fp["_banks_hint"]
    if fp.get("_note"):
        payload["note"] = fp["_note"]

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(payload)},
    ]

    tctx = dict(trace_ctx or {})
    tctx["phase"] = "responder"
    tctx.setdefault("fact_pack", fp)

    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").lower()
    if provider == "timeweb":
        model = settings.TIMEWEB_AI_MODEL
    elif provider == "fireworks":
        model = settings.FIREWORKS_MODEL
    else:
        model = settings.OLLAMA_MODEL
    max_tokens = _budget("BRAIN")

    logger.info(
        "ResponderCall | trace_id=%s | provider=%s | model=%s | intents=%s | scenarios=%s",
        tctx.get("trace_id", ""), provider, model,
        candidate_intents[:3], candidate_scenarios[:2],
    )

    try:
        raw = await ask_llm(messages, max_tokens=max_tokens, trace_ctx=tctx)
        try:
            result = _extract_json(raw)
        except ValueError:
            logger.warning(
                "ConversationResponder | JSON parse failed | trace_id=%s | raw=%s",
                tctx.get("trace_id"), raw[:200],
            )
            return _default_result()

        result = _normalize(result)
        logger.info(
            "ConversationResponder | trace_id=%s | intent=%s | reply_len=%s | handoff=%s | clarify=%s",
            tctx.get("trace_id"), result.get("intent"),
            len(result.get("reply") or ""),
            result["handoff"]["needed"],
            result.get("needs_clarification"),
        )
        return result

    except Exception:
        logger.exception("conversation_responder failed | trace_id=%s", tctx.get("trace_id"))
        return _default_result()


# ---------------------------------------------------------------------------
# brain_result adapter — makes responder output compatible with existing
# validator/repair/handoff logic in message_processor.py
# ---------------------------------------------------------------------------

def responder_to_brain_result(responder: dict, known_slots: dict | None = None) -> dict:
    """Wrap responder output as a brain_result-compatible dict."""
    handoff = responder.get("handoff") or {"needed": False, "reason": None}
    needs_clarification = responder.get("needs_clarification", False)

    if handoff.get("needed"):
        action = "handoff"
    elif needs_clarification:
        action = "ask_clarification"
    else:
        action = "answer"

    intent = responder.get("intent") or "answer"

    return {
        "reply": responder.get("reply"),
        "action": action,
        "needs_tool": {"name": "none", "args": {}},
        "state_update": {
            "active_task": None,
            "last_bank": (known_slots or {}).get("_last_bank"),
            "last_topic": intent,
            "pending_question": None,
            "last_answer_summary": responder.get("next_step"),
            "sales_context": None,
        },
        "handoff": handoff,
        "stop": False,
        "confidence": 0.9,
    }
