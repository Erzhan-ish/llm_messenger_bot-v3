"""LLM Planner — selects the best dialog action from ActionCatalog.

Returns structured JSON (not user-facing text):
  {"action": "compare_tariffs", "confidence": 0.91, "reason": "...",
   "missing_slots": [], "scenario_switch": {"needed": false, "to": null}}

The planner MUST NOT write final user-facing text. That is the executor's job.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_PLANNER_SCHEMA = """{
  "action": "<action_id from available_actions>",
  "confidence": 0.0,
  "reason": "<1-2 sentences>",
  "missing_slots": [],
  "scenario_switch": {"needed": false, "to": null}
}"""

_PLANNER_SYSTEM = """Ты — планировщик диалога. Контекст: сервис помогает арбитражным управляющим открыть банковский счёт должнику в банкротной процедуре.

ЗАДАЧА: выбрать ОДНО действие из available_actions, которое лучше всего отвечает на сообщение пользователя.

ПРАВИЛА:
1. Выбирай ТОЛЬКО из available_actions. Никаких других действий.
2. Если пользователь спрашивает какие банки / с какими банками работаете → list_partner_banks.
3. Если пользователь спрашивает тарифы / сравни / условия / почём → compare_tariffs (если тип должника известен) или list_partner_banks.
4. Если пользователь говорит "да" / "хорошо" / "давайте" ПОСЛЕ предложения сравнить тарифы → compare_tariffs.
5. Если пользователь спрашивает "зачем через вас" / "в чём выгода" / "почему не напрямую" → answer_value_proposition.
6. Если тип должника (ЮЛ/ФЛ) неизвестен и нужен для ответа → ask_debtor_type.
7. Если пользователь жалуется на повторы → handle_repetition.
8. Если пользователь говорит что ответили не на тот вопрос → handle_confusion.
9. Если пользователь прощается / благодарит в конце → close_dialog.
10. Светские разговоры → small_talk.

Возвращай ТОЛЬКО JSON по OUTPUT_SCHEMA. Никакого текста вне JSON. Никакого markdown.""".strip()


def _build_planner_prompt(
    user_message: str,
    recent_dialog: list[dict],
    known_slots: dict,
    available_actions: list[str],
    candidate_intents: list[str],
    active_scenario: str | None,
    last_bot_offer: str | None,
    kb_facts_summary: str,
    required_next_step: str | None,
) -> str:
    dialog_lines: list[str] = []
    for m in (recent_dialog or [])[-6:]:
        role = m.get("role", "")
        text = (m.get("text") or "")[:200]
        dialog_lines.append(f"{role}: {text}")

    # Expose only non-private, non-empty slots (hide internal _* slots)
    public_slots = {
        k: v for k, v in known_slots.items()
        if v and not k.startswith("_") and k not in ("ready_to_open",)
    }

    payload = {
        "user_message": user_message,
        "recent_dialog": dialog_lines,
        "active_scenario": active_scenario or "unknown",
        "known_slots": public_slots,
        "candidate_intents": candidate_intents,
        "last_bot_offer": (last_bot_offer or "")[:200],
        "required_next_step": required_next_step or "",
        "available_actions": available_actions,
        "kb_facts_summary": kb_facts_summary,
    }
    ctx = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"<DIALOG_CONTEXT>\n{ctx}\n</DIALOG_CONTEXT>\n\n"
        f"<OUTPUT_SCHEMA>\n{_PLANNER_SCHEMA}\n</OUTPUT_SCHEMA>\n\n"
        "Выбери одно действие из available_actions. Верни только JSON."
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_RESULT_KEYS = frozenset({"action", "confidence", "reason"})


def _extract_planner_json(raw: str) -> dict:
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

    raise ValueError(f"no valid planner JSON in: {raw[:200]!r}")


def _normalize(obj: dict) -> dict:
    obj.setdefault("action", "list_partner_banks")
    obj.setdefault("confidence", 0.5)
    obj.setdefault("reason", "")
    obj.setdefault("missing_slots", [])
    sw = obj.get("scenario_switch")
    if not isinstance(sw, dict):
        obj["scenario_switch"] = {"needed": False, "to": None}
    else:
        sw.setdefault("needed", False)
        sw.setdefault("to", None)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_llm_planner(
    user_text: str,
    recent_dialog: list[dict],
    slots: dict,
    available_actions: list[str],
    candidate_intents: list[str],
    active_scenario: str | None,
    kb_facts: list[dict],
    trace_ctx: dict | None = None,
) -> dict:
    """Call LLM planner and return normalized action selection dict.

    On any failure, returns a safe fallback (first available action).
    """
    from app.llm.providers import ask_llm
    from app.config import settings

    # Summarize KB facts briefly for the planner
    kb_parts: list[str] = []
    for f in (kb_facts or [])[:5]:
        if isinstance(f, dict):
            text = (f.get("text") or f.get("content") or "")[:200]
            if text:
                kb_parts.append(text)
    kb_summary = "\n".join(kb_parts) if kb_parts else ""

    last_bot_offer = (
        slots.get("_last_bot_question")
        or slots.get("_last_answer_summary")
        or ""
    )
    required_next_step = slots.get("_required_next_step") or ""

    prompt = _build_planner_prompt(
        user_message=user_text,
        recent_dialog=recent_dialog,
        known_slots=slots,
        available_actions=available_actions,
        candidate_intents=candidate_intents,
        active_scenario=active_scenario,
        last_bot_offer=last_bot_offer,
        kb_facts_summary=kb_summary,
        required_next_step=required_next_step,
    )

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    tctx = dict(trace_ctx or {})
    tctx["phase"] = "planner"

    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").lower()
    max_tokens = (
        settings.OLLAMA_ANALYZER_MAX_TOKENS if provider == "ollama"
        else settings.TIMEWEB_ANALYZER_MAX_TOKENS
    )

    try:
        raw = await ask_llm(messages, max_tokens=max_tokens, trace_ctx=tctx)
        result = _extract_planner_json(raw)
        result = _normalize(result)

        # Clamp action to available list
        if available_actions and result["action"] not in available_actions:
            logger.warning(
                "LLMPlanner | action=%s not in available_actions — using fallback",
                result["action"],
            )
            result["action"] = available_actions[0]

        logger.info(
            "PlannerCall | trace_id=%s | available_actions=%s | selected_action=%s"
            " | confidence=%.2f | reason=%s",
            tctx.get("trace_id", ""),
            available_actions,
            result["action"],
            result.get("confidence", 0.0),
            (result.get("reason") or "")[:120],
        )
        return result

    except Exception as exc:
        logger.warning("LLMPlanner failed (%s) — fallback to first available action", exc)
        fallback = available_actions[0] if available_actions else "list_partner_banks"
        return _normalize({"action": fallback, "reason": f"planner_error:{exc}"})
