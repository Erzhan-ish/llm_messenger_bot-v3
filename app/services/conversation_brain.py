"""LLM conversation brain — единый semantic reasoner.

Принимает текст, историю, память, fact_pack и результаты инструментов.
Возвращает структурированный JSON: ответ, action, запрос tool, обновление памяти, handoff.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.config import llm_token_budget as _budget
from app.config import settings
from app.llm.providers import ask_llm
from app.logging import logger

# ---------------------------------------------------------------------------
# Brain system prompt — loaded from file, cached in memory
# ---------------------------------------------------------------------------

_prompt_cache: dict[str, tuple[str, str]] = {}  # path -> (content, hash_prefix)

# Project root: app/services/conversation_brain.py → parent×3 → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_prompt_path(path_str: str) -> Path:
    """Resolve prompt path relative to project root when not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_brain_prompt() -> tuple[str, str, str]:
    """Load brain system prompt from LLM_SYSTEM_PROMPT_PATH.

    Returns (content, resolved_path_str, sha256[:12]).
    Caches after first load. Raises FileNotFoundError if path is missing.
    Relative paths are anchored to the project root, not CWD.
    """
    path_str = (settings.LLM_SYSTEM_PROMPT_PATH or "").strip()
    if not path_str:
        raise RuntimeError("LLM_SYSTEM_PROMPT_PATH is not set in config")

    resolved = _resolve_prompt_path(path_str)
    cache_key = str(resolved)

    if cache_key in _prompt_cache:
        content, sha = _prompt_cache[cache_key]
        return content, cache_key, sha

    if not resolved.is_file():
        raise FileNotFoundError(f"Brain system prompt not found: {resolved!r}")

    content = resolved.read_text(encoding="utf-8").strip()
    sha = hashlib.sha256(content.encode()).hexdigest()[:12]
    _prompt_cache[cache_key] = (content, sha)
    logger.info("BrainPrompt loaded | path={} | hash={} | chars={}", cache_key, sha, len(content))
    return content, cache_key, sha


# Keep the old name as an alias for any direct references in tests
def _load_brain_prompt() -> tuple[str, str, str]:
    return load_brain_prompt()


# Placeholder so that 'BRAIN_SYSTEM_PROMPT' imports don't break existing tests.
# The real value is loaded lazily via load_brain_prompt().
BRAIN_SYSTEM_PROMPT = ""  # populated on first call to run_conversation_brain



REPAIR_PROMPT = """
Ты — менеджер-консультант «В плюсе». Твой предыдущий ответ содержит ошибку.
Исправь ответ, используя только предоставленные fact_pack и tool_results.
Не придумывай цифры. Если данных нет — напиши, что уточнишь.

Правила исправления:
- Если ошибка promised_action_without_handoff: не пиши "откроем" / "сделаем" — скажи "поможем оформить, нужны данные".
- Если ошибка wrong_topic_fact / repeated_intro: перепиши ответ без запрещённых фраз.
- Если ошибка missing_primary_fact: добавь нужный факт из fact_pack.answer_contract.must_include.
- Если ошибка repeated_intro: убери приветствие, начни сразу по сути.
- Если ошибка did_not_explain_reason: объясни причину простыми словами, не повторяй прошлый ответ.
- Соблюдай fact_pack.answer_contract: do_not_include — эти слова/фразы запрещены.

Верни только исправленный текст ответа, без JSON и без объяснений.
""".strip()


_BRAIN_RESULT_KEYS = frozenset({"reply", "action", "needs_tool", "state_update", "handoff", "stop"})
_INPUT_CONTEXT_KEYS = frozenset({"user_message", "recent_dialog", "memory", "kb_facts", "fact_pack"})

_REPAIR_JSON_PROMPT = (
    "Ты вернул невалидный JSON или скопировал входной контекст.\n"
    "Верни только один JSON-объект по схеме: "
    "{reply, action, needs_tool, state_update, handoff, stop, confidence}.\n"
    "Не копируй user_message, recent_dialog, memory, kb_facts, fact_pack.\n"
    "Никакого markdown, никакого текста вне JSON."
)


def _looks_like_brain_result(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    if not (_BRAIN_RESULT_KEYS & obj.keys()):
        return False
    # Reject input context objects that leaked brain keys
    if _INPUT_CONTEXT_KEYS & obj.keys() and not ("reply" in obj or "action" in obj):
        return False
    return True


def _extract_json(raw: str) -> dict:
    """Устойчивый парсер brain JSON с тремя fallback стратегиями."""
    raw = (raw or "").strip()

    # 1. Direct parse
    try:
        obj = json.loads(raw)
        if _looks_like_brain_result(obj):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Fenced ```json block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if _looks_like_brain_result(obj):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Scan all { positions and collect valid brain-result candidates
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    idx = 0
    while idx < len(raw):
        pos = raw.find("{", idx)
        if pos == -1:
            break
        try:
            obj, _ = decoder.raw_decode(raw, pos)
            if _looks_like_brain_result(obj):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass
        idx = pos + 1

    if candidates:
        # Prefer the object that has reply or action directly
        for c in candidates:
            if "reply" in c or "action" in c:
                return c
        return candidates[0]

    raise ValueError(f"no valid brain JSON: {raw[:200]!r}")


def normalize_brain_result(obj: dict) -> dict:
    """Нормализовать структуру brain result — заполнить все недостающие поля."""
    # handoff: bool → dict
    handoff = obj.get("handoff")
    if isinstance(handoff, bool):
        obj["handoff"] = {"needed": handoff, "reason": None}
    elif not isinstance(handoff, dict):
        obj["handoff"] = {"needed": False, "reason": None}
    else:
        obj["handoff"].setdefault("needed", False)
        obj["handoff"].setdefault("reason", None)

    # needs_tool: None / "none" / bare string → dict
    needs_tool = obj.get("needs_tool")
    if needs_tool is None or needs_tool == "none":
        obj["needs_tool"] = {"name": "none", "args": {}}
    elif isinstance(needs_tool, str):
        obj["needs_tool"] = {"name": needs_tool, "args": {}}
    elif isinstance(needs_tool, dict):
        obj["needs_tool"].setdefault("name", "none")
        obj["needs_tool"].setdefault("args", {})

    obj.setdefault("confidence", 0.0)
    obj.setdefault("action", "answer")
    obj.setdefault("reply", None)
    obj.setdefault("stop", False)
    obj.setdefault("state_update", {})
    return obj


def _default_brain_response() -> dict:
    return normalize_brain_result({
        "reply": None,
        "action": "answer",
        "needs_tool": {"name": "none", "args": {}},
        "state_update": {
            "active_task": None,
            "last_bank": None,
            "last_topic": None,
            "pending_question": None,
            "last_answer_summary": None,
            "sales_context": None,
        },
        "handoff": {"needed": False, "reason": None},
        "stop": False,
        "confidence": 0.0,
    })


async def _repair_brain_json(raw: str, trace_ctx: dict | None = None) -> dict | None:
    """Попросить LLM исправить невалидный JSON brain result."""
    if not raw or not raw.strip():
        return None
    messages = [
        {"role": "system", "content": _REPAIR_JSON_PROMPT},
        {"role": "user", "content": f"Исправь:\n{raw[:3000]}"},
    ]
    tctx = dict(trace_ctx or {})
    tctx["phase"] = "json_repair"
    try:
        repaired_raw = await ask_llm(messages, max_tokens=_budget("BRAIN"), trace_ctx=tctx)
        return normalize_brain_result(_extract_json(repaired_raw))
    except Exception:
        logger.exception("_repair_brain_json failed")
        return None


_OUTPUT_SCHEMA = """{
  "reply": "текст клиенту или null",
  "action": "answer|ask_clarification|request_data|handoff|stop",
  "needs_tool": {"name": "calculate_transfer_fee|search_kb|none", "args": {}},
  "state_update": {
    "active_task": {}, "last_bank": null, "last_topic": null,
    "pending_question": null, "last_answer_summary": null,
    "sales_context": {"client_type": null, "selected_bank": null,
                      "need": null, "price_priority": null, "ready_to_open": false}
  },
  "handoff": {"needed": false, "reason": null},
  "stop": false,
  "confidence": 0.0
}"""


def _build_user_message(payload: dict) -> str:
    ctx = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"<CONTEXT_DO_NOT_COPY>\n{ctx}\n</CONTEXT_DO_NOT_COPY>\n\n"
        f"<OUTPUT_SCHEMA>\n{_OUTPUT_SCHEMA}\n</OUTPUT_SCHEMA>\n\n"
        "Верни только один валидный JSON-объект по OUTPUT_SCHEMA.\n"
        "Не копируй CONTEXT_DO_NOT_COPY.\n"
        "Не возвращай user_message, recent_dialog, memory, kb_facts или fact_pack.\n"
        "Никакого markdown. Никакого текста вне JSON."
    )


def _scenario_facts_to_text(scenario_facts: dict, top_n: int = 2) -> str:
    """Convert scenario_facts to a flat fact list the LLM can reliably use.

    Mistral-nemo:12b returns malformed output when receiving deep nested JSON
    or bracket-header formats. A plain numbered list avoids that.
    """
    if not scenario_facts:
        return ""
    facts: list[str] = []
    for i, (sid, profile) in enumerate(scenario_facts.items()):
        if i >= top_n:
            break
        if not isinstance(profile, dict):
            continue
        pricing = profile.get("pricing") or {}
        for fee_val in pricing.values():
            if isinstance(fee_val, dict) and fee_val.get("fact"):
                facts.append(fee_val["fact"])
        for item in (profile.get("constraints") or [])[:3]:
            facts.append(item)
        for item in (profile.get("availability") or [])[:5]:
            facts.append(item)
        for item in (profile.get("features") or [])[:3]:
            facts.append(item)
        for item in (profile.get("bonus") or [])[:2]:
            facts.append(item)
        for item in (profile.get("answer_hints") or [])[:2]:
            facts.append(item)
    if not facts:
        return ""
    return "Обязательные факты по запросу:\n" + "\n".join(f"- {f}" for f in facts)


async def run_conversation_brain(
    user_text: str,
    recent_dialog: list[dict],
    memory: dict,
    kb_facts: list[dict],
    tool_results: dict | None = None,
    fact_pack: dict | None = None,
    trace_ctx: dict | None = None,
) -> dict:
    """Вызвать LLM-мозг и вернуть структурированное решение."""
    fp = dict(fact_pack) if fact_pack else {}
    sf = fp.get("scenario_facts")
    if sf:
        sf_text = _scenario_facts_to_text(sf)
        if sf_text:
            fp["scenario_facts"] = sf_text
        else:
            fp.pop("scenario_facts", None)
    payload = {
        "user_message": user_text,
        "recent_dialog": recent_dialog[-12:],
        "memory": memory,
        "fact_pack": fp,
        "raw_kb_facts": kb_facts[:8],
        "tool_results": tool_results,
    }

    brain_prompt, prompt_path, prompt_hash = load_brain_prompt()

    provider = (getattr(settings, "LLM_PROVIDER", "stub") or "stub").lower()
    model = settings.TIMEWEB_AI_MODEL if provider == "timeweb" else settings.OLLAMA_MODEL
    scenario_ids = [
        m.get("scenario_id", "?")
        for m in (fp.get("scenario_matches") or [])
    ]

    from app.services.llm_trace import make_trace_id
    tctx = dict(trace_ctx or {})
    if "trace_id" not in tctx:
        tctx["trace_id"] = make_trace_id()
    tctx.setdefault("phase", "brain")
    tctx["fact_pack"] = fp

    logger.info(
        "BrainCall | trace_id={} | provider={} | model={} | prompt={}#{} | "
        "user_len={} | dialog={} | kb_facts={} | scenarios={} | fp_keys={}",
        tctx["trace_id"], provider, model, prompt_path, prompt_hash,
        len(user_text), len(recent_dialog),
        len(kb_facts), scenario_ids,
        sorted(fp.keys()),
    )

    messages = [
        {"role": "system", "content": brain_prompt},
        {"role": "user", "content": _build_user_message(payload)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("BRAIN"), trace_ctx=tctx)
        try:
            result = _extract_json(raw)
        except ValueError:
            logger.warning(
                "ConversationBrain | trace_id={} | JSON parse failed — trying repair",
                tctx["trace_id"],
            )
            result = await _repair_brain_json(raw, trace_ctx=tctx) or _default_brain_response()
        result = normalize_brain_result(result)
        logger.info(
            "ConversationBrain | trace_id={} | action={} | reply_len={} | needs_tool={} | handoff={} | conf={}",
            tctx["trace_id"],
            result.get("action"),
            len(result.get("reply") or ""),
            (result.get("needs_tool") or {}).get("name"),
            (result.get("handoff") or {}).get("needed"),
            result.get("confidence"),
        )
        return result
    except Exception:
        logger.exception("conversation_brain failed | trace_id={}", tctx.get("trace_id"))
        return _default_brain_response()


async def conversation_brain_repair(
    previous_reply: str,
    validation_error: str,
    user_text: str,
    memory: dict,
    kb_facts: list[dict],
    tool_results: dict | None = None,
    fact_pack: dict | None = None,
    trace_ctx: dict | None = None,
) -> str | None:
    """Попросить LLM исправить неверный текст ответа."""
    payload = {
        "previous_reply": previous_reply,
        "error": validation_error,
        "user_message": user_text,
        "fact_pack": fact_pack or {},
        "raw_kb_facts": kb_facts[:6],
        "tool_results": tool_results,
        "memory": memory,
    }
    messages = [
        {"role": "system", "content": REPAIR_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    tctx = dict(trace_ctx or {})
    tctx["phase"] = "repair"
    tctx.setdefault("fact_pack", fact_pack or {})
    try:
        raw = await ask_llm(messages, max_tokens=_budget("BRAIN"), trace_ctx=tctx)
        from app.processing.utils import cleanup_text
        return cleanup_text(raw)
    except Exception:
        logger.exception("conversation_brain_repair failed | trace_id={}", tctx.get("trace_id"))
        return None
