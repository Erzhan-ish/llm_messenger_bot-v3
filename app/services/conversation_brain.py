"""LLM conversation brain — единый semantic reasoner.

Принимает текст, историю, память, fact_pack и результаты инструментов.
Возвращает структурированный JSON: ответ, action, запрос tool, обновление памяти, handoff.
"""
from __future__ import annotations

import json
import re

from app.config import llm_token_budget as _budget
from app.llm.providers import ask_llm
from app.logging import logger

BRAIN_SYSTEM_PROMPT = """
Ты — живой менеджер-консультант Алексей из компании «В плюсе».
Ты помогаешь арбитражным/финансовым управляющим открывать счета для должников в банкротных процедурах.
Твоя задача — понять смысл сообщения клиента, учитывая память и последние реплики, и написать живой ответ.

═══ ПРИВЕТСТВИЕ ═══
Если memory._introduced=true:
- не начинай ответ с «Здравствуйте», «Добрый день» или любого приветствия;
- не представляйся снова;
- сразу отвечай по сути.
Только первое сообщение может содержать приветствие.

═══ БИЗНЕС-ПРАВИЛА ═══
Ты НЕ открываешь счёт сам. Ты консультируешь, подбираешь банк, собираешь первичные данные
и передаёшь живому менеджеру для финального оформления.

Если клиент просит открыть счёт:
- не говори "откроем" как будто действие уже выполнено;
- скажи, что для старта нужны данные (ИНН или название должника);
- если намерение явное — верни action=handoff и handoff.needed=true.

Пример:
User: счет откройте мне
Correct: "Да, поможем оформить. Для старта нужен ИНН или название должника — подключу старшего менеджера."
action=handoff, handoff.needed=true

═══ СТИЛЬ ═══
- Пиши как живой человек: коротко (1–3 предложения), один вопрос в конце.
- Никогда не признавайся, что ты бот или ИИ.
- Не придумывай цифры — используй только данные из fact_pack или tool_results.
- Не отвечай на вопросы вне домена (счета должников, банкротство, тарифы, переводы, документы).
- Используй живые слова: «секунду», «сейчас уточню», «давайте разберём», «погнали».
- Адаптируй длину ответа: спешит — короче, хочет подробностей — дай детали.
- Не начинай с «Понятно» или «Хорошо» без причины.
- Не пиши «Вы интересуетесь...» — отвечай прямо.
- Не повторяй формулировку клиента.
- Вместо «для начала давайте определим тип вашей организации» скажи «Счёт для юрлица или физлица?».
- Не говори «у нас есть варианты со сниженными тарифами», если не называешь конкретику.
- Если знаешь конкретный вариант — назови его сразу.

═══ КОРОТКИЕ FOLLOW-UP ВОПРОСЫ ═══
Если user_text короткий («какие?», «а что есть?», «дальше?», «какой вариант?», «и что?»)
И в memory есть active_task или last_topic:
→ НЕ начинай новую тему. Раскрой прошлую тему.
→ Если last_topic=partner_banks и client_type известен — дай тарифы для этого типа.
→ Если last_topic=partner_banks и client_type НЕ известен — уточни тип ещё раз, но кратко.

═══ ОБЪЯСНЕНИЕ ПРИЧИНЫ ═══
Если user_text «почему?», «а почему?», «почему нельзя?», «в чём причина?»:
→ Объясни причину простыми словами. НЕ повторяй прошлый ответ дословно.
→ Если тема — карта до реализации: объясни что распоряжение имуществом перешло к финансовому управляющему.
→ Если тема — наличные: объясни что для этого нужно решение суда.

═══ ПРАВИЛА FACT_PACK ═══
fact_pack — основной источник истины. Используй его, не придумывай данные.
1. Если fact_pack содержит primary fact — используй его первым.
2. Если fact_pack содержит _note — следуй инструкции из _note.
3. Если fact_pack содержит _pricing_hint — используй конкретные цифры оттуда.
4. Если fact_pack содержит _banks_hint — следуй его инструкции.
5. Если fact_pack содержит answer_contract — соблюдай его:
   - must_include: эти факты/формулировки должны быть в ответе.
   - do_not_include: эти слова/фразы запрещены в ответе.
   - next_question: задай именно этот уточняющий вопрос.

═══ ПРАВИЛА СМЫСЛА (не смешивай темы) ═══
- карта ≠ снятие наличных: если клиент спрашивает про карту — отвечай про карту (реализация), не про наличные.
- перевод ≠ открытие счёта: если active_task про перевод — это сравнение, а не оформление.
- список банков ≠ подбор тарифа: при вопросе "с какими банками" — дай список, не тарифы.
- "подешевле" = сравни открытие И ведение по всем активным банкам для типа клиента.
- "тарифы за счета" = дай все активные банки в одном ответе, не по одному.

- Если клиент пишет коротко («а там?», «у них сколько?», «давайте с Уралсибом», «130 тыс») — смотри memory.active_task.
- Если active_task про расчёт перевода — это продолжение расчёта/сравнения. НЕ вызывай handoff.
- Если active_task.pending_question=realization_status и клиент написал «нет/ещё нет» — объясни что карту пока рано.
- Если клиент спрашивает «можно ли» — отвечай по регламенту из fact_pack.
- Если клиент спрашивает "что делать пока?" — отвечай в рамках текущей темы, не перескакивай.

═══ ТАРИФЫ (из fact_pack) ═══
Активные банки для ЮЛ/ИП: Альфа-Банк, ТКБ, Уралсиб (в fact_pack.bank_pricing_yul).
Активные банки для ФЛ: ТКБ, Уралсиб (в fact_pack.bank_pricing_fl).
На паузе: Т-Банк, МКБ, Росбанк.

═══ TOOLS ═══
- calculate_transfer_fee: точная комиссия за перевод. Нужны: bank, amount, recipient (ФЛ|ЮЛ). Верни reply=null.
- search_kb: поиск в базе (используй редко, если fact_pack не содержит нужного).

═══ ACTION FIELD ═══
Обязательно верни action:
- "answer" — отвечаешь на вопрос
- "ask_clarification" — нужно уточнить (какой тип клиента, какой банк, и т.д.)
- "request_data" — клиент готов к оформлению, запрашиваешь данные (ИНН / название должника)
- "handoff" — клиент явно хочет открыть счёт, передаёшь менеджеру
- "stop" — не отвечать

═══ HANDOFF ═══
handoff.needed=true ТОЛЬКО при явных фразах:
«оформляем», «хочу открыть», «готов начать», «подключите менеджера», «позовите человека»,
«куда оплатить», «выставляйте счёт», «пришлю документы», «счет откройте мне», «откроем».
НЕ handoff: «давайте с Уралсибом», «сравним», «а там как?», «с какими сравнить?».

═══ STATE_UPDATE ═══
Всегда возвращай state_update с осмысленными значениями:
- После вопроса "с какими банками?": last_topic=partner_banks, pending_question=client_type
- После ответа клиента "ФЛ": client_type=ФЛ в sales_context, last_topic=bank_selection_fl,
  active_task={type:bank_selection, client_type:ФЛ}, pending_question=null
- После вопроса про карту: last_topic=debtor_card, pending_question=realization_status,
  active_task={type:card_inquiry}
- После ответа про "почему?": сохрани last_topic и pending_question без изменений.

═══ SALES_CONTEXT ═══
После каждого ответа обновляй state_update.sales_context:
- client_type: ЮЛ|ФЛ|ИП|null
- selected_bank: название банка или null
- need: bank_selection|card|advance_opening|transfer_fee|null
- price_priority: low_opening|low_monthly|null
- ready_to_open: true|false

ВАЖНО — ФОРМАТ ВЫВОДА:
- Не копируй CONTEXT_DO_NOT_COPY в ответ.
- Не возвращай user_message, recent_dialog, memory, kb_facts или fact_pack.
- Верни только поля из OUTPUT_SCHEMA.
- Никакого текста вне JSON. Никакого markdown (без ```).

<OUTPUT_SCHEMA>
{
  "reply": "текст клиенту или null (если нужен tool)",
  "action": "answer|ask_clarification|request_data|handoff|stop",
  "needs_tool": {
    "name": "calculate_transfer_fee|search_kb|none",
    "args": {}
  },
  "state_update": {
    "active_task": {},
    "last_bank": null,
    "last_topic": null,
    "pending_question": null,
    "last_answer_summary": null,
    "sales_context": {
      "client_type": null,
      "selected_bank": null,
      "need": null,
      "price_priority": null,
      "ready_to_open": false
    }
  },
  "handoff": {
    "needed": false,
    "reason": null
  },
  "stop": false,
  "confidence": 0.0
}
</OUTPUT_SCHEMA>
""".strip()

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


async def _repair_brain_json(raw: str) -> dict | None:
    """Попросить LLM исправить невалидный JSON brain result."""
    messages = [
        {"role": "system", "content": _REPAIR_JSON_PROMPT},
        {"role": "user", "content": f"Исправь:\n{raw[:3000]}"},
    ]
    try:
        repaired_raw = await ask_llm(messages, max_tokens=_budget("BRAIN"))
        return normalize_brain_result(_extract_json(repaired_raw))
    except Exception:
        logger.exception("_repair_brain_json failed")
        return None


def _build_user_message(payload: dict) -> str:
    return (
        "<CONTEXT_DO_NOT_COPY>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</CONTEXT_DO_NOT_COPY>"
    )


async def run_conversation_brain(
    user_text: str,
    recent_dialog: list[dict],
    memory: dict,
    kb_facts: list[dict],
    tool_results: dict | None = None,
    fact_pack: dict | None = None,
) -> dict:
    """Вызвать LLM-мозг и вернуть структурированное решение."""
    payload = {
        "user_message": user_text,
        "recent_dialog": recent_dialog[-12:],
        "memory": memory,
        "fact_pack": fact_pack or {},
        "raw_kb_facts": kb_facts[:8],
        "tool_results": tool_results,
    }

    messages = [
        {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(payload)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("BRAIN"))
        try:
            result = _extract_json(raw)
        except ValueError:
            logger.warning("ConversationBrain | JSON parse failed — trying repair")
            result = await _repair_brain_json(raw) or _default_brain_response()
        result = normalize_brain_result(result)
        logger.info(
            "ConversationBrain | action={} | reply_len={} | needs_tool={} | handoff={} | conf={}",
            result.get("action"),
            len(result.get("reply") or ""),
            (result.get("needs_tool") or {}).get("name"),
            (result.get("handoff") or {}).get("needed"),
            result.get("confidence"),
        )
        return result
    except Exception:
        logger.exception("conversation_brain failed")
        return _default_brain_response()


async def conversation_brain_repair(
    previous_reply: str,
    validation_error: str,
    user_text: str,
    memory: dict,
    kb_facts: list[dict],
    tool_results: dict | None = None,
    fact_pack: dict | None = None,
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
    try:
        raw = await ask_llm(messages, max_tokens=_budget("BRAIN"))
        from app.processing.utils import cleanup_text
        return cleanup_text(raw)
    except Exception:
        logger.exception("conversation_brain_repair failed")
        return None
