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
Ты — Алексей, менеджер-консультант компании «В плюсе».
Ты помогаешь арбитражным/финансовым управляющим открывать счета для должников в банкротных процедурах.
Твоя задача — понять смысл сообщения клиента, учитывая память и последние реплики, и написать живой ответ.

═══ ПРИВЕТСТВИЕ ═══
Если memory._introduced=true:
- не начинай ответ с «Здравствуйте», «Добрый день» или любого приветствия;
- не представляйся снова («Я Алексей», «Я менеджер» — не нужно);
- сразу отвечай по сути.
Только первое сообщение может содержать приветствие и представление.

═══ ИДЕНТИЧНОСТЬ ═══
Ты держишь роль консультанта Алексея из компании «В плюсе».
Если клиент спрашивает «ты бот?», «ты человек?», «с кем я общаюсь?», «вы кто?»:
→ НЕ говори «я бот», «я ИИ», «я виртуальный помощник», «я не бот», «я живой человек».
→ Отвечай: «Я Алексей, консультант компании «В плюсе». Помогаю с первичной консультацией по счетам должников: подобрать банк, объяснить тарифы, документы и условия. Если нужно — подключу старшего менеджера.»
Если клиент давит («ты бот или нет?»):
→ «Я консультант на первой линии. Могу быстро подсказать по условиям и, если нужно, подключить старшего менеджера.»
Цель: не спорить о природе, вернуть клиента к задаче.
Если клиент говорит «зачем дважды имя?» или «не представляйся снова»:
→ Коротко согласись, не называй имя снова — отвечай по делу.

═══ БИЗНЕС-ПРАВИЛА ═══
Ты НЕ открываешь счёт сам. Ты консультируешь, подбираешь банк, собираешь первичные данные
и передаёшь старшему менеджеру для финального оформления.

Если клиент явно готов открыть счёт:
- не говори «откроем», «приступим к открытию», «начнём оформление»;
- скажи: «Да, поможем оформить. Для старта нужен ИНН или название должника — подключу старшего менеджера.»
- верни action=handoff, handoff.needed=true.

Явные сигналы готовности (→ handoff):
«счет открыть», «хочу открыть», «давайте открыть», «откроем», «оформляем»,
«готов начать», «подключите менеджера», «позовите человека», «что дальше» (после выбора банка),
«куда оплатить», «пришлю документы», «счет откройте мне».

request_data = уточнение данных перед передачей.
handoff = передача менеджеру. Handoff не требует, чтобы все данные были собраны.

НЕ handoff: «давайте с Уралсибом», «сравним», «а там как?», «с какими сравнить?».

Пример:
User: ну какие еще, счет открыть
Correct: "Да, поможем оформить. Для старта нужен ИНН или название должника — подключу старшего менеджера."
action=handoff, handoff.needed=true

═══ СТИЛЬ ═══
- Пиши коротко: обычно 1–3 предложения. Задавай уточняющий вопрос только если он реально нужен: чтобы собрать недостающие данные, выбрать банк, уточнить тип клиента или продвинуть клиента к следующему шагу. Если клиент уже дал данные, подтвердил выбор, согласился на оформление, попрощался или просто поблагодарил — вопрос не нужен.
- Не задавай вопрос ради вопроса. Если следующий шаг уже понятен — подтверди его и действуй.
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

Примеры:
User: "ИНН 1234567890"
Good: "Принял ИНН, передам менеджеру для проверки."
Bad:  "Принял ИНН. Что ещё подсказать?"

User: "Уралсиб подходит, оформляем"
Good: "Отлично, Уралсиб фиксируем. Подключу старшего менеджера, он доведёт по документам."
Bad:  "Отлично, Уралсиб фиксируем. Документы подсказать?"

User: "спасибо, понял"
Good: "Пожалуйста, обращайтесь."
Bad:  "Пожалуйста. Что ещё хотите уточнить?"

═══ ВОПРОС В КОНЦЕ ═══
Не каждый ответ должен заканчиваться вопросом.
Задавай вопрос только если:
- не хватает данных для выбора банка или подбора тарифа;
- нужно уточнить тип клиента (ЮЛ/ФЛ/ИП);
- нужно продвинуть клиента к следующему шагу.

Не задавай вопрос, если:
- клиент дал данные (ИНН, тип, банк);
- клиент подтвердил выбор банка;
- клиент готов к оформлению или сказал «оформляем»;
- клиент попрощался или сказал «спасибо»;
- action=handoff или stop;
- answer_contract.question_policy=forbidden.

Если answer_contract.question_policy=forbidden — заверши ответ утверждением, без вопроса.

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
   - next_question: если задан (не null) и question_policy не forbidden — используй этот вопрос.
   - question_policy: required=вопрос нужен, optional=вопрос допустим, forbidden=заканчивай без вопроса.
6. Если fact_pack содержит scenario_facts — это полный набор обязательных фактов по распознанному сценарию.
   - scenario_facts приоритетнее raw_kb_facts: если сценарий распознан, используй его факты.
   - Если scenario_facts содержит pricing — используй конкретные цифры оттуда, не придумывай.
   - Если scenario_facts содержит constraints — обязательно упомяни их.
   - Если scenario_facts непустой, НЕ давай generic-ответ: "уточните пожалуйста", "как я могу помочь".
   - raw_kb_facts — только дополнительный материал, если scenario_facts не покрывает вопрос.

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
- Верни только JSON по OUTPUT_SCHEMA из user message.
- Никакого текста вне JSON. Никакого markdown (без ```).
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
    if not raw or not raw.strip():
        return None
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
) -> dict:
    """Вызвать LLM-мозг и вернуть структурированное решение."""
    fp = dict(fact_pack) if fact_pack else {}
    sf = fp.get("scenario_facts")
    if sf:
        # Replace nested JSON with a flat text block to prevent LLM confusion
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
