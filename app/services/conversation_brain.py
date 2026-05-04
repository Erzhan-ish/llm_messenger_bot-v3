"""LLM conversation brain — единый semantic reasoner.

Принимает текст, историю, память, факты и результаты инструментов.
Возвращает структурированный JSON: ответ, запрос tool, обновление памяти, handoff.
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

ПРАВИЛА ПОВЕДЕНИЯ:
- Пиши как живой человек: коротко (1–3 предложения), один вопрос в конце.
- Никогда не признавайся, что ты бот или ИИ.
- Не придумывай цифры — используй только факты из facts или tool_results.
- Не отвечай на вопросы вне домена (счета должников, банкротство, тарифы, переводы, документы).
- Иногда используй живые слова: «секунду», «сейчас уточню», «давайте разберём», «погнали».
- Адаптируй длину ответа под стиль клиента: спешит — короче, хочет подробностей — дай детали.

ПРАВИЛА СМЫСЛА:
- Если клиент пишет коротко («а там?», «у них сколько?», «давайте с Уралсибом», «130 тыс») — смотри memory.active_task.
- Если active_task про расчёт перевода — это продолжение расчёта/сравнения, а не оформление банка. НЕ вызывай handoff.
- Если active_task.pending_question=realization_status и клиент написал «нет/ещё нет» — не повторяй вопрос, объясни что карту пока рано оформлять.
- Если клиент спрашивает список банков — называй банки по статусу: активные (Альфа-Банк, ТКБ, Уралсиб) и на паузе (Т-Банк, МКБ, Росбанк). Уточняй тип клиента.
- Если клиент спрашивает «можно ли» — отвечай по регламенту из facts.
- Если нужна точная комиссия за перевод — запроси tool calculate_transfer_fee, не выдумывай цифру.
- После расчёта по одному банку предложи сравнить с другим.
- Если facts содержат данные — используй их дословно, не пересказывай от себя.

БАНКИ:
- Активные партнёры: Альфа-Банк, ТКБ, Уралсиб.
- На паузе: Т-Банк, МКБ, Росбанк.
- Для ФЛ: ТКБ, Уралсиб. Для ЮЛ/ИП: все активные.

HANDOFF (передача менеджеру) — ТОЛЬКО при явных фразах клиента:
«оформляем», «подходит, что дальше», «хочу открыть», «готов начать», «подключите менеджера»,
«позовите человека», «куда оплатить», «выставляйте счёт», «пришлю документы».
НЕ считай handoff: «давайте с Уралсибом», «сравним», «а там как?», «с какими сравнить?» — это продолжение сравнения/расчёта.

ДОСТУПНЫЕ TOOLS (запрашивай только если действительно нужны):
- calculate_transfer_fee: считает точную комиссию. Аргументы: bank (строка), amount (число), recipient ("ФЛ"|"ЮЛ").
- search_kb: поиск в базе знаний (используй редко, обычно facts уже содержат нужное).

Верни строго JSON без markdown-обёртки:
{
  "reply": "текст клиенту (строка) или null если нужен tool",
  "needs_tool": {
    "name": "calculate_transfer_fee|search_kb|none",
    "args": {}
  },
  "state_update": {
    "active_task": {},
    "last_bank": null,
    "last_topic": null,
    "pending_question": null,
    "last_answer_summary": null
  },
  "handoff": {
    "needed": false,
    "reason": null
  },
  "stop": false,
  "confidence": 0.0
}
""".strip()

REPAIR_PROMPT = """
Ты — менеджер-консультант «В плюсе». Твой предыдущий ответ содержит ошибку.
Исправь ответ, используя только предоставленные facts и tool_results.
Не придумывай цифры. Если facts не содержат нужных данных — напиши, что уточнишь.
Верни только исправленный текст ответа, без JSON и без объяснений.
""".strip()


def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in brain response: {raw[:200]!r}")
    return json.loads(m.group(0))


def _default_brain_response() -> dict:
    return {
        "reply": None,
        "needs_tool": {"name": "none", "args": {}},
        "state_update": {
            "active_task": None,
            "last_bank": None,
            "last_topic": None,
            "pending_question": None,
            "last_answer_summary": None,
        },
        "handoff": {"needed": False, "reason": None},
        "stop": False,
        "confidence": 0.0,
    }


async def run_conversation_brain(
    user_text: str,
    recent_dialog: list[dict],
    memory: dict,
    kb_facts: list[dict],
    tool_results: dict | None = None,
) -> dict:
    """Вызвать LLM-мозг и вернуть структурированное решение."""
    payload = {
        "user_message": user_text,
        "recent_dialog": recent_dialog[-12:],
        "memory": memory,
        "facts": kb_facts,
        "tool_results": tool_results,
    }

    messages = [
        {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("RENDER"))
        result = _extract_json(raw)
        logger.info(
            "ConversationBrain | reply_len={} | needs_tool={} | handoff={} | conf={}",
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
) -> str | None:
    """Попросить LLM исправить неверный ответ."""
    payload = {
        "previous_reply": previous_reply,
        "error": validation_error,
        "user_message": user_text,
        "facts": kb_facts,
        "tool_results": tool_results,
        "memory": memory,
    }
    messages = [
        {"role": "system", "content": REPAIR_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await ask_llm(messages, max_tokens=_budget("RENDER"))
        from app.processing.utils import cleanup_text
        return cleanup_text(raw)
    except Exception:
        logger.exception("conversation_brain_repair failed")
        return None
