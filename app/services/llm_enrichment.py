"""LLM enrichment helpers — lightweight contextual calls for specific scenarios.

Three use-cases, each with its own fallback to static behaviour:
  1. llm_extract_missing_slots  — fills client_type / bank_name from complex sentences regex misses
  2. llm_objection_reply        — generates a contextual objection response (replaces static dict)
  3. llm_ack_reply              — generates a natural 1-sentence continuation after "хм" / short ACK
"""
from __future__ import annotations

import json
import re
from typing import Optional

from app.llm.providers import ask_llm
from app.logging import logger


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw.strip())
    except Exception:
        m = _JSON_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# 1. Slot extraction (client_type + bank_name)
# ---------------------------------------------------------------------------

_SLOT_PROMPT = """
Из сообщения клиента извлеки два поля. Верни ТОЛЬКО JSON, без пояснений.

"client_type": "ФЛ" | "ЮЛ" | "ИП" | null
  ФЛ — физическое лицо: должник-физик, частное лицо, "физик", "человек" (как должник), банкрот-физик,
       "у кого списывают долги", "сопровождаю человека", "должник — человек"
  ЮЛ — ООО, АО, НКО, организация, компания, юридическое лицо
  ИП — индивидуальный предприниматель, ИП
  ВАЖНО: тип определяется по ДОЛЖНИКУ, а не по тому, кто пишет.
         Арбитражный управляющий (АУ) сопровождает должника — смотри тип должника.

"bank_name": "Альфа-Банк" | "ТКБ" | "Уралсиб" | "МКБ" | "Росбанк" | "Т-Банк" | null
  Только если банк явно упомянут. Синонимы: "альфа"→"Альфа-Банк", "тинькофф"→"Т-Банк", "транскапитал"→"ТКБ".

Если поле не определяется — null.

Примеры:
Сообщение: "Я ИП, хочу открыть счёт"
{"client_type": "ИП", "bank_name": null}

Сообщение: "Открываем счёт для ООО в Альфе"
{"client_type": "ЮЛ", "bank_name": "Альфа-Банк"}

Сообщение: "Сопровождаю человека, у которого списывают долги через суд"
{"client_type": "ФЛ", "bank_name": null}

Сообщение: "Должник — физлицо, банкрот, нужен счёт"
{"client_type": "ФЛ", "bank_name": null}

Сообщение: "У клиента ООО, хотим в ТКБ"
{"client_type": "ЮЛ", "bank_name": "ТКБ"}

Формат: {"client_type": "ФЛ", "bank_name": null}
""".strip()

_CT_NORM = {
    "фл": "ФЛ", "ФЛ": "ФЛ",
    "юл": "ЮЛ", "ЮЛ": "ЮЛ",
    "ип": "ИП", "ИП": "ИП",
    "физическое лицо": "ФЛ",
    "юридическое лицо": "ЮЛ",
    "индивидуальный предприниматель": "ИП",
}
_BANK_NORM = {"Альфа-Банк", "ТКБ", "Уралсиб", "МКБ", "Росбанк", "Т-Банк"}


async def llm_extract_missing_slots(user_text: str, slots: dict) -> None:
    """
    LLM-based slot extraction for complex sentences that regex misses.
    Mutates slots in-place. Only fills fields that are currently empty.
    Called only when client_type is missing and message has substantive content.
    """
    try:
        raw = await ask_llm(
            [{"role": "system", "content": _SLOT_PROMPT},
             {"role": "user",   "content": user_text}],
            max_tokens=40,
        )
        data = _parse_json(raw or "")
        if not isinstance(data, dict):
            return

        if not slots.get("client_type"):
            ct_raw = str(data.get("client_type") or "").strip()
            ct = _CT_NORM.get(ct_raw) or _CT_NORM.get(ct_raw.lower())
            if ct:
                slots["client_type"] = ct
                logger.info("LLM slot extraction: client_type={}", ct)

        if not slots.get("bank_name"):
            bn = str(data.get("bank_name") or "").strip()
            if bn in _BANK_NORM:
                slots["bank_name"] = bn
                logger.info("LLM slot extraction: bank_name={}", bn)

    except Exception:
        logger.debug("llm_extract_missing_slots failed (ignored)")


# ---------------------------------------------------------------------------
# 2. Objection reply
# ---------------------------------------------------------------------------

_OBJECTION_LABELS: dict[str, str] = {
    "expensive":   "цена кажется высокой",
    "no_visit":    "клиент спрашивает, нужно ли ехать в банк",
    "think":       "клиент говорит «надо подумать» / не готов сейчас",
    "competitor":  "клиент говорит что у конкурентов дешевле",
    "unsure":      "клиент сомневается или не уверен",
}

_OBJECTION_PROMPT = """
Ты — Алексей, менеджер «В плюсе». Отвечаешь на возражение клиента.
Пиши живо, по-человечески, 1–2 предложения. Конкретно, без «конечно», «отлично», «понял».
Не предлагай ссылки, звонки или прайс-листы. Без списков.
""".strip()


async def llm_objection_reply(
    objection_type: str,
    user_text: str,
    slots: dict,
) -> str:
    """
    Generate a contextual objection reply using LLM.
    Falls back to static _objection_reply on failure.
    """
    bank        = slots.get("_last_bank") or slots.get("bank_name") or ""
    client_type = slots.get("client_type") or ""
    last_items  = slots.get("_last_items") or []
    last_bot    = (slots.get("_last_bot_text") or "").strip()
    obj_label   = _OBJECTION_LABELS.get(objection_type, objection_type)

    pricing_parts = []
    for i in last_items:
        label = i.get("label", "")
        val   = i.get("value", "")
        if label and val:
            if val.strip() in ("0 руб./мес.", "0 руб.", "0"):
                val = "бесплатно"
            pricing_parts.append(f"{label} {val}")
    pricing_str = ", ".join(pricing_parts)

    ctx_lines = []
    if client_type:
        ctx_lines.append(f"Тип клиента: {client_type}")
    if bank:
        ctx_lines.append(f"Банк: {bank}")
    if pricing_str:
        ctx_lines.append(f"Тарифы: {pricing_str}")
    if last_bot:
        ctx_lines.append(f"Твой предыдущий ответ: «{last_bot[:200]}»")
    ctx_lines.append(f"Возражение клиента: «{user_text}»")
    ctx_lines.append(f"Тип возражения: {obj_label}")

    try:
        raw = await ask_llm(
            [{"role": "system", "content": _OBJECTION_PROMPT},
             {"role": "user",   "content": "\n".join(ctx_lines)}],
            max_tokens=120,
        )
        from app.processing.utils import cleanup_text
        text = cleanup_text(raw or "").strip()
        if text and len(text) > 10:
            logger.info("LLM objection reply: type={}", objection_type)
            return text
    except Exception:
        logger.debug("llm_objection_reply failed (ignored)")

    # Fallback to static
    from app.processing.plan_builder import _objection_reply
    return _objection_reply(objection_type, seed=user_text)


# ---------------------------------------------------------------------------
# 3. ACK contextual reply
# ---------------------------------------------------------------------------

_ACK_PROMPT = """
Ты — Алексей, менеджер «В плюсе». Клиент произнёс короткую нейтральную реакцию после твоего ответа.
Напиши 1 короткое живое продолжение разговора — без повтора предыдущего ответа.
Без «Конечно», «Отлично», «Хорошо», «Понял». Без оценочных слов.
""".strip()


async def llm_ack_reply(user_text: str, slots: dict) -> Optional[str]:
    """
    Generate a contextual 1-sentence continuation for ACK/хм/мм when there's prior context.
    Returns None if context is insufficient or LLM fails — caller uses static fallback.
    """
    last_bot = (slots.get("_last_bot_text") or "").strip()
    if not last_bot or len(last_bot) < 20:
        return None

    client_type = slots.get("client_type") or ""
    bank        = slots.get("_last_bank") or slots.get("bank_name") or ""

    ctx_lines = [f"Твой предыдущий ответ: «{last_bot[:280]}»"]
    if client_type:
        ctx_lines.append(f"Тип клиента: {client_type}")
    if bank:
        ctx_lines.append(f"Банк: {bank}")
    ctx_lines.append(f"Реакция клиента: «{user_text}»")

    try:
        raw = await ask_llm(
            [{"role": "system", "content": _ACK_PROMPT},
             {"role": "user",   "content": "\n".join(ctx_lines)}],
            max_tokens=80,
        )
        from app.processing.utils import cleanup_text
        text = cleanup_text(raw or "").strip()
        if text and len(text) > 5:
            logger.info("LLM ack reply generated")
            return text
    except Exception:
        logger.debug("llm_ack_reply failed (ignored)")

    return None
