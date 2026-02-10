# app/processing/message_processor.py
from __future__ import annotations

import os
import random
import re
from typing import Optional

from app.context.session_manager import get_or_create_session, reset_session
from app.processing.state_detector import detect_state, DialogState
from app.storage.repositories.messages_repo import save_message, get_messages_by_session
from app.storage.repositories.sessions_repo import (
    set_negative_handled,
    get_slots,
    set_slots,
    set_dialog_state,
    mark_escalated,
    set_client_need,
    get_client_need,
)
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import OutboundDispatcher
from app.services.transcription_service import transcribe_audio
from app.logging import logger

from app.services.client_need_detector import detect_client_need
from app.escalation.service import escalate_to_manager

from app.processing.slots import DEFAULT_SLOTS, extract_slots
from app.processing.slot_questions import QUESTIONS
from app.processing.triggers import (
    AGGRESSIVE_REPLIES,
    NEGATIVE_REPLIES,
    END_DIALOG_PHRASES,
    SHORT_NEUTRAL,
    TRIGGERS,
)
from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.context.context_builder import build_context
from app.knowledge_base.service import get_kb_snippets
from app.llm.providers import ask_llm


manager_nickname = "Алексей"
scenario = "INBOUND_QUESTION"


# ----------------------------
# Intent / mode
# ----------------------------
def detect_onboarding_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in TRIGGERS)


def normalize_mode(slots: dict, text: str) -> str:
    mode = slots.get("_mode") or "INFO"
    if mode != "ONBOARDING" and detect_onboarding_intent(text):
        mode = "ONBOARDING"
    return mode


# ----------------------------
# Slot parsing helpers
# ----------------------------
def parse_documents_ready(text: str) -> Optional[bool]:
    t = (text or "").strip().lower()
    yes = {"да", "есть", "готовы", "готово", "имеются", "имеется", "собраны", "собрал"}
    no = {"нет", "не готовы", "не готово", "не готов", "пока нет", "ещё нет", "еще нет"}
    if t in yes:
        return True
    if t in no:
        return False
    return None


def is_timing_question(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in ["как долго", "сколько", "срок", "за сколько", "в течение"])


def next_missing_slot_for_onboarding(slots: dict) -> Optional[str]:
    required = ["account_type", "debtor_type", "procedure_type", "documents_ready"]
    for k in required:
        if slots.get(k) is None:
            return k
    return None


def is_ready_for_escalation(slots: dict) -> bool:
    required = ["account_type", "debtor_type", "procedure_type", "documents_ready"]
    return all(slots.get(k) is not None for k in required)


# ----------------------------
# Safety: forbid email/docs sending + forbid questions from LLM
# ----------------------------
EMAIL_OR_SEND_RE = re.compile(
    r"(?is)\b("
    r"какая\s+почта|какой\s+email|контактн(ая|ую)\s+почт|электронн(ая|ую)\s+почт|"
    r"на(ш|шу)\s+почт|на\s+email|на\s+e-?mail|"
    r"пришл(ите|и)|скин(ьте|ь)|отправ(ьте|ь)|перешл(ите|и)|прилож(ите|и)"
    r")\b"
)

DOC_LIST_ASK_RE = re.compile(
    r"(?is)\b("
    r"какие\s+документ(ы|ов)|какие\s+именно\s+документ(ы|ов)|переч(исл|ень)\s+документ|"
    r"укажите.*документ|что\s+за\s+документ"
    r")\b"
)

TIMING_SENT_RE = re.compile(r"(?is)\b(3\s*[–-]\s*5|3-5)\s+рабоч(их|ие)\s+дн")


def sanitize_llm_reply(reply: str, user_text: str) -> str:
    """
    1) удаляем предложения про почту/отправку
    2) удаляем вопросы "какие документы"
    3) удаляем повтор про сроки, если пользователь не спрашивал про сроки
    """
    if not reply:
        return "Принял."

    parts = re.split(r"(?<=[.!?])\s+", reply.strip())
    kept = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue

        if EMAIL_OR_SEND_RE.search(s):
            continue
        if DOC_LIST_ASK_RE.search(s):
            continue
        if (not is_timing_question(user_text)) and TIMING_SENT_RE.search(s):
            continue

        kept.append(s)

    out = " ".join(kept).strip()
    return out or "Принял."


def strip_questions(text: str) -> str:
    """
    Полностью удаляем любые предложения с '?'.
    Вопросы пользователю задаёт только код через QUESTIONS.
    """
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [p.strip() for p in parts if p.strip() and "?" not in p]
    return " ".join(kept).strip()


# ----------------------------
# Send helper: intro once reliably
# ----------------------------
async def send_bot(session, channel: str, external_user_id: str, text: str, slots: dict) -> dict:
    # интро сохраняем в slots, это надежнее, чем session.dialog_state
    if not slots.get("_introduced"):
        text = f"Это {manager_nickname}.\n\n" + (text or "")
        slots["_introduced"] = True
        await set_slots(session.id, slots)

    await save_message(session.id, "bot", text, channel)
    await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=text)
    return slots


async def process_message(message):
    print("RUNNING message_processor FROM:", __file__, "PID:", os.getpid())

    if isinstance(message, dict):
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)

    # 0️⃣ Dedup
    if await is_duplicate_message(
        channel=message.channel,
        external_user_id=message.external_user_id,
        external_message_id=message.message_id,
    ):
        return

    # 1️⃣ Rate limit
    try:
        await check_rate_limit(
            channel=message.channel,
            external_user_id=message.external_user_id,
            limit=5,
            window_seconds=10,
        )
    except RateLimitExceeded:
        return

    # 2️⃣ /reset
    if message.text and message.text.strip() == "/reset":
        await reset_session(message.channel, message.external_user_id)
        await OutboundDispatcher.send(
            channel=message.channel,
            external_user_id=message.external_user_id,
            text="Контекст диалога сброшен. Начнём заново.",
        )
        return

    # 3️⃣ Session
    session = await get_or_create_session(
        channel=message.channel,
        external_user_id=message.external_user_id,
    )

    # 4️⃣ Audio → text
    if message.message_type == "audio" and not message.text:
        try:
            message.text = await transcribe_audio(message.audio_path)
        except Exception:
            slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
            slots["_mode"] = slots.get("_mode") or "INFO"
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Не получилось распознать голосовое. Напишите текстом.",
                slots,
            )
            return

    # 5️⃣ Save inbound
    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    user_text = message.text or ""
    text_norm = user_text.strip().lower()

    # 6️⃣ Завершение диалога
    if text_norm in END_DIALOG_PHRASES:
        slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
        await send_bot(
            session,
            message.channel,
            message.external_user_id,
            "Тогда остановимся. Если появятся вопросы — напишите.",
            slots,
        )
        return

    # 7️⃣ Негатив / агрессия
    if text_norm in SHORT_NEUTRAL:
        state = DialogState.IN_PROGRESS
    else:
        state = detect_state(user_text)

    if state in (DialogState.NEGATIVE, DialogState.AGGRESSIVE):
        if not session.negative_handled:
            slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
            reply = random.choice(NEGATIVE_REPLIES if state == DialogState.NEGATIVE else AGGRESSIVE_REPLIES)
            await send_bot(session, message.channel, message.external_user_id, reply, slots)
            await set_negative_handled(session.id, True)
        return

    # 8️⃣ Slots + mode
    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots = extract_slots(user_text, slots)

    mode = normalize_mode(slots, user_text)
    slots["_mode"] = mode

    # сохраняем documents_ready только в ONBOARDING и только если ещё нет значения
    if mode == "ONBOARDING" and slots.get("documents_ready") is None:
        dr = parse_documents_ready(user_text)
        if dr is not None:
            slots["documents_ready"] = dr

    await set_slots(session.id, slots)

    # 8.1 FAST PATH: сроки в INFO
    if mode == "INFO" and is_timing_question(user_text):
        reply = "Обычно открытие счёта занимает 3–5 рабочих дней после получения всех необходимых данных."
        if slots.get("account_type") is None:
            reply += "\n\n" + QUESTIONS.get("account_type", "Какой счёт вас интересует: основной, задатковый, залоговый или специальный?")
        await send_bot(session, message.channel, message.external_user_id, reply, slots)
        return

    # 9️⃣ ONBOARDING: если всё готово — эскалация СРАЗУ и без LLM
    if mode == "ONBOARDING":
        fresh_slots = await get_slots(session.id) or slots
        if is_ready_for_escalation(fresh_slots) and not getattr(session, "escalated", False):
            logger.info("ESCALATING session_id=%s | slots=%s", session.id, fresh_slots)

            await mark_escalated(session.id)
            try:
                await escalate_to_manager(session.id)
            except Exception:
                logger.exception("escalate_to_manager failed")

            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Информацию зафиксировал. Дальше продолжу по вашему кейсу здесь — если нужно, уточню детали.",
                fresh_slots,
            )
            return

        # иначе спрашиваем следующий слот и выходим (LLM не вызываем, чтобы не было мусора)
        missing = next_missing_slot_for_onboarding(fresh_slots)
        if missing:
            q = QUESTIONS.get(missing, "Уточните, пожалуйста, деталь по вашему запросу.")
            await send_bot(session, message.channel, message.external_user_id, q, fresh_slots)
            return

    # 10️⃣ INFO: здесь можно использовать LLM для “человечности”, но без вопросов
    history_context = await build_context(session.id)
    kb_snippets = get_kb_snippets(user_text, top_k=5) or ""

    system_prompt = build_manager_system_prompt().replace("{MANAGER_NICKNAME}", manager_nickname)

    user_payload = [
        f"SCENARIO={scenario}",
        f"MANAGER_NICKNAME={manager_nickname}",
    ]
    if kb_snippets:
        user_payload.append("KB:\n" + kb_snippets)
    if history_context:
        user_payload.append("История диалога:\n" + history_context)
    user_payload.append("Сообщение клиента:\n" + user_text)

    user_payload.append(
        """
Ответь по сути сообщения клиента как менеджер.

ЖЁСТКО запрещено:
- спрашивать или упоминать email/почту;
- просить прислать/скинуть/отправить что-либо;
- спрашивать "какие документы" / просить перечислить документы;
- задавать любые вопросы клиенту (вопрос задаст код).

Если клиент спрашивает про сроки — ответь сроком.
"""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_payload)},
    ]

    try:
        llm_reply = await ask_llm(messages)
    except Exception:
        llm_reply = "Принял."

    reply = sanitize_llm_reply(llm_reply, user_text)
    reply = strip_questions(reply)  # убираем любые вопросы от LLM

    # если в INFO мы всё ещё не знаем тип счёта — задаём 1 вопрос кодом
    if slots.get("account_type") is None:
        q = QUESTIONS.get("account_type")
        if q:
            reply = (reply.rstrip() + "\n\n" + q).strip()

    await send_bot(session, message.channel, message.external_user_id, reply, slots)

    # 11️⃣ Определение потребности (фоном)
    if not await get_client_need(session.id):
        msgs = await get_messages_by_session(session.id)
        dialog_text = "\n".join(f"{m['role']}: {m['text']}" for m in msgs if m["text"])
        need = await detect_client_need(dialog_text)
        if need != "UNKNOWN":
            await set_client_need(session.id, need)
