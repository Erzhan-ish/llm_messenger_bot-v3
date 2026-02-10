# app/processing/message_processor.py
from __future__ import annotations

import os
import random
from typing import Optional

from app.context.session_manager import get_or_create_session, reset_session
from app.processing.state_detector import detect_state, DialogState
from app.storage.repositories.messages_repo import save_message, get_messages_by_session
from app.storage.repositories.sessions_repo import (
    set_negative_handled,
    get_slots,
    set_slots,
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

manager_nickname = "Алексей"


# ----------------------------
# Helpers
# ----------------------------
def detect_onboarding_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in TRIGGERS)


def normalize_mode(slots: dict, text: str) -> str:
    mode = slots.get("_mode") or "INFO"
    if mode != "ONBOARDING" and detect_onboarding_intent(text):
        mode = "ONBOARDING"
    return mode


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
    # строго то, что нужно для эскалации
    required = ["account_type", "debtor_type", "procedure_type", "documents_ready"]
    for k in required:
        if slots.get(k) is None:
            return k
    return None


def is_ready_for_escalation(slots: dict) -> bool:
    required = ["account_type", "debtor_type", "procedure_type", "documents_ready"]
    return all(slots.get(k) is not None for k in required)


async def send_bot(session, channel: str, external_user_id: str, text: str, slots: dict) -> dict:
    """
    1) Интро храним в slots["_introduced"] — это надежнее, чем session.dialog_state
    2) Всегда сохраняем bot message
    """
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
            await send_bot(session, message.channel, message.external_user_id, "Не получилось распознать голосовое. Напишите текстом.", slots)
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

    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots = extract_slots(user_text, slots)

    # режим
    mode = normalize_mode(slots, user_text)
    slots["_mode"] = mode

    # если в ONBOARDING и ждём documents_ready — парсим "да/нет"
    if mode == "ONBOARDING" and slots.get("documents_ready") is None:
        dr = parse_documents_ready(user_text)
        if dr is not None:
            slots["documents_ready"] = dr

    await set_slots(session.id, slots)

    # 6️⃣ Завершение диалога
    if text_norm in END_DIALOG_PHRASES:
        await send_bot(session, message.channel, message.external_user_id, "Тогда остановимся. Если появятся вопросы — напишите.", slots)
        return

    # 7️⃣ Негатив / агрессия
    if text_norm in SHORT_NEUTRAL:
        state = DialogState.IN_PROGRESS
    else:
        state = detect_state(user_text)

    if state in (DialogState.NEGATIVE, DialogState.AGGRESSIVE):
        if not session.negative_handled:
            reply = random.choice(NEGATIVE_REPLIES if state == DialogState.NEGATIVE else AGGRESSIVE_REPLIES)
            await send_bot(session, message.channel, message.external_user_id, reply, slots)
            await set_negative_handled(session.id, True)
        return

    # ============================
    # INFO: консультация
    # ============================
    if mode == "INFO":
        # если вопрос про сроки — отвечаем сразу и не дёргаем LLM
        if is_timing_question(user_text):
            reply = "Обычно открытие счёта занимает 3–5 рабочих дней после получения всех необходимых данных."
            if slots.get("account_type") is None:
                reply += "\n\n" + QUESTIONS.get("account_type", "Какой счёт вас интересует: основной, задатковый, залоговый или специальный?")
            await send_bot(session, message.channel, message.external_user_id, reply, slots)
            return

        # если человек просто назвал тип счета — уточняем намерение (не зацикливаемся)
        if slots.get("account_type") is not None and not detect_onboarding_intent(user_text):
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Понял. Вы хотите просто уточнить условия/сроки или планируете открыть счёт?",
                slots,
            )
            return

        # иначе — спросим тип счета (как “снятие неопределенности”)
        if slots.get("account_type") is None:
            await send_bot(session, message.channel, message.external_user_id, QUESTIONS.get("account_type"), slots)
            return

        # fallback
        await send_bot(session, message.channel, message.external_user_id, "Принял. Уточню детали.", slots)
        return

    # ============================
    # ONBOARDING: сбор слотов без LLM
    # (никаких почт, никаких “какие документы”, никакой самодеятельности)
    # ============================
    missing = next_missing_slot_for_onboarding(slots)
    if missing:
        q = QUESTIONS.get(missing, "Уточните, пожалуйста, деталь по вашему запросу.")
        await send_bot(session, message.channel, message.external_user_id, q, slots)
        return

    # всё собрано -> эскалация
    if is_ready_for_escalation(slots) and not getattr(session, "escalated", False):
        logger.info("ESCALATING session_id=%s | slots=%s", session.id, slots)

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
            slots,
        )
        return

    # safety fallback
    await send_bot(session, message.channel, message.external_user_id, "Принял. Продолжаю.", slots)

    # 13️⃣ Определение потребности (фоном)
    if not await get_client_need(session.id):
        msgs = await get_messages_by_session(session.id)
        dialog_text = "\n".join(f"{m['role']}: {m['text']}" for m in msgs if m["text"])
        need = await detect_client_need(dialog_text)
        if need != "UNKNOWN":
            await set_client_need(session.id, need)
