from app.context.session_manager import get_or_create_session, reset_session
from app.context.context_builder import build_context
from app.knowledge_base.service import get_kb_snippets
from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.processing.state_detector import detect_state, DialogState
from app.storage.repositories.messages_repo import save_message
from app.storage.repositories.sessions_repo import update_session_status,  set_dialog_state, set_negative_handled
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import OutboundDispatcher
from app.services.transcription_service import transcribe_audio
from app.logging import logger
import random
import datetime
from app.services.client_need_detector import detect_client_need
from app.storage.repositories.sessions_repo import set_client_need
from app.storage.repositories.messages_repo import get_messages_by_session
from app.escalation.service import is_ready_for_escalation, escalate_to_manager


from app.storage.repositories.sessions_repo import (
    update_session_status,
    set_dialog_state,
    mark_escalated,
)
from app.escalation.service import (
    is_ready_for_escalation,
    escalate_to_manager
)
from app.llm.providers import ask_llm

manager_nickname = "Алексей"
scenario = "INBOUND_QUESTION"

AGGRESSIVE_REPLIES = [
    "Понял. Прекращаю общение.",
    "Сообщения прекращаю.",
    "Принял. Больше писать не буду.",
    "Хорошо, прекращаю контакт.",
    "Общение завершено.",
]


NEGATIVE_REPLIES = [
    "Понял, больше не беспокою. Если понадобится — напишите.",
    "Хорошо, не буду писать. Обращайтесь при необходимости.",
    "Принял, прекращаю сообщения. Если появятся вопросы — на связи.",
]

def apply_intro_once(session, reply: str) -> str:
    if session.dialog_state == "new":
        return f"Это {manager_nickname}.\n\n" + reply
    return reply


async def process_message(message):
    if isinstance(message, dict):
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)
    # 0️⃣ Dedup
    if await is_duplicate_message(
            channel=message.channel,
            external_user_id=message.external_user_id,
            external_message_id=message.message_id,
    ):
        logger.warning(
            "Duplicate message ignored | channel={} | message_id={}",
            message.channel,
            message.message_id,
        )
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
        logger.warning(
            "Rate limit hit | user_id={} | channel={}",
            message.external_user_id,
            message.channel,
        )
        return

    # 2️⃣ Команда /reset
    if message.text and message.text.strip() == "/reset":
        await reset_session(
            channel=message.channel,
            external_user_id=message.external_user_id,
        )

        await OutboundDispatcher.send(
            channel=message.channel,
            external_user_id=message.external_user_id,
            text="Контекст диалога сброшен. Начнём заново.",
        )

        logger.info(
            "Session reset | user_id={} | channel={}",
            message.external_user_id,
            message.channel,
        )
        return

    logger.info(
        "Processing message | user_id={} | message_id={}",
        message.external_user_id,
        message.message_id,
    )

    # 3️⃣ Session
    session = await get_or_create_session(
        channel=message.channel,
        external_user_id=message.external_user_id,
    )

    if message.message_type == "audio" and not (message.text and message.text.strip()):
        if not message.audio_path:
            logger.error("Audio message without audio_path | msg_id={}", message.message_id)
            return

        try:
            stt_text = await transcribe_audio(message.audio_path)
        except Exception:
            logger.exception("STT failed | msg_id={}", message.message_id)
            stt_text = None

        if not stt_text:
            reply = "Не смог распознать голосовое. Напишите, пожалуйста, текстом."
            await OutboundDispatcher.send(channel=message.channel, external_user_id=message.external_user_id, text=reply)
            return

        message.text = stt_text

    # 4️⃣ Save inbound
    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    # 5️⃣ Context
    history_context = await build_context(session.id)

    # СНАЧАЛА KB
    kb_snippets = get_kb_snippets(message.text, top_k=5) or ""

    system_prompt = build_manager_system_prompt().replace(
        "{MANAGER_NICKNAME}", manager_nickname
    )

    user_payload_parts = []
    user_payload_parts.append(f"SCENARIO={scenario}")
    user_payload_parts.append(f"MANAGER_NICKNAME={manager_nickname}")

    if kb_snippets:
        user_payload_parts.append("KB:\n" + kb_snippets)

    if history_context:
        user_payload_parts.append("История диалога:\n" + history_context)

    user_payload_parts.append("Текущее сообщение клиента:\n" + (message.text or ""))

    # ПОСЛЕ добавления KB — собираем payload
    user_payload = "\n\n".join(user_payload_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    state = detect_state(message.text)

    if state == DialogState.NEGATIVE:
        if not bool(session.negative_handled):
            reply = random.choice(NEGATIVE_REPLIES)

            await save_message(
                session_id=session.id,
                role="bot",
                text=reply,
                channel=message.channel
            )

            await OutboundDispatcher.send(
                channel=message.channel,
                external_user_id=message.external_user_id,
                text=reply,
            )
            await set_negative_handled(session.id, True)
            session.negative_handled = True
        return


    if state == DialogState.AGGRESSIVE:
        if not bool(session.negative_handled):
            reply = random.choice(AGGRESSIVE_REPLIES)

            await save_message(
                session_id=session.id,
                role="bot",
                text=reply,
                channel=message.channel
            )

            await OutboundDispatcher.send(
                channel=message.channel,
                external_user_id=message.external_user_id,
                text=reply,
            )
            await set_negative_handled(session.id, True)
            session.negative_handled = True
        return


   # 6️⃣ LLM
    try:
        reply = await ask_llm(messages)
    except Exception:
        logger.exception("LLM failed")
        reply = "Я на связи. Давайте продолжим чуть позже."

    # 🔹 Представление — ДО сохранения
    reply = apply_intro_once(session, reply)

    # 🔹 Если представились — обновляем dialog_state
    if session.dialog_state == "new":
        await set_dialog_state(session.id, "introduced")
        session.dialog_state = "introduced"  # ВАЖНО: обновить объект

    # 7️⃣ Save outbound
    await save_message(
        session_id=session.id,
        role="bot",
        text=reply,
        channel=message.channel,
    )

    # 🔹 Попытка определить потребность клиента
    if session.client_need is None:
        messages = await get_messages_by_session(session.id)
        dialog_text = "\n".join(
            f"{m['role']}: {m['text']}" for m in messages if m["text"]
        )

        need = await detect_client_need(dialog_text)

        if need != "UNKNOWN":
            await set_client_need(session.id, need)
            session.client_need = need

    # 8️⃣ State (бизнес-состояние)
    state = detect_state(message.text)
    await update_session_status(session.id, state.value)

    # 9️⃣ Outbound
    await OutboundDispatcher.send(
        channel=message.channel,
        external_user_id=message.external_user_id,
        text=reply,
    )

    # 🔟 ЭСКАЛАЦИЯ
    if is_ready_for_escalation(session):
        await escalate_to_manager(session)

        await mark_escalated(session.id)
        session.escalated_at = datetime.utcnow()