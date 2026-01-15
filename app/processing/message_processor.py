from app.context.session_manager import get_or_create_session, reset_session
from app.context.context_builder import build_context
from app.llm.providers.ollama import ask_ollama
from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.processing.state_detector import detect_state
from app.storage.repositories.messages_repo import save_message
from app.storage.repositories.sessions_repo import update_session_status
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import OutboundDispatcher
from app.services.transcription_service import transcribe_audio
from app.logging import logger

from app.llm.providers import ask_llm

async def process_message(message):
    if isinstance(message, dict):
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)
    # 0️⃣ Dedup
    if await is_duplicate_message(
        channel=message.channel,
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

    system_prompt = build_manager_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]

    if history_context:
        messages.append({
            "role": "system",
            "content": f"История диалога:\n{history_context}",
        })

    messages.append({
        "role": "user",
        "content": message.text,
    })

    # 6️⃣ LLM
    try:
        reply = await ask_llm(messages)
    except Exception:
        logger.exception("LLM failed")
        reply = "Я на связи. Давайте продолжим чуть позже."

    # 7️⃣ Save outbound
    await save_message(
        session_id=session.id,
        role="bot",
        text=reply,
        channel=message.channel,
    )

    # 8️⃣ State
    state = detect_state(message.text)
    await update_session_status(session.id, state.value)

    # 9️⃣ Outbound (ЕДИНАЯ ТОЧКА)
    await OutboundDispatcher.send(
        channel=message.channel,
        external_user_id=message.external_user_id,
        text=reply,
    )

    logger.info(
        "Reply sent | user_id={} | session_id={} | state={}",
        message.external_user_id,
        session.id,
        state.value,
    )
