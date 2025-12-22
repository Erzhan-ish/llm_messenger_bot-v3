from app.context.session_manager import get_or_create_session
from app.context.context_builder import build_context
from app.llm.providers.ollama import ask_ollama
from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.processing.state_detector import detect_state
from app.storage.repositories.messages_repo import save_message
from app.storage.repositories.sessions_repo import update_session_status
from app.processing.dedup import is_duplicate_message
from app.logging import logger
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import send_message
from app.context.session_manager import reset_session
from app.delivery.telegram_sender import send_telegram_message


async def process_message(message):
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

    try:
        await check_rate_limit(
            channel=message.channel,
            external_user_id=message.user_id,
            limit=5,
            window_seconds=10,
        )
    except RateLimitExceeded as e:
        logger.warning(
            "Rate limit hit | user_id={} | channel={}",
            message.user_id,
            message.channel,
        )
        return


    if message.text and message.text.strip() == "/reset":
        await reset_session(
            channel=message.channel,
            external_user_id=message.user_id,
        )

        await send_telegram_message(
            user_id=message.user_id,
            text="Контекст диалога сброшен. Начнём заново.",
        )

        logger.info(
            "Session reset | user_id={} | channel={}",
            message.user_id,
            message.channel,
        )
        return


    logger.info(
        "Processing message | user_id={} | message_id={}",
        message.user_id,
        message.message_id,
    )

    # 1️⃣ user + session
    session = await get_or_create_session(
        channel="telegram",
        external_user_id=message.user_id,
    )

    # 2️⃣ сохраняем входящее
    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel="telegram",
        external_message_id=message.message_id,
    )

    # 3️⃣ контекст истории
    history_context = await build_context(session.id)

    # 4️⃣ system prompt менеджера
    system_prompt = build_manager_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]

    if history_context:
        messages.append({
            "role": "system",
            "content": f"История диалога:\n{history_context}"
        })

    messages.append({
        "role": "user",
        "content": message.text
    })

    # 5️⃣ Ollama
    reply = await ask_ollama(messages)

    # 6️⃣ сохраняем ответ бота
    await save_message(
        session_id=session.id,
        role="bot",
        text=reply,
        channel="telegram",
    )

    # 7️⃣ состояние по сообщению пользователя
    state = detect_state(message.text)
    await update_session_status(session.id, state.value)

    # 8️⃣ отправка
    await send_message(
        channel=message.channel,
        user_id=message.user_id,
        text=reply,
    )

    logger.info(
        "Reply sent | user_id={} | session_id={} | state={}",
        message.user_id,
        session.id,
        state.value,
    )
