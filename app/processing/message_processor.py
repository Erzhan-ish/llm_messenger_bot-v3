from app.context.session_manager import get_or_create_session
from app.context.context_builder import build_context
from app.llm.providers.ollama import ask_ollama
from app.processing.state_detector import detect_state
from app.delivery.telegram_sender import send_telegram_message
from app.logging import logger


async def process_message(message):
    logger.info(
        "Processing message | user_id={} | message_id={}",
        message.user_id,
        message.message_id,
    )

    session = await get_or_create_session(int(message.user_id))
    context = await build_context(session.id)

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": message.text},
    ]

    reply = await ask_ollama(messages)

    state = detect_state(reply)

    await send_telegram_message(
        user_id=message.user_id,
        text=reply,
    )

    logger.info(
        "Reply sent | user_id={} | state={}",
        message.user_id,
        state,
    )
