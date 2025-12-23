from app.outbound.dispatcher import OutboundDispatcher
from app.storage.repositories.messages_repo import mark_followup_sent


async def send_followup(
    telegram_user_id: str,
    original_message_id: int,
):
    text = (
        "Коллега, добрый день!\n\n"
        "Понимаю, что могли быть заняты.\n"
        "Подскажите, актуально ли сейчас рассмотреть процедуру?"
    )

    await OutboundDispatcher.send_telegram(
        user_id=telegram_user_id,
        text=text,
    )

    await mark_followup_sent(original_message_id)
