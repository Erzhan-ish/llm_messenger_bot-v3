from app.outbound.dispatcher import OutboundDispatcher
from app.storage.repositories.messages_repo import mark_followup_sent


FOLLOWUP_TEXT = (
    "Коллега, добрый день. "
    "Ранее обсуждали возможность работы по процедурам. "
    "Подскажите, актуально сейчас?"
)


async def send_followup(user):
    await OutboundDispatcher.send(
        channel=user.channel,
        external_user_id=user.external_user_id,
        text=FOLLOWUP_TEXT,
    )

    await mark_followup_sent(user.id)
