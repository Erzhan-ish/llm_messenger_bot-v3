from datetime import datetime
from app.channels.base import UnifiedMessage
from app.processing.message_processor import process_message

class TelegramAdapter:
    async def handle(self, update):
        msg = update.message

        unified = UnifiedMessage(
            channel="telegram",
            user_id=str(msg.from_.id),
            message_id=str(msg.message_id),
            text=msg.text,
            created_at=datetime.utcnow()
        )

        await process_message(unified)


