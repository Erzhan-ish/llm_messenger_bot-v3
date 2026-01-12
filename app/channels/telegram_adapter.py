from datetime import datetime
from app.channels.base import UnifiedMessage
from app.processing.message_processor import process_message
from app.logging import logger


class TelegramAdapter:
    @staticmethod
    async def handle(update):
        msg = update.message

        unified = UnifiedMessage(
            channel="telegram",
            external_user_id=str(msg.from_.id),
            message_id=str(msg.message_id),
            message_type="text",
            text=msg.text,
            audio_path=None,
            created_at=datetime.utcnow(),
        )

        await process_message(unified)


    @staticmethod
    async def handle_start(external_user_id: str, payload: str | None):
        """
        payload — то, что пришло после /start
        например: invite_12345
        """

        logger.info(
            "Telegram /start | external_user_id={} | payload={}",
            external_user_id,
            payload,
        )

        text = (
            "Здравствуйте!\n"
            "Я помощник по работе с банковскими счетами в процедурах банкротства.\n\n"
            "Могу задать несколько вопросов и посмотреть, "
            "можем ли быть полезны друг другу."
        )

        unified = UnifiedMessage(
            channel="telegram",
            external_user_id=external_user_id,
            message_id=f"start:{payload or 'direct'}",
            message_type="text",
            text=text,
            audio_path=None,
            created_at=datetime.utcnow(),
        )

        await process_message(unified)



