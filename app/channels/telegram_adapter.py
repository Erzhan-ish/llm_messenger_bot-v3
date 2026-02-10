# app/channels/telegram_adapter.py
from datetime import datetime
from app.channels.base import UnifiedMessage
from app.logging import logger


class TelegramAdapter:
    @staticmethod
    async def from_payload(payload: dict) -> UnifiedMessage:
        """
        Используется webhook'ом Telegram (raw JSON)
        """
        message = payload.get("message")

        if not message:
            raise ValueError("Telegram payload without message")

        # имитируем объект с .message
        class Update:
            pass

        class Msg:
            pass

        update = Update()
        msg = Msg()

        msg.message_id = message["message_id"]
        msg.text = message.get("text")
        msg.chat = type("Chat", (), {"id": message["chat"]["id"]})

        update.message = msg

        return await TelegramAdapter.to_unified(update)

    @staticmethod
    async def to_unified(update) -> UnifiedMessage:
        msg = update.message

        return UnifiedMessage(
            channel="telegram",
            external_user_id=str(msg.chat.id),
            message_id=str(msg.message_id),
            message_type="text",
            text=msg.text,
            created_at=datetime.utcnow(),
        )

    @staticmethod
    def build_start_message(external_user_id: str, payload: str | None) -> UnifiedMessage:
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

        return UnifiedMessage(
            channel="telegram",
            external_user_id=external_user_id,
            message_id=f"start:{payload or 'direct'}",
            message_type="text",
            text=text,
            created_at=datetime.utcnow(),
        )
