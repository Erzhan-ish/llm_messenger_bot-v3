from app.outbound.telegram import send_telegram
# from app.outbound.whatsapp import send_whatsapp


async def send_message(channel: str, user_id: str, text: str):
    if channel == "telegram":
        await send_telegram(user_id, text)
        return

    if channel == "whatsapp":
        # await send_whatsapp(...)
        raise NotImplementedError

    raise ValueError(f"Unknown channel: {channel}")

from app.delivery.telegram_sender import send_telegram_message
from app.logging import logger


class OutboundDispatcher:
    @staticmethod
    async def send_telegram(
        user_id: str,
        text: str,
    ):
        logger.info(
            "Outbound send | channel=telegram | user_id={}",
            user_id,
        )

        await send_telegram_message(
            user_id=user_id,
            text=text,
        )