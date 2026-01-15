from app.outbound.whatsapp import send_whatsapp
from app.outbound.telegram import send_telegram
from app.logging import logger
from app.config import settings

class OutboundDispatcher:
    @staticmethod
    async def send(
        channel: str,
        external_user_id: str,
        text: str,
    ):
        logger.info(
            "Outbound dispatch | channel={} | user_id={}",
            channel,
            external_user_id,
        )

        # Локальный тест: ничего не отправляем наружу, только лог
        if settings.OUTBOUND_PROVIDER == "stub":
            logger.info("Outbound STUB | channel={} | user_id={} | text={}", channel, external_user_id, text)
            return

        if channel == "telegram":
            await send_telegram(external_user_id, text)
            return

        if channel == "whatsapp":
            await send_whatsapp(
                external_user_id=external_user_id,
                text=text,
            )
            return

        raise ValueError(f"Unknown channel: {channel}")

