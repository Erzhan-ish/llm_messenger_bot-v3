from app.outbound.whatsapp import send_whatsapp
from app.outbound.telegram import send_telegram
from app.logging import logger


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

        if channel == "telegram":
            await send_telegram(external_user_id, text)
            return

        if channel == "whatsapp":
            await send_whatsapp(
                channel=channel,
                external_user_id=external_user_id,
                text=text,
            )
            return

        raise ValueError(f"Unknown channel: {channel}")

