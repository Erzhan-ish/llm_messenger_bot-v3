from app.outbound.whatsapp import send_whatsapp
from app.outbound.telegram import send_telegram, send_telegram_typing
from app.outbound.wazzup import send_wazzup
from app.logging import logger
from app.config import settings
from app.outbound.wazzup import send_wazzup, send_wazzup_typing

class OutboundDispatcher:
    @staticmethod
    async def send(channel: str, external_user_id: str, text: str):
        logger.info("Outbound dispatch | channel={} | user_id={}", channel, external_user_id)

        if settings.OUTBOUND_PROVIDER == "stub":
            logger.info("Outbound STUB | channel={} | user_id={} | text={}", channel, external_user_id, text)
            return

        if channel == "telegram":
            await send_telegram(external_user_id, text)
            return

        if channel == "whatsapp":
            await send_whatsapp(external_user_id=external_user_id, text=text)
            return

        if channel == "wazzup":
            await send_wazzup(external_user_id=external_user_id, text=text)
            return

        if channel == "telegram_personal":
            from app.outbound.pyrogram_telegram import send_pyrogram_telegram
            await send_pyrogram_telegram(external_user_id, text)
            return

        raise ValueError(f"Unknown channel: {channel}")

    @staticmethod
    async def send_typing(channel: str, external_user_id: str) -> None:
        if settings.OUTBOUND_PROVIDER == "stub":
            return

        if channel == "telegram":
            await send_telegram_typing(external_user_id)
            return

        if channel == "wazzup":
            await send_wazzup_typing(external_user_id)
            return

        if channel == "telegram_personal":
            from app.outbound.pyrogram_telegram import send_pyrogram_typing
            await send_pyrogram_typing(external_user_id)
            return

        return