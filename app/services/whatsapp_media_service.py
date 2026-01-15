from app.channels.base import UnifiedMessage
from app.logging import logger

from app.services.whatsapp_media import download_whatsapp_media  # см. ниже


async def ensure_audio_downloaded(msg: UnifiedMessage) -> UnifiedMessage:
    if msg.message_type != "audio":
        return msg

    if msg.audio_path:
        return msg

    if msg.channel != "whatsapp":
        # Telegram: ожидаем, что adapter сам выставит audio_path (или будет отдельный telegram_media)
        logger.warning("Audio without audio_path | channel={} | msg_id={}", msg.channel, msg.message_id)
        return msg

    if not getattr(msg, "media_id", None):
        logger.warning("WhatsApp audio without media_id | msg_id={}", msg.message_id)
        return msg

    audio_path = await download_whatsapp_media(media_id=msg.media_id)
    msg.audio_path = audio_path
    return msg
