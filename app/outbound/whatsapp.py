from datetime import datetime, timedelta
from app.logging import logger
from app.storage.repositories.sessions_repo import get_last_inbound_time
from app.delivery.whatsapp_sender import (
    send_whatsapp_text,
    send_whatsapp_template,
)

WHATSAPP_24H = timedelta(hours=24)


async def send_whatsapp(
    user_id: str,          # internal user.id
    external_user_id: str, # phone number
    text: str,
):
    last_inbound = await get_last_inbound_time(user_id)

    now = datetime.utcnow()

    # 24h окно
    if last_inbound and now - last_inbound <= WHATSAPP_24H:
        logger.info(
            "WhatsApp outbound | free window | user_id={}",
            user_id,
        )
        await send_whatsapp_text(
            to=external_user_id,
            text=text,
        )
        return

    # вне окна → только template
    logger.info(
        "WhatsApp outbound | template required | user_id={}",
        user_id,
    )

    await send_whatsapp_template(
        to=external_user_id,
        template_name="followup_generic",
        # ⚠️ параметры ТОЛЬКО если шаблон их поддерживает
    )
