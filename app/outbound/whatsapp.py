from datetime import datetime, timedelta
from app.logging import logger
from app.storage.repositories.sessions_repo import get_last_inbound_time
from app.delivery.whatsapp_sender import send_whatsapp_text, send_whatsapp_template

WHATSAPP_24H = timedelta(hours=24)

async def send_whatsapp(external_user_id: str, text: str) -> None:
    last_inbound = await get_last_inbound_time(
        channel="whatsapp",
        external_user_id=external_user_id,
    )

    now = datetime.utcnow()

    # 24h окно → можно свободный текст
    if last_inbound and (now - last_inbound) <= WHATSAPP_24H:
        logger.info("WhatsApp outbound | free window | user_id={}", external_user_id)
        await send_whatsapp_text(user_id=external_user_id, text=text)
        return

    # вне окна → только шаблон (pre-approved template)
    logger.info("WhatsApp outbound | template required | user_id={}", external_user_id)
    await send_whatsapp_template(
        user_id=external_user_id,
        template_name="followup_generic",
        # params только если реально есть переменные в шаблоне
        params=None,
    )
