import httpx
from app.config import settings
from app.logging import logger

WHATSAPP_API_URL = (
    f"https://graph.facebook.com/v19.0/"
    f"{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
)


def _normalize_user_id(user_id: str) -> str:
    """
    WhatsApp Cloud API принимает номер ТОЛЬКО в формате:
    79991234567 (только цифры, без +)
    """
    user_id = user_id.strip()
    if user_id.startswith("+"):
        user_id = user_id[1:]

    if not user_id.isdigit():
        raise ValueError("WhatsApp user_id must contain digits only")

    return user_id


async def send_whatsapp_text(user_id: str, text: str):
    """
    Отправка обычного текста (ТОЛЬКО в 24h окне)
    """
    user_id = _normalize_user_id(user_id)

    payload = {
        "messaging_product": "whatsapp",
        "to": user_id,
        "type": "text",
        "text": {
            "body": text,
            "preview_url": False,
        },
    }

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            WHATSAPP_API_URL,
            json=payload,
            headers=headers,
        )

    if r.status_code not in (200, 201):
        logger.error(
            "WhatsApp text send failed | status={} | body={}",
            r.status_code,
            r.text,
        )
        raise RuntimeError("WhatsApp text send failed")

    data = r.json()
    meta_id = data.get("messages", [{}])[0].get("id")

    logger.info(
        "WhatsApp text sent | user_id={} | meta_id={}",
        user_id,
        meta_id,
    )


async def send_whatsapp_template(
    user_id: str,
    template_name: str,
    params: dict | None = None,
):
    """
    Отправка template message (ПОСЛЕ 24h)
    """
    user_id = _normalize_user_id(user_id)

    components = []

    if params:
        components.append(
            {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": str(v)}
                    for v in params.values()
                ],
            }
        )

    payload = {
        "messaging_product": "whatsapp",
        "to": user_id,
        "type": "template",
        "template": {
            "name": template_name,
            # ⚠️ ДОЛЖНО совпадать с языком,
            # в котором шаблон одобрен в Meta
            "language": {
                "code": settings.WHATSAPP_TEMPLATE_LANG,
            },
            "components": components,
        },
    }

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            WHATSAPP_API_URL,
            json=payload,
            headers=headers,
        )

    if r.status_code not in (200, 201):
        logger.error(
            "WhatsApp template send failed | status={} | body={}",
            r.status_code,
            r.text,
        )
        raise RuntimeError("WhatsApp template send failed")

    data = r.json()
    meta_id = data.get("messages", [{}])[0].get("id")

    logger.info(
        "WhatsApp template sent | user_id={} | template={} | meta_id={}",
        user_id,
        template_name,
        meta_id,
    )
