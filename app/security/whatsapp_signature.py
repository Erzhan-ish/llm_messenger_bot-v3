import hmac
import hashlib
from typing import Optional

from app.config import settings


def verify_whatsapp_signature(
    raw_body: bytes,
    header_value: Optional[str],
) -> bool:
    """
    Meta присылает X-Hub-Signature-256: "sha256=<hex>"
    Считаем HMAC-SHA256 по raw_body, ключ = META_APP_SECRET,
    сравниваем с тем, что в заголовке.
    """
    if not settings.META_APP_SECRET:
        return False

    if not header_value:
        return False

    prefix = "sha256="
    if not header_value.startswith(prefix):
        return False

    received = header_value[len(prefix):].strip()
    computed = hmac.new(
        key=settings.META_APP_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, received)
